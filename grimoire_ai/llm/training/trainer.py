"""Training loop for the GrimoireTransformer.

Design decisions
----------------

Optimizer: AdamW
    ``torch.optim.AdamW`` implements decoupled weight decay (Loshchilov &
    Hutter, 2019), which keeps the weight decay update separate from the
    adaptive learning rate.  This is important for generalisation: the
    original ``Adam + L2`` regularisation conflates weight decay with the
    gradient magnitude, causing large-gradient parameters to receive less
    decay than small-gradient ones.  AdamW fixes this.

    Weight decay is applied only to weight matrices (2-D parameters).
    Bias terms, RMSNorm scale vectors, and the embedding table receive
    zero weight decay — regularising these can harm training stability.

Learning rate schedule: linear warmup + cosine decay
    The learning rate starts at 0, increases linearly to ``peak_lr`` over
    ``warmup_steps`` steps, then follows a cosine curve down to
    ``min_lr = 0.1 × peak_lr``.  The warmup prevents large, noisy
    gradient updates at initialisation when the loss landscape is steep.

Gradient clipping: max norm 1.0
    Clips the global gradient norm to 1.0 before every optimizer step.
    Without clipping, a single bad batch can produce an explosive gradient
    that permanently damages the model weights.  This is non-negotiable
    for from-scratch transformer training.

Gradient accumulation
    The effective batch size is ``batch_size × accumulate_steps``.
    Gradients are accumulated over ``accumulate_steps`` forward/backward
    passes before a single optimizer step.  This simulates a larger batch
    without requiring the GPU/CPU to hold more activations in memory at once.

Mixed precision (fp16 AMP)
    When training on CUDA, ``torch.autocast`` runs the forward pass in
    fp16, which roughly halves memory usage and exploits RTX tensor cores.
    A ``GradScaler`` multiplies the loss by a large scale factor before
    the backward pass to prevent fp16 underflow (very small gradients
    rounding to zero), then unscales before the optimizer step.
    On CPU the scaler is a no-op and the autocast context is skipped.
"""

import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
from grimoire_ai.llm.training.checkpoint import load_checkpoint, save_checkpoint


class Trainer:
    """Manages the full training loop for a ``GrimoireTransformer``.

    Instantiate with a model and dataset, then call ``train()``.  The
    trainer handles device placement, the optimizer, LR scheduling,
    gradient clipping, mixed-precision scaling, gradient accumulation,
    periodic logging, and checkpointing.

    Attributes:
        model: The ``GrimoireTransformer`` being trained.
        config: The model's ``TransformerConfig``.
        device: ``"cuda"`` or ``"cpu"`` as a string.
        peak_lr: Maximum learning rate reached after the warmup phase.
        min_lr: Minimum LR at the end of cosine decay (10 % of peak).
        warmup_steps: Number of optimizer steps over which LR rises linearly.
        total_steps: Total number of optimizer steps for the full training run.
        batch_size: Number of sequences per forward pass.
        accumulate_steps: Number of forward/backward passes per optimizer step.
        log_every: Log training metrics every this many optimizer steps.
        save_every: Save a checkpoint every this many optimizer steps.
        checkpoint_dir: Directory where checkpoints are written.
        _use_amp: Whether fp16 automatic mixed precision is active.
        _scaler: ``GradScaler`` for AMP loss scaling (CUDA only).
        _optimizer: ``AdamW`` optimizer.
        _scheduler: Lambda-based LR scheduler.
        _step: Current global optimizer step count.
        _on_log: Optional callback invoked at each log point with
            ``(step, avg_loss, lr)``.  Used by the training UI to stream
            live progress without polling stdout.
        _val_loader: ``DataLoader`` over the validation set, or ``None`` when
            no ``val_dataset`` was supplied (eval disabled).
        _eval_every: Run a validation pass every this many optimizer steps.
        _eval_batches: Cap the number of validation batches per eval pass
            (0 = use the entire validation set).
        _on_eval: Optional callback invoked after each validation pass with
            ``(step, val_loss, elapsed)``.
    """

    def __init__(
        self,
        model: GrimoireTransformer,
        train_dataset: Dataset,
        peak_lr: float = 3e-4,
        warmup_steps: int = 500,
        total_steps: int = 10_000,
        batch_size: int = 4,
        accumulate_steps: int = 8,
        log_every: int = 50,
        save_every: int = 1000,
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None,
        num_workers: int = 0,
        val_dataset: Optional[Dataset] = None,
        eval_every: int = 0,
        eval_batches: int = 0,
        on_log: Optional[Callable[[int, float, float], None]] = None,
        on_save: Optional[Callable[[int, float], None]] = None,
        on_done: Optional[Callable[[int, float], None]] = None,
        on_eval: Optional[Callable[[int, float, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Set up the trainer, optimizer, scheduler, and data loader.

        Args:
            model: An initialised ``GrimoireTransformer``.  Will be moved
                to ``device`` automatically.
            train_dataset: A ``TokenizedDataset`` (or any
                ``torch.utils.data.Dataset`` returning ``(input_ids,
                target_ids)`` pairs).
            peak_lr: Peak learning rate after warmup.  3e-4 is a safe
                default for AdamW at this model scale.
            warmup_steps: Number of optimizer steps for linear LR warmup.
            total_steps: Total number of optimizer steps for the run.
                Training stops when this count is reached.
            batch_size: Number of sequences per micro-batch (one forward
                pass before accumulation).
            accumulate_steps: Number of micro-batches per optimizer step.
                Effective batch size = ``batch_size × accumulate_steps``.
            log_every: Print a log line every this many optimizer steps.
            save_every: Write a checkpoint every this many optimizer steps.
            checkpoint_dir: Directory for checkpoint files.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
            num_workers: Number of DataLoader worker processes.  Keep at 0
                on Windows to avoid multiprocessing issues.
            val_dataset: Optional held-out ``Dataset`` (same item format as
                ``train_dataset``).  When provided, a validation loss is
                computed periodically so train/val divergence (overfitting)
                is visible.  When ``None`` (the default) no evaluation runs
                and behaviour is unchanged.
            eval_every: Run a validation pass every this many optimizer
                steps.  ``0`` (the default) falls back to ``save_every`` so
                eval lines up with checkpoints.  Ignored if ``val_dataset``
                is ``None``.
            eval_batches: Maximum number of validation batches to average per
                eval pass.  ``0`` (the default) uses the entire validation
                set.  A small cap (e.g. 50) keeps eval cheap on large
                held-out sets while still giving a stable estimate.
            on_log: Optional callable invoked at each log interval with
                ``(step: int, avg_loss: float, lr: float)``.  When ``None``
                (the default) only stdout is used — existing behaviour is
                unchanged.  The training UI registers a callback here to
                stream live loss updates without polling stdout.
            on_save: Optional callable invoked each time a checkpoint is
                written, with ``(step: int, elapsed: float)`` where
                ``elapsed`` is total seconds since training started.
            on_done: Optional callable invoked once when training finishes,
                with ``(total_steps: int, elapsed: float)``.
            on_eval: Optional callable invoked after each validation pass,
                with ``(step: int, val_loss: float, elapsed: float)`` where
                ``elapsed`` is total seconds since training started.  Used by
                the training UI to plot a validation curve alongside train.
        """
        self.config = model.config
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * 0.1
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.accumulate_steps = accumulate_steps
        self.log_every = log_every
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self._step = 0
        self._last_avg_loss: float = 0.0
        self._last_val_loss: float = float("nan")
        self._eval_every = eval_every if eval_every > 0 else save_every
        self._eval_batches = eval_batches
        self._on_log = on_log
        self._on_save = on_save
        self._on_done = on_done
        self._on_eval = on_eval
        self._stop_event = stop_event

        # --- Device setup -----------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._use_amp = device == "cuda"

        if device == "cuda":
            # Let cuDNN auto-tune kernel selection for the fixed input shapes
            # used during training.  One-time overhead at first step; free
            # speed improvement thereafter.
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)

        # torch.compile() traces the model graph and emits optimised CUDA
        # kernels (operator fusion, reduced memory traffic).  Falls back
        # silently on CPU or if compilation is unavailable.
        #
        # IMPORTANT: the compiled wrapper is kept as a SEPARATE handle used
        # only for the forward pass.  ``self.model`` stays the raw module so
        # that ``state_dict()`` / ``load_state_dict()`` keys are NOT prefixed
        # with ``_orig_mod.`` — otherwise checkpoints would be incompatible
        # with the inference engine and with non-compiled resume runs.
        # torch.compile shares the underlying parameters, so loading into the
        # raw module also updates the weights the compiled handle executes.
        self._forward_model = self.model
        if device == "cuda" and hasattr(torch, "compile"):
            if hasattr(torch, "_dynamo"):
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.verbose = False
            self._forward_model = torch.compile(self.model)

        # GradScaler is a no-op on CPU but we instantiate it uniformly
        # to avoid branching in the training loop.
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # --- Optimizer --------------------------------------------------
        # Separate parameters into two groups:
        # - weight matrices (2-D): apply weight decay
        # - all others (biases, RMSNorm weights, embedding): no decay
        decay_params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for name, p in model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]
        self._optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": 0.1},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # --- LR scheduler: linear warmup + cosine decay -----------------
        def _lr_lambda(step: int) -> float:
            """Compute the LR multiplier at a given optimizer step.

            Returns a value in [min_ratio, 1.0] where the LR at step s is
            ``peak_lr × _lr_lambda(s)``.  During warmup the multiplier
            rises linearly from 0 to 1.  After warmup it follows a cosine
            curve from 1 down to ``min_lr / peak_lr``.
            """
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = self.min_lr / self.peak_lr
            return min_ratio + (1.0 - min_ratio) * cosine

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda=_lr_lambda
        )

        # --- DataLoader -------------------------------------------------
        self._loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=PaddingCollator(pad_id=PAD_ID),
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            drop_last=True,
        )

        # Validation loader (optional).  Not shuffled, so the eval estimate is
        # over a fixed slice of held-out data and comparable across steps.
        self._val_loader: Optional[DataLoader] = None
        if val_dataset is not None:
            self._val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=PaddingCollator(pad_id=PAD_ID),
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
                drop_last=False,
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> None:
        """Run the training loop until ``total_steps`` optimizer steps.

        Each optimizer step consists of ``accumulate_steps`` micro-batches.
        Gradients are accumulated across micro-batches; a single optimizer
        step is taken after the last micro-batch in each group.

        The loop cycles the DataLoader indefinitely until ``total_steps``
        is reached, so training can run for more steps than there are
        batches in the dataset.

        Args:
            resume_from: Optional path to a checkpoint ``.pt`` file.  If
                provided, model weights, optimizer state, scaler state, and
                the step counter are restored before training begins.
        """
        if resume_from is not None:
            self._load_resume(resume_from)

        self.model.train()
        self._optimizer.zero_grad()

        data_iter   = iter(self._loader)
        micro_count = 0
        running_loss = 0.0
        t_start = time.time()
        t0 = t_start

        compiled = self._forward_model is not self.model
        print(
            f"Training on {self.device.upper()} | "
            f"AMP={'on' if self._use_amp else 'off'} | "
            f"compile={'on' if compiled else 'off'} | "
            f"params={self.model.num_parameters():,} | "
            f"effective batch={self.batch_size * self.accumulate_steps}"
        )

        while self._step < self.total_steps:
            if self._stop_event is not None and self._stop_event.is_set():
                break

            # Fetch the next micro-batch, cycling the loader if exhausted.
            try:
                input_ids, target_ids, attention_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self._loader)
                input_ids, target_ids, attention_mask = next(data_iter)

            # non_blocking=True overlaps the H→D transfer with GPU work
            # when pin_memory=True on the DataLoader (CUDA only, no-op on CPU).
            non_blocking = self.device == "cuda"
            input_ids      = input_ids.to(self.device, non_blocking=non_blocking)
            target_ids     = target_ids.to(self.device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)

            # --- Forward pass with optional AMP --------------------------
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self._use_amp,
            ):
                logits = self._forward_model(input_ids, attention_mask=attention_mask)
                # logits: (batch, seq_len, vocab_size) → flatten for cross_entropy
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=PAD_ID,
                )
                # Scale loss by 1/accumulate_steps so the gradient is the
                # mean over all micro-batches in this optimizer step.
                loss = loss / self.accumulate_steps

            # --- Backward pass ------------------------------------------
            self._scaler.scale(loss).backward()
            running_loss += loss.item()
            micro_count  += 1

            # --- Optimizer step (every accumulate_steps micro-batches) ---
            if micro_count == self.accumulate_steps:
                # Unscale before clipping so the clip threshold is in the
                # original (unscaled) gradient space.
                self._scaler.unscale_(self._optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._scheduler.step()
                self._optimizer.zero_grad()

                self._step  += 1
                micro_count  = 0

                # --- Logging -------------------------------------------
                if self._step % self.log_every == 0:
                    elapsed_interval = time.time() - t0
                    elapsed_total    = time.time() - t_start
                    lr_now    = self._scheduler.get_last_lr()[0]
                    # running_loss is the sum of per-optimizer-step losses
                    # since the last log point; divide to report the mean.
                    self._last_avg_loss = running_loss / self.log_every
                    print(
                        f"step {self._step:>6} / {self.total_steps} | "
                        f"loss {self._last_avg_loss:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"{elapsed_interval:.1f}s"
                    )
                    if self._on_log is not None:
                        self._on_log(self._step, self._last_avg_loss, lr_now, elapsed_total)
                    running_loss = 0.0
                    t0 = time.time()

                # --- Checkpointing -------------------------------------
                if self._step % self.save_every == 0:
                    elapsed_total = time.time() - t_start
                    ckpt_path = self.checkpoint_dir / f"step_{self._step:07d}.pt"
                    save_checkpoint(
                        path=str(ckpt_path),
                        model=self.model,
                        optimizer=self._optimizer,
                        scheduler=self._scheduler,
                        step=self._step,
                        config_dict=self.config.to_dict(),
                        train_loss=self._last_avg_loss,
                        scaler=self._scaler if self._use_amp else None,
                    )
                    print(f"  → checkpoint saved: {ckpt_path}")
                    if self._on_save is not None:
                        self._on_save(self._step, elapsed_total)

                # --- Evaluation ----------------------------------------
                if self._val_loader is not None and self._step % self._eval_every == 0:
                    val_loss = self.evaluate()
                    self._last_val_loss = val_loss
                    elapsed_total = time.time() - t_start
                    print(
                        f"  → eval step {self._step:>6} | "
                        f"val loss {val_loss:.4f}"
                    )
                    if self._on_eval is not None:
                        self._on_eval(self._step, val_loss, elapsed_total)
                    # evaluate() flips the model to eval mode; restore train.
                    self.model.train()
                    # Restart the running-loss interval timer so the eval pass
                    # is not counted as training time in the next log line.
                    t0 = time.time()

        elapsed_total = time.time() - t_start
        stopped_early = self._stop_event is not None and self._stop_event.is_set()
        if stopped_early:
            print(f"\nTraining stopped at step {self._step} | elapsed: {elapsed_total:.1f}s")
        else:
            print(f"\nTraining complete. Final step: {self._step} | total time: {elapsed_total:.1f}s")
        if self._on_done is not None:
            self._on_done(self._step, elapsed_total)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute the mean cross-entropy loss over the validation set.

        Runs the model in eval mode (dropout disabled) with no gradient
        tracking.  AMP autocast is applied on CUDA exactly as in training so
        the number is comparable to the training loss.  Padding positions are
        ignored via ``ignore_index=PAD_ID``.

        The raw (uncompiled) module is used for the forward pass to avoid a
        separate ``torch.compile`` recompilation for the eval-mode graph; it
        shares parameters with the compiled training handle, so the weights
        are identical.

        Only the first ``eval_batches`` validation batches are averaged when
        that cap is set (0 = the whole validation set).  Because the loader is
        not shuffled, the same slice is evaluated each call, making successive
        validation losses directly comparable.

        Returns:
            The mean per-batch validation loss as a float, or ``nan`` if no
            validation set was configured.
        """
        if self._val_loader is None:
            return float("nan")

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        non_blocking = self.device == "cuda"
        for input_ids, target_ids, attention_mask in self._val_loader:
            input_ids      = input_ids.to(self.device, non_blocking=non_blocking)
            target_ids     = target_ids.to(self.device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)

            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self._use_amp,
            ):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=PAD_ID,
                )

            total_loss += loss.item()
            n_batches  += 1
            if self._eval_batches and n_batches >= self._eval_batches:
                break

        if was_training:
            self.model.train()

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Resume helper
    # ------------------------------------------------------------------

    def _load_resume(self, path: str) -> None:
        """Restore model, optimizer, scaler, and step from a checkpoint.

        Args:
            path: Path to a checkpoint ``.pt`` file produced by
                ``save_checkpoint``.
        """
        print(f"Resuming from checkpoint: {path}")
        ckpt = load_checkpoint(path)

        self.model.load_state_dict(ckpt["model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None and self._use_amp:
            self._scaler.load_state_dict(ckpt["scaler"])

        self._step = ckpt["step"]
        if "scheduler" in ckpt:
            self._scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Legacy checkpoints without scheduler state: replay steps to
            # approximate the correct LR (less accurate but better than nothing).
            for _ in range(self._step):
                self._scheduler.step()

        print(f"  Resumed at step {self._step} / {self.total_steps}")
