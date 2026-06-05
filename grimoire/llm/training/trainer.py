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
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from grimoire.llm.data.collator import PaddingCollator
from grimoire.llm.model.transformer import GrimoireTransformer
from grimoire.llm.tokenizer.special_tokens import PAD_ID
from grimoire.llm.training.checkpoint import load_checkpoint, save_checkpoint


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
        on_log: Optional[Callable[[int, float, float], None]] = None,
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
            on_log: Optional callable invoked at each log interval with
                ``(step: int, avg_loss: float, lr: float)``.  When ``None``
                (the default) only stdout is used — existing behaviour is
                unchanged.  The training UI registers a callback here to
                stream live loss updates without polling stdout.
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
        self._on_log = on_log

        # --- Device setup -----------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._use_amp = device == "cuda"
        self.model = model.to(device)

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
        t0 = time.time()

        print(
            f"Training on {self.device.upper()} | "
            f"AMP={'on' if self._use_amp else 'off'} | "
            f"params={self.model.num_parameters():,} | "
            f"effective batch={self.batch_size * self.accumulate_steps}"
        )

        while self._step < self.total_steps:
            # Fetch the next micro-batch, cycling the loader if exhausted.
            try:
                input_ids, target_ids, attention_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self._loader)
                input_ids, target_ids, attention_mask = next(data_iter)

            input_ids      = input_ids.to(self.device)
            target_ids     = target_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # --- Forward pass with optional AMP --------------------------
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self._use_amp,
            ):
                logits = self.model(input_ids, attention_mask=attention_mask)
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
                    elapsed   = time.time() - t0
                    lr_now    = self._scheduler.get_last_lr()[0]
                    # running_loss is the sum of per-optimizer-step losses
                    # since the last log point; divide to report the mean.
                    self._last_avg_loss = running_loss / self.log_every
                    print(
                        f"step {self._step:>6} / {self.total_steps} | "
                        f"loss {self._last_avg_loss:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"{elapsed:.1f}s"
                    )
                    if self._on_log is not None:
                        self._on_log(self._step, self._last_avg_loss, lr_now)
                    running_loss = 0.0
                    t0 = time.time()

                # --- Checkpointing -------------------------------------
                if self._step % self.save_every == 0:
                    ckpt_path = self.checkpoint_dir / f"step_{self._step:07d}.pt"
                    save_checkpoint(
                        path=str(ckpt_path),
                        model=self.model,
                        optimizer=self._optimizer,
                        step=self._step,
                        config_dict=self.config.to_dict(),
                        train_loss=self._last_avg_loss,
                        scaler=self._scaler if self._use_amp else None,
                    )
                    print(f"  → checkpoint saved: {ckpt_path}")

        print(f"\nTraining complete. Final step: {self._step}")

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
        # Advance the scheduler to match the restored step count.
        for _ in range(self._step):
            self._scheduler.step()

        print(f"  Resumed at step {self._step} / {self.total_steps}")
