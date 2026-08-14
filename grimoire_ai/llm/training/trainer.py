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

Mixed precision (bf16/fp16 AMP)
    When training on CUDA, ``torch.autocast`` runs the forward pass in a
    reduced-precision dtype, which roughly halves memory usage and exploits
    RTX tensor cores. The dtype is chosen per-GPU via
    ``torch.cuda.is_bf16_supported()``:

    - bf16 (Ampere and newer, e.g. RTX 3050): bf16 has fp32's exponent
      range, so it never underflows -- no loss scaling needed, and
      ``GradScaler`` is instantiated but permanently disabled
      (``enabled=False``) on this path.
    - fp16 (pre-Ampere GPUs, where bf16 tensor cores aren't available): the
      original path. A ``GradScaler`` multiplies the loss by a large scale
      factor before the backward pass to prevent fp16 underflow (very small
      gradients rounding to zero), then unscales before the optimizer step.
      Whenever the scale factor overshoots, that optimizer step is skipped
      entirely -- a real (if infrequent) source of wasted compute that bf16
      doesn't have.

    On CPU and MPS (Apple Silicon) the scaler is a no-op and the autocast
    context is skipped — training still runs on the GPU on MPS, just
    without mixed precision (see ``docs/speed_optimization.md`` item #6 for
    scoping MPS's own bf16-without-GradScaler path, not yet implemented).

RETRO neighbor retrieval (optional, docs/architecture_optimization.md item #3)
    When ``neighbor_ids`` is supplied (an array from
    ``scripts/build_retrieval_neighbors.py``, aligned to
    ``train_dataset``'s window order), ``train_dataset`` is wrapped in
    ``NeighborAugmentedDataset`` and every batch also carries retrieved
    neighbor token ids, fed into ``GrimoireTransformer.forward`` for the
    Chunked Cross-Attention sublayers configured via
    ``model.config.retro_layers`` to attend to. Retrieval itself is never
    computed live during training — the retrieval database is frozen for
    the whole run, so neighbors are looked up once, offline, ahead of time.
    Requires ``model.config.retro_layers`` to be set; passing
    ``neighbor_ids`` for a model with no CCA sublayers raises immediately
    rather than silently wasting the precomputed data.

    ``evaluate()`` does NOT currently consume neighbor data — validation
    loss is computed on the plain self-attention path even when RETRO is
    enabled for training. Unlike Multi-Token Prediction's auxiliary loss
    (a training-only crutch never used at inference), CCA output genuinely
    changes what the model predicts, so this is a real scope gap rather
    than a deliberate comparability choice — it means ``val_loss`` under-
    represents what a RETRO-enabled model actually does once retrieval is
    wired into inference too. Left as-is for now since closing it needs a
    second (validation-aligned) neighbor-ids array, not just a code change.

Multi-Token Prediction (optional, docs/architecture_optimization.md item #2)
    When ``model.config.n_predict > 0``, an auxiliary loss is added: each
    of the model's ``n_predict`` extra heads predicts a token further
    ahead than the primary next-token objective (head ``i`` predicts
    ``i + 2`` positions ahead). Targets for these extra offsets are built
    from the *same* ``(input_ids, target_ids)`` pair already produced by
    the dataset — no dataset changes needed — by shifting further into
    ``input_ids`` and padding the tail of each window with
    ``ignore_index=-100`` where no target that far ahead exists yet (the
    last few positions of a window). Because pretraining windows overlap
    (``TokenizedDataset``'s default 50% stride), the predictions lost to
    this tail-masking in one window are covered as mid-window predictions
    in an adjacent one.

    The auxiliary loss is combined into the total training loss as
    ``primary_loss + mtp_loss_weight * mean(per_head_loss)`` — a small
    weight (default 0.3, matching DeepSeek-V3's reported value) so the
    auxiliary signal augments rather than dominates the primary objective.
    ``evaluate()`` deliberately does NOT include the MTP term: ``val_loss``
    stays pure next-token cross-entropy so it's directly comparable across
    MTP-enabled and MTP-disabled runs, and so early stopping/checkpoint
    selection isn't skewed by an auxiliary term nobody is optimising for at
    inference time.

    Combining the two: when both ``neighbor_ids`` and
    ``model.config.n_predict > 0`` are active, the training-loop forward
    call passes ``neighbor_ids`` unconditionally and requests
    ``return_mtp_logits=True`` only when ``n_predict > 0`` — see the single
    ``self._forward_model(...)`` call site in ``train()``, which is the one
    place these two features' code paths actually intersect.
"""

import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.neighbor_dataset import NeighborAugmentedDataset, collate_with_neighbors
from grimoire_ai.llm.device import select_device, torch_has_triton
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
from grimoire_ai.llm.training.checkpoint import (
    load_checkpoint,
    resize_checkpoint_vocab,
    resize_optimizer_vocab_state,
    save_checkpoint,
)


def _mtp_target(full_ids: torch.Tensor, offset: int, seq_len: int) -> torch.Tensor:
    """Build one Multi-Token Prediction head's targets from the token stream.

    ``full_ids`` is ``input_ids`` with the one extra real token from
    ``target_ids`` appended (length ``seq_len + 1``) — the longest run of
    genuine corpus tokens available without touching the dataset. For an
    MTP head predicting ``offset`` positions ahead (``offset >= 2``; ``1``
    is the primary head, handled separately with the existing
    ``target_ids``), position ``t``'s target is ``full_ids[t + offset]``
    when that index exists, else there is no real token that far ahead in
    this window and the position is masked with ``-100`` (ignored by
    ``cross_entropy``, identical to how padding is already handled).

    Args:
        full_ids: ``(batch, seq_len + 1)`` — ``input_ids`` with
            ``target_ids[:, -1:]`` appended.
        offset: How many positions ahead this head predicts. Must be
            ``>= 2``.
        seq_len: The window length (``input_ids.shape[1]``) — the returned
            tensor always has this length so it aligns with the head's
            logits.

    Returns:
        ``(batch, seq_len)`` integer tensor, ``-100`` past the point where
        no real target token exists yet in this window.
    """
    shifted = full_ids[:, offset:]  # (batch, seq_len + 1 - offset), possibly empty
    pad_len = seq_len - shifted.shape[1]
    if pad_len <= 0:
        return shifted[:, :seq_len]
    pad = torch.full(
        (shifted.shape[0], pad_len), -100, dtype=shifted.dtype, device=shifted.device
    )
    return torch.cat([shifted, pad], dim=1)


def _select_amp_dtype(use_amp: bool) -> tuple[torch.dtype, bool]:
    """Pick the AMP dtype and whether ``GradScaler`` should actually scale.

    bf16 has fp32's exponent range and never underflows, so it needs no
    loss scaling -- preferred whenever the GPU's tensor cores support it
    (Ampere+, via ``torch.cuda.is_bf16_supported()``). Older CUDA GPUs fall
    back to fp16 + ``GradScaler``, the original path (fp16's narrow
    exponent range underflows without loss scaling).

    Args:
        use_amp: Whether AMP is active at all (``device == "cuda"``). When
            ``False``, returns fp16/disabled without querying CUDA -- safe
            to call on any device, including one with no GPU at all.

    Returns:
        ``(dtype, grad_scaler_enabled)``. ``grad_scaler_enabled`` is
        ``False`` whenever ``dtype`` is bf16 or ``use_amp`` is ``False``;
        ``True`` only for the fp16-on-CUDA fallback.
    """
    if use_amp and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, use_amp


class Trainer:
    """Manages the full training loop for a ``GrimoireTransformer``.

    Instantiate with a model and dataset, then call ``train()``.  The
    trainer handles device placement, the optimizer, LR scheduling,
    gradient clipping, mixed-precision scaling, gradient accumulation,
    periodic logging, and checkpointing.

    Attributes:
        model: The ``GrimoireTransformer`` being trained.
        config: The model's ``TransformerConfig``.
        device: ``"cuda"``, ``"mps"``, or ``"cpu"`` as a string.
        peak_lr: Maximum learning rate reached after the warmup phase.
        min_lr: Minimum LR at the end of cosine decay (10 % of peak).
        warmup_steps: Number of optimizer steps over which LR rises linearly.
        total_steps: Total number of optimizer steps for the full training run.
        batch_size: Number of sequences per forward pass.
        accumulate_steps: Number of forward/backward passes per optimizer step.
        log_every: Log training metrics every this many optimizer steps.
        save_every: Save a checkpoint every this many optimizer steps.
        checkpoint_dir: Directory where checkpoints are written.
        _use_amp: Whether automatic mixed precision is active (CUDA only).
        _amp_dtype: ``torch.bfloat16`` when the GPU's tensor cores support
            it (``torch.cuda.is_bf16_supported()``), else ``torch.float16``.
            Only consulted when ``_use_amp`` is ``True``.
        _grad_scaler_enabled: Whether ``_scaler`` actually scales losses.
            ``False`` whenever ``_amp_dtype`` is bf16 (never needed — bf16
            has fp32's exponent range) or AMP is off entirely; ``True`` only
            for the fp16-on-CUDA fallback path, where loss scaling prevents
            underflow.
        _scaler: ``GradScaler`` for AMP loss scaling. Instantiated
            unconditionally but a no-op unless ``_grad_scaler_enabled``.
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
        mtp_loss_weight: Weight applied to the mean Multi-Token Prediction
            auxiliary loss before adding it to the primary loss. Only takes
            effect when ``model.config.n_predict > 0``; otherwise unused.
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
        early_stop_enabled: bool = False,
        early_stop_patience: int = 3,
        early_stop_bootstraps: int = 1000,
        early_stop_alpha: float = 0.05,
        swa_enabled: bool = False,
        swa_start_frac: float = 0.75,
        sample_weights: Optional["Sequence[float]"] = None,
        neighbor_ids: Optional[np.ndarray] = None,
        gradient_checkpointing: bool = False,
        compile_mode: Optional[str] = None,
        mtp_loss_weight: float = 0.3,
        on_log: Optional[Callable[[int, float, float], None]] = None,
        on_save: Optional[Callable[[int, float], None]] = None,
        on_done: Optional[Callable[[int, float], None]] = None,
        on_eval: Optional[Callable[[int, float, float], None]] = None,
        stop_event: Optional[threading.Event] = None,
        model_state_dict_fn: Optional[Callable[[], dict]] = None,
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
            device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (auto-detect:
                CUDA, then MPS on Apple Silicon, then CPU).
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
        # Tracks the step a checkpoint was last written at, so train() can
        # guarantee a final save covering the true end state even when the
        # exit step (natural completion, early stop, or user stop) doesn't
        # land on a save_every boundary. -1 means "nothing saved this call".
        self._last_saved_step = -1
        self._last_avg_loss: float = 0.0
        self._last_val_loss: float = float("nan")
        self._eval_every = eval_every if eval_every > 0 else save_every
        self._eval_batches = eval_batches
        # Per-batch losses from the most recent evaluate() call, used by the
        # bootstrap early-stopping check (empty until the first eval).
        self._last_val_batch_losses: list[float] = []
        self._early_stopper = None
        if early_stop_enabled:
            from grimoire_ai.llm.training.early_stopping import EarlyStopper
            self._early_stopper = EarlyStopper(
                patience=early_stop_patience,
                n_boot=early_stop_bootstraps,
                alpha=early_stop_alpha,
            )
        # --- Stochastic Weight Averaging (SWA) --------------------------
        # When enabled, the parameters from the tail of training (the last
        # 1 - swa_start_frac of steps, where the LR is small and the optimiser
        # is bouncing around a minimum) are averaged into a separate model.
        # The average tends to sit in a flatter, better-generalising basin
        # than any single iterate.  The averaged weights are saved as
        # ``swa.pt`` at the end of the run.  The model uses RMSNorm and has no
        # BatchNorm running statistics, so no post-hoc BN recalibration is
        # needed.
        self._swa_enabled = swa_enabled
        self._swa_start = int(total_steps * swa_start_frac) if swa_enabled else None
        self._swa_model = None       # created lazily once swa_start is reached
        self._swa_n = 0              # number of snapshots averaged so far
        self._on_log = on_log
        self._on_save = on_save
        self._on_done = on_done
        self._on_eval = on_eval
        self._model_state_dict_fn = model_state_dict_fn
        self._stop_event = stop_event

        # --- Device setup -----------------------------------------------
        # CUDA > MPS (Apple Silicon) > CPU. The CUDA-only optimizations below
        # (cuDNN benchmark, torch.compile, AMP/GradScaler, pinned memory) are
        # deliberately not extended to MPS — GradScaler is CUDA-specific in
        # PyTorch and MPS autocast support is still immature — so MPS runs
        # the plain fp32 path, getting its speedup purely from GPU placement.
        device = select_device(device)
        self.device = device
        self._use_amp = device == "cuda"
        self._amp_dtype, self._grad_scaler_enabled = _select_amp_dtype(self._use_amp)

        if device == "cuda":
            # Let cuDNN auto-tune kernel selection for the fixed input shapes
            # used during training.  One-time overhead at first step; free
            # speed improvement thereafter.
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
        self.gradient_checkpointing = gradient_checkpointing

        if mtp_loss_weight < 0:
            raise ValueError(f"mtp_loss_weight ({mtp_loss_weight}) must be non-negative.")
        self.mtp_loss_weight = mtp_loss_weight

        # torch.compile() traces the model graph and emits optimised CUDA
        # kernels (operator fusion, reduced memory traffic).  Falls back
        # silently on CPU or if compilation is unavailable.
        #
        # compile_mode (docs/speed_optimization.md item #3): None (default)
        # uses torch.compile's default mode -- fast to warm up, the right
        # choice for short runs (fine-tuning, a few hundred steps) where
        # extra warmup cost may not be recouped. "max-autotune" spends much
        # longer per-shape autotuning kernels in exchange for faster steady-
        # state steps -- worth it for pretraining's thousands of steps at
        # one fixed (batch_size, seq_len) shape, where that cost amortizes.
        # Left opt-in rather than switched by default: the actual crossover
        # point depends on hardware, not just step count, so this needs an
        # A/B on the GPU it will actually run on before becoming a default.
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
                # ``suppress_errors`` only stops the BackendCompilerFailed
                # exception from propagating — dynamo still unconditionally
                # logs a WARNING with the full traceback for every frame it
                # falls back on.  This is expected and harmless when Triton
                # itself isn't installed (e.g. Windows), so raise the logger
                # level only in that known case — leave it untouched
                # otherwise so a *genuine* compile regression on a working
                # Triton setup still surfaces a visible warning instead of
                # silently degrading to eager.
                if not torch_has_triton():
                    try:
                        torch._logging.set_logs(
                            dynamo=logging.ERROR, inductor=logging.ERROR
                        )
                    except Exception:
                        pass
            self._forward_model = torch.compile(self.model, mode=compile_mode)
        self.compile_mode = compile_mode

        # GradScaler is a no-op whenever _grad_scaler_enabled is False (CPU,
        # MPS, or CUDA-with-bf16) but we instantiate it uniformly to avoid
        # branching in the training loop.
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._grad_scaler_enabled)

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

        # --- Optional RETRO neighbor retrieval ---------------------------
        # See the module docstring's "RETRO neighbor retrieval" section.
        self._neighbor_ids = neighbor_ids
        collate_fn: Callable = PaddingCollator(pad_id=PAD_ID)
        if neighbor_ids is not None:
            if self.config.retro_layers is None:
                raise ValueError(
                    "neighbor_ids was given but model.config.retro_layers is "
                    "None -- the model has no Chunked Cross-Attention "
                    "sublayers to consume them. Set retro_layers when "
                    "building the model, or omit neighbor_ids."
                )
            train_dataset = NeighborAugmentedDataset(train_dataset, neighbor_ids)
            collate_fn = collate_with_neighbors

        # --- DataLoader -------------------------------------------------
        # With per-window importance weights, draw windows in proportion to
        # their difficulty (a WeightedRandomSampler) instead of uniformly.
        # Harder windows — those a short warm-up run still predicts poorly —
        # are sampled more often, focusing compute where the model has the most
        # to learn.  Without weights, fall back to plain uniform shuffling.
        sampler = self._build_weighted_sampler(sample_weights, train_dataset)
        self._loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
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

    @staticmethod
    def _build_weighted_sampler(
        sample_weights: Optional[Sequence[float]],
        train_dataset: Dataset,
    ) -> Optional[WeightedRandomSampler]:
        """Build a difficulty-weighted sampler, or ``None`` for uniform sampling.

        Args:
            sample_weights: One non-negative weight per dataset window, aligned
                to ``train_dataset`` order.  ``None`` disables weighting.
            train_dataset: The training dataset (used to validate the length).

        Returns:
            A ``WeightedRandomSampler`` over the dataset, or ``None`` when no
            weights were supplied.

        Raises:
            ValueError: If the weights length does not match the dataset, or if
                the weights are non-finite, negative, or sum to zero.
        """
        if sample_weights is None:
            return None

        w = np.asarray(list(sample_weights), dtype=np.float64)
        n = len(train_dataset)  # type: ignore[arg-type]
        if w.shape[0] != n:
            raise ValueError(
                f"sample_weights has {w.shape[0]} entries but the training "
                f"dataset has {n} windows. The weights file must be scored on "
                f"the same corpus, seq_len, stride, and split as training."
            )
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("sample_weights must be finite and non-negative.")
        if w.sum() <= 0:
            raise ValueError("sample_weights must not all be zero.")

        return WeightedRandomSampler(
            weights=torch.as_tensor(w, dtype=torch.double),
            num_samples=n,
            replacement=True,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None, start_step: int = 0) -> None:
        """Run the training loop until ``total_steps`` optimizer steps.

        Each optimizer step consists of ``accumulate_steps`` micro-batches.
        Gradients are accumulated across micro-batches; a single optimizer
        step is taken after the last micro-batch in each group.

        The loop cycles the DataLoader indefinitely until ``total_steps``
        is reached, so training can run for more steps than there are
        batches in the dataset.

        Args:
            resume_from: Optional path to a checkpoint ``.pt`` file written
                by *this* ``Trainer`` (e.g. via ``save_every``). If provided,
                model weights, optimizer state, scaler state, and the step
                counter are restored before training begins; ``start_step``
                is ignored in this case.
            start_step: Resume point when the caller restored model/optimizer
                state through some *other* mechanism than ``resume_from``
                (e.g. a LoRA adapter's own checkpoint format) and only needs
                this ``Trainer`` to pick up the step count and LR schedule
                from where that state left off. ``total_steps`` is the
                absolute target, not an additional count -- starting at step
                400 with ``total_steps=1000`` runs 600 more steps. Has no
                effect when ``resume_from`` is given. There is no saved
                scheduler state to restore in this path, so the LR schedule
                is approximated by replaying ``start_step`` calls to it, the
                same fallback ``_load_resume`` uses for legacy checkpoints
                that predate scheduler-state saving.
        """
        if resume_from is not None:
            self._load_resume(resume_from)
        elif start_step:
            self._step = start_step
            for _ in range(start_step):
                self._scheduler.step()
            print(f"  Starting at step {self._step} / {self.total_steps}")

        self.model.train()
        self._optimizer.zero_grad()

        data_iter   = iter(self._loader)
        micro_count = 0
        running_loss = 0.0
        steps_since_log = 0
        t_start = time.time()
        t0 = t_start

        compiled = self._forward_model is not self.model
        amp_status = (
            f"on ({'bf16' if self._amp_dtype == torch.bfloat16 else 'fp16'})"
            if self._use_amp else "off"
        )
        compile_status = f"on ({self.compile_mode or 'default'})" if compiled else "off"
        print(
            f"Training on {self.device.upper()} | "
            f"AMP={amp_status} | "
            f"compile={compile_status} | "
            f"params={self.model.num_parameters():,} | "
            f"effective batch={self.batch_size * self.accumulate_steps}"
        )

        while self._step < self.total_steps:
            if self._stop_event is not None and self._stop_event.is_set():
                break

            # Fetch the next micro-batch, cycling the loader if exhausted.
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self._loader)
                batch = next(data_iter)
            if self._neighbor_ids is not None:
                input_ids, target_ids, attention_mask, neighbor_ids = batch
            else:
                input_ids, target_ids, attention_mask = batch
                neighbor_ids = None

            # non_blocking=True overlaps the H→D transfer with GPU work
            # when pin_memory=True on the DataLoader (CUDA only, no-op on CPU).
            non_blocking = self.device == "cuda"
            input_ids      = input_ids.to(self.device, non_blocking=non_blocking)
            target_ids     = target_ids.to(self.device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)
            if neighbor_ids is not None:
                neighbor_ids = neighbor_ids.to(self.device, non_blocking=non_blocking)

            # --- Forward pass with optional AMP --------------------------
            with torch.autocast(
                device_type=self.device,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                n_predict = self.config.n_predict
                if n_predict > 0:
                    logits, mtp_logits = self._forward_model(
                        input_ids, attention_mask=attention_mask,
                        neighbor_ids=neighbor_ids, return_mtp_logits=True,
                    )
                else:
                    logits = self._forward_model(
                        input_ids, attention_mask=attention_mask, neighbor_ids=neighbor_ids,
                    )
                # logits: (batch, seq_len, vocab_size) → flatten for cross_entropy
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )
                if n_predict > 0:
                    # See _mtp_target and the module docstring's "Multi-Token
                    # Prediction" section for how these targets are derived
                    # from input_ids/target_ids without any dataset changes.
                    seq_len = input_ids.shape[1]
                    full_ids = torch.cat([input_ids, target_ids[:, -1:]], dim=1)
                    mtp_loss = sum(
                        F.cross_entropy(
                            mtp_logits[i].view(-1, self.config.vocab_size),
                            _mtp_target(full_ids, i + 2, seq_len).view(-1),
                            ignore_index=-100,
                        )
                        for i in range(n_predict)
                    ) / n_predict
                    loss = loss + self.mtp_loss_weight * mtp_loss
                # Scale loss by 1/accumulate_steps so the gradient is the
                # mean over all micro-batches in this optimizer step.
                loss = loss / self.accumulate_steps

            # --- Backward pass ------------------------------------------
            self._scaler.scale(loss).backward()
            # Accumulate the unscaled loss for logging; multiply back by
            # accumulate_steps so the logged value reflects the true per-step
            # cross-entropy, not the 1/accumulate_steps-scaled training loss.
            running_loss += loss.item() * self.accumulate_steps
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
                steps_since_log += 1

                # --- SWA snapshot --------------------------------------
                # Fold the current weights into the running average once we
                # are past the SWA start step (tail of training).
                if self._swa_enabled and self._step >= self._swa_start:
                    self._update_swa()

                # --- Logging -------------------------------------------
                if self._step % self.log_every == 0:
                    elapsed_interval = time.time() - t0
                    elapsed_total    = time.time() - t_start
                    lr_now    = self._scheduler.get_last_lr()[0]
                    # running_loss is the sum of per-micro-batch losses since
                    # the last log point — steps_since_log optimizer steps
                    # each comprising accumulate_steps micro-batches — so
                    # divide by their product to report the true mean
                    # cross-entropy.  steps_since_log is usually log_every,
                    # but can be smaller right after a resume (the counter
                    # restarts at 0 while self._step starts mid-interval).
                    self._last_avg_loss = running_loss / (steps_since_log * self.accumulate_steps)
                    print(
                        f"step {self._step:>6} / {self.total_steps} | "
                        f"loss {self._last_avg_loss:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"{elapsed_interval:.1f}s"
                    )
                    if self._on_log is not None:
                        self._on_log(self._step, self._last_avg_loss, lr_now, elapsed_total)
                    running_loss = 0.0
                    steps_since_log = 0
                    t0 = time.time()

                # --- Checkpointing -------------------------------------
                if self._step % self.save_every == 0:
                    self._save_checkpoint_now(t_start)

                # --- Evaluation ----------------------------------------
                if self._val_loader is not None and self._step % self._eval_every == 0:
                    val_loss = self.evaluate()
                    self._last_val_loss = val_loss
                    elapsed_total = time.time() - t_start
                    print(
                        f"  -> eval step {self._step:>6} | "
                        f"val loss {val_loss:.4f}"
                    )
                    if self._on_eval is not None:
                        self._on_eval(self._step, val_loss, elapsed_total)

                    # --- Bootstrap early stopping --------------------------
                    # Stop when the apparent improvement in validation loss has
                    # stayed within its bootstrap noise band for `patience`
                    # consecutive evals — i.e. further training is no longer
                    # producing statistically meaningful gains.
                    if self._early_stopper is not None and self._early_stop_check():
                        print(
                            f"  -> early stop at step {self._step}: "
                            f"no significant val-loss improvement for "
                            f"{self._early_stopper.patience} evals "
                            f"(best {self._early_stopper.best_loss:.4f})"
                        )
                        # evaluate() flips the model to eval mode; restore train
                        # for symmetry even though we are about to exit.
                        self.model.train()
                        break

                    # evaluate() flips the model to eval mode; restore train.
                    self.model.train()
                    # Restart the running-loss interval timer so the eval pass
                    # is not counted as training time in the next log line.
                    t0 = time.time()

        # Guarantee the true end state is always recoverable: natural
        # completion, early stopping, and user-initiated stop can all land on
        # a step that isn't a save_every boundary, silently dropping however
        # many steps came after the last periodic save.
        if self._last_saved_step != self._step:
            self._save_checkpoint_now(t_start)

        # Save the averaged SWA weights, if any snapshots were collected.
        if self._swa_enabled:
            self._finalize_swa()

        elapsed_total = time.time() - t_start
        stopped_early = self._stop_event is not None and self._stop_event.is_set()
        if stopped_early:
            print(f"\nTraining stopped at step {self._step} | elapsed: {elapsed_total:.1f}s")
        else:
            print(f"\nTraining complete. Final step: {self._step} | total time: {elapsed_total:.1f}s")
        if self._on_done is not None:
            self._on_done(self._step, elapsed_total)

    def _save_checkpoint_now(self, t_start: float) -> None:
        """Write a checkpoint at the current step and fire the on_save callback.

        Shared by the periodic save_every check and the unconditional final
        save in train(), so both produce identical files/log lines and both
        update _last_saved_step to avoid a redundant duplicate write.
        """
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
            model_state_dict=self._model_state_dict_fn() if self._model_state_dict_fn else None,
        )
        self._last_saved_step = self._step
        print(f"  -> checkpoint saved: {ckpt_path}")
        if self._on_save is not None:
            self._on_save(self._step, elapsed_total)

    # ------------------------------------------------------------------
    # Stochastic Weight Averaging
    # ------------------------------------------------------------------

    def _update_swa(self) -> None:
        """Fold the current model weights into the running SWA average.

        The ``AveragedModel`` is created lazily on the first call so that no
        extra memory is reserved when SWA is disabled or never reached (e.g.
        when early stopping fires before ``swa_start``).
        """
        from torch.optim.swa_utils import AveragedModel
        if self._swa_model is None:
            self._swa_model = AveragedModel(self.model)
        else:
            self._swa_model.update_parameters(self.model)
        self._swa_n += 1

    def _finalize_swa(self) -> None:
        """Write the averaged SWA weights to ``{checkpoint_dir}/swa.pt``.

        No-op (with a note) when no snapshots were collected, which happens if
        the run ended before ``swa_start`` — e.g. early stopping triggered
        first.
        """
        if self._swa_model is None:
            print("  -> SWA: no snapshots collected (run ended before swa_start); nothing saved.")
            return
        swa_path = self.checkpoint_dir / "swa.pt"
        swa_inner = self._swa_model.module
        swa_sd_fn = self._model_state_dict_fn and getattr(swa_inner, "merged_state_dict", None)
        save_checkpoint(
            path=str(swa_path),
            model=swa_inner,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            step=self._step,
            config_dict=self.config.to_dict(),
            train_loss=self._last_avg_loss,
            scaler=self._scaler if self._use_amp else None,
            model_state_dict=swa_sd_fn() if swa_sd_fn else None,
        )
        print(f"  -> SWA: averaged {self._swa_n} snapshot(s) saved to {swa_path}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute the mean cross-entropy loss over the validation set.

        Runs the model in eval mode (dropout disabled) with no gradient
        tracking.  AMP autocast is applied on CUDA exactly as in training so
        the number is comparable to the training loss.  Padding positions are
        ignored via ``ignore_index=-100``.

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
        batch_losses: list[float] = []
        non_blocking = self.device == "cuda"
        for input_ids, target_ids, attention_mask in self._val_loader:
            input_ids      = input_ids.to(self.device, non_blocking=non_blocking)
            target_ids     = target_ids.to(self.device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)

            with torch.autocast(
                device_type=self.device,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    target_ids.view(-1),
                    ignore_index=-100,
                )

            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            n_batches  += 1
            if self._eval_batches and n_batches >= self._eval_batches:
                break

        if was_training:
            self.model.train()

        # Stored for the bootstrap early-stopping check in train().
        self._last_val_batch_losses = batch_losses
        return total_loss / max(n_batches, 1)

    def _early_stop_check(self) -> bool:
        """Feed the latest eval's per-batch losses to the early stopper.

        Returns ``True`` when the stopper signals that validation loss has
        stopped improving beyond the bootstrap noise band.  Returns ``False``
        when early stopping is disabled or no per-batch losses are available.
        """
        if self._early_stopper is None or not self._last_val_batch_losses:
            return False
        return self._early_stopper.update(self._last_val_batch_losses)

    # ------------------------------------------------------------------
    # Resume helper
    # ------------------------------------------------------------------

    def _load_resume(self, path: str) -> None:
        """Restore model, optimizer, scaler, and step from a checkpoint.

        If ``self.model`` was built with a larger ``vocab_size`` than the
        checkpoint (e.g. after ``BytePairEncoder.extend()``), the
        checkpoint's embedding/output_head weight and the optimizer's
        momentum buffers for that parameter are grown to match before
        loading — see ``resize_checkpoint_vocab`` and
        ``resize_optimizer_vocab_state``.  A *smaller* vocab_size is
        rejected with ``ValueError`` rather than silently truncating
        learned rows.

        Args:
            path: Path to a checkpoint ``.pt`` file produced by
                ``save_checkpoint``.

        Raises:
            ValueError: If ``self.model``'s vocab_size is smaller than the
                checkpoint's.
        """
        print(f"Resuming from checkpoint: {path}")
        ckpt = load_checkpoint(path)

        new_vocab_size = self.model.config.vocab_size
        ckpt_cfg = ckpt.get("config")
        ckpt_vocab_size = ckpt_cfg.get("vocab_size") if isinstance(ckpt_cfg, dict) else None
        vocab_size_changed = ckpt_vocab_size is not None and new_vocab_size != ckpt_vocab_size
        if vocab_size_changed:
            print(f"  Vocabulary size changed ({ckpt_vocab_size:,} -> "
                  f"{new_vocab_size:,}); resizing checkpoint embeddings ...")
            ckpt = resize_checkpoint_vocab(ckpt, new_vocab_size)

        self.model.load_state_dict(ckpt["model"])
        self._optimizer.load_state_dict(ckpt["optimizer"])
        if vocab_size_changed:
            # AdamW's restored momentum buffers still have the checkpoint's
            # old (smaller) row count; pad them with zeros for the newly
            # added vocabulary ids before any optimizer.step() runs.
            resize_optimizer_vocab_state(
                self._optimizer, self.model.embedding.weight, new_vocab_size
            )
        # Checkpoints are always loaded with map_location="cpu"; move optimizer
        # state tensors (momentum, variance buffers) to the training device so
        # AdamW doesn't mix CPU state with CUDA parameters on the first step.
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
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
