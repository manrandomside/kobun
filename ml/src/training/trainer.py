"""Training loop orchestrator for Kobun Phase 1 classification.

:class:`Trainer` wires together the previously shipped components
(dataset, transforms, classifier, loss, metrics tracker, callbacks,
config) into a single ``fit()`` call. It supports:

- AMP via :mod:`torch.amp` on CUDA (silently disabled on CPU).
- Warmup via :class:`~torch.optim.lr_scheduler.SequentialLR` composing
  :class:`~torch.optim.lr_scheduler.LinearLR` with the main schedule.
- MixUp batch mixing, pushed in through the ``mixup`` constructor arg
  rather than the DataLoader so soft-label targets survive collation.
- W&B logging with a graceful fallback: if ``wandb.init`` raises (no
  API key, no network, offline mode misconfigured) the loop still
  runs to completion and a warning is logged. ``WANDB_MODE=disabled``
  is also honored; ``wandb.log`` becomes a no-op in that case.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)
from torch.utils.data import DataLoader

import wandb

from ml.src.data.transforms import Mixup
from ml.src.evaluation.metrics import MetricsTracker
from ml.src.training.callbacks import (
    CheckpointManager,
    EarlyStopping,
    build_callbacks,
)
from ml.src.utils.config import KobunConfig, OptimizerConfig, SchedulerConfig


logger = logging.getLogger(__name__)


_BATCH_LOG_INTERVAL: int = 50


def build_optimizer(
    config: OptimizerConfig, model: nn.Module
) -> Optimizer:
    """Construct an optimizer over the trainable parameters of ``model``.

    Args:
        config: Validated :class:`OptimizerConfig`.
        model: Module whose trainable parameters are handed to the
            optimizer.

    Returns:
        An :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If ``config.name`` is not one of the supported keys.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = config.name
    if name == "adamw":
        return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "adam":
        return Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(
    config: SchedulerConfig,
    optimizer: Optimizer,
    num_epochs: int,
) -> Optional[LRScheduler]:
    """Build a per-epoch LR schedule, optionally prefixed with warmup.

    Warmup composition uses :class:`SequentialLR` so a single
    ``scheduler.step()`` per epoch drives both phases. ``LinearLR``
    ramps from ``start_factor=0.01`` to full LR over
    ``warmup_epochs``, then the main schedule takes over.

    Args:
        config: Validated :class:`SchedulerConfig`.
        optimizer: Optimizer to wrap.
        num_epochs: Total training epochs, used to size
            :class:`CosineAnnealingLR`.

    Returns:
        A scheduler, or ``None`` when ``config.name == "none"`` and no
        warmup is requested.
    """
    name = config.name
    warmup = config.warmup_epochs

    def _main() -> Optional[LRScheduler]:
        if name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=max(1, num_epochs - warmup),
                eta_min=config.min_lr,
            )
        if name == "step":
            return StepLR(
                optimizer, step_size=config.step_size, gamma=config.gamma
            )
        if name == "none":
            return None
        raise ValueError(f"unknown scheduler: {name!r}")

    if warmup <= 0:
        return _main()

    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup
    )
    main_sched = _main()
    if main_sched is None:
        return warmup_sched
    return SequentialLR(
        optimizer, [warmup_sched, main_sched], milestones=[warmup]
    )


class Trainer:
    """Coordinates a full training run from a :class:`KobunConfig`.

    Construction wires optimizer, scheduler, callbacks, AMP scaler,
    and device placement. :meth:`fit` runs the epoch loop; W&B is
    initialized lazily there so unit tests can instantiate a Trainer
    without touching the network.
    """

    def __init__(
        self,
        config: KobunConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        class_weights: Optional[torch.Tensor] = None,
        mixup: Optional[Mixup] = None,
        device: str = "auto",
    ) -> None:
        self.config: KobunConfig = config
        self.device: str = self._resolve_device(device)

        self.model: nn.Module = model.to(self.device)
        # Loss holds class-weight buffers (registered on SoftCrossEntropyLoss
        # and nn.CrossEntropyLoss(weight=...)); move with the model.
        self.loss_fn: nn.Module = loss_fn.to(self.device)
        self.class_weights: Optional[torch.Tensor] = (
            class_weights.to(self.device) if class_weights is not None else None
        )
        self.mixup: Optional[Mixup] = mixup
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        self.optimizer: Optimizer = build_optimizer(
            config.training.optimizer, self.model
        )
        self.scheduler: Optional[LRScheduler] = build_scheduler(
            config.training.scheduler,
            self.optimizer,
            config.training.epochs,
        )

        self.use_amp: bool = (
            config.training.mixed_precision and self.device.startswith("cuda")
        )
        if config.training.mixed_precision and not self.use_amp:
            logger.warning(
                "mixed_precision=True requested but device is %r; "
                "AMP is disabled. AMP only runs on CUDA.",
                self.device,
            )
        self.scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda") if self.use_amp else None
        )

        self.checkpoint_manager: CheckpointManager
        self.early_stopping: Optional[EarlyStopping]
        self.checkpoint_manager, self.early_stopping = build_callbacks(config)

        self._wandb_active: bool = False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.use_amp:
            return torch.amp.autocast(
                device_type="cuda", dtype=torch.float16
            )
        return contextlib.nullcontext()

    def _init_wandb(self) -> None:
        """Initialize W&B, tolerating init failures and disabled mode."""
        lc = self.config.logging
        try:
            wandb.init(
                project=lc.wandb_project,
                entity=lc.wandb_entity,
                name=lc.wandb_run_name,
                tags=list(lc.wandb_tags),
                config=self.config.model_dump(),
            )
            self._wandb_active = wandb.run is not None
        except Exception as exc:
            logger.warning(
                "W&B init failed (%s); continuing without run logging.",
                exc,
            )
            self._wandb_active = False
        if self._wandb_active:
            try:
                wandb.watch(self.model, log="gradients", log_freq=100)
            except Exception as exc:
                logger.warning(
                    "wandb.watch failed (%s); skipping gradient hooks.", exc
                )

    def _wandb_log(self, payload: dict[str, Any]) -> None:
        if self._wandb_active:
            wandb.log(payload)

    def fit(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Summary dict with keys ``best_metric``,
            ``best_checkpoint_path``, ``total_epochs_trained``, and
            ``early_stopped``.
        """
        self._init_wandb()

        tc = self.config.training
        ec = self.config.evaluation
        monitored_key = ec.metric  # "balanced_accuracy" | "top1_accuracy"

        total_epochs_trained = 0
        early_stopped = False

        try:
            for epoch in range(tc.epochs):
                train_metrics = self._train_epoch(epoch)
                total_epochs_trained = epoch + 1

                should_eval = (
                    epoch % ec.eval_every_n_epochs == 0
                    or epoch == tc.epochs - 1
                )
                if should_eval:
                    val_metrics = self._evaluate()
                    monitored = float(val_metrics[monitored_key])
                    checkpoint_data = self._save_checkpoint_data(
                        epoch, monitored
                    )
                    saved_path = self.checkpoint_manager.update(
                        epoch, monitored, checkpoint_data
                    )
                    log_payload: dict[str, Any] = {
                        "epoch": epoch,
                        **train_metrics,
                        "val_balanced_accuracy": val_metrics[
                            "balanced_accuracy"
                        ],
                        "val_top1_accuracy": val_metrics["top1_accuracy"],
                        "val_top3_accuracy": val_metrics["top3_accuracy"],
                        "val_per_class_min": val_metrics["per_class_min"],
                        "val_per_class_mean": val_metrics["per_class_mean"],
                        "val_per_class_std": val_metrics["per_class_std"],
                    }
                    if saved_path is not None:
                        log_payload["checkpoint_saved"] = str(saved_path)
                    self._wandb_log(log_payload)

                    if self.early_stopping is not None:
                        if self.early_stopping.step(monitored):
                            logger.info(
                                "Early stopping at epoch %d "
                                "(best %s=%.4f)",
                                epoch,
                                monitored_key,
                                self.early_stopping.best_metric,
                            )
                            early_stopped = True
                            break
                else:
                    self._wandb_log({"epoch": epoch, **train_metrics})

                if self.scheduler is not None:
                    self.scheduler.step()
        finally:
            if self._wandb_active:
                wandb.finish()

        best = self.checkpoint_manager.get_best()
        best_metric = best[0] if best is not None else float("nan")
        best_path = str(best[1]) if best is not None else None
        return {
            "best_metric": best_metric,
            "best_checkpoint_path": best_path,
            "total_epochs_trained": total_epochs_trained,
            "early_stopped": early_stopped,
        }

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """One epoch over ``train_loader``."""
        self.model.train()
        running_loss = 0.0
        running_samples = 0
        clip = self.config.training.gradient_clip_norm
        apply_mixup = self.mixup is not None and self.mixup.is_applied

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if apply_mixup:
                images, labels = self.mixup(images, labels)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Grad clipping requires unscaled grads; do it after
                # unscale_ but before scaler.step().
                if clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip
                    )
                self.optimizer.step()

            batch_size = images.shape[0]
            running_loss += float(loss.detach().item()) * batch_size
            running_samples += batch_size

            if (batch_idx + 1) % _BATCH_LOG_INTERVAL == 0:
                self._wandb_log(
                    {
                        "train_batch_loss": float(loss.detach().item()),
                        "train_batch_step": epoch * len(self.train_loader)
                        + batch_idx,
                    }
                )

        avg_loss = running_loss / max(1, running_samples)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return {"train_loss": avg_loss, "train_lr": current_lr}

    def _evaluate(self) -> dict[str, float]:
        """One pass over ``val_loader`` returning flat metric dict."""
        self.model.eval()
        tracker = MetricsTracker(num_classes=49, topk=(1, 3))
        with torch.no_grad(), self._autocast():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                tracker.update(logits, labels)

        results = tracker.compute()
        per_class = results["per_class_accuracy"]

        return {
            "balanced_accuracy": float(results["balanced_accuracy"]),
            "top1_accuracy": float(results["top1_accuracy"]),
            "top3_accuracy": float(results["top3_accuracy"]),
            "per_class_min": float(np.nanmin(per_class)),
            "per_class_mean": float(np.nanmean(per_class)),
            "per_class_std": float(np.nanstd(per_class)),
        }

    def _save_checkpoint_data(
        self, epoch: int, metric: float
    ) -> dict[str, Any]:
        """Assemble the dict passed to :meth:`CheckpointManager.update`."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict()
                if self.scaler is not None
                else None
            ),
            "epoch": epoch,
            "metric": metric,
            "config": self.config.model_dump(),
        }
