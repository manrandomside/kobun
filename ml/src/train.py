"""CLI entry point for Kobun Phase 1 classification training.

Run via ``python -m ml.src.train --config ml/configs/<name>.yaml``. All
hyperparameters come from the YAML; the CLI only carries orchestration
knobs (dry run, resume, per-run overrides for W&B sweeps, seed/device
overrides).

Flow:

1. Load + validate YAML via :func:`load_config`.
2. Apply ``--override key.path=value`` pairs through
   :func:`override_config` (revalidates; typos fail).
3. Seed RNGs, make a dated checkpoint subdirectory, and snapshot the
   *effective* config to ``<ckpt_dir>/config.yaml`` for audit.
4. Build data loaders with the val-transform fix (see decisions in
   commit message: create_train_val_split's Subset inherits the train
   dataset's augmented transform, biasing val metrics; we reload the
   underlying arrays with the eval transform and Subset by the same
   indices).
5. Build model / loss / optional MixUp, short-circuit on ``--dry-run``.
6. Instantiate :class:`Trainer`, optionally restore from ``--resume``,
   call ``fit()``, log summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset

from ml.src.data.datasets import KuzushijiDataset, create_train_val_split
from ml.src.data.transforms import Mixup, build_transforms
from ml.src.models.classifier import build_classifier, count_parameters
from ml.src.training.losses import build_loss, compute_class_weights
from ml.src.training.trainer import Trainer
from ml.src.utils.config import (
    KobunConfig,
    load_config,
    override_config,
    save_config,
    set_seeds,
)


logger = logging.getLogger("ml.src.train")


def configure_logging(level: str = "INFO") -> None:
    """Set up root logging format. DEBUG=1 env overrides the level."""
    if os.environ.get("DEBUG") == "1":
        level = "DEBUG"
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _coerce_scalar(raw: str) -> Any:
    """Type-coerce an override RHS: int > float > bool > null > str.

    Order matters: ``"256"`` parses as int (not float), ``"3e-3"``
    falls through to float. ``bool(str)`` is not used — Python's
    truthiness would map every non-empty string to True; instead we
    match the literal tokens ``true`` / ``false`` case-insensitively.
    ``null`` / ``none`` yield ``None`` so overrides can clear optional
    fields.
    """
    lower = raw.lower()
    if lower in ("null", "none"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_overrides(override_strings: list[str]) -> dict[str, Any]:
    """Parse ``["a.b=1", "c=true"]`` into ``{"a.b": 1, "c": True}``.

    Args:
        override_strings: List of ``"key.path=value"`` items from the
            CLI.

    Returns:
        Flat dotted-path dict suitable for
        :func:`ml.src.utils.config.override_config`.

    Raises:
        ValueError: If any entry is missing ``=`` or has an empty key.
    """
    result: dict[str, Any] = {}
    for item in override_strings:
        if "=" not in item:
            raise ValueError(
                f"override {item!r} must be in 'key.path=value' form"
            )
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"override {item!r} has empty key")
        result[key] = _coerce_scalar(raw)
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. ``argv`` exposed for tests."""
    parser = argparse.ArgumentParser(
        prog="python -m ml.src.train",
        description="Train a Kobun Phase 1 classifier from a YAML config.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a Kobun training YAML config.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a .pt checkpoint to resume from (model/optimizer/"
        "scheduler/scaler state restored; epoch counter restarts at 0).",
    )
    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="Dotted-path overrides applied after loading the YAML. "
        "Re-validated through Pydantic so typos fail.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build config, data, model, and loss, then exit before "
        "entering the training loop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override data.seed from the config (reproducibility knob).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override trainer device: 'auto', 'cuda', or 'cpu'.",
    )
    return parser.parse_args(argv)


def _log_config_summary(config: KobunConfig) -> None:
    tc = config.training
    dc = config.data
    mc = config.model
    logger.info("Config summary:")
    logger.info("  model.name          = %s", mc.name)
    logger.info("  model.pretrained    = %s", mc.pretrained)
    logger.info("  data.batch_size     = %d", dc.batch_size)
    logger.info("  data.val_fraction   = %.3f", dc.val_fraction)
    logger.info("  data.seed           = %d", dc.seed)
    logger.info("  data.augment.mixup  = %.3f", dc.augmentations.mixup_alpha)
    logger.info("  training.epochs     = %d", tc.epochs)
    logger.info(
        "  training.optimizer  = %s lr=%.2e wd=%.2e",
        tc.optimizer.name,
        tc.optimizer.lr,
        tc.optimizer.weight_decay,
    )
    logger.info(
        "  training.scheduler  = %s warmup=%d",
        tc.scheduler.name,
        tc.scheduler.warmup_epochs,
    )
    logger.info(
        "  training.loss       = %s smoothing=%.2f weights=%s",
        tc.loss.type,
        tc.loss.label_smoothing,
        tc.loss.class_weight_scheme,
    )
    logger.info("  training.amp        = %s", tc.mixed_precision)


def _build_run_dir(config: KobunConfig) -> Path:
    """Create a timestamped checkpoint subdirectory, return its path."""
    base = Path(config.logging.checkpoint_dir)
    run_name = config.logging.wandb_run_name
    if run_name:
        run_dir = base / run_name
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = base / f"{config.model.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _build_loaders(
    config: KobunConfig,
    device: str,
) -> tuple[DataLoader, DataLoader, KuzushijiDataset]:
    """Build train / val loaders with the val-transform fix applied.

    Returns the train_loader, val_loader, and the training-transform
    dataset (used for class-count-based loss weights). The val Subset
    indexes into a second KuzushijiDataset instance loaded with the
    eval transform so the val metric is not polluted by augmentation.
    """
    dc = config.data
    train_tf = build_transforms(
        image_size=dc.image_size,
        mode="train",
        rotation_deg=dc.augmentations.random_rotation_deg,
        erasing_prob=dc.augmentations.random_erasing_prob,
        channels=dc.channels,
    )
    eval_tf = build_transforms(
        image_size=dc.image_size,
        mode="eval",
        channels=dc.channels,
    )
    train_full = KuzushijiDataset(
        data_dir=Path(dc.data_dir),
        split="train",
        transform=train_tf,
        load_classmap=False,
    )
    train_sub, val_sub_train_tf = create_train_val_split(
        train_full, val_fraction=dc.val_fraction, seed=dc.seed
    )

    val_full_eval = KuzushijiDataset(
        data_dir=Path(dc.data_dir),
        split="train",
        transform=eval_tf,
        load_classmap=False,
    )
    val_sub = Subset(val_full_eval, val_sub_train_tf.indices)

    pin_memory = device != "cpu"
    train_loader = DataLoader(
        train_sub,
        batch_size=dc.batch_size,
        shuffle=True,
        num_workers=dc.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=dc.batch_size,
        shuffle=False,
        num_workers=dc.num_workers,
        pin_memory=pin_memory,
    )
    logger.info(
        "Loaders: train=%d samples, val=%d samples, batch=%d, workers=%d",
        len(train_sub),
        len(val_sub),
        dc.batch_size,
        dc.num_workers,
    )
    return train_loader, val_loader, train_full


def _restore_checkpoint(
    trainer: Trainer, checkpoint_path: str
) -> None:
    """Load model/optimizer/scheduler/scaler state from a .pt file."""
    logger.info("Resuming from %s", checkpoint_path)
    ckpt = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
    trainer.model.load_state_dict(ckpt["model_state_dict"])
    trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None and trainer.scheduler:
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict") is not None and trainer.scaler:
        trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
    resumed_epoch = ckpt.get("epoch", -1) + 1
    logger.warning(
        "Resumed state loaded (was at epoch %d). Trainer.fit() does not "
        "accept a starting_epoch yet, so the epoch counter restarts at 0 "
        "for logging. LR/optimizer/scaler state continues correctly since "
        "their internal step counters were restored.",
        resumed_epoch - 1,
    )


def main(args: argparse.Namespace) -> int:
    """Run one training invocation. Return exit code (0 OK, 1 error)."""
    configure_logging()

    try:
        config = load_config(args.config)
        logger.info("Loaded config from %s", args.config)

        if args.override:
            overrides = parse_overrides(args.override)
            logger.info("Applying %d override(s): %s", len(overrides), overrides)
            config = override_config(config, overrides)

        if args.seed is not None:
            config.data.seed = args.seed
            logger.info("Seed overridden to %d", args.seed)

        set_seeds(config.data.seed)

        run_dir = _build_run_dir(config)
        config.logging.checkpoint_dir = str(run_dir)
        save_config(config, run_dir / "config.yaml")
        logger.info("Run directory: %s", run_dir)

        _log_config_summary(config)

        device = args.device or "auto"
        resolved_device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        logger.info("Target device: %s (requested=%s)", resolved_device, device)

        train_loader, val_loader, train_full = _build_loaders(
            config, resolved_device
        )

        model = build_classifier(
            architecture=config.model.name,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            in_channels=config.model.in_channels,
            drop_rate=config.model.drop_rate,
        )
        logger.info(
            "Model %s: %d trainable params",
            config.model.name,
            count_parameters(model, trainable_only=True),
        )

        class_weights: Optional[torch.Tensor] = None
        if config.training.loss.type in ("weighted_ce", "soft_ce"):
            counts = train_full.get_class_counts()
            class_weights = compute_class_weights(
                counts, scheme=config.training.loss.class_weight_scheme
            )
        loss_fn = build_loss(
            loss_type=config.training.loss.type,
            class_weights=class_weights,
            label_smoothing=config.training.loss.label_smoothing,
            num_classes=config.model.num_classes,
        )

        mixup: Optional[Mixup] = None
        if config.data.augmentations.mixup_alpha > 0:
            mixup = Mixup(
                alpha=config.data.augmentations.mixup_alpha,
                num_classes=config.model.num_classes,
            )
            logger.info(
                "MixUp enabled with alpha=%.3f",
                config.data.augmentations.mixup_alpha,
            )

        if args.dry_run:
            logger.info("Dry run complete: config valid, data pipeline OK.")
            return 0

        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            class_weights=class_weights,
            mixup=mixup,
            device=device,
        )

        if args.resume:
            _restore_checkpoint(trainer, args.resume)

        summary = trainer.fit()
        logger.info("Training complete. Summary:")
        logger.info("  best_metric             = %s", summary["best_metric"])
        logger.info(
            "  best_checkpoint_path    = %s", summary["best_checkpoint_path"]
        )
        logger.info(
            "  total_epochs_trained    = %d", summary["total_epochs_trained"]
        )
        logger.info("  early_stopped           = %s", summary["early_stopped"])
        return 0

    except Exception:
        logger.error("Training failed:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main(parse_args()))
