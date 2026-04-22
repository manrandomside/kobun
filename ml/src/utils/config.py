"""Config-driven training utilities for Kobun classification.

Training is fully driven by YAML config files in ``ml/configs/`` — there
are no hardcoded hyperparameters in ``train.py``. This keeps Phase 1
experiments reproducible and makes W&B sweeps a matter of overriding a
few keys rather than editing code.

The schema is expressed as a tree of Pydantic v2 models rooted at
:class:`KobunConfig`. Pydantic enforces types, numeric bounds, and
``Literal`` enums, and every model sets ``extra="forbid"`` so a
misspelled key in the YAML (for example ``epoch`` instead of
``epochs``) fails validation loudly instead of silently defaulting.

Public helpers:

- :func:`load_config` / :func:`save_config` — YAML round-trip with
  validation.
- :func:`override_config` — dotted-path overrides for sweeps, returned
  as a new instance so the original config is never mutated.
- :func:`set_seeds` — one-shot RNG seeding across ``random``, ``numpy``,
  and ``torch`` for reproducibility.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field


_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(BaseModel):
    """Architecture and classifier head parameters."""

    model_config = _STRICT

    name: Literal["baseline_cnn", "resnet50", "vit_base"]
    pretrained: bool = True
    num_classes: int = 49
    in_channels: int = 3
    drop_rate: float = 0.0


class AugmentationsConfig(BaseModel):
    """Per-sample augmentation knobs for the training transform."""

    model_config = _STRICT

    random_rotation_deg: float = 10.0
    random_erasing_prob: float = 0.25
    mixup_alpha: float = 0.0  # 0.0 disables mixup


class DataConfig(BaseModel):
    """Dataset, dataloader, and augmentation configuration."""

    model_config = _STRICT

    dataset: Literal["kuzushiji49"] = "kuzushiji49"
    data_dir: str = "ml/data/kuzushiji-49"
    image_size: int = 28
    channels: Literal[1, 3] = 3
    batch_size: int = 128
    num_workers: int = 4
    val_fraction: float = 0.15
    seed: int = 42
    augmentations: AugmentationsConfig = Field(
        default_factory=AugmentationsConfig
    )


class OptimizerConfig(BaseModel):
    """Optimizer family and shared hyperparameters."""

    model_config = _STRICT

    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # SGD only


class SchedulerConfig(BaseModel):
    """Learning-rate schedule."""

    model_config = _STRICT

    name: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    step_size: int = 30  # step only
    gamma: float = 0.1  # step only


class LossConfig(BaseModel):
    """Loss function selection, forwarded to :func:`ml.src.training.losses.build_loss`."""

    model_config = _STRICT

    type: Literal["ce", "weighted_ce", "soft_ce"] = "soft_ce"
    label_smoothing: float = 0.1
    class_weight_scheme: Literal[
        "inverse", "inverse_sqrt", "effective"
    ] = "inverse_sqrt"


class TrainingConfig(BaseModel):
    """Outer training loop parameters."""

    model_config = _STRICT

    epochs: int = 50
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    gradient_clip_norm: Optional[float] = None
    mixed_precision: bool = False  # AMP for ViT on P100
    early_stopping_patience: Optional[int] = None


class EvaluationConfig(BaseModel):
    """Evaluation cadence and primary metric."""

    model_config = _STRICT

    metric: Literal["balanced_accuracy", "top1_accuracy"] = "balanced_accuracy"
    test_time_augmentation: bool = False
    eval_every_n_epochs: int = 1


class LoggingConfig(BaseModel):
    """W&B + checkpointing + HF Hub upload targets."""

    model_config = _STRICT

    wandb_project: str = "kobun-classification"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = Field(default_factory=list)
    save_top_k: int = 3
    checkpoint_dir: str = "./checkpoints"
    hf_hub_repo: Optional[str] = None


class KobunConfig(BaseModel):
    """Top-level Kobun training config.

    Only :attr:`model` is required; every other section has sensible
    defaults matching the Phase 1 ResNet-50 recipe from the project
    spec (Bagian E).

    :attr:`config_path` is auto-populated by :func:`load_config` for
    traceability (so checkpoints can log which YAML they came from)
    and is stripped back out by :func:`save_config`.
    """

    model_config = _STRICT

    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    config_path: Optional[str] = None


def load_config(config_path: str | Path) -> KobunConfig:
    """Load and validate a Kobun training config from YAML.

    Args:
        config_path: Path to a YAML file whose top-level keys match the
            :class:`KobunConfig` schema.

    Returns:
        Validated :class:`KobunConfig` with :attr:`config_path` set to
        the absolute-or-relative path that was loaded.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the YAML root is not a mapping.
        pydantic.ValidationError: If any field fails validation
            (including unknown keys thanks to ``extra="forbid"``).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"config YAML root must be a mapping; got {type(data).__name__}"
        )
    cfg = KobunConfig.model_validate(data)
    cfg.config_path = str(path)
    return cfg


def save_config(config: KobunConfig, output_path: str | Path) -> None:
    """Serialize a :class:`KobunConfig` back to YAML.

    The :attr:`KobunConfig.config_path` field is excluded so saved
    snapshots stay portable. Nested keys preserve declaration order
    for readability rather than being sorted alphabetically.

    Args:
        config: Config to serialize.
        output_path: Destination path. Parent directories are created
            on demand.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude={"config_path"}, mode="python")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )


def override_config(
    config: KobunConfig, overrides: dict[str, Any]
) -> KobunConfig:
    """Apply dotted-path overrides and return a revalidated config.

    Designed for W&B sweeps: keys like ``"training.optimizer.lr"`` are
    resolved by walking the dumped dict and replacing the leaf value.
    The result is re-validated through :meth:`KobunConfig.model_validate`
    so overrides are type-checked and cannot smuggle in unknown fields
    (``extra="forbid"`` still applies).

    Args:
        config: Original config. Not mutated.
        overrides: Mapping from dotted paths to replacement values.
            Example: ``{"training.optimizer.lr": 3e-3, "data.batch_size": 256}``.

    Returns:
        A new :class:`KobunConfig` with overrides applied and the
        original :attr:`config_path` preserved.

    Raises:
        ValueError: If an override path traverses a non-mapping value.
        pydantic.ValidationError: If the merged config fails validation.
    """
    data = config.model_dump(exclude={"config_path"}, mode="python")
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        if not all(parts):
            raise ValueError(f"empty segment in override path {dotted!r}")
        node = data
        for segment in parts[:-1]:
            nxt = node.get(segment)
            if nxt is None:
                nxt = {}
                node[segment] = nxt
            elif not isinstance(nxt, dict):
                raise ValueError(
                    f"override path {dotted!r} traverses non-mapping "
                    f"value at {segment!r}"
                )
            node = nxt
        node[parts[-1]] = value
    new_cfg = KobunConfig.model_validate(data)
    new_cfg.config_path = config.config_path
    return new_cfg


def set_seeds(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` for reproducibility.

    Also flips cuDNN into deterministic mode
    (``cudnn.deterministic=True``, ``cudnn.benchmark=False``). This
    costs roughly 5-15% throughput on conv-heavy workloads but is
    required for the "reproduce from scratch" standard in spec
    Bagian K Phase 5.

    Args:
        seed: Integer seed broadcast to every RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
