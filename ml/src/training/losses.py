"""Loss functions for imbalanced Kuzushiji-49 classification.

Kuzushiji-49 has a ~25:1 class imbalance (min ~777, max ~6000 samples),
so the Phase 1 training recipe stacks three regularizers on top of plain
cross-entropy:

1. **Class-weighted loss** — :func:`compute_class_weights` produces per-class
   weights following either inverse frequency, inverse-square-root
   (Clanuwat 2018 convention, the project default), or the
   Class-Balanced "effective number of samples" formulation from Cui et
   al. 2019 (arXiv:1901.05555).
2. **Label smoothing** — prevents the classifier from driving logits to
   infinity on the dominant classes (Szegedy et al. 2016).
3. **Mixup compatibility** — :class:`SoftCrossEntropyLoss` accepts either
   hard integer labels (``(B,)``) or soft label distributions
   (``(B, num_classes)``) produced by :class:`ml.src.data.transforms.Mixup`,
   dispatching by target rank so the same loss object works for both
   training paths.

Choose :class:`SoftCrossEntropyLoss` (or ``build_loss("soft_ce")``) when
Mixup is enabled; the standard ``nn.CrossEntropyLoss`` does not accept
the probability-vector targets that Mixup produces.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ClassWeightScheme = Literal["inverse", "inverse_sqrt", "effective"]
LossType = Literal["ce", "weighted_ce", "soft_ce"]


def compute_class_weights(
    class_counts: np.ndarray,
    scheme: ClassWeightScheme = "inverse_sqrt",
    beta: float = 0.9999,
) -> torch.Tensor:
    """Compute per-class loss weights from a count vector.

    Three schemes are supported. ``inverse_sqrt`` is the project default
    because it matches the Clanuwat (2018) Kuzushiji reference recipe
    and produces weights in a tighter range than plain inverse
    frequency.

    Args:
        class_counts: Integer-like array of shape ``(num_classes,)`` with
            the per-class sample counts. Typically the output of
            :meth:`ml.src.data.datasets.KuzushijiDataset.get_class_counts`.
        scheme: Weighting formula.

            - ``"inverse"`` — ``w_i = N / (K * count_i)``. This is
              sklearn's ``class_weight="balanced"`` rule. Simple but
              can produce extreme weights when a class is very rare.
            - ``"inverse_sqrt"`` — ``w_i = 1 / sqrt(count_i)``,
              renormalized so ``sum(w) == K``. Softer than ``inverse``;
              follows Clanuwat 2018 Section 4.2.
            - ``"effective"`` — Class-Balanced Loss from Cui et al. 2019
              (arXiv:1901.05555): ``w_i = (1 - beta) / (1 - beta**count_i)``
              renormalized so ``sum(w) == K``. Smoothly interpolates
              between uniform (``beta -> 0``) and inverse frequency
              (``beta -> 1``).

        beta: Hyperparameter for ``"effective"``. Ignored otherwise.
            ``0.9999`` is the value recommended in the paper for
            long-tailed corpora.

    Returns:
        ``torch.float32`` tensor of shape ``(num_classes,)`` ready to
        pass as the ``weight`` argument of a cross-entropy loss.

    Raises:
        ValueError: If ``class_counts`` contains a zero (undefined
            weight), if ``scheme`` is unknown, or if ``beta`` is outside
            ``(0, 1)`` for the effective scheme.

    Example:
        >>> counts = np.array([6000, 1200, 777, 3000])
        >>> w = compute_class_weights(counts, scheme="inverse_sqrt")
        >>> float(w.sum())  # doctest: +ELLIPSIS
        4.0...
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(
            f"class_counts must be 1-D, got shape {counts.shape}"
        )
    if (counts <= 0).any():
        raise ValueError(
            "class_counts must be strictly positive; "
            f"found {int((counts <= 0).sum())} non-positive entries"
        )

    num_classes = counts.shape[0]

    if scheme == "inverse":
        weights = counts.sum() / (num_classes * counts)
    elif scheme == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.sum() * num_classes
    elif scheme == "effective":
        if not 0.0 < beta < 1.0:
            raise ValueError(
                f"beta must be in (0, 1) for the effective scheme, got {beta}"
            )
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(
            "scheme must be one of 'inverse', 'inverse_sqrt', 'effective'; "
            f"got {scheme!r}"
        )

    return torch.tensor(weights, dtype=torch.float32)


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy loss that also accepts soft target distributions.

    ``nn.CrossEntropyLoss`` in PyTorch 2.x technically accepts float
    probability targets, but the interaction with ``label_smoothing``
    and ``weight`` under soft targets is not well-specified. This module
    makes the two paths explicit:

    - When ``target`` is 1-D integer (shape ``(B,)``), it delegates to
      :func:`F.cross_entropy` with the provided ``weight`` and
      ``label_smoothing``. This matches standard training.
    - When ``target`` is 2-D float (shape ``(B, num_classes)`` — what
      :class:`Mixup` produces), it computes the weighted soft-label
      cross-entropy manually:
      ``loss = -sum_i(w_i * target_i * log_softmax(logits)_i)``, averaged
      over the batch, with label smoothing folded into the target
      distribution before the dot product.

    Args:
        weight: Optional per-class weight tensor of shape
            ``(num_classes,)``. Usually the output of
            :func:`compute_class_weights`.
        label_smoothing: Label smoothing factor in ``[0, 1)``. ``0``
            disables it.

    Example:
        >>> crit = SoftCrossEntropyLoss(label_smoothing=0.1)
        >>> logits = torch.randn(4, 49)
        >>> hard = torch.randint(0, 49, (4,))
        >>> soft = torch.softmax(torch.randn(4, 49), dim=1)
        >>> _ = crit(logits, hard)
        >>> _ = crit(logits, soft)
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1); got {label_smoothing}"
            )
        # Register as buffer so .to(device) moves the weight with the module.
        if weight is not None:
            self.register_buffer("weight", weight.to(torch.float32))
        else:
            self.weight: Optional[torch.Tensor] = None
        self.label_smoothing: float = label_smoothing

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if target.dim() == 1:
            return F.cross_entropy(
                logits,
                target,
                weight=self.weight,
                label_smoothing=self.label_smoothing,
            )
        if target.dim() == 2:
            return self._soft_target_forward(logits, target)
        raise ValueError(
            "SoftCrossEntropyLoss target must be 1-D (hard labels) or "
            f"2-D (soft labels); got shape {tuple(target.shape)}"
        )

    def _soft_target_forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if target.shape != logits.shape:
            raise ValueError(
                "soft target shape must match logits; got "
                f"target {tuple(target.shape)} vs logits {tuple(logits.shape)}"
            )
        num_classes = logits.shape[1]
        target = target.to(logits.dtype)

        # Redistribute a fraction of mass uniformly across classes.
        # Mirrors nn.CrossEntropyLoss's smoothing rule for the hard case.
        if self.label_smoothing > 0:
            ls = self.label_smoothing
            target = target * (1.0 - ls) + ls / num_classes

        log_probs = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            log_probs = log_probs * self.weight.to(
                device=log_probs.device, dtype=log_probs.dtype
            ).unsqueeze(0)
        return -(target * log_probs).sum(dim=1).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Convenience loss for hard labels with label smoothing only.

    A thin wrapper over :func:`F.cross_entropy` for the common case
    where class weighting and soft targets are not needed. Prefer
    :class:`SoftCrossEntropyLoss` when Mixup is enabled.

    Args:
        smoothing: Label smoothing factor in ``[0, 1)``.
        num_classes: Retained for API symmetry with other losses; not
            needed by ``F.cross_entropy`` itself.
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 49) -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(
                f"smoothing must be in [0, 1); got {smoothing}"
            )
        self.smoothing: float = smoothing
        self.num_classes: int = num_classes

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.smoothing)


def build_loss(
    loss_type: LossType = "soft_ce",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    num_classes: int = 49,
) -> nn.Module:
    """Construct a classification loss by name.

    Args:
        loss_type: Which loss to instantiate.

            - ``"ce"`` — plain :class:`nn.CrossEntropyLoss`. Baseline
              sanity; ignores ``class_weights`` and ``label_smoothing``.
            - ``"weighted_ce"`` — :class:`nn.CrossEntropyLoss` with
              ``weight`` and ``label_smoothing``. Fast path for
              hard-label training only; not Mixup-compatible.
            - ``"soft_ce"`` — :class:`SoftCrossEntropyLoss`. Accepts
              both hard and soft targets and is the recommended
              default when Mixup is used anywhere in the training loop.

        class_weights: Per-class weight tensor for the weighted / soft
            variants. Produced by :func:`compute_class_weights`.
        label_smoothing: Label smoothing factor; ignored for ``"ce"``.
        num_classes: Retained for API symmetry.

    Returns:
        An :class:`nn.Module` mapping ``(logits, target)`` to a scalar
        loss tensor.

    Raises:
        ValueError: If ``loss_type`` is unknown.

    Example:
        >>> import numpy as np
        >>> counts = np.array([6000, 1200, 777, 3000])
        >>> weights = compute_class_weights(counts)
        >>> loss = build_loss("soft_ce", class_weights=weights, label_smoothing=0.1,
        ...                   num_classes=4)
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if loss_type == "soft_ce":
        return SoftCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    raise ValueError(
        f"loss_type must be one of 'ce', 'weighted_ce', 'soft_ce'; "
        f"got {loss_type!r}"
    )
