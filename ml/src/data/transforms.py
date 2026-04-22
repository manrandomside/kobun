"""Augmentation pipelines for Kuzushiji-49 classification.

Two public surfaces:

- :func:`build_transforms` returns an :class:`albumentations.Compose` that
  accepts a ``(28, 28)`` uint8 grayscale array (what
  :class:`ml.src.data.datasets.KuzushijiDataset` emits) and returns a
  normalized ``torch.Tensor`` in CHW float32 form. Albumentations handles
  the 2D-to-HWC broadcast internally, so the callers do not have to
  reshape.
- :class:`Mixup` is the batch-level mixing operator. It lives here for
  topical grouping, but it is applied inside the training loop (on an
  already-batched ``(B, C, H, W)`` tensor), not inside the DataLoader.

Default normalization statistics follow the convention of the downstream
backbone: ImageNet RGB stats when ``channels=3`` (matches
pretrained ResNet/ViT from ``timm``) and mean/std ``0.5`` when
``channels=1``. Call :func:`compute_dataset_stats` if you want to swap in
K49-specific statistics later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

if TYPE_CHECKING:
    from ml.src.data.datasets import KuzushijiDataset


# ImageNet statistics — the canonical choice for pretrained backbones.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Neutral single-channel stats: centers [0, 1] intensities to [-1, 1].
_GRAYSCALE_MEAN: tuple[float] = (0.5,)
_GRAYSCALE_STD: tuple[float] = (0.5,)

_NATIVE_IMAGE_SIZE: int = 28


def build_transforms(
    image_size: int = 28,
    mode: Literal["train", "eval"] = "train",
    rotation_deg: float = 10.0,
    erasing_prob: float = 0.25,
    channels: Literal[1, 3] = 3,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> A.Compose:
    """Build an Albumentations pipeline for K49 classification.

    The returned ``Compose`` is invoked as ``compose(image=arr)["image"]``
    and yields a ``torch.Tensor`` of shape ``(channels, image_size,
    image_size)`` with dtype ``float32``. Input arrays may be either
    ``(28, 28)`` or ``(28, 28, 1)`` uint8; Albumentations handles both.

    Args:
        image_size: Target spatial size. A resize step is inserted only
            when this differs from the native ``28`` so the common path
            stays free of an unnecessary interpolation.
        mode: ``"train"`` enables rotation, small affine, and coarse
            dropout (cutout). ``"eval"`` runs resize + normalize + tensor
            conversion only — no stochastic ops.
        rotation_deg: Max rotation angle in degrees. K49 glyphs survive
            moderate rotation; ``~10`` is the range used by the Clanuwat
            reference implementations.
        erasing_prob: Probability of applying :class:`CoarseDropout`.
            Mapped from the spec's ``random_erasing`` knob.
        channels: ``3`` tiles the grayscale input across RGB so
            ImageNet-pretrained backbones see the expected layout; ``1``
            keeps it single-channel (for the baseline CNN).
        mean: Per-channel normalization mean. Defaults to ImageNet when
            ``channels=3`` and ``(0.5,)`` when ``channels=1``.
        std: Per-channel normalization std. Defaults follow the same
            rule as ``mean``.

    Returns:
        An :class:`albumentations.Compose` suitable for passing as the
        ``transform`` argument of :class:`KuzushijiDataset`.

    Raises:
        ValueError: If ``mode`` or ``channels`` is outside the documented
            set, or if the lengths of ``mean`` / ``std`` disagree with
            ``channels``.
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels!r}")

    resolved_mean, resolved_std = _resolve_normalize_stats(channels, mean, std)

    ops: list[A.BasicTransform] = []

    if image_size != _NATIVE_IMAGE_SIZE:
        ops.append(A.Resize(height=image_size, width=image_size))

    if mode == "train":
        # Rotation with a constant black fill preserves the MNIST-style
        # background convention (K49 is white-ink on black). ``fill``
        # replaces the deprecated ``value`` kwarg in Albumentations 1.4+.
        ops.append(
            A.Rotate(
                limit=rotation_deg,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.8,
            )
        )
        ops.append(
            A.Affine(
                translate_percent=(0.0, 0.05),
                scale=(0.9, 1.1),
                shear=(-5, 5),
                p=0.5,
            )
        )
        # Cutout equivalent. The modern *_range kwargs replace the older
        # max_holes / min_holes / max_height style.
        ops.append(
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(4, 8),
                hole_width_range=(4, 8),
                p=erasing_prob,
            )
        )

    if channels == 3:
        # A.ToRGB broadcasts a single-channel (H, W) or (H, W, 1) image
        # to (H, W, 3) by tiling. Doing it inside Compose keeps the
        # dataset-side code style-agnostic.
        ops.append(A.ToRGB(p=1.0))

    ops.append(A.Normalize(mean=resolved_mean, std=resolved_std))
    ops.append(ToTensorV2())

    return A.Compose(ops)


def _resolve_normalize_stats(
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if mean is None:
        mean = _IMAGENET_MEAN if channels == 3 else _GRAYSCALE_MEAN
    if std is None:
        std = _IMAGENET_STD if channels == 3 else _GRAYSCALE_STD
    if len(mean) != channels or len(std) != channels:
        raise ValueError(
            f"mean/std length must equal channels={channels}; "
            f"got mean={mean}, std={std}"
        )
    return mean, std


class Mixup:
    """Batch-level MixUp regularizer.

    MixUp interpolates pairs of samples within a batch and their one-hot
    labels using a mixing coefficient drawn from ``Beta(alpha, alpha)``.
    ``alpha=0.2`` is the standard setting for image classification and
    matches the project spec (Phase 1 ResNet-50 recipe).

    Unlike the per-sample ops in :func:`build_transforms`, MixUp requires
    the full batch tensor, so it is invoked inside the training step
    rather than through the DataLoader. Keeping it in this module groups
    the augmentation logic in one place.
    """

    def __init__(self, alpha: float = 0.2, num_classes: int = 49) -> None:
        """Initialize the mixer.

        Args:
            alpha: Beta distribution parameter. ``0`` disables MixUp (see
                :attr:`is_applied`).
            num_classes: Number of target classes used to build soft
                label vectors. Defaults to the K49 size.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.alpha: float = alpha
        self.num_classes: int = num_classes

    @property
    def is_applied(self) -> bool:
        """True when :meth:`__call__` would actually perturb the batch."""
        return self.alpha > 0

    def __call__(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp to a batch.

        Args:
            images: ``(B, C, H, W)`` float tensor.
            labels: ``(B,)`` integer class indices.

        Returns:
            A tuple ``(mixed_images, mixed_labels)``. ``mixed_images``
            has the same shape as the input; ``mixed_labels`` has shape
            ``(B, num_classes)`` with rows summing to ``1.0``.

        Raises:
            ValueError: If ``images`` / ``labels`` batch sizes disagree.
        """
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "images and labels must share batch size; got "
                f"{images.shape[0]} vs {labels.shape[0]}"
            )
        labels_onehot = F.one_hot(labels, self.num_classes).to(images.dtype)
        if not self.is_applied:
            return images, labels_onehot

        lam = float(np.random.beta(self.alpha, self.alpha))
        perm = torch.randperm(images.shape[0], device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[perm]
        mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_onehot[perm]
        return mixed_images, mixed_labels


def compute_dataset_stats(dataset: "KuzushijiDataset") -> tuple[float, float]:
    """Compute per-pixel mean and std over a K49 dataset split.

    Results are scaled to the ``[0, 1]`` range to match what
    :class:`albumentations.Normalize` expects, and can be fed back into
    :func:`build_transforms` via ``mean=(m,)`` / ``std=(s,)`` for
    single-channel setups. This is a convenience for future K49-native
    normalization experiments; by default we ship ImageNet stats to keep
    pretrained backbones happy.

    Args:
        dataset: Any :class:`KuzushijiDataset` instance. Reads the raw
            ``images`` array directly, bypassing the ``transform``.

    Returns:
        Tuple ``(mean, std)`` of Python ``float`` values in ``[0, 1]``.
    """
    pixels = dataset.images.astype(np.float32, copy=False) / 255.0
    return float(pixels.mean()), float(pixels.std())
