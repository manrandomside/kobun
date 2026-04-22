"""Classification model builders for the Kobun Phase 1 pipeline.

Three architectures are wired up behind a single factory:

- ``baseline_cnn``: a compact 3-conv / 2-FC network trained from scratch as
  a sanity baseline (~137K parameters).
- ``resnet50``: ResNet-50 via ``timm``, the main pretrained baseline.
- ``vit_base``: ViT-Base via ``timm``, configured for the native K49 input
  (``img_size=28``, ``patch_size=4``) and trained from scratch — the
  published ``patch16_224`` checkpoints cannot transfer cleanly to this
  patch / image-size combination, so pretrained weights are unsupported
  for this variant. See :func:`build_classifier` for details.

All builders return an :class:`nn.Module` whose forward pass accepts
``(B, C, H, W)`` float tensors from :func:`ml.src.data.transforms.build_transforms`
and emits ``(B, num_classes)`` logits.
"""

from __future__ import annotations

import logging
from typing import Literal

import timm
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)

Architecture = Literal["baseline_cnn", "resnet50", "vit_base"]

# K49 is 28x28; a 4x4 patch grid gives 7x7 tokens, which is the smallest
# sensible ViT configuration before the sequence length collapses.
_VIT_IMG_SIZE: int = 28
_VIT_PATCH_SIZE: int = 4


class BaselineCNN(nn.Module):
    """Compact 3-conv + 2-FC CNN used as a Phase 1 sanity baseline.

    The architecture follows the spec's "warmup" model:
    ``conv -> conv -> conv -> avg pool -> fc -> fc``. BatchNorm is placed
    between conv and ReLU (standard post-conv / pre-activation in the
    classic sense) to stabilize training on the imbalanced K49 corpus.
    An adaptive pool before the classifier keeps the FC shape independent
    of minor changes to the input size.

    With ``in_channels=3`` the network lands at ~137K parameters, which
    stays within the "small sanity net" envelope the spec calls out
    (~100K) while remaining a clean, readable design.

    Attributes:
        features: The convolutional trunk.
        classifier: The FC head producing class logits.
    """

    def __init__(
        self,
        num_classes: int = 49,
        in_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the baseline CNN.

        Args:
            num_classes: Size of the classification head.
            in_channels: ``1`` for grayscale, ``3`` for RGB-tiled input
                matching :func:`build_transforms` with ``channels=3``.
            dropout: Dropout probability applied between the two FC
                layers. Set to ``0`` to disable.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_classifier(
    architecture: Architecture,
    num_classes: int = 49,
    pretrained: bool = True,
    in_channels: int = 3,
    drop_rate: float = 0.0,
) -> nn.Module:
    """Construct a Phase 1 classifier by name.

    Args:
        architecture: One of ``"baseline_cnn"``, ``"resnet50"``, or
            ``"vit_base"``.
        num_classes: Output head size. Defaults to 49 (K49).
        pretrained: If True, load ImageNet-pretrained weights. Ignored
            for ``baseline_cnn`` (from scratch by construction) and for
            ``vit_base`` (see note below); a warning is logged in both
            cases when ``True`` is passed.
        in_channels: Input channels. Defaults to 3 so models slot in
            directly behind :func:`build_transforms` with ``channels=3``.
            Set to 1 for single-channel grayscale inputs.
        drop_rate: Forwarded to ``timm.create_model`` as the classifier
            dropout rate for ResNet / ViT. For ``baseline_cnn`` it
            overrides the default 0.3 dropout when non-zero.

    Returns:
        An :class:`nn.Module` mapping ``(B, in_channels, H, W)`` to
        ``(B, num_classes)`` logits.

    Raises:
        ValueError: If ``architecture`` is not one of the supported keys.

    Note:
        The ViT branch uses ``timm``'s ``vit_base_patch16_224`` shell but
        overrides ``img_size=28`` and ``patch_size=4`` so the model runs
        on native K49 inputs (7x7 token grid). timm only ships
        ``patch16`` checkpoints at ``img_size=224``, so the pretrained
        weights' patch embedding and positional embedding cannot be
        reused; we build from scratch instead. If you want pretrained
        ViT benefits, upscale the input to 224x224 in
        ``build_transforms`` and use a separate non-Kobun call into
        ``timm.create_model`` — not supported here by design.
    """
    if architecture == "baseline_cnn":
        if pretrained:
            logger.warning(
                "pretrained=True is ignored for baseline_cnn; "
                "the baseline has no published weights."
            )
        dropout = drop_rate if drop_rate > 0 else 0.3
        return BaselineCNN(
            num_classes=num_classes,
            in_channels=in_channels,
            dropout=dropout,
        )

    if architecture == "resnet50":
        return timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=drop_rate,
        )

    if architecture == "vit_base":
        if pretrained:
            logger.warning(
                "pretrained=True is ignored for vit_base: the "
                "patch_size=4 / img_size=28 config is incompatible with "
                "timm's patch16_224 checkpoints."
            )
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_channels,
            drop_rate=drop_rate,
            img_size=_VIT_IMG_SIZE,
            patch_size=_VIT_PATCH_SIZE,
        )

    raise ValueError(
        f"architecture must be one of "
        f"'baseline_cnn', 'resnet50', 'vit_base'; got {architecture!r}"
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the parameter count of a model.

    Args:
        model: Any ``nn.Module``.
        trainable_only: If True, count only parameters with
            ``requires_grad=True``. Useful when parts of a backbone are
            frozen during fine-tuning.

    Returns:
        Total parameter count as a Python ``int``.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 28, 28),
) -> str:
    """Return a human-readable summary with params and I/O shapes.

    Performs a dummy forward on a zero tensor to capture the output
    shape, restoring ``training`` state afterward so repeated summaries
    stay idempotent. The forward is wrapped in ``torch.no_grad``.

    Args:
        model: The module to describe.
        input_shape: Shape of a single dummy batch, including the batch
            dimension. Defaults to a two-sample, 3-channel, 28x28 probe.

    Returns:
        A multi-line string with architecture name, total /
        trainable parameter counts, and input / output shapes. If the
        forward fails (e.g. shape mismatch) the output line reports the
        error instead of raising.
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    name = model.__class__.__name__

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(*input_shape)
            out = model(dummy)
        output_shape = tuple(out.shape)
        output_repr = f"{output_shape}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        output_repr = f"<forward failed: {type(exc).__name__}: {exc}>"
    finally:
        model.train(was_training)

    return (
        f"Architecture: {name}\n"
        f"Total params:     {total:>12,}\n"
        f"Trainable params: {trainable:>12,}\n"
        f"Input shape:  {tuple(input_shape)}\n"
        f"Output shape: {output_repr}"
    )
