"""Lifecycle hooks for the Kobun training loop.

Two callbacks, both designed to work standalone but composed by
:class:`ml.src.training.trainer.Trainer`:

- :class:`CheckpointManager` keeps a rolling top-K snapshot of model
  state based on a monitored metric (e.g., validation balanced
  accuracy). Uses a heap so the replace-worst-kept operation is
  O(log K) even for large K.
- :class:`EarlyStopping` watches the same monitored metric and signals
  the loop to stop after ``patience`` consecutive epochs without
  improvement.

The :func:`build_callbacks` factory reads the Phase 1 config and
returns both with sane defaults: checkpoints always on, early stopping
disabled unless ``config.training.early_stopping_patience`` is set.
"""

from __future__ import annotations

import heapq
import logging
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from ml.src.utils.config import KobunConfig


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Keep the top-K checkpoints by a monitored metric.

    Internally holds a min-heap keyed so that the "worst kept"
    checkpoint always sits at the root. When a new checkpoint scores
    better than the root, the root is popped, its ``.pt`` file deleted,
    and the new checkpoint saved in its place.

    A monotonic counter is pushed alongside the metric as a tie-breaker
    so the heap never has to compare :class:`pathlib.Path` objects
    directly (which is not ordered on all platforms).

    Attributes:
        checkpoint_dir: Directory where ``.pt`` files are written.
        k: Maximum number of checkpoints to keep.
        mode: ``"max"`` for higher-is-better metrics (accuracy),
            ``"min"`` for lower-is-better (loss).
        filename_template: :meth:`str.format`-style template with
            ``{epoch}`` and ``{metric}`` placeholders.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        k: int = 3,
        mode: Literal["max", "min"] = "max",
        filename_template: str = "epoch{epoch:03d}_metric{metric:.4f}.pt",
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.k: int = k
        self.mode: Literal["max", "min"] = mode
        self.filename_template: str = filename_template
        # Heap entries: (sort_key, counter, metric_value, path).
        self._heap: list[tuple[float, int, float, Path]] = []
        self._counter: int = 0

    def _sort_key(self, metric: float) -> float:
        """Lower is worse. Root of the min-heap is always the next evict."""
        return metric if self.mode == "max" else -metric

    def update(
        self,
        epoch: int,
        metric_value: float,
        checkpoint_data: dict[str, Any],
    ) -> Optional[Path]:
        """Consider a new checkpoint for inclusion.

        Args:
            epoch: Epoch index, used in the filename.
            metric_value: Monitored metric for this checkpoint.
            checkpoint_data: Dict passed straight to :func:`torch.save`.
                Typically contains ``model_state_dict``,
                ``optimizer_state_dict``, ``scheduler_state_dict``,
                ``scaler_state_dict``, ``epoch``, ``metric``, and
                ``config``.

        Returns:
            The path written, or ``None`` if the checkpoint was not
            in the top-K and therefore not saved.
        """
        key = self._sort_key(metric_value)
        keep = len(self._heap) < self.k or key > self._heap[0][0]
        if not keep:
            return None

        filename = self.filename_template.format(
            epoch=epoch, metric=metric_value
        )
        path = self.checkpoint_dir / filename
        torch.save(checkpoint_data, path)

        heapq.heappush(
            self._heap, (key, self._counter, metric_value, path)
        )
        self._counter += 1

        if len(self._heap) > self.k:
            _, _, _, evicted = heapq.heappop(self._heap)
            evicted.unlink(missing_ok=True)

        return path

    def get_best(self) -> Optional[tuple[float, Path]]:
        """Return ``(best_metric, best_path)`` or ``None`` if empty."""
        if not self._heap:
            return None
        # The "best" is the entry with the largest sort_key — i.e., the
        # one farthest from eviction. Heap root is the worst kept.
        best = max(self._heap, key=lambda item: item[0])
        return best[2], best[3]

    def get_top_k(self) -> list[tuple[float, Path]]:
        """Return kept checkpoints sorted best-first."""
        return [
            (metric, path)
            for _, _, metric, path in sorted(
                self._heap, key=lambda item: item[0], reverse=True
            )
        ]

    def reset(self) -> None:
        """Clear the in-memory heap. Existing ``.pt`` files are untouched."""
        self._heap.clear()
        self._counter = 0


class EarlyStopping:
    """Signal the loop to stop after no improvement for ``patience`` epochs.

    Attributes:
        patience: Number of consecutive non-improving epochs tolerated.
        mode: ``"max"`` for higher-is-better, ``"min"`` for lower.
        min_delta: Minimum absolute change in the monitored metric to
            count as an improvement. Guards against noise.
        stopped: ``True`` once the patience has been exceeded.
        epochs_without_improvement: Current counter.
        best_metric: Best metric seen so far, or ``None`` before the
            first :meth:`step`.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: Literal["max", "min"] = "max",
        min_delta: float = 0.0,
    ) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be positive; got {patience}")
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min'; got {mode!r}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0; got {min_delta}")
        self.patience: int = patience
        self.mode: Literal["max", "min"] = mode
        self.min_delta: float = min_delta
        self.stopped: bool = False
        self.epochs_without_improvement: int = 0
        self.best_metric: Optional[float] = None

    def _is_improvement(self, metric_value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return metric_value > self.best_metric + self.min_delta
        return metric_value < self.best_metric - self.min_delta

    def step(self, metric_value: float) -> bool:
        """Record a new metric reading and return ``True`` if stopping."""
        if self._is_improvement(metric_value):
            self.best_metric = metric_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.patience:
            self.stopped = True
        return self.stopped

    def reset(self) -> None:
        self.stopped = False
        self.epochs_without_improvement = 0
        self.best_metric = None


def build_callbacks(
    config: KobunConfig,
) -> tuple[CheckpointManager, Optional[EarlyStopping]]:
    """Construct the standard callback pair from a :class:`KobunConfig`.

    :class:`CheckpointManager` is always returned. :class:`EarlyStopping`
    is returned only when ``config.training.early_stopping_patience`` is
    set; otherwise the second element is ``None`` and the caller should
    skip the early-stop branch.

    Args:
        config: Validated :class:`KobunConfig`.

    Returns:
        Tuple ``(checkpoint_manager, early_stopping_or_none)``.
    """
    ckpt = CheckpointManager(
        checkpoint_dir=config.logging.checkpoint_dir,
        k=config.logging.save_top_k,
        mode="max",
    )
    if config.training.early_stopping_patience is None:
        return ckpt, None
    early = EarlyStopping(
        patience=config.training.early_stopping_patience,
        mode="max",
    )
    return ckpt, early
