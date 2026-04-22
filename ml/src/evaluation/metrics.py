"""Classification metrics for Kuzushiji-49 evaluation.

Implements the Phase 1 evaluation suite from the Kobun spec (Bagian F):
balanced accuracy, top-k accuracy, per-class accuracy, confusion matrix,
and top confused pairs for error analysis.

The balanced accuracy formula follows Clanuwat et al. 2018 (the K49 paper):

    accs = []
    for cls in range(num_classes):
        mask = (y_true == cls)
        accs.append((y_pred == cls)[mask].mean())
    balanced_acc = np.mean(accs)

:func:`sklearn.metrics.balanced_accuracy_score` is used as the trusted
implementation; this module wraps it for consistent input handling
(numpy or torch tensor, CPU/GPU) and adds utilities oriented toward
per-class analysis and confusion-pair mining that the vanilla sklearn
API does not surface directly.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


NormalizeMode = Literal["true", "pred", "all"]


def _to_numpy_1d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce a 1-D label array (numpy or torch tensor) to int64 numpy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D label array, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def balanced_accuracy(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: int = 49,
) -> float:
    """Compute balanced accuracy (mean per-class recall).

    Follows the K49 paper formula (Clanuwat et al. 2018). Classes that
    never appear in ``y_true`` are excluded from the mean, matching the
    default behavior of :func:`sklearn.metrics.balanced_accuracy_score`.
    ``num_classes`` is retained for API symmetry with the other metrics
    in this module.

    Args:
        y_true: Ground-truth labels, 1-D numpy array or torch tensor.
        y_pred: Predicted labels (argmax of logits), 1-D, same length
            as ``y_true``.
        num_classes: Total class count (default 49 for K49). Not used
            by the underlying sklearn call; retained for API symmetry.

    Returns:
        Balanced accuracy as a ``float`` in ``[0.0, 1.0]``.

    Raises:
        ValueError: If inputs are not 1-D or shapes disagree.

    Example:
        >>> import numpy as np
        >>> bacc = balanced_accuracy(
        ...     np.array([0, 1, 2, 0]), np.array([0, 1, 2, 0])
        ... )
        >>> float(bacc)
        1.0
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must share shape; "
            f"got {yt.shape} vs {yp.shape}"
        )
    _ = num_classes
    return float(balanced_accuracy_score(yt, yp))


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """Top-k accuracy: fraction of samples whose label is in the top-k.

    Single-pass implementation using :meth:`torch.Tensor.topk`.

    Args:
        logits: Pre-softmax scores, shape ``(B, num_classes)``.
        targets: Integer labels, shape ``(B,)``.
        k: Number of top predictions to consider. Clipped to
            ``num_classes`` if larger.

    Returns:
        Accuracy in ``[0.0, 1.0]``.

    Raises:
        ValueError: If shapes are inconsistent.

    Example:
        >>> import torch
        >>> logits = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.1, 0.2]])
        >>> targets = torch.tensor([1, 0])
        >>> topk_accuracy(logits, targets, k=1)
        1.0
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (B, num_classes); got {tuple(logits.shape)}"
        )
    if targets.ndim != 1:
        raise ValueError(
            f"targets must be 1-D (B,); got {tuple(targets.shape)}"
        )
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"batch size mismatch: logits {logits.shape[0]} vs "
            f"targets {targets.shape[0]}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = (topk == targets.to(topk.device).unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def per_class_accuracy(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: int = 49,
) -> np.ndarray:
    """Per-class recall; ``NaN`` for classes absent from ``y_true``.

    Useful for identifying the worst classes (Phase 1 target: all
    above 85%). Callers can filter with ``~np.isnan(arr)`` or
    aggregate with :func:`numpy.nanmean`.

    Args:
        y_true: Ground-truth labels, 1-D.
        y_pred: Predicted labels, 1-D, same length as ``y_true``.
        num_classes: Length of the returned array.

    Returns:
        ``np.float64`` array of shape ``(num_classes,)`` where
        ``arr[c]`` is the recall for class ``c``, or ``NaN`` if no
        sample in ``y_true`` has label ``c``.

    Example:
        >>> import numpy as np
        >>> pca = per_class_accuracy(
        ...     np.array([0, 0, 1]), np.array([0, 1, 1]), num_classes=3
        ... )
        >>> pca[0], pca[1], bool(np.isnan(pca[2]))
        (0.5, 1.0, True)
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must share shape; "
            f"got {yt.shape} vs {yp.shape}"
        )
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        mask = yt == c
        if mask.any():
            out[c] = float((yp[mask] == c).mean())
    return out


def compute_confusion_matrix(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: int = 49,
    normalize: Optional[NormalizeMode] = None,
) -> np.ndarray:
    """Confusion matrix with consistent input handling and fixed shape.

    Thin wrapper over :func:`sklearn.metrics.confusion_matrix` that
    accepts torch tensors and always returns a full
    ``(num_classes, num_classes)`` matrix, even if some classes are
    absent from the batch.

    Args:
        y_true: Ground-truth labels, 1-D.
        y_pred: Predicted labels, 1-D.
        num_classes: Matrix dimension.
        normalize: sklearn normalization mode.

            - ``None`` — raw counts.
            - ``"true"`` — each row sums to 1 (per-true-class view).
            - ``"pred"`` — each column sums to 1.
            - ``"all"`` — whole matrix sums to 1.

    Returns:
        Matrix of shape ``(num_classes, num_classes)``. Rows are true
        labels, columns are predicted labels.
    """
    yt = _to_numpy_1d(y_true)
    yp = _to_numpy_1d(y_pred)
    labels = np.arange(num_classes)
    return confusion_matrix(yt, yp, labels=labels, normalize=normalize)


def find_top_confused_pairs(
    confusion_matrix: np.ndarray,
    top_n: int = 10,
    exclude_diagonal: bool = True,
) -> list[tuple[int, int, int]]:
    """Find the largest off-diagonal entries in a confusion matrix.

    These are the most common confusion patterns — e.g., pairs like
    (い, り) or (こ, ご) in K49 — and are the starting point for the
    Phase 5 error analysis.

    Args:
        confusion_matrix: Square 2-D matrix, typically the raw-count
            output of :func:`compute_confusion_matrix` with
            ``normalize=None``.
        top_n: Number of pairs to return. Fewer are returned if the
            matrix has fewer non-zero off-diagonal entries.
        exclude_diagonal: Drop ``(c, c)`` entries (correct predictions)
            before ranking.

    Returns:
        List of ``(true_class, pred_class, count)`` tuples sorted in
        descending order by count.

    Raises:
        ValueError: If ``confusion_matrix`` is not square 2-D.
    """
    if confusion_matrix.ndim != 2 or (
        confusion_matrix.shape[0] != confusion_matrix.shape[1]
    ):
        raise ValueError(
            f"confusion_matrix must be square 2-D; "
            f"got shape {confusion_matrix.shape}"
        )
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}")

    cm = confusion_matrix.copy()
    if exclude_diagonal:
        np.fill_diagonal(cm, 0)

    flat = cm.ravel()
    n = min(top_n, int((flat > 0).sum()))
    if n == 0:
        return []

    # argpartition locates the top-n unsorted, then sort just those.
    idx = np.argpartition(flat, -n)[-n:]
    idx = idx[np.argsort(-flat[idx])]
    num_classes = cm.shape[0]
    return [
        (int(i // num_classes), int(i % num_classes), int(flat[i]))
        for i in idx
    ]


class MetricsTracker:
    """Accumulate classification predictions across eval batches.

    Stores the top-``max(topk)`` predictions (not full logits) per
    batch as ``int64`` tensors on CPU. This keeps memory low for a
    full K49 validation pass — roughly 30K samples x 3 predictions
    fits in under 1 MB — and prevents the tracker from holding GPU
    memory between batches, which matters on the shared Kaggle P100.

    Usage::

        tracker = MetricsTracker(num_classes=49)
        for batch in dataloader:
            logits = model(batch.images)
            tracker.update(logits, batch.labels)
        results = tracker.compute()
    """

    def __init__(
        self,
        num_classes: int = 49,
        topk: tuple[int, ...] = (1, 3),
    ) -> None:
        if not topk:
            raise ValueError("topk must be a non-empty tuple")
        if any(k <= 0 for k in topk):
            raise ValueError(f"topk values must be positive; got {topk}")
        self.num_classes: int = num_classes
        self.topk: tuple[int, ...] = tuple(sorted(set(topk)))
        self._max_k: int = max(self.topk)
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def update(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Ingest one batch. Top-k predictions are moved to CPU."""
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be 2-D; got shape {tuple(logits.shape)}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"targets must be 1-D; got shape {tuple(targets.shape)}"
            )
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"batch size mismatch: logits {logits.shape[0]} vs "
                f"targets {targets.shape[0]}"
            )
        k = min(self._max_k, logits.shape[1])
        topk = logits.topk(k, dim=1).indices.detach().cpu().to(torch.int64)
        self._preds.append(topk)
        self._targets.append(targets.detach().cpu().to(torch.int64))

    def compute(self) -> dict[str, float | np.ndarray]:
        """Aggregate buffered batches into the metrics dict.

        Returns:
            Dict with keys:

            - ``"top{k}_accuracy"`` — one entry per configured ``k``.
            - ``"balanced_accuracy"`` — float.
            - ``"per_class_accuracy"`` — ``(num_classes,)`` array
              (``NaN`` for absent classes).
            - ``"confusion_matrix"`` — ``(num_classes, num_classes)``
              raw-count matrix.

        Raises:
            RuntimeError: If :meth:`update` was never called.
        """
        if not self._targets:
            raise RuntimeError(
                "MetricsTracker.compute called with no batches buffered"
            )
        preds = torch.cat(self._preds, dim=0)
        targets = torch.cat(self._targets, dim=0)
        top1 = preds[:, 0].numpy()
        targets_np = targets.numpy()

        results: dict[str, float | np.ndarray] = {}
        for k in self.topk:
            correct = (preds[:, :k] == targets.unsqueeze(1)).any(dim=1)
            results[f"top{k}_accuracy"] = float(correct.float().mean().item())
        results["balanced_accuracy"] = balanced_accuracy(
            targets_np, top1, num_classes=self.num_classes
        )
        results["per_class_accuracy"] = per_class_accuracy(
            targets_np, top1, num_classes=self.num_classes
        )
        results["confusion_matrix"] = compute_confusion_matrix(
            targets_np, top1, num_classes=self.num_classes
        )
        return results

    def reset(self) -> None:
        """Clear all buffers so the tracker can be re-used across epochs."""
        self._preds.clear()
        self._targets.clear()
