"""Dataset loaders for the Kuzushiji-49 classification corpus.

Kuzushiji-49 ships as four NumPy ``.npz`` archives: 232,365 training and
38,547 test images at 28x28 grayscale across 49 classes (48 hiragana plus the
iteration mark). The full corpus is roughly 50 MB, so we load the chosen
split eagerly into memory and return views from ``__getitem__``; this is
faster than repeated disk I/O and fits comfortably on an 8 GB laptop.

Augmentation is delegated to the caller via a ``transform`` callable. Both
the Albumentations style (``fn(image=img)["image"]``) and the torchvision /
positional style (``fn(img)``) are supported. The style is auto-detected the
first time ``__getitem__`` applies a transform and then cached, so the
dispatch cost is paid once per dataset instance.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


NUM_CLASSES = 49
DEFAULT_DATA_DIR = Path("ml/data/kuzushiji-49")

_SPLIT_FILES: dict[str, tuple[str, str]] = {
    "train": ("k49-train-imgs.npz", "k49-train-labels.npz"),
    "test": ("k49-test-imgs.npz", "k49-test-labels.npz"),
}
_CLASSMAP_FILENAME = "k49_classmap.csv"

_TRANSFORM_MODE_KEYWORD = "keyword"
_TRANSFORM_MODE_POSITIONAL = "positional"


class KuzushijiDataset(Dataset):
    """Kuzushiji-49 classification dataset backed by in-memory NumPy arrays.

    All images for the chosen split are loaded once in ``__init__``. The
    ``transform`` argument, if provided, is applied lazily per sample in
    ``__getitem__``.

    Attributes:
        data_dir: Directory the arrays were loaded from.
        split: Either ``"train"`` or ``"test"``.
        transform: The per-sample transform callable, or ``None``.
        images: ``np.ndarray`` of shape ``(N, 28, 28)`` with dtype ``uint8``.
        labels: ``np.ndarray`` of shape ``(N,)`` with dtype ``int64``.
        num_classes: Always 49 for this corpus.
        classmap: DataFrame indexed by class id (0..48) with columns
            ``codepoint`` (e.g. ``U+3042``) and ``char`` (e.g. ``あ``).
            ``None`` when ``load_classmap=False``.
    """

    num_classes: int = NUM_CLASSES

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        split: Literal["train", "test"] = "train",
        transform: Optional[Callable[..., Any]] = None,
        load_classmap: bool = True,
    ) -> None:
        """Load a single split into memory.

        Args:
            data_dir: Directory containing the four K49 ``.npz`` files plus
                the optional class-map CSV.
            split: Which split to materialize.
            transform: Optional per-sample callable applied in
                ``__getitem__``.
            load_classmap: If True, load ``k49_classmap.csv`` into
                ``self.classmap`` for Unicode / glyph lookups.

        Raises:
            ValueError: If ``split`` is not ``"train"`` or ``"test"``, or if
                image and label counts disagree after loading.
            FileNotFoundError: If any required ``.npz`` archive (or the
                class-map when requested) is missing.
        """
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"split must be one of {list(_SPLIT_FILES)}, got {split!r}"
            )

        self.data_dir: Path = Path(data_dir)
        self.split: str = split
        self.transform: Optional[Callable[..., Any]] = transform

        img_name, lbl_name = _SPLIT_FILES[split]
        self.images: np.ndarray = self._load_npz(self.data_dir / img_name).astype(
            np.uint8, copy=False
        )
        self.labels: np.ndarray = self._load_npz(self.data_dir / lbl_name).astype(
            np.int64, copy=False
        )

        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError(
                "image/label length mismatch: "
                f"{self.images.shape[0]} images vs {self.labels.shape[0]} labels"
            )

        self.classmap: Optional[pd.DataFrame] = None
        if load_classmap:
            self.classmap = self._load_classmap(
                self.data_dir / _CLASSMAP_FILENAME
            )

        # Resolved lazily on the first transform call. Caches the calling
        # convention so we pay the inspect.signature cost only once.
        self._transform_mode: Optional[str] = None

    @staticmethod
    def _load_npz(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(
                f"expected K49 array at {path}. "
                "Run `python ml/scripts/download_data.py --dataset classification` "
                "to materialize the dataset."
            )
        with np.load(path) as archive:
            return archive["arr_0"]

    @staticmethod
    def _load_classmap(path: Path) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(
                f"expected class-map at {path}. "
                "Pass load_classmap=False to skip, or restore the Kaggle extract."
            )
        return pd.read_csv(path, index_col="index")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        image = self.images[idx]
        label = int(self.labels[idx])
        if self.transform is None:
            return image, label
        return self._apply_transform(image), label

    def _apply_transform(self, image: np.ndarray) -> Any:
        if self._transform_mode is None:
            self._transform_mode = self._detect_transform_mode()
        assert self.transform is not None
        if self._transform_mode == _TRANSFORM_MODE_KEYWORD:
            return self.transform(image=image)["image"]
        return self.transform(image)

    def _detect_transform_mode(self) -> str:
        """Pick ``keyword`` for Albumentations-style transforms, else ``positional``.

        Albumentations ``Compose.__call__`` signatures expose ``**data`` as a
        VAR_KEYWORD, and hand-written functions named with an ``image``
        parameter are treated the same. Everything else (torchvision
        ``Compose``, lambdas taking ``img``) falls back to positional.
        """
        transform = self.transform
        try:
            params = inspect.signature(transform).parameters
        except (TypeError, ValueError):
            return _TRANSFORM_MODE_POSITIONAL
        if "image" in params:
            return _TRANSFORM_MODE_KEYWORD
        for param in params.values():
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                return _TRANSFORM_MODE_KEYWORD
        return _TRANSFORM_MODE_POSITIONAL

    def get_class_counts(self) -> np.ndarray:
        """Return the per-class sample counts as a ``(num_classes,)`` int64 array."""
        return np.bincount(self.labels, minlength=self.num_classes).astype(np.int64)

    def get_char(self, label: int) -> str:
        """Return the hiragana glyph for the given class id.

        Args:
            label: Class id in the range ``[0, num_classes)``.

        Returns:
            The hiragana character string from ``k49_classmap.csv``.

        Raises:
            KeyError: If the class-map was not loaded, or ``label`` is absent.
        """
        if self.classmap is None:
            raise KeyError(
                "classmap not loaded — construct KuzushijiDataset with "
                "load_classmap=True to enable glyph lookups"
            )
        if label not in self.classmap.index:
            raise KeyError(f"label {label} not in classmap")
        return str(self.classmap.loc[label, "char"])


def create_train_val_split(
    dataset: KuzushijiDataset,
    val_fraction: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[Subset, Subset]:
    """Split a :class:`KuzushijiDataset` into stratified train/val subsets.

    Both returned subsets share the underlying images, labels, and transform;
    only the index sets differ, so no pixel data is copied.

    Args:
        dataset: Source dataset, typically the training split.
        val_fraction: Fraction of samples held out for validation. Must be in
            the open interval ``(0, 1)``.
        seed: RNG seed for deterministic splits.
        stratify: If True, preserve per-class proportions across the split.
            Strongly recommended given K49's ~25:1 class imbalance.

    Returns:
        A tuple ``(train_subset, val_subset)`` of
        :class:`torch.utils.data.Subset` objects.

    Raises:
        ValueError: If ``val_fraction`` is outside ``(0, 1)``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1), got {val_fraction}"
        )
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=dataset.labels if stratify else None,
    )
    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
    )
