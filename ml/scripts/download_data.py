"""Download Kuzushiji datasets for Phase 1 classification and Phase 2 detection.

The classification corpus (Kuzushiji-49) is fetched from the Kaggle dataset
mirror ``anokas/kuzushiji``, which bundles the official CODH K49 release
alongside Kuzushiji-MNIST. We switched to this mirror after the upstream CODH
server ``codh.rois.ac.jp`` went offline in November 2025 for a migration
whose timeline is still unresolved as of April 2026. Attribution and the
original license (CC BY-SA 4.0, Clanuwat et al. 2018) are unaffected —
``anokas/kuzushiji`` is a byte-identical mirror of the official release.

The detection corpus is the 2019 Kaggle "Kuzushiji Recognition" competition.
Both paths authenticate against ``~/.kaggle/kaggle.json`` and require that the
authenticated user has accepted the respective dataset / competition terms.

The K49 archive is auto-extracted and the bundled K-MNIST arrays are pruned,
since this project trains only on K49. The detection archive is left as a zip
and extracted later by the preprocessing layer.

Usage:
    python ml/scripts/download_data.py --dataset classification
    python ml/scripts/download_data.py --dataset detection
    python ml/scripts/download_data.py --dataset all --force
    python ml/scripts/download_data.py --dataset classification --data-dir /tmp/kobun
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


KAGGLE_DATASET_CLASSIFICATION = "anokas/kuzushiji"
KAGGLE_COMPETITION_DETECTION = "kuzushiji-recognition"

CLASSIFICATION_SUBDIR = "kuzushiji-49"
DETECTION_SUBDIR = "kuzushiji-recognition"

K49_REQUIRED_FILES: tuple[str, ...] = (
    "k49-train-imgs.npz",
    "k49-train-labels.npz",
    "k49-test-imgs.npz",
    "k49-test-labels.npz",
)

KMNIST_FILES_TO_DELETE: tuple[str, ...] = (
    "kmnist-train-imgs.npz",
    "kmnist-train-labels.npz",
    "kmnist-test-imgs.npz",
    "kmnist-test-labels.npz",
)

# Default data root sits next to this script's ml/ parent so that running the
# script from any CWD still writes into <repo>/ml/data/.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _verify(path: Path) -> bool:
    """Return True when ``path`` exists as a non-empty regular file."""
    return path.is_file() and path.stat().st_size > 0


def _authenticated_kaggle_api() -> Any:
    """Import the ``kaggle`` client and authenticate it, or fail with guidance.

    Kept as a helper so the import cost (and the config-file read that the
    Kaggle client performs at auth time) is paid only when a dataset path is
    actually invoked. ``--help`` never triggers it.

    Returns:
        The authenticated ``kaggle.api`` singleton.

    Raises:
        SystemExit: If the package is missing or credentials are invalid.
    """
    try:
        from kaggle import api as kaggle_api
    except ImportError as exc:
        raise SystemExit(
            "`kaggle` package not installed. "
            "Run `pip install -r ml/requirements.txt` inside the project venv."
        ) from exc

    try:
        kaggle_api.authenticate()
    except Exception as exc:
        raise SystemExit(
            "Kaggle authentication failed. "
            "Place a valid API token at `~/.kaggle/kaggle.json` "
            "(chmod 600 on *nix, restrict ACLs on Windows). "
            "Regenerate at https://www.kaggle.com/settings."
        ) from exc

    return kaggle_api


def download_classification(data_dir: Path, force: bool) -> None:
    """Download Kuzushiji-49 arrays via the Kaggle ``anokas/kuzushiji`` mirror.

    The mirror bundles both Kuzushiji-MNIST (10 classes) and Kuzushiji-49
    (49 classes). After auto-extract we prune the K-MNIST ``.npz`` arrays;
    the ``*_classmap.csv`` files are small and harmless, so they are kept.

    Args:
        data_dir: Root data directory. Files are written to
            ``<data_dir>/kuzushiji-49/``.
        force: If True, redownload and re-extract even when files are present.

    Raises:
        SystemExit: On download, extraction, or integrity check failure.
    """
    target_dir = data_dir / CLASSIFICATION_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[classification] target: {target_dir}")

    expected = [target_dir / name for name in K49_REQUIRED_FILES]
    if all(_verify(p) for p in expected) and not force:
        print(
            "[classification] skip (all 4 K49 .npz files already present; "
            "use --force to redownload)"
        )
        return

    kaggle_api = _authenticated_kaggle_api()

    print(
        f"[classification] fetch Kaggle dataset "
        f"'{KAGGLE_DATASET_CLASSIFICATION}' (K49 + K-MNIST mirror)"
    )
    try:
        kaggle_api.dataset_download_files(
            KAGGLE_DATASET_CLASSIFICATION,
            path=str(target_dir),
            unzip=True,
            force=force,
            quiet=False,
        )
    except Exception as exc:
        raise SystemExit(
            f"[classification] Kaggle download failed: {exc}. "
            "Confirm the dataset is accessible at "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET_CLASSIFICATION}."
        ) from exc

    for name in KMNIST_FILES_TO_DELETE:
        stray = target_dir / name
        if stray.exists():
            stray.unlink()
            print(f"[classification] pruned {name}")

    missing = [p.name for p in expected if not _verify(p)]
    if missing:
        raise SystemExit(
            f"[classification] integrity check failed — missing/empty: {missing}"
        )
    print(f"[classification] done ({len(expected)} K49 file(s) in {target_dir})")


def download_detection(data_dir: Path, force: bool) -> None:
    """Download the Kaggle Kuzushiji Recognition competition archive.

    Args:
        data_dir: Root data directory. Files are written to
            ``<data_dir>/kuzushiji-recognition/``.
        force: If True, redownload even when the directory already has files.

    Raises:
        SystemExit: On missing credentials, rejected rules, or API failure.
    """
    target_dir = data_dir / DETECTION_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[detection] target: {target_dir}")

    if not force:
        existing = [p for p in target_dir.iterdir() if p.is_file()]
        if existing:
            print(
                f"[detection] skip (found {len(existing)} file(s) in {target_dir}; "
                "use --force to redownload)"
            )
            return

    kaggle_api = _authenticated_kaggle_api()

    print(
        f"[detection] fetch competition '{KAGGLE_COMPETITION_DETECTION}' "
        "via Kaggle API"
    )
    try:
        kaggle_api.competition_download_files(
            KAGGLE_COMPETITION_DETECTION,
            path=str(target_dir),
            quiet=False,
            force=force,
        )
    except Exception as exc:
        raise SystemExit(
            f"[detection] Kaggle download failed: {exc}. "
            "Confirm you have accepted the competition rules at "
            f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION_DETECTION}/rules."
        ) from exc

    downloaded = [p for p in target_dir.iterdir() if p.is_file()]
    if not downloaded:
        raise SystemExit(f"[detection] no files landed in {target_dir}")
    print(f"[detection] done ({len(downloaded)} file(s) in {target_dir})")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description=(
            "Download Kuzushiji datasets: K49 classification (Kaggle mirror) "
            "and/or Kuzushiji Recognition detection (Kaggle competition)."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("classification", "detection", "all"),
        required=True,
        help="Which dataset to fetch.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory for downloaded data (default: ml/data/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if target files already exist.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point. Parses arguments and dispatches to dataset downloaders."""
    args = _parse_args(argv)
    data_dir: Path = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"data root: {data_dir}")

    if args.dataset in ("classification", "all"):
        download_classification(data_dir, force=args.force)
    if args.dataset in ("detection", "all"):
        download_detection(data_dir, force=args.force)

    print("all requested downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
