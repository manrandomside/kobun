"""Download Kuzushiji datasets for Phase 1 classification and Phase 2 detection.

The classification corpus (Kuzushiji-49) is fetched from the CODH public mirror
as four NumPy `.npz` files. The detection corpus is the 2019 Kaggle
"Kuzushiji Recognition" competition, retrieved via the Kaggle API and requires
a valid `~/.kaggle/kaggle.json` with competition rules accepted for the
authenticated user.

Archives are written as-is; extraction is deferred to the preprocessing layer.

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
from typing import Iterable

import requests
from tqdm import tqdm


CODH_BASE_URL = "http://codh.rois.ac.jp/kmnist/dataset/k49"
K49_FILES: tuple[str, ...] = (
    "k49-train-imgs.npz",
    "k49-train-labels.npz",
    "k49-test-imgs.npz",
    "k49-test-labels.npz",
)

KAGGLE_COMPETITION = "kuzushiji-recognition"

CLASSIFICATION_SUBDIR = "kuzushiji-49"
DETECTION_SUBDIR = "kuzushiji-recognition"

# Default data root sits next to this script's ml/ parent so that running the
# script from any CWD still writes into <repo>/ml/data/.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CHUNK_BYTES = 1 << 15
HTTP_TIMEOUT_SECONDS = 30


def _verify(path: Path) -> bool:
    """Return True when ``path`` exists as a non-empty regular file."""
    return path.is_file() and path.stat().st_size > 0


def _download_file(url: str, dest: Path) -> None:
    """Stream a single URL to ``dest`` with a tqdm progress bar.

    The response body is written to ``<dest>.part`` first and renamed on
    success, so an interrupted run never leaves a zero-length ``dest`` that
    later looks complete to the skip-if-present check.

    Args:
        url: Absolute URL to fetch.
        dest: Final destination path on disk.

    Raises:
        requests.RequestException: If any transport-level error occurs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with (
            tmp.open("wb") as file_handle,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                leave=False,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                file_handle.write(chunk)
                progress.update(len(chunk))
    tmp.replace(dest)


def download_classification(data_dir: Path, force: bool) -> None:
    """Download the Kuzushiji-49 classification arrays from CODH.

    Args:
        data_dir: Root data directory. Files are written to
            ``<data_dir>/kuzushiji-49/``.
        force: If True, redownload files even when already present.

    Raises:
        SystemExit: On network failure or integrity check failure.
    """
    target_dir = data_dir / CLASSIFICATION_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[classification] target: {target_dir}")

    for filename in K49_FILES:
        dest = target_dir / filename
        if _verify(dest) and not force:
            print(f"[classification] skip {filename} (already present)")
            continue
        url = f"{CODH_BASE_URL}/{filename}"
        print(f"[classification] fetch {url}")
        try:
            _download_file(url, dest)
        except requests.RequestException as exc:
            raise SystemExit(
                f"[classification] download failed for {url}: {exc}"
            ) from exc
        if not _verify(dest):
            raise SystemExit(
                f"[classification] integrity check failed for {dest}"
            )
    print("[classification] done")


def download_detection(data_dir: Path, force: bool) -> None:
    """Download the Kaggle Kuzushiji Recognition competition archive.

    Delegates to the official ``kaggle`` Python client. The client is imported
    lazily so that ``--dataset classification`` never pays its import or
    auth-loading cost.

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

    try:
        from kaggle import api as kaggle_api
    except ImportError as exc:
        raise SystemExit(
            "[detection] `kaggle` package not installed. "
            "Run `pip install -r ml/requirements.txt` inside the project venv."
        ) from exc

    try:
        kaggle_api.authenticate()
    except Exception as exc:
        raise SystemExit(
            "[detection] Kaggle authentication failed. "
            "Place a valid API token at `~/.kaggle/kaggle.json` "
            "(chmod 600 on *nix, restrict ACLs on Windows). "
            "Regenerate at https://www.kaggle.com/settings."
        ) from exc

    print(f"[detection] fetch competition '{KAGGLE_COMPETITION}' via Kaggle API")
    try:
        kaggle_api.competition_download_files(
            KAGGLE_COMPETITION,
            path=str(target_dir),
            quiet=False,
            force=force,
        )
    except Exception as exc:
        raise SystemExit(
            f"[detection] Kaggle download failed: {exc}. "
            "Confirm you have accepted the competition rules at "
            f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules."
        ) from exc

    downloaded = [p for p in target_dir.iterdir() if p.is_file()]
    if not downloaded:
        raise SystemExit(f"[detection] no files landed in {target_dir}")
    print(f"[detection] done ({len(downloaded)} file(s) in {target_dir})")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description=(
            "Download Kuzushiji datasets: K49 classification (CODH) and/or "
            "Kuzushiji Recognition detection (Kaggle competition)."
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
