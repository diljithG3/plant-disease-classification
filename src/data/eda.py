"""Reusable EDA utilities: class distribution, image sampling, and a full-dataset image manifest
(size, mode, file size, corruption check). Notebooks call into these rather than reimplementing them."""

import random
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def list_class_dirs(data_path) -> list[Path]:
    return sorted(p for p in Path(data_path).iterdir() if p.is_dir())


def list_class_files(class_dir: Path) -> list[Path]:
    # sorted(), not raw iterdir() order: Path.iterdir() order is filesystem-dependent
    # (NTFS and ext4 can disagree), and build_splits() shuffles this list with a fixed
    # seed -- an unsorted input would silently produce a different split on a
    # different OS/filesystem even with the same seed, breaking the "split never
    # drifts between machines" guarantee the cached splits.csv is supposed to give.
    return sorted(p for p in class_dir.iterdir() if p.is_file())


def class_counts(data_path) -> pd.Series:
    counts = {d.name: len(list_class_files(d)) for d in list_class_dirs(data_path)}
    return pd.Series(counts, name="count").sort_index()


def sample_image_paths(data_path, n_per_class: int = 3, seed: int = 42) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    samples = {}
    for class_dir in list_class_dirs(data_path):
        files = list_class_files(class_dir)
        samples[class_dir.name] = rng.sample(files, min(n_per_class, len(files)))
    return samples


def _inspect_image(path: Path, data_path: Path) -> dict:
    # Relative to data_path, not absolute -- same reasoning as splits.py: an absolute
    # path only works on the one machine that built the manifest.
    row = {
        "path": path.relative_to(data_path).as_posix(),
        "class": path.parent.name,
        "file_size_kb": path.stat().st_size / 1024,
    }
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            row["width"], row["height"] = img.size
            row["mode"] = img.mode
        row["readable"] = True
    except (UnidentifiedImageError, OSError):
        row["width"] = row["height"] = row["mode"] = None
        row["readable"] = False
    return row


def build_manifest(data_path, cache_path=None, force: bool = False) -> pd.DataFrame:
    """Scans every image once (header-only reads, no full pixel decode) and records
    width/height/mode/file size/readability. Cached to `cache_path` since a full scan
    over tens of thousands of images is slow to repeat on every notebook run."""
    if cache_path and Path(cache_path).exists() and not force:
        return pd.read_csv(cache_path)

    data_path = Path(data_path)
    class_dirs = list_class_dirs(data_path)
    all_files = [f for d in class_dirs for f in list_class_files(d)]

    rows = [_inspect_image(f, data_path) for f in tqdm(all_files, desc="Scanning images")]
    df = pd.DataFrame(rows)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df
