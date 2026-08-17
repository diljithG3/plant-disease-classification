"""Builds and caches a fixed, per-class-stratified train/val/test split.

The split is persisted to disk (not just seeded-and-recomputed) so every later phase
trains/evaluates on exactly the same images -- required for the Phase 8 comparison
across approaches to mean anything.
"""

import json
import random
from pathlib import Path

import pandas as pd

from .eda import list_class_dirs, list_class_files


def build_splits(data_path, ratios: dict, seed: int, cache_path=None, force: bool = False) -> pd.DataFrame:
    if cache_path and Path(cache_path).exists() and not force:
        return pd.read_csv(cache_path)

    assert abs(sum(ratios.values()) - 1.0) < 1e-6, f"split ratios must sum to 1, got {ratios}"

    data_path = Path(data_path)
    rng = random.Random(seed)
    rows = []
    for class_dir in list_class_dirs(data_path):
        files = list_class_files(class_dir)
        rng.shuffle(files)

        n = len(files)
        n_train = round(n * ratios["train"])
        n_val = round(n * ratios["val"])
        # remainder -> test, so rounding never drops or duplicates a file
        assignment = ["train"] * n_train + ["val"] * n_val
        assignment += ["test"] * (n - len(assignment))

        for f, split in zip(files, assignment):
            # Relative to data_path (POSIX separators, so it reads back correctly on
            # both Windows and Colab), not absolute -- an absolute path only ever
            # works on the one machine it was built on, which defeats the entire
            # point of caching a split "that never drifts between machines."
            rows.append({"path": f.relative_to(data_path).as_posix(), "class": class_dir.name, "split": split})

    df = pd.DataFrame(rows)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def build_class_to_idx(data_path, cache_path=None, force: bool = False) -> dict:
    if cache_path and Path(cache_path).exists() and not force:
        with open(cache_path) as f:
            return json.load(f)

    class_to_idx = {d.name: i for i, d in enumerate(list_class_dirs(data_path))}

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(class_to_idx, f, indent=2)

    return class_to_idx
