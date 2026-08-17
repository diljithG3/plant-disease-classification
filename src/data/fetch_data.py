"""Downloads the PlantVillage dataset (raw/<source> variant) into the configured data root.

Works identically for local (~/plant_disease_dataset by default, or DATA_ROOT) and
Colab (/content/dataset) -- the target path is entirely config-driven, only the
config's resolved root differs.

Usage:
    python -m src.data.fetch_data
    python -m src.data.fetch_data --force   # re-download even if data already present
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config

EXPECTED_CLASS_COUNT = {"color": 38, "grayscale": 38, "segmented": 38}


def _force_rmtree(path) -> None:
    """shutil.rmtree(path) alone: git marks .git/objects/pack/* read-only, and Windows'
    os.unlink refuses to remove a read-only file regardless of the containing
    directory's permissions (Linux only cares about the directory's permissions, so
    this never showed up there). Clear every file/dir's read-only bit first so cleaning
    up the temporary clone directory doesn't crash after a successful download."""
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.chmod(os.path.join(root, name), stat.S_IWRITE)
    shutil.rmtree(path)


def is_already_downloaded(target_dir: Path, source: str) -> bool:
    if not target_dir.exists():
        return False
    class_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
    return len(class_dirs) >= EXPECTED_CLASS_COUNT.get(source, 1)


def fetch_dataset(cfg: dict, force: bool = False) -> Path:
    data_root = Path(cfg["data"]["root"])
    dataset_name = cfg["data"]["dataset_name"]
    source = cfg["data"]["source"]
    repo_url = cfg["data"]["repo_url"]

    target_dir = data_root / dataset_name / source

    if is_already_downloaded(target_dir, source) and not force:
        n = len(list(target_dir.iterdir()))
        print(f"Dataset already present at {target_dir} ({n} class folders). Skipping download.")
        return target_dir

    if target_dir.exists():
        _force_rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    clone_dir = data_root / "_plantvillage_clone_tmp"
    if clone_dir.exists():
        _force_rmtree(clone_dir)

    print(f"Cloning {repo_url} (sparse checkout: raw/{source} only) ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(clone_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set", f"raw/{source}"],
        check=True,
    )

    source_dir = clone_dir / "raw" / source
    if not source_dir.exists():
        raise RuntimeError(
            f"Expected {source_dir} after sparse checkout but it's missing. "
            "The upstream repo layout may have changed -- check "
            f"{repo_url} manually."
        )

    shutil.move(str(source_dir), str(target_dir))
    _force_rmtree(clone_dir)

    n = len(list(target_dir.iterdir()))
    print(f"Done. {n} class folders at {target_dir}")
    expected = EXPECTED_CLASS_COUNT.get(source)
    if expected and n != expected:
        print(f"WARNING: expected {expected} class folders, found {n}. Inspect {target_dir} before proceeding.")

    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Fetch the PlantVillage dataset.")
    parser.add_argument("--force", action="store_true", help="Re-download even if data already exists.")
    args = parser.parse_args()

    cfg = load_config()
    print(f"Environment: {cfg['env']}")
    fetch_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
