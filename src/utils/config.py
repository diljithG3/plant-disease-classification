"""Loads configs/config.yaml and resolves environment-specific values (e.g. data root)."""

import os
from pathlib import Path

import yaml

from .env import current__env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = current__env()
    # .expanduser() resolves a leading ~ to the current user's home dir (works
    # unchanged if DATA_ROOT or the configured default is already an absolute path).
    data_root = Path(os.environ.get("DATA_ROOT") or cfg["data"]["root"][env]).expanduser()

    cfg["env"] = env
    cfg["data"]["root"] = str(data_root)
    cfg["data"]["path"] = str(data_root / cfg["data"]["dataset_name"] / cfg["data"]["source"])
    cfg["data"]["artifacts_dir"] = str(PROJECT_ROOT / cfg["data"]["artifacts_dir"])

    for key in ("experiments_dir", "logs_dir", "checkpoints_dir"):
        cfg["logging"][key] = str(PROJECT_ROOT / cfg["logging"][key])

    return cfg


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2))
