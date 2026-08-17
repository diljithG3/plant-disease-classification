"""Checkpoint save/load. Every checkpoint carries class_to_idx alongside the model
weights so a checkpoint is never ambiguous about which index maps to which class --
important once Phase 8 loads checkpoints from four different training runs."""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, epoch: int, best_metric: float, class_to_idx: dict, extra: dict = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "class_to_idx": class_to_idx,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location=None) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
