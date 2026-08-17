"""Picks the training device. 'auto' (the config default) means: use CUDA if
available (Colab GPU runtime), otherwise CPU (local debugging)."""

import torch


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
