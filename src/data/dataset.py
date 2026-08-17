"""PyTorch Dataset over the fixed split produced by splits.py, plus a DataLoader
factory and per-class training weights (Phase 1 found a 36x class imbalance)."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class PlantDiseaseDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, class_to_idx: dict, data_path, transform=None):
        self.df = split_df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        # splits.csv stores paths relative to data_path (portable across machines --
        # see splits.py) -- data_path is the current environment's resolved root
        # (cfg["data"]["path"]), joined back on here at load time.
        self.data_path = Path(data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # .convert("RGB") is required: Phase 1's manifest found one stray RGBA image
        # among 54,304 RGB images -- without this it crashes whichever batch it lands in.
        image = Image.open(self.data_path / row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[row["class"]]
        return image, label


def compute_class_weights(train_df: pd.DataFrame, class_to_idx: dict) -> torch.Tensor:
    """Inverse-frequency weight per class, computed from the TRAIN split only.
    Phase 3 decides how to apply this (weighted loss and/or WeightedRandomSampler)."""
    counts = train_df["class"].value_counts()
    weights = torch.zeros(len(class_to_idx))
    for cls, idx in class_to_idx.items():
        weights[idx] = 1.0 / counts.get(cls, 1)
    return weights / weights.sum() * len(class_to_idx)


def make_weighted_sampler(train_df: pd.DataFrame, class_to_idx: dict) -> WeightedRandomSampler:
    """Alternative to a class-weighted loss: oversamples rare classes at the batch
    level instead. Phase 4 picks whichever (or both) actually helps minority-class
    recall -- this just makes the mechanism available."""
    class_weights = compute_class_weights(train_df, class_to_idx)
    sample_weights = train_df["class"].map(lambda c: class_weights[class_to_idx[c]].item()).values
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def make_dataloaders(
    splits_df: pd.DataFrame,
    class_to_idx: dict,
    train_transform,
    eval_transform,
    batch_size: int,
    num_workers: int,
    data_path,
    train_sampler: WeightedRandomSampler | None = None,
) -> dict[str, DataLoader]:
    """train_sampler, if given, replaces the default shuffle=True for the train split
    (e.g. pass make_weighted_sampler(...) to oversample rare classes)."""
    loaders = {}
    for split in ("train", "val", "test"):
        split_df = splits_df[splits_df["split"] == split]
        transform = train_transform if split == "train" else eval_transform
        dataset = PlantDiseaseDataset(split_df, class_to_idx, data_path, transform=transform)
        if split == "train" and train_sampler is not None:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
            )
    return loaders
