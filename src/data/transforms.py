"""Train/eval transforms. Held fixed from here through Phase 8 -- preprocessing must
never be a confound when comparing the from-scratch CNN against the transfer-learning
variants."""

import torch
from torchvision import transforms


def get_train_transforms(cfg: dict) -> transforms.Compose:
    image_size = cfg["image_size"]
    aug = cfg["augmentation"]
    norm = cfg["normalization"]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
            transforms.RandomRotation(aug["rotation_degrees"]),
            transforms.ColorJitter(**aug["color_jitter"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


def get_eval_transforms(cfg: dict) -> transforms.Compose:
    image_size = cfg["image_size"]
    norm = cfg["normalization"]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
    )


def denormalize(tensor: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Inverts Normalize for display -- e.g. plt.imshow(denormalize(img, cfg).permute(1, 2, 0))."""
    norm = cfg["normalization"]
    mean = torch.tensor(norm["mean"]).view(3, 1, 1)
    std = torch.tensor(norm["std"]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
