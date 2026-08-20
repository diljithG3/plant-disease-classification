"""Phase 5-7's transfer-learning backbones: an ImageNet-pretrained ResNet50, with a
fresh linear head for this dataset's classes and progressively more of the backbone
left trainable (Phase 5: none, Phase 6: layer4 only, Phase 7: all of it). Expects
input already preprocessed to ImageNet stats (224x224, ImageNet mean/std) -- Phase
2's transforms (src/data/transforms.py) already produce exactly that, on purpose."""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_feature_extractor(num_classes: int) -> nn.Module:
    """Phase 5: every backbone parameter has requires_grad=False; only the
    newly-created `fc` layer (replacing ImageNet's 1000-way head) is trainable. See
    notebooks/05_feature_extraction.ipynb for training/results."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_full_finetune(num_classes: int) -> nn.Module:
    """Phase 7: every parameter is trainable, including the entire pretrained
    backbone (not just layer4, per Phase 6) -- the least conservative of the three
    transfer-learning variants, so it needs the lowest backbone learning rate of
    the three (see notebooks/07_full_finetune.ipynb) to avoid wrecking ImageNet's
    pretrained features before the freshly-initialized `fc` head has learned
    anything useful to backpropagate. Nothing to freeze here -- every parameter
    already has requires_grad=True by default."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_partial_finetune(num_classes: int) -> nn.Module:
    """Phase 6: same starting point as build_resnet50_feature_extractor, except
    layer4 -- the last of ResNet50's four residual blocks -- is left trainable
    alongside the fresh `fc` head. layer1-3's generic low-level features (edges,
    textures) stay frozen; only layer4's higher-level features and the head adapt to
    this dataset. See notebooks/06_partial_finetune.ipynb for training/results."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
