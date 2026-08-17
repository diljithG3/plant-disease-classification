"""Phase 5's transfer-learning backbone: an ImageNet-pretrained ResNet50 with the
entire backbone frozen and a fresh linear head for this dataset's classes. See
notebooks/05_feature_extraction.ipynb for training/results."""

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_feature_extractor(num_classes: int) -> nn.Module:
    """Every backbone parameter has requires_grad=False; only the newly-created `fc`
    layer (replacing ImageNet's 1000-way head) is trainable. Expects input already
    preprocessed to ImageNet stats (224x224, ImageNet mean/std) -- Phase 2's
    transforms (src/data/transforms.py) already produce exactly that, on purpose."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
