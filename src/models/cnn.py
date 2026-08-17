"""Phase 4's from-scratch CNN. Deliberately minimal to start -- this is the baseline
later iterations (batchnorm, dropout, depth) are compared against, not the final
architecture. See notebooks/04_cnn_from_scratch.ipynb for the iteration log."""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Three conv blocks (channels double as spatial size halves via MaxPool), global
    average pool, linear head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 56 -> 28
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class SimpleCNNBatchNorm(nn.Module):
    """Same architecture as SimpleCNN, with BatchNorm2d after each conv (before the
    ReLU, the standard ordering) -- the one deliberate change under test against the
    baseline. bias=False on each Conv2d since BatchNorm's own learned shift makes the
    conv bias redundant."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class SimpleCNNBatchNormDeep(nn.Module):
    """SimpleCNNBatchNorm + one more conv block (128 -> 256), continuing the same
    channel-doubling/spatial-halving pattern -- the depth iteration, built on top of
    the already-adopted BatchNorm change, not the plain baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),      # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),     # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),   # 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 28 -> 14
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))
