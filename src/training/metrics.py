"""Metrics computed identically for every phase (4-8): accuracy alone would hide the
36x class imbalance Phase 1 found, so macro-F1, per-class recall, and a confusion
matrix are computed every time, not just accuracy."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

# Same blue sequential ramp used in notebooks/01_eda.ipynb and 02_data_pipeline.ipynb
# (project's dataviz convention) -- kept consistent so every notebook's charts read as
# one system.
_BLUE_SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("project_blue_sequential", _BLUE_SEQUENTIAL)


def compute_metrics(y_true, y_pred, num_classes: int) -> dict:
    labels = list(range(num_classes))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names, ax=None, normalize: bool = True):
    """Row-normalized by default (each row = recall for that true class), since raw
    counts on a 36x-imbalanced dataset are dominated by the largest classes."""
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))

    if ax is None:
        side = max(6, len(class_names) * 0.35)
        _, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(cm, cmap=BLUE_CMAP, vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax
