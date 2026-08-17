"""Core train/eval loops. Model-agnostic on purpose -- identical for the from-scratch
CNN (Phase 4) and every transfer-learning variant (Phase 5-7), so Phase 8's
comparison is never confounded by the training code itself differing per approach."""

import torch
from tqdm.auto import tqdm

from .metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    n = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, mininterval=1.0):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return running_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int) -> dict:
    model.eval()
    running_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    for xb, yb in tqdm(loader, desc="eval", leave=False, mininterval=1.0):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(yb.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = running_loss / n
    return metrics
