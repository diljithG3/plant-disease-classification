"""The reusable training harness. Built once here in Phase 3, called unmodified by
Phases 4-7 -- every model variant trains through the exact same loop, checkpointing,
logging, and metrics, which is what makes the Phase 8 comparison apples-to-apples."""

import time
from pathlib import Path

from . import engine
from .checkpoint import load_checkpoint, save_checkpoint
from .logging_utils import RunLogger


def train_model(
    model,
    loaders: dict,
    criterion,
    optimizer,
    device,
    run_name: str,
    cfg: dict,
    class_to_idx: dict,
    max_epochs: int,
    scheduler=None,
    resume_from=None,
) -> list[dict]:
    num_classes = len(class_to_idx)
    checkpoints_dir = Path(cfg["logging"]["checkpoints_dir"]) / run_name
    checkpoint_metric = cfg["training"]["checkpoint_metric"]
    patience = cfg["training"]["early_stopping_patience"]

    logger = RunLogger(run_name, cfg["logging"]["logs_dir"], resume=bool(resume_from))
    model.to(device)

    start_epoch = 0
    best_metric = -float("inf")
    epochs_without_improvement = 0

    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, map_location=device)
        start_epoch = payload["epoch"] + 1
        best_metric = payload["best_metric"]
        print(
            f"Resumed from {resume_from}: starting at epoch {start_epoch}, "
            f"best {checkpoint_metric} so far = {best_metric:.4f}"
        )

    history = []
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_loss = engine.train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_metrics = engine.evaluate(model, loaders["val"], criterion, device, num_classes)
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - t0

        row = {
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        logger.log_epoch(epoch, row)
        history.append({"epoch": epoch, **row})

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        current = val_metrics[checkpoint_metric]
        if current > best_metric:
            best_metric = current
            epochs_without_improvement = 0
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, best_metric, class_to_idx)
        else:
            epochs_without_improvement += 1

        # Saved *after* the best_metric update above, not before -- last.pt must record
        # the fully up-to-date best_metric (including this epoch's result), or a resume
        # starts from a stale, lower best_metric than what best.pt actually holds. That
        # would let a later epoch that's worse than the true best get accepted as a new
        # "improvement" and silently overwrite best.pt with a worse checkpoint.
        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, best_metric, class_to_idx)

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no {checkpoint_metric} improvement for {patience} epochs)")
            break

    logger.close()
    return history
