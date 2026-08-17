"""TensorBoard + CSV logging, one run per run_name. CSV rows are appended
incrementally (not written once at the end) so a Colab disconnect mid-training
doesn't lose the history already logged."""

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, run_name: str, logs_dir, resume: bool = False):
        self.run_dir = Path(logs_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"

        has_existing_data = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if has_existing_data and not resume:
            raise FileExistsError(
                f"{self.csv_path} already has logged epochs, but this run wasn't started with "
                f"resume_from. Continuing would silently append a second run's rows onto the "
                f"first (and overlay its TensorBoard scalars at the same epoch numbers), "
                f"corrupting both. Pick a new run_name, delete {self.run_dir}, or pass "
                f"resume_from=... to train_model() to continue this run intentionally."
            )

        self.tb_writer = SummaryWriter(log_dir=str(self.run_dir))
        self._header_written = has_existing_data

    def log_epoch(self, epoch: int, row: dict) -> None:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(key, value, epoch)
        self.tb_writer.flush()

        write_header = not self._header_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", *row.keys()])
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow({"epoch": epoch, **row})

    def close(self) -> None:
        self.tb_writer.close()
