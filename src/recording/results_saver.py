"""ResultsSaver – writes simulation results into structured run directories."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.recording.recorder import PROCESS_LOG_DTYPE


class ResultsSaver:
    """Writes the process and order logs of one run as CSV files."""

    def __init__(self, base_dir: Path, experiment_name: str, process_name: str) -> None:
        self._timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._batch_dir = (
            Path(base_dir) / experiment_name / f"data_{process_name}_{self._timestamp}"
        )
        self._batch_dir.mkdir(parents=True, exist_ok=True)

    @property
    def batch_dir(self) -> Path:
        return self._batch_dir

    def save_run(self, run_number: int, process_log: np.ndarray, order_log: List[Dict]) -> None:
        prefix = f"run{run_number}"
        self._save_events(prefix, process_log)
        self._save_orders(prefix, order_log)

    def _save_events(self, prefix: str, process_log: np.ndarray) -> None:
        filepath = self._batch_dir / f"{prefix}_events_{self._timestamp}.csv"
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(PROCESS_LOG_DTYPE.names)
            for row in process_log:
                # Integer time contract: every duration is rounded exactly once,
                # at the emitting module's boundary — logged ≡ executed, raw.
                # The value order must match PROCESS_LOG_DTYPE, which also
                # supplies the header above.
                writer.writerow([
                    row["order_id"],
                    f"{row['timestamp_event_start']:.6f}",
                    row["station"],
                    row["station_type"],
                    f"{row['time_processing']:.6f}",
                    row["is_breakdown"],
                    f"{row['net_process_time']:.6f}",
                    f"{row['repair_time']:.6f}",
                ])

    def _save_orders(self, prefix: str, order_log: List[Dict]) -> None:
        filepath = self._batch_dir / f"{prefix}_orders_{self._timestamp}.csv"
        if not order_log:
            filepath.touch()
            return
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(order_log[0].keys()))
            writer.writeheader()
            writer.writerows(order_log)
