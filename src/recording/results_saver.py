"""ResultsSaver – writes simulation results into structured run directories."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.recording.recorder import PROCESS_LOG_DTYPE


# Synthetic order_id markers for breakdown events (filtered as special cases
# by the prepare_*_data streams) — must NEVER receive a multi-seed prefix.
_SPECIAL_ORDER_IDS = frozenset({"BREAKDOWN"})


class ResultsSaver:
    """Writes run CSVs. With ``order_id_prefix`` set, every order_id is prefixed
    (e.g. ``"s101_"``) so that GroundSim runs of several seeds can share one
    parent directory without prepare_transition_data.py (which groups by
    order_id) mixing routes across seeds.
    """

    def __init__(
        self,
        base_dir: Path,
        experiment_name: str,
        process_name: str,
        *,
        order_id_prefix: Optional[str] = None,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._order_id_prefix = order_id_prefix
        self._timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        self._batch_dir = (
            self._base_dir
            / experiment_name
            / f"data_{process_name}_{self._timestamp}"
        )
        self._batch_dir.mkdir(parents=True, exist_ok=True)

    def _prefixed(self, order_id: str) -> str:
        if self._order_id_prefix is None or order_id in _SPECIAL_ORDER_IDS:
            return order_id
        return f"{self._order_id_prefix}{order_id}"

    @property
    def batch_dir(self) -> Path:
        return self._batch_dir

    def save_run(
        self, run_number: int, process_log: np.ndarray, order_log: List[Dict],
    ) -> Dict[str, Path]:
        prefix = f"run{run_number}"
        events_path = self._save_events(prefix, process_log)
        orders_path = self._save_orders(prefix, order_log)
        return {"events": events_path, "orders": orders_path}

    def _save_events(self, prefix: str, process_log: np.ndarray) -> Path:
        filename = f"{prefix}_events_{self._timestamp}.csv"
        filepath = self._batch_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(PROCESS_LOG_DTYPE.names)
            for row in process_log:
                # Integer time contract: every duration is rounded exactly once,
                # at the emitting module's boundary — logged ≡ executed, raw.
                # The value order must match PROCESS_LOG_DTYPE, which also
                # supplies the header above.
                writer.writerow([
                    self._prefixed(str(row["order_id"])),
                    f"{row['timestamp_event_start']:.6f}",
                    row["station"],
                    row["station_type"],
                    f"{row['time_processing']:.6f}",
                    row["is_breakdown"],
                    f"{row['net_process_time']:.6f}",
                    f"{row['repair_time']:.6f}",
                ])

        return filepath

    def _save_orders(self, prefix: str, order_log: List[Dict]) -> Path:
        filename = f"{prefix}_orders_{self._timestamp}.csv"
        filepath = self._batch_dir / filename

        if not order_log:
            filepath.touch()
            return filepath

        header = list(order_log[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            if self._order_id_prefix is None:
                writer.writerows(order_log)
            else:
                for o in order_log:
                    row = dict(o)
                    row["order_id"] = self._prefixed(str(row["order_id"]))
                    writer.writerow(row)

        return filepath