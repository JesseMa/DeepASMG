"""Recorder – logs all station passes and order lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.simulation.order import Order


PROCESS_LOG_DTYPE = np.dtype([
    ("order_id", "U32"),
    ("timestamp_event_start", "f8"),
    ("station", "U16"),
    ("station_type", "U8"),            # "machine" or "buffer"
    ("time_processing", "f8"),
    ("is_breakdown", "?"),
    ("net_process_time", "f8"),
    ("repair_time", "f8"),
])


@dataclass(slots=True)
class ProcessRecord:
    """A single station pass."""

    order_id: str
    timestamp_event_start: float
    station: str
    station_type: str
    time_processing: float
    is_breakdown: bool = False
    net_process_time: float = 0.0
    repair_time: float = 0.0


class Recorder:
    """Collects all simulation data for later analysis."""

    def __init__(self) -> None:
        self._process_log: List[ProcessRecord] = []
        self._orders: Dict[str, Order] = {}

    def record_process_step(
        self,
        order_id: str,
        timestamp_start: float,
        station_id: str,
        station_type: str,
        process_time: float,
        is_breakdown: bool = False,
        repair_time: float = 0.0,
    ) -> None:
        """Log one station pass; process_time is wall clock and includes repair_time."""
        net_time = process_time - repair_time
        self._process_log.append(
            ProcessRecord(
                order_id=order_id,
                timestamp_event_start=timestamp_start,
                station=station_id,
                station_type=station_type,
                time_processing=process_time,
                is_breakdown=is_breakdown,
                net_process_time=max(0.0, net_time),
                repair_time=repair_time,
            )
        )

    def register_order(self, order: Order) -> None:
        self._orders[order.id] = order

    def complete_order(self, order_id: str, timestamp: float) -> None:
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not registered")
        self._orders[order_id].timestamp_completion = timestamp

    def get_process_log_array(self, warmup_time: float = 0.0) -> np.ndarray:
        """Return the process log as a structured NumPy array (warmup filtered)."""
        filtered = [r for r in self._process_log if r.timestamp_event_start >= warmup_time]
        if not filtered:
            return np.array([], dtype=PROCESS_LOG_DTYPE)

        return np.array(
            [
                (
                    r.order_id, r.timestamp_event_start, r.station,
                    r.station_type, r.time_processing, r.is_breakdown,
                    r.net_process_time, r.repair_time,
                )
                for r in filtered
            ],
            dtype=PROCESS_LOG_DTYPE,
        )

    def get_order_log(self, warmup_time: float = 0.0) -> List[Dict]:
        """Return the order log as a list of dicts (warmup filtered)."""
        results = []
        for order in self._orders.values():
            if order.timestamp_creation < warmup_time:
                continue
            entry = {
                "order_id": order.id,
                "timestamp_creation": order.timestamp_creation,
                "timestamp_completion": order.timestamp_completion,
            }
            entry.update(order.features)
            results.append(entry)
        return results

    def reset(self) -> None:
        self._process_log.clear()
        self._orders.clear()