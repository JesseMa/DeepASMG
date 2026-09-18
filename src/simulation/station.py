"""Station – unified runtime object for all station types."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


@dataclass
class _DepartureEntry:
    order: "Order"
    target_station_id: str


class Station:
    def __init__(self, config: "StationConfig") -> None:
        self.config = config
        self.id = config.id

        self._processing_slots: Dict[str, "Order"] = {}

        self._departure_queue: deque[_DepartureEntry] = deque()

        self._incoming_waitlist: deque[str] = deque()
        self._waitlist_set: Set[str] = set()

        self.is_down: bool = False

        self.ttf_remaining: Optional[float] = None

        self.accumulated_op_time: float = 0.0
        self.cycle_start_wall_time: float = 0.0
        self.jobs_in_cycle: int = 0

    @property
    def occupied_slot_count(self) -> int:
        return len(self._processing_slots) + len(self._departure_queue)

    @property
    def station_type(self) -> str:
        return "machine" if self.config.is_machine else "buffer"

    @property
    def is_full(self) -> bool:
        return self.occupied_slot_count >= self.config.capacity

    @property
    def is_available(self) -> bool:
        if self.is_down:
            return False
        return not self.is_full

    def start_processing(self, order: "Order") -> None:
        if not self.is_available:
            raise RuntimeError(
                f"Station {self.id} unavailable "
                f"(processing={len(self._processing_slots)}, "
                f"departure={len(self._departure_queue)}, "
                f"capacity={self.config.capacity}, "
                f"down={self.is_down})"
            )
        self._processing_slots[order.id] = order

    def finish_processing(self, order_id: str) -> "Order":
        if order_id not in self._processing_slots:
            raise RuntimeError(
                f"Station {self.id}: no active order '{order_id}'"
            )
        return self._processing_slots.pop(order_id)

    def add_to_departure(self, order: "Order", target_station_id: str) -> None:
        self._departure_queue.append(
            _DepartureEntry(order=order, target_station_id=target_station_id)
        )

    def peek_departure(self) -> Tuple["Order", str]:
        entry = self._departure_queue[0]
        return entry.order, entry.target_station_id

    def pop_departure(self) -> Tuple["Order", str]:
        entry = self._departure_queue.popleft()
        return entry.order, entry.target_station_id


    @property
    def has_departure(self) -> bool:
        return bool(self._departure_queue)

    def get_departure_target(self) -> Optional[str]:
        if not self.has_departure:
            return None
        return self._departure_queue[0].target_station_id

    def add_to_waitlist(self, source_station_id: str) -> None:
        if source_station_id not in self._waitlist_set:
            self._incoming_waitlist.append(source_station_id)
            self._waitlist_set.add(source_station_id)

    def pop_waitlist(self) -> str:
        source_id = self._incoming_waitlist.popleft()
        self._waitlist_set.remove(source_id)
        return source_id

    @property
    def has_waiting(self) -> bool:
        return bool(self._incoming_waitlist)

    def get_utilization(self, current_time: float) -> float:
        wall_clock = current_time - self.cycle_start_wall_time
        if wall_clock <= 0:
            return 1.0
        return min(1.0, self.accumulated_op_time / wall_clock)

