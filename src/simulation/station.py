"""Station – unified runtime object for all station types.

Capacity: occupied = len(processing_slots) + len(departure_queue) ≤ capacity;
capacity = 0 means unlimited.

TTF-decrement paradigm: ttf_remaining counts operating seconds to the next
failure (None = cannot fail); the cycle accumulators are reset at repair end,
not at breakdown, so a cycle spans repair-end to repair-end.
"""

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

        # Incoming waitlist: upstream stations wanting to send here (FIFO)
        self._incoming_waitlist: deque[str] = deque()
        self._waitlist_set: Set[str] = set()

        self.is_down: bool = False

        self.ttf_remaining: Optional[float] = None

        # Cycle wear accumulators; feed the repair-duration model at breakdown.
        self.accumulated_op_time: float = 0.0
        self.cycle_start_wall_time: float = 0.0
        self.jobs_in_cycle: int = 0

    @property
    def active_processing_count(self) -> int:
        return len(self._processing_slots)

    @property
    def departure_count(self) -> int:
        return len(self._departure_queue)

    @property
    def occupied_slot_count(self) -> int:
        return self.active_processing_count + self.departure_count

    def has_capacity(self) -> bool:
        if self.config.capacity == 0:
            return True
        return self.occupied_slot_count < self.config.capacity

    @property
    def station_type(self) -> str:
        """Recorded station kind: "machine" or "buffer"."""
        return "machine" if self.config.is_machine else "buffer"

    @property
    def is_available(self) -> bool:
        return self.has_capacity() and not self.is_down

    def start_processing(self, order: "Order") -> None:
        if not self.is_available:
            raise RuntimeError(
                f"Station {self.id} unavailable "
                f"(processing={self.active_processing_count}, "
                f"departure={self.departure_count}, "
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
        """Remove and return the front entry; frees a slot."""
        entry = self._departure_queue.popleft()
        return entry.order, entry.target_station_id


    @property
    def has_departure(self) -> bool:
        return len(self._departure_queue) > 0

    def get_departure_target(self) -> Optional[str]:
        if not self.has_departure:
            return None
        return self._departure_queue[0].target_station_id

    def add_to_waitlist(self, source_station_id: str) -> None:
        """Add a station to the waitlist; repeat calls are no-ops."""
        if source_station_id not in self._waitlist_set:
            self._incoming_waitlist.append(source_station_id)
            self._waitlist_set.add(source_station_id)

    def pop_waitlist(self) -> str:
        source_id = self._incoming_waitlist.popleft()
        self._waitlist_set.remove(source_id)
        return source_id

    @property
    def has_waiting(self) -> bool:
        return len(self._incoming_waitlist) > 0

    def get_utilization(self, current_time: float) -> float:
        """Utilization in the current cycle: op_time / wall_clock, in [0, 1]; 1.0 if wall_clock=0."""
        wall_clock = current_time - self.cycle_start_wall_time
        if wall_clock <= 0:
            return 1.0
        return min(1.0, self.accumulated_op_time / wall_clock)

