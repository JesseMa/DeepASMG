"""DeadlockManager - cycle detection and overflow buffer.

Circular waits (A->B->C->A) arise when every station in a chain of departure
targets is at capacity. They are broken by overflow displacement: an order is
parked without occupying a slot until its target station becomes free.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.recording.recorder import Recorder
    from src.simulation.order import Order
    from src.simulation.station import Station


class DeadlockManager:
    """Deadlock detection and overflow resolution."""

    def __init__(self, max_resolutions_per_step: int = 50) -> None:
        self._overflow_orders: Dict[str, deque["Order"]] = defaultdict(deque)
        self._max_resolutions_per_step = max_resolutions_per_step
        self.reset()

    def reset(self) -> None:
        self._overflow_orders.clear()
        self._deadlock_count: int = 0
        self._resolutions_this_step: int = 0

    def reset_step_counter(self) -> None:
        self._resolutions_this_step = 0

    @property
    def deadlock_count(self) -> int:
        """Total number of deadlocks resolved since simulation start."""
        return self._deadlock_count

    @property
    def overflow_size(self) -> int:
        return sum(len(q) for q in self._overflow_orders.values())

    def detect_cycle(
        self,
        stations: Dict[str, "Station"],
        source_id: str,
        target_id: str,
    ) -> Optional[List[str]]:
        """Station IDs of the cycle closed by the edge source→target, else None.

        Follows the chain of departure targets from target (target waits on X,
        X waits on Y, ...); returning to source means a cycle.
        """
        if source_id == target_id:
            return [source_id]

        visited = [source_id, target_id]
        current_id = target_id
        max_chain = len(stations) + 1

        for _ in range(max_chain):
            current_station = stations.get(current_id)
            if current_station is None:
                return None

            next_target = current_station.get_departure_target()

            if next_target is None:
                return None
            if next_target not in stations:
                return None  # end token or unknown station
            if next_target == source_id:
                return visited

            next_station = stations[next_target]
            if next_station.is_available:
                return None  # chain resolves on its own

            if next_target in visited:
                return None  # sub-cycle not involving source – irrelevant

            visited.append(next_target)
            current_id = next_target

        return None

    def resolve_deadlock(
        self,
        source_station: "Station",
        recorder: "Recorder",
        current_time: float,
    ) -> bool:
        """Move the front departure entry to overflow, freeing its slot.

        The displaced order is delivered by drain_overflow() once its target
        station becomes free. Returns False when the per-step resolution limit
        is hit or the station has no departure entry.
        """
        if self._resolutions_this_step >= self._max_resolutions_per_step:
            return False
        if not source_station.has_departure:
            return False

        order, original_target = source_station.pop_departure()
        self._overflow_orders[original_target].append(order)
        self._deadlock_count += 1
        self._resolutions_this_step += 1

        recorder.record_process_step(
            order_id=order.id,
            timestamp_start=current_time,
            station_id=source_station.id,
            station_type="overflow",
            process_time=0.0,
            is_breakdown=False,
            repair_time=0.0,
        )
        return True

    def drain_overflow(
        self,
        target_station: "Station",
        accept_order_fn: Callable[["Station", "Order"], None],
    ) -> None:
        """Deliver waiting overflow orders to the now-free target station.

        accept_order_fn occupies a slot and schedules PROCESS_COMPLETE.
        """
        queue = self._overflow_orders.get(target_station.id)
        if not queue:
            return

        while queue and target_station.is_available:
            order = queue.popleft()
            accept_order_fn(target_station, order)

        if not queue:
            del self._overflow_orders[target_station.id]