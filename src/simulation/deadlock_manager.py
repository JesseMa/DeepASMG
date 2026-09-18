"""DeadlockManager - cycle detection and overflow buffer."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.recording.recorder import Recorder
    from src.simulation.order import Order
    from src.simulation.station import Station


class DeadlockManager:

    def __init__(self) -> None:
        self._overflow_orders: Dict[str, deque["Order"]] = defaultdict(deque)
        self.reset()

    def reset(self) -> None:
        self._overflow_orders.clear()
        self._deadlock_count: int = 0

    @property
    def deadlock_count(self) -> int:
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

        if source_id == target_id:
            return [source_id]

        target = stations.get(target_id)
        if target is None or not target.is_full:
            return None

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
                return None
            if next_target == source_id:
                return visited

            next_station = stations[next_target]
            if not next_station.is_full:
                return None

            if next_target in visited:
                return None

            visited.append(next_target)
            current_id = next_target

        return None

    def resolve_deadlock(
        self,
        source_station: "Station",
        recorder: "Recorder",
        current_time: float,
    ) -> bool:

        if not source_station.has_departure:
            return False

        order, original_target = source_station.pop_departure()
        self._overflow_orders[original_target].append(order)
        self._deadlock_count += 1

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

        queue = self._overflow_orders.get(target_station.id)
        if not queue:
            return

        while queue and target_station.is_available:
            order = queue.popleft()
            accept_order_fn(target_station, order)

        if not queue:
            del self._overflow_orders[target_station.id]