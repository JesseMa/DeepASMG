"""Event types and event queue for the discrete event simulation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional


class EventType(Enum):
    PROCESS_COMPLETE = auto()
    MACHINE_REPAIRED = auto()


@dataclass(slots=True)
class Event:
    time: int
    type: EventType
    station_id: Optional[str] = None
    order_id: Optional[str] = None
    data: Any = None


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple] = []
        self._seq: int = 0

    def schedule(
        self,
        time: float,
        event_type: EventType,
        station_id: Optional[str] = None,
        order_id: Optional[str] = None,
        data: Any = None,
    ) -> None:
        if float(time) != int(time):
            raise ValueError(
                f"Fail fast: event time must be whole seconds (integer time "
                f"contract), got {time} for {event_type} at {station_id}."
            )
        event = Event(
            time=int(time),
            type=event_type,
            station_id=station_id,
            order_id=order_id,
            data=data,
        )
        heapq.heappush(self._heap, (int(time), self._seq, event))
        self._seq += 1

    def pop(self) -> Event:
        return heapq.heappop(self._heap)[2]

    def peek_time(self) -> int:
        return self._heap[0][0]

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0