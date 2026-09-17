"""Event types and event queue for the discrete event simulation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class EventType(Enum):
    PROCESS_COMPLETE = auto()
    MACHINE_REPAIRED = auto()


@dataclass(order=True)
class Event:
    """A single event, ordered by time, then by _seq (FIFO for equal times)."""

    time: float
    _seq: int = field(compare=True, repr=False)
    type: EventType = field(compare=False)
    station_id: Optional[str] = field(default=None, compare=False)
    order_id: Optional[str] = field(default=None, compare=False)
    data: Any = field(default=None, compare=False)


class EventQueue:
    """Min-heap of events, ordered by (time, _seq).

    Integer time contract: every scheduled time is a whole second, so events
    that fall in the same tick are genuinely simultaneous and their order is
    decided by _seq — insertion order, i.e. FIFO — not by sub-second precision
    the model does not have.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []
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
            time=time,
            _seq=self._seq,
            type=event_type,
            station_id=station_id,
            order_id=order_id,
            data=data,
        )
        self._seq += 1
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float:
        return self._heap[0].time

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0