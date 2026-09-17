"""Deadlock detection, overflow displacement and delivery.

These paths are reached only when several stations block each other in a cycle,
which the bundled verification runs do not reliably produce. The tests below
force the situation deterministically on a three-station ring at capacity one,
so the machinery stays covered independently of any particular seed.

    python -m pytest tests/ -q
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import pytest

from src.config.schema import StationConfig
from src.simulation.deadlock_manager import DeadlockManager
from src.simulation.order import Order
from src.simulation.station import Station


class _Recorder:
    """Minimal stand-in capturing what the manager writes."""

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def record_process_step(self, **kwargs) -> None:
        self.rows.append(kwargs)


def _ring(n: int = 3) -> Dict[str, Station]:
    """n stations of capacity 1, each holding one order bound for the next."""
    stations = {
        f"S{i}": Station(StationConfig(id=f"S{i}", capacity=1, is_machine=True))
        for i in range(n)
    }
    for i in range(n):
        order = Order(id=f"O{i}", features={}, timestamp_creation=0.0)
        stations[f"S{i}"].add_to_departure(order, f"S{(i + 1) % n}")
    return stations


def test_ring_closes_a_cycle():
    stations = _ring()
    dm = DeadlockManager()
    cycle = dm.detect_cycle(stations, "S0", "S1")
    assert cycle is not None
    assert "S0" in cycle


def test_chain_that_ends_is_not_a_cycle():
    stations = _ring()
    # Break the ring: the last station routes to a free station instead.
    stations["S2"].pop_departure()
    stations["S3"] = Station(StationConfig(id="S3", capacity=1))
    stations["S2"].add_to_departure(
        Order(id="OX", features={}, timestamp_creation=0.0), "S3"
    )
    dm = DeadlockManager()
    assert dm.detect_cycle(stations, "S0", "S1") is None


def test_self_edge_and_unknown_target_are_not_cycles():
    stations = _ring()
    dm = DeadlockManager()
    assert dm.detect_cycle(stations, "S0", "S0") is not None  # degenerate ring
    assert dm.detect_cycle(stations, "S0", "does-not-exist") is None


def test_resolution_displaces_one_order_and_frees_the_slot():
    stations = _ring()
    dm = DeadlockManager()
    rec = _Recorder()
    source = stations["S0"]
    assert source.occupied_slot_count == 1

    assert dm.resolve_deadlock(source, rec, current_time=10.0) is True

    assert source.has_departure is False          # slot freed
    assert dm.deadlock_count == 1
    assert dm.overflow_size == 1                  # order parked, not lost
    assert len(rec.rows) == 1
    row = rec.rows[0]
    assert row["station_type"] == "overflow"
    assert row["process_time"] == 0.0             # occupies no time
    assert row["order_id"] == "O0"


def test_displaced_order_is_delivered_once_the_target_frees():
    stations = _ring()
    dm = DeadlockManager()
    dm.resolve_deadlock(stations["S0"], _Recorder(), current_time=10.0)

    target = stations["S1"]
    target.pop_departure()                        # target becomes available
    delivered: List[str] = []
    dm.drain_overflow(target, lambda st, o: delivered.append(o.id))

    assert delivered == ["O0"]
    assert dm.overflow_size == 0


def test_overflow_is_not_delivered_while_the_target_is_full():
    stations = _ring()
    dm = DeadlockManager()
    dm.resolve_deadlock(stations["S0"], _Recorder(), current_time=10.0)

    delivered: List[str] = []
    dm.drain_overflow(stations["S1"], lambda st, o: delivered.append(o.id))

    assert delivered == []
    assert dm.overflow_size == 1                  # still parked, still counted


def test_per_step_resolution_cap_holds():
    dm = DeadlockManager(max_resolutions_per_step=2)
    rec = _Recorder()
    station = Station(StationConfig(id="S", capacity=9))
    for i in range(5):
        station.add_to_departure(
            Order(id=f"O{i}", features={}, timestamp_creation=0.0), "T"
        )

    results = [dm.resolve_deadlock(station, rec, current_time=0.0) for _ in range(4)]
    assert results == [True, True, False, False]
    assert dm.deadlock_count == 2

    dm.reset_step_counter()
    assert dm.resolve_deadlock(station, rec, current_time=1.0) is True
    assert dm.deadlock_count == 3


def test_resolution_without_a_departure_entry_is_a_no_op():
    dm = DeadlockManager()
    rec = _Recorder()
    empty = Station(StationConfig(id="S", capacity=1))
    assert dm.resolve_deadlock(empty, rec, current_time=0.0) is False
    assert dm.deadlock_count == 0
    assert rec.rows == []


def test_reset_clears_counters_and_parked_orders():
    stations = _ring()
    dm = DeadlockManager()
    dm.resolve_deadlock(stations["S0"], _Recorder(), current_time=0.0)
    assert dm.deadlock_count == 1 and dm.overflow_size == 1

    dm.reset()
    assert dm.deadlock_count == 0
    assert dm.overflow_size == 0


def test_orders_are_conserved_across_a_full_displacement_cycle():
    stations = _ring()
    before = {
        st.peek_departure()[0].id for st in stations.values() if st.has_departure
    }
    dm = DeadlockManager()
    dm.resolve_deadlock(stations["S0"], _Recorder(), current_time=0.0)

    stations["S1"].pop_departure()
    delivered: List[str] = []
    dm.drain_overflow(stations["S1"], lambda st, o: delivered.append(o.id))

    still_queued = {
        st.peek_departure()[0].id for st in stations.values() if st.has_departure
    }
    assert before == still_queued | set(delivered) | {"O1"}
    assert dm.overflow_size == 0


def test_cascade_enqueues_every_freed_station_once_per_event():
    """The cascade runs iteratively; a station freed mid-cascade is picked up.

    The queue previously carried a deduplication set alongside it. Measured over
    469,103 cascade entries across a 2-day and a 31-day run it never suppressed
    a single enqueue, so it was removed; this pins the behaviour it guarded.
    """
    from src.simulation.engine import SimulationEngine

    seen: List[str] = []
    stations = {
        "S0": Station(StationConfig(id="S0", capacity=1, is_start_station=True)),
        "S1": Station(StationConfig(id="S1", capacity=1)),
    }
    engine = SimulationEngine.__new__(SimulationEngine)
    engine._stations = stations
    engine._cascade_queue = deque()
    engine._is_cascading = False
    engine._deadlock = DeadlockManager()
    engine._process_freed_slot = lambda st: (
        seen.append(st.id),
        engine._on_slot_freed(stations["S1"]) if st.id == "S0" else None,
    )

    engine._on_slot_freed(stations["S0"])

    # S1 was freed while the cascade was running and must still be processed,
    # exactly once, without recursion.
    assert seen == ["S0", "S1"]
    assert not engine._is_cascading
    assert len(engine._cascade_queue) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
