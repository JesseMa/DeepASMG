"""The arrival index as a routing key: counted by the kernel, resolved by the
configuration, counted again by the preparation, and honoured by the truth
vector the routing comparison scores against.

    python -m pytest tests/ -q
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.config.schema import StationConfig
from src.dynamics.GroundSim.ground_transition import GroundTransition
from src.evaluation.routing_evaluation import config_true_vector
from src.experiments.sim_runner import get_process_config
from src.fitting.deep_data_preparation.prepare_transition_data import build_transition_samples
from src.simulation.deadlock_manager import DeadlockManager
from src.simulation.engine import SimulationEngine
from src.simulation.order import Order
from src.simulation.station import Station

FEATS = {"modell": "C", "feature_a": "a.2", "feature_b": "b.4"}


def _order(visits=None):
    return Order(id="o", features=dict(FEATS), timestamp_creation=0.0, visits=visits or {})


def test_visit_row_wins_on_the_third_arrival_only():
    m5 = {s.id: s for s in get_process_config().stations}["M5"]
    assert _order({"M5": 1}).resolve_transition_key(m5.transitions, station_id="M5") == "C_b.4"
    assert _order({"M5": 2}).resolve_transition_key(m5.transitions, station_id="M5") == "C_b.4"
    assert _order({"M5": 3}).resolve_transition_key(m5.transitions, station_id="M5") == "visit_3"
    assert _order().resolve_transition_key(m5.transitions, station_id="M5") == "C_b.4"


def test_config_true_vector_honours_the_visit():
    pc = get_process_config()
    assert config_true_vector(pc, "M5", FEATS, visit=1) == pytest.approx({"End": 0.48, "B5": 0.52})
    assert config_true_vector(pc, "M5", FEATS, visit=3) == {"End": 1.0}


def test_ground_transition_ends_the_route_on_the_third_visit():
    tr = GroundTransition(np.random.default_rng(0))
    tr.initialize({s.id: s for s in get_process_config().stations})
    avail = {"End", "B5"}
    third = [tr.predict("M5", _order({"M5": 3}), avail) for _ in range(50)]
    assert third == [None] * 50
    first = [tr.predict("M5", _order({"M5": 1}), avail) for _ in range(2000)]
    assert first.count("B5") / 2000 == pytest.approx(0.52, abs=0.03)


def test_engine_counts_arrivals_per_station():
    engine = SimulationEngine.__new__(SimulationEngine)
    engine._events = type("Q", (), {"schedule": lambda self, **k: None})()
    engine._current_time = 0
    engine._process_time = type("P", (), {"predict": lambda self, *a: 5.0})()
    station = Station(StationConfig(id="M5", capacity=3, is_machine=True,
                                    process_times={"C": (10.0, 1.0)}))
    order = _order()
    for n in (1, 2, 3):
        engine._accept_order(station, order)
        station.finish_processing(order.id)
        assert order.visits["M5"] == n


def test_preparation_counts_prior_rows_per_order_and_station():
    rows = [("o1", FEATS, "M5", "B5"), ("o2", FEATS, "M5", "End"),
            ("o1", FEATS, "M5", "B5"), ("o1", FEATS, "M5", "End")]
    samples = build_transition_samples(rows, n_hist_slots=2)
    assert [s.visit for s in samples] == ["1", "1", "2", "3"]


def test_sequential_rule_counts_under_the_resolved_key():
    """The round-robin counter and the draw path must name the key alike, so a
    visit-indexed row and a product row never split one counter in two."""
    cfg = StationConfig(
        id="S", capacity=1,
        transitions={"visit_2": {"X": 1.0}, "A": {"B1": 0.5, "B2": 0.5}},
        sequential_routing={"A": {"cycle": 2, "targets": ["B1", "B2"]}},
    )
    tr = GroundTransition(np.random.default_rng(0))
    tr.initialize({"S": cfg})
    o = Order(id="o", features={"modell": "A"}, timestamp_creation=0.0, visits={"S": 1})
    first = [tr.predict("S", o, {"B1", "B2", "X"}) for _ in range(4)]
    assert first == ["B1", "B1", "B2", "B2"]
    assert set(tr._seq_counters["S"]) == {"A"}


def test_cascade_enqueues_every_freed_station_once_per_event():
    """The cascade runs iteratively; a station freed mid-cascade is picked up."""
    seen = []
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
    assert seen == ["S0", "S1"]
    assert not engine._is_cascading and len(engine._cascade_queue) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
