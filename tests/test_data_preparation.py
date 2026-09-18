"""What the preparation turns the log into, and what the reference fits on it."""

from __future__ import annotations

import numpy as np
import pytest

from src.fitting.deep_data_preparation.data_io import RawEvent
from src.fitting.deep_data_preparation.prepare_process_time_data import filter_and_join
from src.fitting.deep_data_preparation.prepare_repair_time_data import extract_repair_samples
from src.fitting.deep_data_preparation.prepare_survival_data import extract_survival_samples
from src.fitting.ref_data_preparation.ref_analyzer import (
    _dequantized_exponential, _dequantized_normal,
)


def _ev(t, station, *, order="O1", stype="machine", dur=10.0, breakdown=False, repair=0.0):
    return RawEvent(order_id="BREAKDOWN" if breakdown else order, timestamp_event_start=t,
                    station=station, station_type=stype, time_processing=dur,
                    is_breakdown=breakdown, net_process_time=0.0 if breakdown else dur,
                    repair_time=repair)


def _one_machine_log():
    log, t = [], 0.0
    for ops, rep in ((30.0, 100.0), (50.0, 200.0), (70.0, 300.0)):
        for _ in range(int(ops // 10)):
            log.append(_ev(t, "M1", dur=10.0))
            t += 10.0
        log.append(_ev(t, "M1", breakdown=True, repair=rep))
        t += rep
    log.append(_ev(t, "M1", dur=10.0))
    return log


def test_first_left_truncated_cycle_is_dropped_but_seeds_the_history():
    samples = extract_survival_samples(_one_machine_log())
    events = [s for s in samples if s.event]
    assert [s.operating_time for s in events] == [50.0, 70.0]
    assert events[0].prev_ttf == 30.0 and events[0].prev_repair_time == 100.0
    assert [s.operating_time for s in samples if not s.event] == [10.0]


def test_first_repair_is_dropped_from_the_repair_samples_too():
    samples = extract_repair_samples(_one_machine_log())
    assert [s.repair_time for s in samples] == [200.0, 300.0]
    assert [s.operating_time_since_last for s in samples] == [50.0, 70.0]


def test_process_time_samples_are_machine_operations_only():
    orders = {"O1": {"modell": "A", "feature_a": "a.1", "feature_b": "b.1"}}
    log = [
        _ev(0.0, "M1", dur=42.0),
        _ev(1.0, "B1", stype="buffer", dur=2.0),
        _ev(2.0, "M1", breakdown=True, repair=50.0),
        _ev(3.0, "overflow", stype="overflow", dur=0.0),
        _ev(4.0, "M2", order="unknown", dur=30.0),
    ]
    joined = filter_and_join(log, orders)
    assert [(e.station, e.net_process_time) for e, _ in joined] == [("M1", 42.0)]


def test_dequantized_normal_recovers_the_latent_moments():
    rng = np.random.default_rng(0)
    xs = np.ceil(rng.normal(54.3, 2.7, size=200_000))
    mean, std = _dequantized_normal(xs)
    assert mean == pytest.approx(54.3, abs=0.02)
    assert std == pytest.approx(2.7, abs=0.02)
    assert _dequantized_normal(np.full(100, 7.0))[1] == 0.0


def test_dequantized_exponential_is_the_geometric_mle():
    rng = np.random.default_rng(1)
    xs = np.ceil(rng.exponential(800.0, size=400_000))
    assert _dequantized_exponential(xs) == pytest.approx(800.0, rel=0.01)
    # tiny scales: the closed form, not the naive mean - 0.5
    xs = np.ceil(rng.exponential(1.3, size=400_000))
    assert _dequantized_exponential(xs) == pytest.approx(1.3, rel=0.03)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
