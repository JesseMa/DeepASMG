"""Every duration a strategy returns is a whole second, rounded exactly once.

The kernel schedules on these values directly, so a fractional duration would
either be rounded a second time or rejected by the event queue. The contract
is pinned at every emitting boundary: the three duration families of the
generator, of the fitted reference and of the learned surrogates, and the
queue that refuses anything else.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config.schema import StationConfig
from src.dynamics.GroundSim.ground_process_time import GroundProcessTime
from src.dynamics.GroundSim.ground_repair import GroundRepair
from src.dynamics.GroundSim.ground_survival import GroundSurvival
from src.dynamics.RefSim.ref_process_time import RefProcessTime
from src.dynamics.RefSim.ref_repair import RefRepair
from src.dynamics.RefSim.ref_survival import RefSurvival, RefSurvivalWeibull
from src.simulation.event import EventQueue, EventType
from src.simulation.order import Order

STATIONS = {
    "M1": StationConfig(id="M1", capacity=1, is_machine=True, mttr=1234.5,
                        ttf_scale_seconds=9876.5, setup_time=7.3,
                        process_times={"A": (12.7, 0.83)}),
    "B1": StationConfig(id="B1", capacity=1, transit_time=2),
}
ORDER = Order(id="o", features={"modell": "A", "feature_a": "a.1", "feature_b": "b.1"},
              timestamp_creation=0.0)


def _whole_seconds(values, *, at_least=1.0):
    values = np.asarray(values, dtype=float)
    assert np.all(values == np.floor(values)), "fractional durations"
    assert np.all(values >= at_least)


def test_ground_strategies_return_whole_seconds_from_fractional_parameters():
    rng = np.random.default_rng(0)
    pt, rt, sv = GroundProcessTime(rng), GroundRepair(rng), GroundSurvival(rng)
    for s in (pt, rt, sv):
        s.initialize(STATIONS)
    _whole_seconds([pt.predict("M1", ORDER, t) for t in range(200)])
    _whole_seconds([rt.predict_repair_time("M1", 500.0, 0.5) for _ in range(200)])
    _whole_seconds([sv.sample_time_to_failure("M1") for _ in range(200)])
    assert sv.sample_time_to_failure("B1") is None
    assert rt.predict_repair_time("B1", 0.0, 0.0) == 0.0


def test_ground_process_time_adds_setup_before_the_single_rounding():
    rng = np.random.default_rng(0)
    pt = GroundProcessTime(rng)
    pt.initialize(STATIONS)
    noon = 12 * 3600.0   # day shift: no night-shift factor on the latent law
    draws = np.array([pt.predict("M1", ORDER, noon) for _ in range(2000)])
    # latent net ~ N(12.7, 0.83) + setup 7.3 = 20.0 -> ceil'd values centre on 20.5
    assert draws.mean() == pytest.approx(20.5, abs=0.15)
    params = pt.distribution_params("M1", ORDER, noon)
    assert params["mu"] == pytest.approx(20.0)


def test_ref_strategies_return_whole_seconds_from_fractional_parameters():
    rng = np.random.default_rng(1)
    pt = RefProcessTime({"M1": (12.7, 0.83)}, rng)
    rt = RefRepair({"M1": 1234.5}, rng, pooled_scale=1000.0)
    sv = RefSurvival({"M1": 9876.5}, rng, pooled_mttf=9000.0)
    svw = RefSurvivalWeibull({"M1": (1.8, 9876.5)}, rng, pooled_params=(1.5, 9000.0))
    for s in (pt, rt, sv, svw):
        s.initialize(STATIONS)
    _whole_seconds([pt.predict("M1", ORDER) for _ in range(200)])
    _whole_seconds([rt.predict_repair_time("M1", 0.0, 0.0) for _ in range(200)])
    _whole_seconds([sv.sample_time_to_failure("M1") for _ in range(200)])
    _whole_seconds([svw.sample_time_to_failure("M1") for _ in range(200)])
    assert sv.sample_time_to_failure("B1") is None and svw.sample_time_to_failure("B1") is None


def test_deep_strategies_ceil_the_sampled_value(monkeypatch):
    """The learned heads emit a continuous law; the strategy rounds its draw once."""
    import torch
    from src.dynamics.DeepSim import deep_process_time, deep_repair, deep_survival

    class _Rng:
        def normal(self, mu, sigma):
            return 37.25
        def random(self):
            return 0.5

    pt = deep_process_time.DeepProcessTime.__new__(deep_process_time.DeepProcessTime)
    pt._rng = _Rng()
    pt._model = object()
    pt._prev_on_machine = {}
    monkeypatch.setattr(pt, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(pt, "_encode_single", lambda *a, **k: np.zeros(1, np.float32))
    monkeypatch.setattr(deep_process_time, "infer_single", lambda m, x: torch.tensor([37.0, 0.0]))
    assert pt.predict("M1", ORDER, 0.0) == 38.0

    rt = deep_repair.DeepRepair.__new__(deep_repair.DeepRepair)
    rt._rng = _Rng()
    rt._regressor_model = object()
    monkeypatch.setattr(rt, "_encode_input", lambda *a, **k: np.zeros(1, np.float32))
    monkeypatch.setattr(deep_repair, "infer_single", lambda m, x: torch.tensor([np.log(1000.0)]))
    got = rt.predict_repair_time("M1", 0.0, 0.0)
    assert got == np.ceil(-np.log(0.5) * 1000.0)

    sv = deep_survival.DeepSurvival.__new__(deep_survival.DeepSurvival)
    sv._rng = _Rng()
    sv._survival_model = object()
    sv._can_fail = {"M1"}
    sv._surv_duration_scale = 1000.0
    monkeypatch.setattr(sv, "_encode_input", lambda *a, **k: np.zeros(1, np.float32))
    monkeypatch.setattr(deep_survival, "infer_single", lambda m, x: torch.tensor([0.0, 0.0]))
    got = sv.sample_time_to_failure("M1")
    assert got == np.ceil(-np.log(0.5) * 1000.0)
    assert sv.sample_time_to_failure("B1") is None


def test_event_queue_refuses_fractional_times_and_pops_time_then_fifo():
    q = EventQueue()
    with pytest.raises(ValueError, match="whole seconds"):
        q.schedule(3.5, EventType.PROCESS_COMPLETE, station_id="M1")
    for t, oid in ((5, "a"), (3, "b"), (3, "c"), (7, "d"), (3.0, "e")):
        q.schedule(t, EventType.PROCESS_COMPLETE, station_id="M1", order_id=oid)
    assert q.peek_time() == 3 and isinstance(q.peek_time(), int)
    order = [(q.pop().order_id) for _ in range(5)]
    assert order == ["b", "c", "e", "a", "d"]
    assert not q


def test_station_config_refuses_fractional_transit_and_zero_capacity():
    with pytest.raises(ValueError, match="whole seconds"):
        StationConfig(id="B", capacity=1, transit_time=2.5)
    with pytest.raises(ValueError, match="capacity"):
        StationConfig(id="B", capacity=0)
    StationConfig(id="B", capacity=1, transit_time=2.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
