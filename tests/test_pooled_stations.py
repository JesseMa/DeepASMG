"""A fallible station without its own training failures is simulated from the pool, never made immortal by accident."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.config.schema import StationConfig
from src.dynamics.RefSim.ref_repair import RefRepair
from src.dynamics.RefSim.ref_survival import RefSurvival, RefSurvivalWeibull

STATIONS = {
    "M1": StationConfig(id="M1", capacity=1, is_machine=True, mttr=3000.0, ttf_scale_seconds=1e5),
    "M2": StationConfig(id="M2", capacity=1, is_machine=True, mttr=3000.0, ttf_scale_seconds=1e5),
    "B1": StationConfig(id="B1", capacity=1, transit_time=2),
}


def _deep_survival(monkeypatch, station_map):
    import torch
    from src.dynamics.DeepSim import deep_survival
    sv = deep_survival.DeepSurvival.__new__(deep_survival.DeepSurvival)
    sv._rng = np.random.default_rng(0)
    sv._survival_model = object()
    sv._surv_station_map = station_map
    sv._surv_feature_dim = len(station_map) + 4
    sv._surv_duration_scale = 1e5
    sv._surv_n_jobs_scale = sv._surv_repair_time_scale = 1.0
    for name in ("_prev_ttf", "_prev_n_jobs", "_mean_ttf", "_n_cycles", "_prev_repair_time"):
        setattr(sv, name, {})
    monkeypatch.setattr(sv, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(deep_survival, "infer_single", lambda m, x: torch.tensor([np.log(5.0), 0.0]))
    return sv


def test_deep_survival_pools_an_unobserved_fallible_station(monkeypatch, caplog):
    sv = _deep_survival(monkeypatch, {"M1": 0})
    with caplog.at_level(logging.WARNING):
        sv.initialize(STATIONS)
    assert "M2" in caplog.text
    ttf = sv.sample_time_to_failure("M2")
    assert ttf is not None and ttf >= 1.0 and ttf == int(ttf)
    assert sv.distribution_params("M2")["family"] == "weibull"
    assert sv.sample_time_to_failure("B1") is None and sv.distribution_params("B1") is None
    # the station block is all zero for the pooled station, one-hot otherwise
    assert sv._encode_input("M2")[0] == 0.0 and sv._encode_input("M1")[0] == 1.0


def test_deep_repair_pools_an_unobserved_fallible_station(monkeypatch, caplog):
    import torch
    from src.dynamics.DeepSim import deep_repair
    rt = deep_repair.DeepRepair.__new__(deep_repair.DeepRepair)
    rt._rng = np.random.default_rng(0)
    rt._regressor_model = object()
    rt._reg_station_map = {"M1": 0}
    rt._reg_feature_dim = 4
    rt._reg_op_time_mean = rt._reg_util_mean = rt._reg_wear_mean = 0.0
    rt._reg_op_time_std = rt._reg_util_std = rt._reg_wear_std = 1.0
    rt._reg_median_ttf = {"M1": 1e5}
    monkeypatch.setattr(rt, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(deep_repair, "infer_single", lambda m, x: torch.tensor([np.log(500.0)]))
    with caplog.at_level(logging.WARNING):
        rt.initialize(STATIONS)
    assert "M2" in caplog.text
    got = rt.predict_repair_time("M2", 1000.0, 0.5)
    assert got >= 1.0 and got == int(got)
    assert rt.distribution_params("M2", 1000.0, 0.5) == {"family": "exponential", "scale": pytest.approx(500.0)}


def test_ref_strategies_pool_an_unobserved_fallible_station(caplog):
    rng = np.random.default_rng(0)
    sv = RefSurvival({"M1": 8e4}, rng, pooled_mttf=9e4)
    rt = RefRepair({"M1": 900.0}, rng, pooled_scale=1100.0)
    svw = RefSurvivalWeibull({"M1": (5.0, 1e5)}, rng, pooled_params=(4.0, 9e4))
    with caplog.at_level(logging.WARNING):
        for s in (sv, rt, svw):
            s.initialize(STATIONS)
    assert caplog.text.count("M2") == 3
    assert sv.distribution_params("M2") == {"family": "exponential", "scale": 9e4}
    assert rt.distribution_params("M2") == {"family": "exponential", "scale": 1100.0}
    assert svw.distribution_params("M2") == {"family": "weibull", "shape": 4.0, "scale": 9e4}
    assert sv.sample_time_to_failure("M2") is not None
    assert sv.sample_time_to_failure("B1") is None


def test_ref_strategies_refuse_a_missing_station_without_a_pool():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no pooled fit"):
        RefSurvival({"M1": 8e4}, rng, pooled_mttf=float("nan")).initialize(STATIONS)
    with pytest.raises(ValueError, match="no pooled fit"):
        RefSurvivalWeibull({"M1": (5.0, 1e5)}, rng, pooled_params=None).initialize(STATIONS)


def test_buffers_listed_in_a_fitted_table_are_never_drawn():
    rng = np.random.default_rng(0)
    sv = RefSurvival({"M1": 8e4, "M2": 8e4, "B1": float("inf")}, rng, pooled_mttf=9e4)
    sv.initialize(STATIONS)
    state = rng.bit_generator.state
    assert sv.sample_time_to_failure("B1") is None
    assert rng.bit_generator.state == state


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
