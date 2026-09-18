"""DeepSurvival — NN-based time-to-failure, Weibull in operating seconds."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import logging

import math

import numpy as np

from src.dynamics.foundation_dynamics import (
    SurvivalStrategy, load_deep_model, infer_single,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig


_logger = logging.getLogger(__name__)


class DeepSurvival(SurvivalStrategy):

    _can_fail: set = frozenset()

    def __init__(
        self,
        survival_model_path: str | Path,
        survival_metadata_path: str | Path,
        rng: np.random.Generator,
    ) -> None:
        self._survival_model_path = Path(survival_model_path)
        self._survival_metadata_path = Path(survival_metadata_path)
        self._rng = rng

        self._survival_model = None
        self._surv_station_map: Dict[str, int] = {}
        self._surv_feature_dim: int = 0
        self._surv_duration_scale: float = 1.0
        self._surv_n_jobs_scale: float = 1.0
        self._surv_repair_time_scale: float = 1.0

        self._prev_ttf: Dict[str, float] = {}
        self._prev_n_jobs: Dict[str, float] = {}
        self._mean_ttf: Dict[str, float] = {}
        self._n_cycles: Dict[str, int] = {}
        self._prev_repair_time: Dict[str, float] = {}

    def _ensure_loaded(self) -> None:
        if self._survival_model is not None:
            return

        self._survival_model, meta = load_deep_model(
            self._survival_model_path, self._survival_metadata_path,
        )

        self._surv_station_map = meta["encoding_maps"]["station"]
        self._surv_feature_dim = meta["feature_dim"]
        self._surv_duration_scale = meta["duration_scale"]
        self._surv_n_jobs_scale = meta["n_jobs_scale"]
        self._surv_repair_time_scale = meta["repair_time_scale"]

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._ensure_loaded()

        self._prev_ttf.clear()
        self._prev_n_jobs.clear()
        self._mean_ttf.clear()
        self._n_cycles.clear()
        self._prev_repair_time.clear()

        self._can_fail = {sid for sid, cfg in stations.items() if cfg.mttr > 0}
        pooled = sorted(self._can_fail - set(self._surv_station_map))
        if pooled:
            _logger.warning(
                "DeepSurvival: no training observations for %s; these are "
                "predicted from the pooled model (all-zero station block). "
                "A rare failure mode is the ordinary case on a real log.",
                ", ".join(pooled),
            )

    def sample_time_to_failure(
        self,
        station_id: str,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        if self._survival_model is None:
            raise RuntimeError(
                "DeepSurvival.sample_time_to_failure: survival "
                "model not loaded. initialize() must run before the first "
                "call."
            )
        if station_id not in self._can_fail:
            return None

        x = self._encode_input(station_id)
        output = infer_single(self._survival_model, x)
        shape = math.exp(output[0].item())
        scale = math.exp(output[1].item())

        u_raw = self._rng.random()
        self._sg_ttf_logu_draws = getattr(self, "_sg_ttf_logu_draws", 0) + 1
        if u_raw < 1e-10:
            self._sg_ttf_logu_guard = getattr(self, "_sg_ttf_logu_guard", 0) + 1
        u = max(u_raw, 1e-10)
        ttf_normalized = scale * (-np.log(u)) ** (1.0 / shape)
        ttf = ttf_normalized * self._surv_duration_scale

        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(float(ttf)))

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        if self._survival_model is None:
            raise RuntimeError("DeepSurvival.distribution_params: model not loaded.")
        if station_id not in self._can_fail:
            return None
        output = infer_single(self._survival_model, self._encode_input(station_id))
        return {"family": "weibull", "shape": math.exp(output[0].item()),
                "scale": math.exp(output[1].item()) * self._surv_duration_scale}

    def notify_cycle_end(
        self,
        station_id: str,
        ttf: float,
        n_jobs: int,
        repair_time: float,
    ) -> None:
        prev_n = self._n_cycles.get(station_id, 0)
        prev_mean = self._mean_ttf.get(station_id, 0.0)
        new_n = prev_n + 1
        self._n_cycles[station_id] = new_n
        self._mean_ttf[station_id] = (prev_mean * prev_n + ttf) / new_n
        self._prev_ttf[station_id] = ttf
        self._prev_n_jobs[station_id] = float(n_jobs)
        self._prev_repair_time[station_id] = repair_time

    def _encode_input(self, station_id: str) -> np.ndarray:
        x = np.zeros(self._surv_feature_dim, dtype=np.float32)
        n_stations = len(self._surv_station_map)

        idx = self._surv_station_map.get(station_id)
        if idx is not None:
            x[idx] = 1.0

        dur_scale = max(self._surv_duration_scale, 1e-6)
        n_jobs_scale = max(self._surv_n_jobs_scale, 1e-6)
        repair_scale = max(self._surv_repair_time_scale, 1e-6)

        x[n_stations]     = self._prev_ttf.get(station_id, 0.0) / dur_scale
        x[n_stations + 1] = self._prev_n_jobs.get(station_id, 0.0) / n_jobs_scale
        x[n_stations + 2] = self._mean_ttf.get(station_id, 0.0) / dur_scale
        x[n_stations + 3] = self._prev_repair_time.get(station_id, 0.0) / repair_scale

        return x
