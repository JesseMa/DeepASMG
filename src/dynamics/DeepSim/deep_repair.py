"""
DeepRepair — NN-based downtime duration, conditional Exponential.

Exponential matches the GroundRepair family, so y > 0 holds structurally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    RepairStrategy, load_deep_model, infer_single,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig


class DeepRepair(RepairStrategy):
    """Repair duration by inversion sampling from an Exponential with the
    predicted scale."""

    def __init__(
        self,
        regressor_model_path: str | Path,
        regressor_metadata_path: str | Path,
        rng: np.random.Generator,
    ) -> None:
        self._regressor_model_path = Path(regressor_model_path)
        self._regressor_metadata_path = Path(regressor_metadata_path)
        self._rng = rng

        self._regressor_model = None
        self._reg_station_map: Dict[str, int] = {}
        self._reg_feature_dim: int = 0
        self._reg_op_time_mean: float = 0.0
        self._reg_op_time_std: float = 1.0
        self._reg_util_mean: float = 0.0
        self._reg_util_std: float = 1.0
        self._reg_wear_mean: float = 0.0
        self._reg_wear_std: float = 1.0
        self._reg_median_ttf: Dict[str, float] = {}

    def _ensure_loaded(self) -> None:
        if self._regressor_model is not None:
            return

        self._regressor_model, reg_meta = load_deep_model(
            self._regressor_model_path, self._regressor_metadata_path,
        )

        self._reg_station_map = reg_meta["encoding_maps"]["station"]
        self._reg_feature_dim = reg_meta["feature_dim"]
        self._reg_op_time_mean = reg_meta["operating_time_mean"]
        self._reg_op_time_std = reg_meta["operating_time_std"]
        self._reg_util_mean = reg_meta["utilization_mean"]
        self._reg_util_std = reg_meta["utilization_std"]
        self._reg_wear_mean = reg_meta["wear_ratio_mean"]
        self._reg_wear_std = reg_meta["wear_ratio_std"]
        self._reg_median_ttf = reg_meta["median_ttf_per_station"]

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._ensure_loaded()

        required = {sid for sid, cfg in stations.items() if cfg.mttr > 0}
        missing = sorted(required - set(self._reg_station_map))
        if missing:
            raise RuntimeError(
                "DeepRepair: encoding map does not cover all "
                "machine stations. "
                f"Missing from the regressor model: {missing}. "
                "Model training and station topology are inconsistent."
            )

    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> float:
        if self._regressor_model is None or station_id not in self._reg_station_map:
            raise RuntimeError(
                f"DeepRepair.predict_repair_time: station "
                f"'{station_id}' not in the regressor encoding map, or model "
                f"not loaded. initialize() should have caught this."
            )

        x = self._encode_input(
            station_id, operating_time_since_last, utilization,
        )
        output = infer_single(self._regressor_model, x)
        log_scale = float(output[0].item())
        scale = float(np.exp(log_scale))
        u_raw = self._rng.random()
        self._sg_repair_logu_draws = getattr(self, "_sg_repair_logu_draws", 0) + 1
        if u_raw < 1e-10:
            self._sg_repair_logu_guard = getattr(self, "_sg_repair_logu_guard", 0) + 1
        u = max(u_raw, 1e-10)
        sample = -np.log(u) * scale
        self._sg_repair_draws = getattr(self, "_sg_repair_draws", 0) + 1
        if sample < 1.0:
            self._sg_repair_clamp = getattr(self, "_sg_repair_clamp", 0) + 1
        return float(np.ceil(float(sample)))  # integer time contract

    def distribution_params(
        self,
        station_id: str,
        operating_time_since_last: float = 0.0,
        utilization: float = 0.0,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed Exponential params (scale = exp(log_scale) = mean)."""
        if self._regressor_model is None or station_id not in self._reg_station_map:
            raise RuntimeError(f"DeepRepair.distribution_params: station '{station_id}' missing.")
        output = infer_single(
            self._regressor_model,
            self._encode_input(station_id, operating_time_since_last, utilization),
        )
        return {"family": "exponential", "scale": float(np.exp(float(output[0].item())))}

    def notify_cycle_end(
        self,
        station_id: str,
        ttf: float,
        n_jobs: int,
        repair_time: float,
    ) -> None:
        """No-op: the feature set does not depend on TTF history."""

    def _encode_input(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
    ) -> np.ndarray:
        """Regressor feature vector: [station_onehot, operating_time,
        utilization, wear_ratio], the three continuous entries normalized with
        the training statistics. wear_ratio is rebuilt at inference time from
        the train-fitted per-station median cycle length; a station without a
        median (fewer than two observed cycles) yields 0.0, matching the cold
        start used during data preparation."""
        x = np.zeros(self._reg_feature_dim, dtype=np.float32)

        n_stations = len(self._reg_station_map)
        idx = self._reg_station_map.get(station_id)
        if idx is not None:
            x[idx] = 1.0

        denom = self._reg_median_ttf.get(station_id, 0.0)
        wear_ratio = (
            operating_time_since_last / max(denom, 1.0) if denom > 0 else 0.0
        )

        x[n_stations] = (
            (operating_time_since_last - self._reg_op_time_mean)
            / max(self._reg_op_time_std, 1e-6)
        )
        x[n_stations + 1] = (
            (utilization - self._reg_util_mean) / max(self._reg_util_std, 1e-6)
        )
        x[n_stations + 2] = (
            (wear_ratio - self._reg_wear_mean) / max(self._reg_wear_std, 1e-6)
        )

        return x
