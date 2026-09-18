"""DeepRepair — NN-based downtime duration, conditional Exponential."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import logging

import numpy as np

from src.dynamics.foundation_dynamics import (
    RepairStrategy, load_deep_model, infer_single,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig


_logger = logging.getLogger(__name__)


class DeepRepair(RepairStrategy):

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
        self._pooled = sorted(required - set(self._reg_station_map))
        if self._pooled:
            _logger.warning(
                "DeepRepair: no training observations for %s; these are "
                "predicted from the pooled model (all-zero station block). "
                "A rare failure mode is the normal case on real logs.",
                ", ".join(self._pooled),
            )

    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> float:
        if self._regressor_model is None:
            raise RuntimeError(
                "DeepRepair.predict_repair_time: model not loaded. "
                "initialize() must run before the first call."
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
        if self._regressor_model is None:
            raise RuntimeError("DeepRepair.distribution_params: model not loaded.")
        output = infer_single(
            self._regressor_model,
            self._encode_input(station_id, operating_time_since_last, utilization),
        )
        return {"family": "exponential", "scale": float(np.exp(float(output[0].item())))}

    def _encode_input(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
    ) -> np.ndarray:
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
