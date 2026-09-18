"""Statistical repair-duration strategy: Exponential from per-station repair means.

A fallible machine without a training repair is simulated from the
fleet-pooled scale, as in ref_survival."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import RepairStrategy
from src.dynamics.RefSim.ref_survival import fill_pooled

if TYPE_CHECKING:
    from src.config.schema import StationConfig


class RefRepair(RepairStrategy):
    """Exponential repair time per station (scale in seconds); wear and utilization ignored."""

    def __init__(
        self,
        repair_times: Dict[str, float],
        rng: np.random.Generator,
        pooled_scale: float,
    ) -> None:
        self._repair_scales = dict(repair_times)
        self._pooled = pooled_scale
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        fill_pooled("RefRepair", self._repair_scales, self._pooled, stations)

    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,  # ignored
        utilization: float,                # ignored
        current_time: float = 0.0,  # noqa: ARG002
    ) -> float:
        scale = self._repair_scales[station_id]
        self._sg_repair_draws = getattr(self, "_sg_repair_draws", 0) + 1
        return float(np.ceil(self._rng.exponential(scale=scale)))  # integer time contract

    def distribution_params(
        self,
        station_id: str,
        operating_time_since_last: float = 0.0,  # noqa: ARG002
        utilization: float = 0.0,  # noqa: ARG002
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed Exponential params (scale = mean)."""
        return {"family": "exponential", "scale": float(self._repair_scales[station_id])}
