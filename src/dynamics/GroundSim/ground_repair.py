"""GroundRepair — wear-dependent downtime prediction."""

from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from src.dynamics.foundation_dynamics import RepairStrategy
if TYPE_CHECKING:
    from src.config.schema import StationConfig


# Stress = weighted relative wear and utilization of the failing cycle.
REPAIR_WEAR_WEIGHT = 0.7
REPAIR_UTIL_WEIGHT = 0.3


class GroundRepair(RepairStrategy):

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

        self._mttr: Dict[str, float] = {}
        self._ttf_scales: Dict[str, float] = {}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._mttr = {}
        self._ttf_scales = {}
        for sid, sc in stations.items():
            if sc.is_machine:
                if sc.mttr < 0:
                    raise ValueError(
                        f"Fail fast: MTTR must not be negative (station {sid})."
                    )
                if sc.mttr > 0:
                    if sc.ttf_scale_seconds <= 0:
                        raise ValueError(
                            f"Fail fast: station '{sid}' has MTTR > 0 but no "
                            f"ttf_scale_seconds > 0."
                        )
                    self._mttr[sid] = sc.mttr
                    self._ttf_scales[sid] = sc.ttf_scale_seconds

    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> float:
        stress = self._stress(station_id, operating_time_since_last, utilization)
        if stress is None:
            return 0.0

        self._sg_stress_calls = getattr(self, "_sg_stress_calls", 0) + 1
        if stress < 0.1:
            self._sg_stress_floor = getattr(self, "_sg_stress_floor", 0) + 1
        effective_mttr = self._mttr[station_id] * max(0.1, stress)
        draw = self._rng.exponential(effective_mttr)
        self._sg_repair_draws = getattr(self, "_sg_repair_draws", 0) + 1
        if draw < 1.0:
            self._sg_repair_clamp = getattr(self, "_sg_repair_clamp", 0) + 1
        return float(np.ceil(draw))  # integer time contract; ceil(x>0) >= 1

    def distribution_params(
        self,
        station_id: str,
        operating_time_since_last: float = 0.0,
        utilization: float = 0.0,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> "Optional[Dict[str, object]]":
        stress = self._stress(station_id, operating_time_since_last, utilization)
        if stress is None:
            return None
        effective_mttr = self._mttr[station_id] * max(0.1, stress)
        return {"family": "exponential", "scale": float(effective_mttr)}

    def _stress(
        self, station_id: str, operating_time_since_last: float, utilization: float,
    ) -> "Optional[float]":
        if station_id not in self._mttr:
            return None
        scale = self._ttf_scales.get(station_id, 1.0)
        wear_ratio = operating_time_since_last / max(scale, 1.0)
        util = max(0.0, min(1.0, utilization))
        return REPAIR_WEAR_WEIGHT * wear_ratio + REPAIR_UTIL_WEIGHT * util
