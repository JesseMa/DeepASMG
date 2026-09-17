"""Statistical repair-duration strategy: Exponential from per-station repair means."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import RepairStrategy

if TYPE_CHECKING:
    from src.config.schema import StationConfig


class RefRepair(RepairStrategy):
    """Exponential repair time per station (scale in seconds); wear and utilization ignored."""

    def __init__(
        self,
        repair_times: Dict[str, float],
        rng: np.random.Generator,
    ) -> None:
        self._repair_scales = repair_times
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = []
        for sid, sc in stations.items():
            if sc.is_machine and sc.mttr > 0 and sid not in self._repair_scales:
                missing.append(sid)
        if missing:
            raise ValueError(
                f"Fail fast: missing repair statistics for stations {missing}. "
                f"Machines would be simulated with no repair time (0s downtime)."
            )

    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,  # ignored
        utilization: float,                # ignored
        current_time: float = 0.0,  # noqa: ARG002
    ) -> float:
        if station_id not in self._repair_scales:
            raise RuntimeError(
                f"Fail fast: no repair statistics for station '{station_id}'. "
                f"initialize() should have prevented this."
            )

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
