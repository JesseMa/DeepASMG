"""Statistical time-to-failure strategies: Exponential (MTTF) and Weibull."""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import SurvivalStrategy

if TYPE_CHECKING:
    from src.config.schema import StationConfig


class RefSurvival(SurvivalStrategy):
    """Exponential TTF from per-station MTTF in operating seconds."""

    def __init__(
        self,
        mttf_data: Dict[str, float],
        rng: np.random.Generator,
    ) -> None:
        self._mttf = mttf_data
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = []
        for sid, sc in stations.items():
            if sc.is_machine and sc.mttr > 0 and sid not in self._mttf:
                missing.append(sid)
        if missing:
            raise ValueError(
                f"Fail fast: missing MTTF data for stations {missing}."
            )

    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        """TTF from Exp(MTTF), floored at 1.0 s; None if the station cannot fail."""
        mttf = self._mttf.get(station_id, float('inf'))
        if mttf == float('inf') or mttf <= 0:
            return None

        ttf = self._rng.exponential(scale=mttf)
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(float(ttf)))  # integer time contract

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """Deployed Exponential params (scale=MTTF). None = failure-free station."""
        mttf = self._mttf.get(station_id, float("inf"))
        if mttf == float("inf") or mttf <= 0:
            return None
        return {"family": "exponential", "scale": float(mttf)}


class RefSurvivalWeibull(SurvivalStrategy):
    """Per-station Weibull TTF in operating seconds.

    Parameters are the right-censored MLE over the training split's TTF spells.
    """

    def __init__(
        self,
        weibull_params: Dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> None:
        self._params = weibull_params
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = []
        for sid, sc in stations.items():
            if sc.is_machine and sc.mttr > 0 and sid not in self._params:
                missing.append(sid)
        if missing:
            raise ValueError(
                f"Fail fast: missing Weibull TTF parameters for stations {missing}."
            )

    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        """Weibull(shape, scale) TTF, one RNG draw, floored at 1.0 s."""
        params = self._params.get(station_id)
        if params is None:
            return None
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return None
        ttf = scale * float(self._rng.weibull(shape))
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(ttf))  # integer time contract

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """Deployed Weibull params (shape, scale). None = failure-free station."""
        params = self._params.get(station_id)
        if params is None:
            return None
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return None
        return {"family": "weibull", "shape": float(shape), "scale": float(scale)}
