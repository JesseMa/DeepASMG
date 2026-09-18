"""Statistical time-to-failure strategies: Exponential (MTTF) and Weibull.

A machine the topology declares fallible (mttr > 0) but whose training slice
holds no usable failure is simulated from the fleet-pooled fit, with a
warning. The topology decides whether a station can fail, the data how often.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import SurvivalStrategy

if TYPE_CHECKING:
    from src.config.schema import StationConfig

_logger = logging.getLogger(__name__)


def fill_pooled(name: str, table: Dict[str, object], pooled: object,
                stations: Dict[str, "StationConfig"]) -> set:
    """Give every fallible station without its own entry the pooled one.

    Returns the set of stations that can fail at all (mttr > 0): the topology
    decides that, whatever the fitted table happens to list.
    """
    can_fail = {sid for sid, sc in stations.items() if sc.mttr > 0}
    missing = sorted(can_fail - set(table))
    if missing:
        if pooled is None or (isinstance(pooled, float) and not np.isfinite(pooled)):
            raise ValueError(
                f"Fail fast: {name}: no training failures for {missing} and no "
                f"pooled fit to fall back on."
            )
        for sid in missing:
            table[sid] = pooled
        _logger.warning(
            "%s: no training failures for %s; simulated from the fleet-pooled fit. "
            "A rare failure mode is the normal case on real logs.",
            name, ", ".join(missing),
        )
    return can_fail


class RefSurvival(SurvivalStrategy):
    """Exponential TTF from per-station MTTF in operating seconds."""

    def __init__(
        self,
        mttf_data: Dict[str, float],
        rng: np.random.Generator,
        pooled_mttf: float,
    ) -> None:
        self._mttf = dict(mttf_data)
        self._pooled = pooled_mttf
        self._rng = rng
        self._can_fail: set = set()

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._can_fail = fill_pooled("RefSurvival", self._mttf, self._pooled, stations)

    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        """TTF from Exp(MTTF); None if the station cannot fail."""
        if station_id not in self._can_fail:
            return None
        ttf = self._rng.exponential(scale=self._mttf[station_id])
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(float(ttf)))  # integer time contract

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """Deployed Exponential params (scale=MTTF). None = failure-free station."""
        if station_id not in self._can_fail:
            return None
        return {"family": "exponential", "scale": float(self._mttf[station_id])}


class RefSurvivalWeibull(SurvivalStrategy):
    """Per-station Weibull TTF in operating seconds.

    Parameters are the interval-censored MLE over the training split's TTF
    spells.
    """

    def __init__(
        self,
        weibull_params: Dict[str, tuple[float, float]],
        rng: np.random.Generator,
        pooled_params: Optional[tuple[float, float]],
    ) -> None:
        self._params = dict(weibull_params)
        self._pooled = pooled_params
        self._rng = rng
        self._can_fail: set = set()

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._can_fail = fill_pooled("RefSurvivalWeibull", self._params, self._pooled, stations)

    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        """Weibull(shape, scale) TTF, one RNG draw; None if the station cannot fail."""
        if station_id not in self._can_fail:
            return None
        shape, scale = self._params[station_id]
        ttf = scale * float(self._rng.weibull(shape))
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(ttf))  # integer time contract

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """Deployed Weibull params (shape, scale). None = failure-free station."""
        if station_id not in self._can_fail:
            return None
        shape, scale = self._params[station_id]
        return {"family": "weibull", "shape": float(shape), "scale": float(scale)}
