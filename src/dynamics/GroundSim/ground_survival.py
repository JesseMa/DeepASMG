"""GroundSurvival — time-to-failure sampling from a bathtub hazard curve."""

from __future__ import annotations
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from src.dynamics.foundation_dynamics import SurvivalStrategy
if TYPE_CHECKING:
    from src.config.schema import StationConfig


def bathtub_cumulative_hazard(
    w: float, p1: float, p2: float,
    h_early: float, h_normal: float, gamma: float,
) -> float:
    """Piecewise-analytic cumulative hazard H(w) of the bathtub model.

    The wear-out exponent is capped at 700 so H stays finite; beyond that
    S = exp(-H) is 0 in double precision either way.
    """
    if w <= 0:
        return 0.0
    h_at_p1 = h_early * p1 / 2.0 + h_normal * p1
    if w <= p1:
        return h_early * (w - w * w / (2.0 * p1)) + h_normal * w
    if w <= p2:
        return h_at_p1 + h_normal * (w - p1)
    h_at_p2 = h_at_p1 + h_normal * (p2 - p1)
    return h_at_p2 + (h_normal / gamma) * (np.exp(min(gamma * (w - p2), 700.0)) - 1.0)


# Bathtub phases in normalized wear W (W = 1 at ttf_scale_seconds): early
# failures decay linearly until p1, the hazard is flat until p2, then grows
# exponentially with rate gamma.
BATHTUB = {"p1": 0.10, "p2": 0.80, "h_early": 0.30, "h_normal": 0.08, "gamma": 8.0}


class GroundSurvival(SurvivalStrategy):
    """
    Bathtub hazard model with TTF decrement.

    Hazard rate h(W) with normalized wear W (W=1.0 at ttf_scale_seconds):
        Phase 1 (0 ≤ W < p₁):  h_early·(1 - W/p₁) + h_normal   [linearly decreasing]
        Phase 2 (p₁ ≤ W < p₂): h_normal                          [constant]
        Phase 3 (W ≥ p₂):      h_normal · exp(γ·(W - p₂))        [exponential]

    TTF = H⁻¹(E) × ttf_scale_seconds with E ~ Exp(1), inverted by bisection.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._ttf_scales: Dict[str, float] = {}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
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
                            f"ttf_scale_seconds > 0. Without a scale, no TTF "
                            f"can be sampled. Configure ttf_scale_seconds "
                            f"or set MTTR to 0 to disable failures."
                        )
                    self._ttf_scales[sid] = sc.ttf_scale_seconds

    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[float]:
        """TTF in whole operating seconds (>= 1); None if the station cannot fail."""
        scale = self._ttf_scales.get(station_id)
        if scale is None or scale <= 0:
            return None

        target_H = self._rng.exponential(1.0)
        W_star = self._inverse_cumulative_hazard(target_H)
        ttf = W_star * scale
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return float(np.ceil(ttf))  # integer time contract; ceil(x>0) >= 1

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """True bathtub TTF parameters (phases + scale); None = cannot fail."""
        scale = self._ttf_scales.get(station_id)
        if scale is None or scale <= 0:
            return None
        return {"family": "bathtub", "scale": float(scale), **BATHTUB}

    @staticmethod
    def _cumulative_hazard(W: float) -> float:
        return bathtub_cumulative_hazard(W, **BATHTUB)

    def _inverse_cumulative_hazard(
        self,
        target_H: float,
        tol: float = 1e-8,
        max_iter: int = 80,
    ) -> float:
        """Invert H(W) = target_H via bisection."""
        lo = 0.0
        hi = 3.0

        while self._cumulative_hazard(hi) < target_H:
            hi *= 2.0

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            if self._cumulative_hazard(mid) < target_H:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break

        return (lo + hi) / 2.0
