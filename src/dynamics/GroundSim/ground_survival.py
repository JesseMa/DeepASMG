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


class GroundSurvival(SurvivalStrategy):
    """
    Bathtub hazard model with TTF decrement.

    Hazard rate h(W) with normalized wear W (W=1.0 at ttf_scale_seconds):
        Phase 1 (0 ≤ W < p₁):  h_early·(1 - W/p₁) + h_normal   [linearly decreasing]
        Phase 2 (p₁ ≤ W < p₂): h_normal                          [constant]
        Phase 3 (W ≥ p₂):      h_normal · exp(γ·(W - p₂))        [exponential]

    TTF = H⁻¹(E) × ttf_scale_seconds with E ~ Exp(1), inverted by bisection.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        early_phase_end: float = 0.10,
        wearout_phase_start: float = 0.80,
        h_early: float = 0.30,
        h_normal: float = 0.08,
        gamma: float = 8.0,
    ) -> None:
        if not (0.0 < early_phase_end < wearout_phase_start):
            raise ValueError(
                "Fail fast: 0 < early_phase_end < wearout_phase_start must hold."
            )
        if wearout_phase_start >= 2.0:
            raise ValueError("Fail fast: wearout_phase_start must be < 2.0.")
        if h_early < 0 or h_normal <= 0:
            raise ValueError("Fail fast: h_early >= 0 and h_normal > 0 required.")
        if gamma <= 0:
            raise ValueError("Fail fast: gamma must be > 0.")

        self._rng = rng

        self._p1 = early_phase_end
        self._p2 = wearout_phase_start
        self._h_early = h_early
        self._h_normal = h_normal
        self._gamma = gamma


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
        """TTF in operating seconds, floored at 1.0 s; None if the station cannot fail."""
        scale = self._ttf_scales.get(station_id)
        if scale is None or scale <= 0:
            return None

        target_H = self._rng.exponential(1.0)
        W_star = self._inverse_cumulative_hazard(target_H)
        ttf = W_star * scale
        self._sg_ttf_draws = getattr(self, "_sg_ttf_draws", 0) + 1
        if ttf < 1.0:
            self._sg_ttf_clamp = getattr(self, "_sg_ttf_clamp", 0) + 1
        return max(1.0, float(ttf))

    def distribution_params(
        self, station_id: str, current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[Dict[str, object]]:
        """True bathtub TTF parameters (phases + scale); None = cannot fail."""
        scale = self._ttf_scales.get(station_id)
        if scale is None or scale <= 0:
            return None
        return {
            "family": "bathtub", "scale": float(scale),
            "p1": float(self._p1), "p2": float(self._p2),
            "h_early": float(self._h_early), "h_normal": float(self._h_normal),
            "gamma": float(self._gamma),
        }

    def _cumulative_hazard(self, W: float) -> float:
        return bathtub_cumulative_hazard(
            W, self._p1, self._p2, self._h_early, self._h_normal, self._gamma,
        )

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
