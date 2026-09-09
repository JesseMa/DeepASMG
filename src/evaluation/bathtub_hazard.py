"""Bathtub hazard in the wear domain, plus TTF scoring (NLL, CRPS).

The hazard is defined over normalized wear W, so the operating-second form is
h_T(t) = h_W(t/scale)/scale with the per-station ttf_scale_seconds.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import quad

from src.dynamics.GroundSim.ground_survival import (
    GroundSurvival, bathtub_cumulative_hazard,
)

HAZARD_GRID_POINTS = 200
# scipy.integrate.quad defaults, pinned explicitly.
BATHTUB_CRPS_EPSABS = 1.49e-8
BATHTUB_CRPS_EPSREL = 1.49e-8


def ground_survival_params() -> Dict[str, float]:
    """Bathtub phase parameters from a default ``GroundSurvival``, so scoring
    and the simulated runs share one set of values."""
    gs = GroundSurvival(np.random.default_rng(0))
    return {
        "p1": gs._p1, "p2": gs._p2,
        "h_early": gs._h_early, "h_normal": gs._h_normal, "gamma": gs._gamma,
    }


def bathtub_hazard_w(w: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Instantaneous hazard h(W) in the wear domain (piecewise, analytic)."""
    w = np.asarray(w, dtype=float)
    p1, p2 = params["p1"], params["p2"]
    h_e, h_n, g = params["h_early"], params["h_normal"], params["gamma"]
    h = np.empty_like(w)
    m1 = w <= p1
    m2 = (w > p1) & (w <= p2)
    m3 = w > p2
    h[m1] = h_e * (1.0 - w[m1] / p1) + h_n
    h[m2] = h_n
    h[m3] = h_n * np.exp(g * (w[m3] - p2))
    return h


def true_hazard_grid(
    ttf_scale_seconds: float,
    t_max: float,
    *,
    n_points: int = HAZARD_GRID_POINTS,
    params: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """True hazard h_T(t) over t∈[0, t_max] in operating seconds."""
    if ttf_scale_seconds <= 0:
        raise ValueError(f"ttf_scale_seconds must be > 0 (got {ttf_scale_seconds}).")
    p = params if params is not None else ground_survival_params()
    t_grid = np.linspace(0.0, float(t_max), n_points)
    h_t = bathtub_hazard_w(t_grid / ttf_scale_seconds, p) / ttf_scale_seconds
    return t_grid, h_t


def cumulative_hazard_w(w: float, params: Dict[str, float]) -> float:
    """Cumulative hazard H(W)=∫₀ᵂ h, from the GroundSim bathtub definition."""
    return bathtub_cumulative_hazard(
        w, params["p1"], params["p2"],
        params["h_early"], params["h_normal"], params["gamma"],
    )


def bathtub_survival(t: float, scale: float, params: Dict[str, float]) -> float:
    """S(t)=exp(−H(t/scale))."""
    return float(np.exp(-cumulative_hazard_w(t / scale, params)))


def bathtub_nll(t: float, scale: float, censored: bool,
                params: Dict[str, float] | None = None) -> float:
    """NLL of the bathtub TTF (operating seconds). Uncensored: −log f = −log h + H;
    censored: −log S = H. (f = h·S, S = exp(−H).)"""
    p = params if params is not None else ground_survival_params()
    w = max(t, 0.0) / scale
    big_h = cumulative_hazard_w(w, p)
    if censored:
        return float(big_h)
    h_t = float(bathtub_hazard_w(np.array([w]), p)[0]) / scale
    return float(-np.log(max(h_t, 1e-300)) + big_h)


def _bathtub_tail_end_w(params: Dict[str, float], h_target: float = 40.0) -> float:
    """Smallest w with H(w) ≥ h_target (S² = e^{−2H} ≤ e^{−80}: negligible).

    H is continuous and strictly increasing, so plain bisection on
    cumulative_hazard_w is exact enough for an integration bound.
    """
    lo, hi = 0.0, 4.0
    while cumulative_hazard_w(hi, params) < h_target:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cumulative_hazard_w(mid, params) < h_target:
            lo = mid
        else:
            hi = mid
    return hi


def bathtub_crps(t: float, scale: float,
                 params: Dict[str, float] | None = None) -> float:
    """CRPS of the bathtub TTF: ∫₀^y F² dx + ∫_y^∞ (1−F)² dx with F = 1 − S.

    The upper integral runs to a FINITE bound w_end where the survival mass is
    negligible (H ≥ 40 ⇒ integrand ≤ e^{−80}); quadrature to infinity does not
    converge reliably for realized TTFs in the mid-range of the distribution.
    The hazard-phase boundaries are passed as explicit
    quadrature break points.
    """
    p = params if params is not None else ground_survival_params()
    yy = max(0.0, float(t))

    def cdf(x: float) -> float:
        return 1.0 - bathtub_survival(x, scale, p)

    w_end = _bathtub_tail_end_w(p)
    t_end = max(w_end * scale, yy * (1.0 + 1e-9) + 1.0)
    breaks = sorted({p["p1"] * scale, p["p2"] * scale})
    lower_pts = [b for b in breaks if 0.0 < b < yy] or None
    upper_pts = [b for b in breaks if yy < b < t_end] or None

    lower, _ = quad(lambda x: cdf(x) ** 2, 0.0, yy, points=lower_pts,
                    limit=200, epsabs=BATHTUB_CRPS_EPSABS, epsrel=BATHTUB_CRPS_EPSREL)
    upper, _ = quad(lambda x: (1.0 - cdf(x)) ** 2, yy, t_end, points=upper_pts,
                    limit=200, epsabs=BATHTUB_CRPS_EPSABS, epsrel=BATHTUB_CRPS_EPSREL)
    return float(lower + upper)
