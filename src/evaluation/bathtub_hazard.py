"""The GroundSim bathtub hazard as an evaluation-side function of time."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.dynamics.GroundSim.ground_survival import BATHTUB, bathtub_cumulative_hazard

HAZARD_GRID_POINTS = 200


def bathtub_hazard_w(w: np.ndarray, params: Dict[str, float]) -> np.ndarray:
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


def cumulative_hazard_w(w: float, params: Dict[str, float]) -> float:
    return bathtub_cumulative_hazard(
        w, params["p1"], params["p2"],
        params["h_early"], params["h_normal"], params["gamma"],
    )


def true_hazard_grid(
    ttf_scale_seconds: float, t_max: float, *, n_points: int = HAZARD_GRID_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    if ttf_scale_seconds <= 0:
        raise ValueError(f"ttf_scale_seconds must be > 0 (got {ttf_scale_seconds}).")
    t_grid = np.linspace(0.0, float(t_max), n_points)
    h_t = bathtub_hazard_w(t_grid / ttf_scale_seconds, BATHTUB) / ttf_scale_seconds
    return t_grid, h_t
