"""Lattice NLL, lattice CRPS and ECE as pure functions, independent of the simulation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

_LOG_HALF = -0.6931471805599453


def interval_nll(log_cdf, log_sf, k: float, *, lo: float = 0.0) -> float:
    if k <= lo:
        raise ValueError(f"Fail fast: a realized duration must exceed {lo}, got {k}.")
    left = max(k - 1.0, lo)
    if float(log_cdf(k)) > _LOG_HALF:
        a, b = float(log_sf(left)), float(log_sf(k))
    else:
        a, b = float(log_cdf(k)), float(log_cdf(left))
    return float(-(a + np.log1p(-np.exp(b - a))))


def survival_nll(log_sf, k: float) -> float:
    return float(-log_sf(k))


_MAX_SUPPORT = 10**8       # three years at one-second resolution
_CHUNK = 2**20


def lattice_crps(cdf, k: float, *, lo: int, hi: int) -> float:
    if hi - lo > _MAX_SUPPORT:
        raise ValueError(
            f"Fail fast: predictive law spans {hi - lo:,} s around k={k}; "
            f"a forecast this wide is a broken model, not a score."
        )
    total = 0.0
    for start in range(lo, hi + 1, _CHUNK):
        j = np.arange(start, min(start + _CHUNK, hi + 1), dtype=float)
        F = np.asarray(cdf(j), dtype=float)
        total += float(np.sum((F - (j >= k)) ** 2))
    return total


def _support_bounds(cdf, k: float, *, step: float, eps: float = 1e-9) -> tuple[int, int]:
    lo = max(int(np.floor(k)) - 1, 0)
    while lo > 0 and float(cdf(lo)) > eps:
        lo = max(int(lo - step), 0)
    hi = int(np.ceil(k)) + 1
    while float(cdf(hi)) < 1.0 - eps:
        hi = int(hi + step)
    return lo, hi


def compute_ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 15,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Dict[str, float]:
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return {"ece": float("nan"), "top_bin_gap": float("nan"), "n": 0}
    edges = np.linspace(lo, hi, n_bins + 1)
    ece = 0.0
    top_gap = float('nan')  # distinguishes 'no data in top bin' from a zero gap
    for i in range(n_bins):
        if i < n_bins - 1:
            in_bin = (conf >= edges[i]) & (conf < edges[i + 1])
        else:
            in_bin = (conf >= edges[i]) & (conf <= edges[i + 1])
        cnt = int(np.sum(in_bin))
        if cnt == 0:
            continue
        acc_b = float(np.mean(acc[in_bin]))
        conf_b = float(np.mean(conf[in_bin]))
        ece += (cnt / n) * abs(acc_b - conf_b)
        if i == n_bins - 1:
            top_gap = acc_b - conf_b
    return {"ece": float(ece), "top_bin_gap": float(top_gap), "n": int(n)}
