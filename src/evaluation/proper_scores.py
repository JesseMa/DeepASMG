"""CRPS, NLL and ECE as pure functions, independent of the simulation."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.special import gamma as gamma_fn, gammainc
from scipy.stats import norm, weibull_min



def crps_normal(mu: float, sigma: float, y: float) -> float:
    """CRPS(N(μ,σ), y) in closed form.

    CRPS = σ·[ z·(2Φ(z) − 1) + 2φ(z) − 1/√π ],  z = (y − μ)/σ.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0 (got {sigma}).")
    z = (y - mu) / sigma
    return float(
        sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    )


def crps_normal_vec(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Array form of ``crps_normal``, broadcasting over the inputs."""
    if np.any(sigma <= 0):
        raise ValueError(f"sigma must be > 0 (got min {np.min(sigma)}).")
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_exponential(scale: float, y: float) -> float:
    """CRPS(Exp(scale=β), y) in closed form, y ≥ 0.

    With β = 1/λ: CRPS = y − 1.5β + 2β·e^{−y/β}. (At y=0: β/2.)
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0 (got {scale}).")
    yy = max(0.0, float(y))
    return float(yy - 1.5 * scale + 2.0 * scale * np.exp(-yy / scale))


def crps_weibull(shape: float, scale: float, y: float) -> float:
    """CRPS(Weibull(k=shape, λ=scale), y) in closed form, y ≥ 0.

        CRPS = y − 2λΓ(1+1/k)·P(1/k, (y/λ)^k) + λΓ(1+1/k)·2^{−1/k}

    with P = scipy.special.gammainc (regularized lower incomplete gamma).
    """
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape/scale must be > 0 (got {shape}, {scale}).")
    yy = max(0.0, float(y))
    mean = scale * float(gamma_fn(1.0 + 1.0 / shape))
    z = (yy / scale) ** shape
    return float(yy - 2.0 * mean * float(gammainc(1.0 / shape, z))
                 + mean * 2.0 ** (-1.0 / shape))


def nll_normal(mu: float, sigma: float, y: float) -> float:
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0 (got {sigma}).")
    return float(0.5 * np.log(2.0 * np.pi * sigma * sigma) + (y - mu) ** 2 / (2.0 * sigma * sigma))


def nll_normal_vec(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np.any(sigma <= 0):
        raise ValueError(f"sigma must be > 0 (got min {np.min(sigma)}).")
    return 0.5 * np.log(2 * np.pi * sigma * sigma) + (y - mu) ** 2 / (2 * sigma * sigma)


def nll_exponential(scale: float, y: float) -> float:
    if scale <= 0:
        raise ValueError(f"scale must be > 0 (got {scale}).")
    return float(np.log(scale) + max(0.0, float(y)) / scale)


def nll_weibull(shape: float, scale: float, y: float) -> float:
    if shape <= 0 or scale <= 0:
        raise ValueError(f"shape/scale must be > 0 (got {shape}, {scale}).")
    logpdf = weibull_min.logpdf(max(1e-12, float(y)), shape, scale=scale)
    return float(-logpdf)

def compute_ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 15,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Dict[str, float]:
    """Confidence ECE with ``n_bins`` equal-width bins over [lo, hi].

    ECE = Σ (|B|/N)·|acc(B) − conf(B)|; empty bins contribute 0.
    """
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
