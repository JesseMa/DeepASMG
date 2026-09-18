"""Lattice NLL, lattice CRPS and ECE as pure functions, independent of the simulation.

Under the integer time contract every strategy returns ceil(X), so the law a
system actually deploys is the lattice law P(Y = k) = F(k) - F(k-1). The
scores here take that law directly: the interval NLL is its log-likelihood and
the lattice CRPS is the CRPS definition evaluated on the step CDF
G(x) = F(floor(x)). Scoring a continuous density against whole-second
observations instead would reward a forecast shifted by half a bin.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

_LOG_HALF = -0.6931471805599453


def interval_nll(log_cdf, log_sf, k: float, *, lo: float = 0.0) -> float:
    """-log P(Y = k) for Y = ceil(X), i.e. -log(F(k) - F(k-1)).

    Formed in log space from whichever side does not cancel: the cdf in the
    lower half of the law, the survival function in the upper half. Far in a
    tail both F(k) and F(k-1) (or both S values) agree to every stored digit
    and their plain difference is zero; log(e^a - e^b) = a + log1p(-e^(b-a))
    keeps the digits. These are exactly the rows on which a surrogate is
    wrong, and the score has to say by how much rather than saturate.
    """
    if k <= lo:
        raise ValueError(f"Fail fast: a realized duration must exceed {lo}, got {k}.")
    left = max(k - 1.0, lo)
    if float(log_cdf(k)) > _LOG_HALF:
        a, b = float(log_sf(left)), float(log_sf(k))
    else:
        a, b = float(log_cdf(k)), float(log_cdf(left))
    return float(-(a + np.log1p(-np.exp(b - a))))


def survival_nll(log_sf, k: float) -> float:
    """-log S(k) for a right-censored whole-second observation."""
    return float(-log_sf(k))


_MAX_SUPPORT = 10**8       # three years at one-second resolution
_CHUNK = 2**20


def lattice_crps(cdf, k: float, *, lo: int, hi: int) -> float:
    """CRPS of the lattice law on the integer grid [lo, hi].

    sum_j (F(j) - 1{j >= k})^2 over the support where F is neither 0 nor 1;
    outside that range the summand vanishes, so the bounds only need to cover
    the probability mass (see _support_bounds). The grid is summed in chunks
    of bounded memory; a law spanning more than _MAX_SUPPORT seconds is not a
    forecast of anything this study measures and is refused.
    """
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
    """Integer range outside which the lattice CRPS summand is below eps.

    Walks outward from the observation in multiples of `step` until F is within
    eps of 0 below and of 1 above, so the sum is exact to that tolerance
    regardless of how wide the distribution is.
    """
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
