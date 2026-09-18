"""The scoring rule must prefer the truth.

Every duration a strategy returns is ceil() of its latent draw, so the law a
system deploys is the lattice law P(Y = k) = F(k) - F(k-1). Scoring a
continuous density against those whole seconds instead is minimized half a bin
away from the truth, which would reward a forecast that has absorbed the
rounding once already and let it beat the generator's own parameters.

These tests pin that the rule in use does not have that defect.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.score_aggregation import continuous_scores

SEED = 20260917
N = 3000


def _observations(mu: float, sigma: float, n: int = N) -> np.ndarray:
    """Whole-second log entries produced by a latent N(mu, sigma)."""
    return np.ceil(np.random.default_rng(SEED).normal(mu, sigma, n))


def _mean_score(mu: float, sigma: float, obs: np.ndarray, index: int) -> float:
    return float(np.mean([
        continuous_scores("normal", {"mu": mu, "sigma": sigma}, y, False)[index]
        for y in obs
    ]))


@pytest.mark.parametrize("index,name", [(0, "CRPS"), (1, "NLL")])
def test_score_is_minimized_at_the_truth(index, name):
    """Shifting the reported mean away from the truth must cost, in both signs."""
    mu, sigma = 74.0, 1.0
    obs = _observations(mu, sigma)
    at_truth = _mean_score(mu, sigma, obs, index)
    for delta in (-0.5, -0.25, 0.25, 0.5):
        shifted = _mean_score(mu + delta, sigma, obs, index)
        assert shifted > at_truth, (
            f"{name} prefers a mean shifted by {delta:+} over the truth"
        )


@pytest.mark.parametrize("sigma", [0.74, 1.0, 2.5, 4.5])
def test_truth_beats_the_naively_quantized_competitor(sigma):
    """The oracle must beat a forecast fitted to the log without dequantizing.

    A model trained on ceil()'d observations as if they were exact converges to
    mean + 1/2 and variance + 1/12. That competitor must lose to the generator's
    own parameters at every station width in this topology.
    """
    mu = 54.0
    obs = _observations(mu, sigma)
    naive_mu, naive_sigma = mu + 0.5, float(np.sqrt(sigma**2 + 1.0 / 12.0))
    for index, name in ((0, "CRPS"), (1, "NLL")):
        truth = _mean_score(mu, sigma, obs, index)
        naive = _mean_score(naive_mu, naive_sigma, obs, index)
        assert truth < naive, (
            f"{name} at sigma={sigma}: the naively quantized forecast wins "
            f"({naive:.4f} < {truth:.4f})"
        )


def test_sharper_and_blunter_forecasts_both_lose():
    """Misreporting the spread must cost in either direction."""
    mu, sigma = 74.0, 1.0
    obs = _observations(mu, sigma)
    at_truth = _mean_score(mu, sigma, obs, 1)
    assert _mean_score(mu, sigma * 0.5, obs, 1) > at_truth
    assert _mean_score(mu, sigma * 2.0, obs, 1) > at_truth


@pytest.mark.parametrize("family,params,y", [
    ("normal", {"mu": 74.0, "sigma": 1.0}, 74.0),
    ("exponential", {"scale": 1800.0}, 1500.0),
    ("weibull", {"shape": 1.8, "scale": 73000.0}, 60000.0),
    ("bathtub", {"scale": 73000.0}, 60000.0),
])
def test_every_family_scores_finitely(family, params, y):
    crps, nll = continuous_scores(family, params, y, False)
    assert np.isfinite(crps) and crps > 0
    assert np.isfinite(nll)


@pytest.mark.parametrize("family,params", [
    ("exponential", {"scale": 1800.0}),
    ("weibull", {"shape": 1.8, "scale": 73000.0}),
    ("bathtub", {"scale": 73000.0}),
])
def test_censored_rows_score_the_survival_term_only(family, params):
    crps, nll = continuous_scores(family, params, 60000.0, True)
    assert crps is None
    assert np.isfinite(nll) and nll > 0


def test_censoring_is_rejected_for_the_normal_family():
    """Processing and repair durations always complete; a censored row there is a bug."""
    with pytest.raises(ValueError, match="censoring is not defined"):
        continuous_scores("normal", {"mu": 74.0, "sigma": 1.0}, 74.0, True)


def test_far_tail_stays_finite():
    """A grossly wrong forecast must score badly, not crash on underflow."""
    _, nll = continuous_scores("normal", {"mu": 50.0, "sigma": 1.0}, 120.0, False)
    assert np.isfinite(nll) and nll > 100


@pytest.mark.parametrize("family,params", [
    ("normal", {"mu": 74.0, "sigma": 1.0}),
    ("exponential", {"scale": 1800.0}),
    ("weibull", {"shape": 1.8, "scale": 73000.0}),
])
def test_the_upper_tail_does_not_saturate(family, params):
    """Worse forecasts must keep scoring worse, far above the predictive mean.

    Taken from the CDF alone the interval probability cancels to zero once
    F(k) and F(k-1) are both 1.0 in double precision, which would clamp every
    gross overshoot to the same value — exactly where a surrogate fails.
    """
    base = params.get("mu", params.get("scale"))
    points = [base * m for m in (2.0, 4.0, 8.0)] if family != "normal" else [
        74.0 + d for d in (10.0, 20.0, 30.0)
    ]
    scores = [continuous_scores(family, params, k, False)[1] for k in points]
    assert all(np.isfinite(s) for s in scores)
    assert scores[0] < scores[1] < scores[2], (
        f"{family}: the tail saturates instead of penalising worse overshoots "
        f"({scores})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
