"""The scoring rule must prefer the truth."""

from __future__ import annotations

import numpy as np
import pytest

from src.dynamics.GroundSim.ground_survival import BATHTUB
from src.evaluation.score_aggregation import continuous_scores

from scipy.stats import norm  # noqa: E402


def _expected_score(truth_mu, truth_sigma, mu, sigma, index) -> float:
    ks = np.arange(np.floor(truth_mu - 9 * truth_sigma), np.ceil(truth_mu + 9 * truth_sigma) + 1)
    w = norm.cdf(ks, truth_mu, truth_sigma) - norm.cdf(ks - 1, truth_mu, truth_sigma)
    scores = np.array([continuous_scores("normal", {"mu": mu, "sigma": sigma}, float(k), False)[index]
                       for k in ks])
    return float(np.sum(w * scores) / np.sum(w))


@pytest.mark.parametrize("index,name", [(0, "CRPS"), (1, "NLL")])
def test_score_is_minimized_at_the_truth(index, name):
    mu, sigma = 74.0, 1.0
    at_truth = _expected_score(mu, sigma, mu, sigma, index)
    for delta in (-0.5, -0.25, 0.25, 0.5):
        shifted = _expected_score(mu, sigma, mu + delta, sigma, index)
        assert shifted > at_truth, (
            f"{name} prefers a mean shifted by {delta:+} over the truth"
        )


@pytest.mark.parametrize("sigma", [0.74, 1.0, 2.5, 4.5])
def test_truth_beats_the_naively_quantized_competitor(sigma):
    mu = 54.0
    naive_mu, naive_sigma = mu + 0.5, float(np.sqrt(sigma**2 + 1.0 / 12.0))
    for index, name in ((0, "CRPS"), (1, "NLL")):
        truth = _expected_score(mu, sigma, mu, sigma, index)
        naive = _expected_score(mu, sigma, naive_mu, naive_sigma, index)
        assert truth < naive, (
            f"{name} at sigma={sigma}: the naively quantized forecast wins "
            f"({naive:.4f} < {truth:.4f})"
        )


def test_sharper_and_blunter_forecasts_both_lose():
    mu, sigma = 74.0, 1.0
    at_truth = _expected_score(mu, sigma, mu, sigma, 1)
    assert _expected_score(mu, sigma, mu, sigma * 0.5, 1) > at_truth
    assert _expected_score(mu, sigma, mu, sigma * 2.0, 1) > at_truth


@pytest.mark.parametrize("family,params,y", [
    ("normal", {"mu": 74.0, "sigma": 1.0}, 74.0),
    ("exponential", {"scale": 1800.0}, 1500.0),
    ("weibull", {"shape": 1.8, "scale": 73000.0}, 60000.0),
    ("bathtub", {"scale": 73000.0, **BATHTUB}, 60000.0),
])
def test_every_family_scores_finitely(family, params, y):
    crps, nll = continuous_scores(family, params, y, False)
    assert np.isfinite(crps) and crps > 0
    assert np.isfinite(nll)


@pytest.mark.parametrize("family,params", [
    ("exponential", {"scale": 1800.0}),
    ("weibull", {"shape": 1.8, "scale": 73000.0}),
    ("bathtub", {"scale": 73000.0, **BATHTUB}),
])
def test_censored_rows_score_the_survival_term_only(family, params):
    crps, nll = continuous_scores(family, params, 60000.0, True)
    assert crps is None
    assert np.isfinite(nll) and nll > 0


def test_censoring_is_rejected_for_the_normal_family():
    with pytest.raises(ValueError, match="censoring is not defined"):
        continuous_scores("normal", {"mu": 74.0, "sigma": 1.0}, 74.0, True)


def test_far_tail_stays_finite():
    _, nll = continuous_scores("normal", {"mu": 50.0, "sigma": 1.0}, 120.0, False)
    assert np.isfinite(nll) and nll > 100


def _reference_nll(family, params, k, censored=False):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    if family == "normal":
        z = lambda t: (mp.mpf(t) - params["mu"]) / params["sigma"]  # noqa: E731
        S = lambda t: mp.ncdf(-z(t))  # noqa: E731
        F = lambda t: mp.ncdf(z(t))  # noqa: E731
    elif family == "exponential":
        S = lambda t: mp.exp(-mp.mpf(t) / params["scale"])  # noqa: E731
        F = lambda t: 1 - S(t)  # noqa: E731
    else:
        S = lambda t: mp.exp(-(mp.mpf(t) / params["scale"]) ** params["shape"])  # noqa: E731
        F = lambda t: 1 - S(t)  # noqa: E731
    if censored:
        return float(-mp.log(S(k)))
    # form the interval from the side that keeps its digits, as the scorer does
    p = F(k) - F(k - 1) if F(k) < mp.mpf("0.5") else S(k - 1) - S(k)
    return float(-mp.log(p))


@pytest.mark.parametrize("family,params,points", [
    ("normal", {"mu": 74.3, "sigma": 0.74}, [45.0, 60.0, 74.0, 90.0, 104.0]),
    ("exponential", {"scale": 800.0}, [1.0, 27589.0, 48000.0, 160000.0]),
    ("weibull", {"shape": 4.80, "scale": 149435.0}, [36.0, 139.0, 392.0, 1686.0, 3e5, 6e5]),
    ("weibull", {"shape": 6.22, "scale": 253288.0}, [36.0, 139.0, 392.0, 1686.0, 9e5]),
])
def test_both_tails_are_exact_at_the_shipped_shapes(family, params, points):
    for k in points:
        got = continuous_scores(family, params, k, False)[1]
        ref = _reference_nll(family, params, k)
        assert got == pytest.approx(ref, rel=1e-9, abs=1e-9), (k, got, ref)


def test_censored_far_tail_is_not_capped():
    got = continuous_scores("weibull", {"shape": 5.0, "scale": 1e5}, 4e5, True)[1]
    assert got == pytest.approx(4.0 ** 5, rel=1e-9)


def test_a_zero_duration_is_refused():
    with pytest.raises(ValueError, match="must exceed"):
        continuous_scores("normal", {"mu": 74.0, "sigma": 1.0}, 0.0, False)


def test_an_absurdly_wide_law_is_refused_rather_than_summed():
    with pytest.raises(ValueError, match="broken model"):
        continuous_scores("normal", {"mu": 35.0, "sigma": 1e12}, 35.0, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
