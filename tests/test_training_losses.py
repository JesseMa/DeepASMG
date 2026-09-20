"""The interval likelihoods the surrogates are trained with."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

torch = pytest.importorskip("torch")

from src.fitting.deep_training.foundation_training import (  # noqa: E402
    GaussianNLLModule, build_mlp_layers, init_output_at_marginal,
)
from src.fitting.deep_training.train_survival import weibull_nll_loss  # noqa: E402


def _gaussian_nll(pred, y):
    class Stub:
        _log_interval_prob = GaussianNLLModule._log_interval_prob
        _nll_loss = GaussianNLLModule._nll_loss
    return Stub()._nll_loss(pred, y)


def _reference_log_interval(mu, sigma, k):
    a, b = (k - 1 - mu) / sigma, (k - mu) / sigma
    if b <= 0:
        return norm.logcdf(b) + np.log1p(-np.exp(norm.logcdf(a) - norm.logcdf(b)))
    if a >= 0:
        return norm.logsf(a) + np.log1p(-np.exp(norm.logsf(b) - norm.logsf(a)))
    return np.log(norm.cdf(b) - norm.cdf(a))


@pytest.mark.parametrize("mu,sigma", [(74.3, 0.74), (36.0, 2.5), (134.0, 8.8), (74.3, 20.0)])
def test_gaussian_interval_loss_matches_scipy_into_both_tails(mu, sigma):
    for z in (-30.0, -8.0, -2.0, 0.0, 0.4, 2.0, 8.0, 30.0):
        k = float(np.round(mu + z * sigma))
        pred = torch.tensor([[mu, 2.0 * np.log(sigma)]], dtype=torch.float32)
        got = _gaussian_nll(pred, torch.tensor([k])).item()
        ref = -_reference_log_interval(mu, sigma, k)
        assert got == pytest.approx(ref, rel=1e-4, abs=1e-4), (z, got, ref)


def test_gaussian_interval_loss_has_gradient_far_from_the_mean():
    for y in (20.0, 70.0, 134.0, 1000.0):
        pred = torch.tensor([[0.7, 0.0]], requires_grad=True)
        _gaussian_nll(pred, torch.tensor([y])).backward()
        d_mu, d_logvar = pred.grad[0].tolist()
        assert d_mu < 0.0, f"y={y}: the mean is not pushed up"
        assert d_logvar < 0.0, f"y={y}: the variance is not pushed up"
        assert np.isfinite(d_mu) and np.isfinite(d_logvar)


def test_gaussian_interval_loss_is_finite_on_every_input():
    zl = torch.tensor([-0.5, 0.0, -1e-3, 5.0, -50.0, 3000.0], requires_grad=True)
    zu = zl + torch.tensor([1.0, 1.0, 2e-3, 20.0, 1.0, 1.0])
    lp = GaussianNLLModule._log_interval_prob(zl, zu)
    lp.sum().backward()
    assert torch.isfinite(lp).all() and torch.isfinite(zl.grad).all()


def test_process_time_module_learns_mixed_scale_targets_from_default_init():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    latent = {0: (36.0, 2.0), 1: (74.0, 1.0), 2: (134.0, 3.0)}
    n = 600
    station = rng.integers(0, 3, size=n)
    x = np.eye(3, dtype=np.float32)[station]
    y = np.ceil(np.array([rng.normal(*latent[s]) for s in station])).astype(np.float32)

    net = build_mlp_layers(3, [32], 2, 0.0, output_activation="softplus_first")
    init_output_at_marginal(net, GaussianNLLModule.marginal_bias(y))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(300):
        opt.zero_grad()
        _gaussian_nll(net(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        out = net(torch.eye(3))
    for s, (mu, sigma) in latent.items():
        assert out[s, 0].item() == pytest.approx(mu, abs=0.5), f"station {s} mean"
        assert np.exp(0.5 * out[s, 1].item()) == pytest.approx(sigma, rel=0.2), f"station {s} sigma"


def test_exponential_interval_loss_matches_closed_form():
    from src.fitting.deep_training.foundation_training import ExponentialNLLModule

    class Stub(ExponentialNLLModule):
        def __init__(self):
            pass
        def __call__(self, x):
            return x
        def log(self, *a, **k):
            pass
    stub = Stub()
    for s in (800.0, 2105.0):
        for k in (1.0, 100.0, s, 5 * s, 40 * s):
            pred = torch.tensor([[np.log(s)]], dtype=torch.float32)
            got = stub._compute_loss((pred, torch.tensor([k])), "val").item()
            ref = (k - 1) / s - np.log(1 - np.exp(-1 / s))
            assert got == pytest.approx(ref, rel=1e-4), (s, k)


def test_weibull_interval_loss_matches_float64_reference_and_keeps_gradient():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    D, shape, lam_s = 150000.0, 5.5, 1.5e5
    log_shape = torch.tensor([np.log(shape)], dtype=torch.float32)
    log_scale = torch.tensor([np.log(lam_s / D)], dtype=torch.float32)
    for k in (36.0, 500.0, 5e4, 1.5e5, 3e5):
        got = weibull_nll_loss(log_shape, log_scale, torch.tensor([k / D]),
                               torch.tensor([1.0]), 1.0 / D).item()
        S = lambda t: mp.exp(-(mp.mpf(t) / lam_s) ** shape)  # noqa: E731
        ref = float(-mp.log(S(k - 1) - S(k)))
        assert got == pytest.approx(ref, rel=2e-3), k
    censored = weibull_nll_loss(log_shape, log_scale, torch.tensor([3e5 / D]),
                                torch.tensor([0.0]), 1.0 / D).item()
    assert censored == pytest.approx((3e5 / lam_s) ** shape, rel=1e-4)

    ls = torch.zeros(3, requires_grad=True)
    lc = torch.zeros(3, requires_grad=True)
    weibull_nll_loss(ls, lc, torch.tensor([36 / D, 1.0, 40.0]),
                     torch.tensor([1.0, 1.0, 0.0]), 1 / D).backward()
    assert torch.isfinite(ls.grad).all() and torch.isfinite(lc.grad).all()
    assert ls.grad[0].item() != 0.0, "an early failure must move the shape"


def test_weibull_loss_stays_finite_with_gradient_when_the_law_calls_the_row_impossible():
    # shape 194 with t/lambda = 1.6: (t/lambda)^k is 1e40, past float32. The old
    # form returned nan and -inf gradients, which poisoned every weight.
    ls = torch.tensor([np.log(194.0)], requires_grad=True)
    lc = torch.tensor([0.0], requires_grad=True)
    loss = weibull_nll_loss(ls, lc, torch.tensor([1.6]), torch.tensor([1.0]), 1e-5)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 1e6
    assert torch.isfinite(ls.grad).all() and torch.isfinite(lc.grad).all()
    assert ls.grad.item() > 0.0, "the impossible row must push the shape down"


def test_survival_repair_feature_is_log_compressed_on_both_sides():
    # An exponential repair tail puts a 20x outlier into the feature; log1p keeps
    # it within a few units, and inference must encode it the same way.
    from src.dynamics.DeepSim.deep_survival import DeepSurvival
    sv = DeepSurvival.__new__(DeepSurvival)
    sv._surv_station_map = {"M1": 0}
    sv._surv_feature_dim = 5
    sv._surv_duration_scale = sv._surv_n_jobs_scale = 1.0
    sv._surv_repair_time_scale = 1884.8
    sv._prev_ttf, sv._prev_n_jobs, sv._mean_ttf = {}, {}, {}
    sv._prev_repair_time = {"M1": 37972.0}
    x = sv._encode_input("M1")
    assert x[4] == pytest.approx(np.log1p(37972.0 / 1884.8), rel=1e-6)
    assert x[4] < 3.1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_warmup_keeps_a_wide_head_finite_at_the_top_of_the_search_range():
    # The failing HPO corner: wide network, lr 7e-3. Without the warm-up Adam's
    # first step collapses sigma and the gradients are NaN within ten steps.
    pl = pytest.importorskip("pytorch_lightning")
    from src.fitting.deep_training.foundation_training import WARMUP_STEPS
    from src.fitting.deep_training.train_process_time import ProcessTimeLightningModule
    from torch.utils.data import DataLoader, TensorDataset
    rng = np.random.default_rng(1)
    n = 4096
    station = rng.integers(0, 3, size=n)
    x = np.eye(3, dtype=np.float32)[station]
    mu = np.array([36.0, 74.0, 134.0])[station]
    y = np.ceil(rng.normal(mu, 2.0)).astype(np.float32)
    pl.seed_everything(0, workers=True)
    mod = ProcessTimeLightningModule(input_dim=3, hidden_dims=[2048, 1024, 512],
                                     learning_rate=7.3e-3, dropout_rate=0.135)
    init_output_at_marginal(mod.net, mod.marginal_bias(y))
    data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train = DataLoader(data, batch_size=256, shuffle=True)
    val = DataLoader(data, batch_size=1024)
    trainer = pl.Trainer(max_steps=WARMUP_STEPS + 50, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         accelerator="cpu", deterministic=True)
    trainer.fit(mod, train, val)
    with torch.no_grad():
        out = mod(torch.eye(3))
    assert torch.isfinite(out).all()
    sigma = torch.exp(0.5 * out[:, 1])
    assert (sigma > 0.5).all() and (sigma < 50.0).all(), f"sigma collapsed or exploded: {sigma.tolist()}"
    assert (out[:, 0] > 20.0).all() and (out[:, 0] < 160.0).all(), f"means left the data range: {out[:, 0].tolist()}"


def test_exported_archive_carries_no_debug_source_ranges(tmp_path):
    import zipfile
    import torch
    from src.fitting.deep_training.foundation_training import strip_debug_info

    net = torch.nn.Sequential(torch.nn.Linear(3, 2))
    path = tmp_path / "m.pt"
    torch.jit.trace(net, torch.zeros(1, 3)).save(str(path))
    assert any(n.endswith(".debug_pkl") for n in zipfile.ZipFile(path).namelist())
    strip_debug_info(path)
    names = zipfile.ZipFile(path).namelist()
    assert not any(n.endswith(".debug_pkl") for n in names)
    x = torch.randn(4, 3)
    assert torch.equal(torch.jit.load(str(path))(x), net(x))
