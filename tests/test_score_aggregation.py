"""How component scores are pooled and compared across systems."""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import routing_evaluation as routing
from src.evaluation.score_aggregation import (
    _mean_se, _paired_mean_se, aggregate_categorical, aggregate_continuous,
)
from src.experiments.sim_runner import get_process_config


def _continuous_frame(seed_means, *, n_per_seed=200, noise=0.01, station="M1"):
    rng = np.random.default_rng(0)
    rows = []
    for seed, m in seed_means.items():
        for _ in range(n_per_seed):
            y = float(np.ceil(rng.normal(m, 1.0)))
            rows.append({"seed": seed, "station": station, "head": "",
                         "params": json.dumps({"family": "normal", "mu": m + rng.normal(0, noise), "sigma": 1.0}),
                         "realized": y, "censored": 0})
    return pd.DataFrame(rows)


def test_standard_error_is_clustered_by_seed():
    df = _continuous_frame({1000: 70.0, 1001: 71.0})
    rows, _ = aggregate_continuous(df, "S", "processing")
    pooled = next(r for r in rows if r["station"] == "POOLED")
    assert pooled["n_seeds_nll"] == 2 and pooled["n_nll"] == 400
    m, se, k = _mean_se({1000: 1.0, 1001: 2.0})
    assert (m, k) == (1.5, 2) and se == pytest.approx(0.5)
    assert _mean_se({})[2] == 0


def test_paired_difference_uses_matched_seeds_and_refuses_unequal_rows():
    d, se, k = _paired_mean_se({1: 3.0, 2: 5.0, 3: 9.0}, {1: 1, 2: 1, 3: 1},
                               {1: 1.0, 2: 2.0, 3: 4.0, 4: 100.0}, {1: 1, 2: 1, 3: 1, 4: 1})
    assert (d, k) == pytest.approx((10.0 / 3.0, 3)) and se > 0
    with pytest.raises(ValueError, match="unequal row sets"):
        _paired_mean_se({1: 3.0}, {1: 10}, {1: 1.0}, {1: 9})


def test_reference_scored_first_yields_paired_columns():
    ref = _continuous_frame({1000: 70.0, 1001: 71.0})
    cand = _continuous_frame({1000: 70.0, 1001: 71.0}, noise=0.5)
    _, ref_map = aggregate_continuous(ref, "GroundSim", "processing")
    rows, _ = aggregate_continuous(cand, "DeepSim", "processing", reference=ref_map)
    pooled = next(r for r in rows if r["station"] == "POOLED")
    assert pooled["n_seeds_paired_nll"] == 2
    assert pooled["nll_paired_diff"] > 0.0   # the noisier forecast loses


def test_categorical_aggregation_is_seed_clustered():
    rows = []
    for seed in (1000, 1001, 1002):
        for i in range(20):
            rows.append({"seed": seed, "station": "M5", "head": "",
                         "params": json.dumps({"probs": {"A": 0.7, "B": 0.3}}),
                         "realized": "A" if (i + seed) % 3 else "B"})
    out = aggregate_categorical(pd.DataFrame(rows), "DeepSim", "transition")
    pooled = next(r for r in out if r["station"] == "POOLED")
    assert pooled["n"] == 60 and pooled["n_seeds"] == 3
    assert 0.0 < pooled["brier_mean"] < 1.0 and np.isfinite(pooled["ece"])


def test_routing_comparison_scores_each_visit_against_its_own_truth(tmp_path):
    pc = get_process_config()
    rows, side, i = [], [], 0
    for variant, feats in routing.all_variants():
        for visit in (1, 2, 3):
            truth = routing.config_true_vector(pc, "M5", feats, visit=visit)
            ctx = f"1000:transition:{i}"
            i += 1
            side.append((ctx, variant, visit))
            rows.append({"context_id": ctx, "station": "M5", "params": json.dumps({"probs": truth})})
    with (tmp_path / "DeepSim__transition.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["context_id", "station", "params"])
        w.writeheader()
        w.writerows(rows)
    with (tmp_path / "_context_variant.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["context_id", "variant", "visit"])
        w.writerows(side)
    vectors = routing.deep_shadow_vectors(tmp_path, "DeepSim")
    scored = routing.compare_rows(pc, vectors, "DeepSim")
    assert len(vectors) == 72 and len(scored) == 48
    assert sorted({r["visit"] for r in scored}) == [1, 2]   # visit 3 is a point mass
    assert max(r["l1"] for r in scored) == 0.0


def test_reference_vectors_cover_exactly_the_realized_keys():
    from src.experiments.sim_runner import SimFactorySet
    pc = get_process_config()
    fs = SimFactorySet()
    keys = {("M5", "C_a.2_b.4", 1), ("M5", "C_a.2_b.4", 2)}
    fitted = routing.ref_fitted_vectors(fs.module("stat", "tr", np.random.default_rng(0)), pc, keys)
    assert set(fitted) == keys
    assert fitted[("M5", "C_a.2_b.4", 1)] == fitted[("M5", "C_a.2_b.4", 2)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
