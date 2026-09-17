"""Aggregate proper scores from the shadow CSVs.

Under censoring the NLL uses the survival term −log S, and CRPS is scored only
on uncensored rows, whose excluded count is reported.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from src.evaluation import proper_scores as ps
from src.evaluation import bathtub_hazard as bathtub

_BATHTUB = bathtub.ground_survival_params()


def _parse(x) -> Optional[dict]:
    if x is None or isinstance(x, float) or (isinstance(x, str) and not x.strip()):
        return None
    try:
        return json.loads(x)
    except (ValueError, TypeError):
        return None


def _family_cdf(family: str, p: dict):
    """CDF of the latent duration law; ceil() of it is what the system deploys."""
    if family == "normal":
        mu, sigma = p["mu"], p["sigma"]
        return lambda x: norm.cdf(x, mu, sigma)
    if family == "exponential":
        b = p["scale"]
        return lambda x: 1.0 - np.exp(-np.maximum(np.asarray(x, float), 0.0) / b)
    if family == "weibull":
        k, lam = p["shape"], p["scale"]
        return lambda x: 1.0 - np.exp(-(np.maximum(np.asarray(x, float), 0.0) / lam) ** k)
    if family == "bathtub":
        scale = p["scale"]
        surv = np.vectorize(
            lambda x: bathtub.bathtub_survival(float(x), scale, _BATHTUB), otypes=[float]
        )
        return lambda x: 1.0 - surv(x)
    raise ValueError(f"Unknown family: {family}")


def _crps_step(family: str, p: dict) -> float:
    """Grid stride for the lattice CRPS support walk, one order below the scale."""
    if family == "normal":
        return max(float(p["sigma"]), 1.0)
    if family == "exponential":
        return max(float(p["scale"]) / 20.0, 1.0)
    return max(float(p["scale"]) / 20.0, 1.0)


def continuous_scores(family: str, p: dict, y: float, censored: bool
                      ) -> Tuple[Optional[float], float]:
    """(CRPS|None if censored, NLL) for a whole-second realization y.

    Every duration a strategy returns is ceil() of its latent draw (integer
    time contract), so the deployed predictive law is the lattice law
    P(Y = k) = F(k) - F(k-1). Both scores are taken on that law: the NLL is its
    log-likelihood, the CRPS is the CRPS definition applied to the step CDF.
    Under censoring the NLL keeps the survival term and CRPS is not scored.
    """
    if censored and family == "normal":
        raise ValueError(
            "Fail fast: censoring is not defined for the normal family "
            "(processing and repair durations always complete)."
        )
    cdf = _family_cdf(family, p)
    if censored:
        return None, ps.survival_nll(cdf, y)
    nll = ps.interval_nll(cdf, y)
    lo, hi = ps._support_bounds(cdf, y, step=_crps_step(family, p))
    return ps.lattice_crps(cdf, y, lo=lo, hi=hi), nll


def _seed_means(by_seed: Dict[int, List[float]]) -> Tuple[Dict[int, float], int]:
    """Per-seed means and the total number of usable observations.

    A seed with no usable value is absent from the mapping rather than NaN.
    """
    means: Dict[int, float] = {}
    n_obs = 0
    for seed, vals in by_seed.items():
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if a.size:
            means[seed] = float(a.mean())
            n_obs += int(a.size)
    return means, n_obs


def _mean_se(seed_means: Dict[int, float]) -> Tuple[float, float, int]:
    """Mean over replications and its seed-clustered standard error.

    The score rows within one replication are not independent: they come from
    one trajectory. Averaging per seed first and taking the spread across the
    k seeds is the resolution the design actually provides; treating the rows
    as independent understates the standard error.
    """
    a = np.asarray(sorted(seed_means.values()), dtype=float)
    k = len(a)
    if k == 0:
        return float("nan"), float("nan"), 0
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    return m, se, k


def _paired_mean_se(
    seed_means: Dict[int, float], ref_means: Dict[int, float]
) -> Tuple[float, float, int]:
    """Mean paired difference against the reference and its standard error.

    Under common random numbers the systems share a seed, so the per-seed
    difference removes the between-seed variation the two share. This is the
    comparison the design was built for, and it is much sharper than the
    unpaired one.
    """
    shared = sorted(set(seed_means) & set(ref_means))
    if not shared:
        return float("nan"), float("nan"), 0
    d = np.asarray([seed_means[s] - ref_means[s] for s in shared], dtype=float)
    k = len(d)
    se = float(d.std(ddof=1) / np.sqrt(k)) if k > 1 else 0.0
    return float(d.mean()), se, k


def _label(row, column: str) -> str:
    """Label cell of a scored row; empty where the column does not apply.

    The CSV round-trip turns an empty field into NaN, which must not become
    the string "nan" (arrival rows have no station, several components have
    no head).
    """
    value = row.get(column, "")
    if value is None or value != value:
        return ""
    return str(value)


def aggregate_continuous(
    df, system: str, component: str, *, reference: Optional[dict] = None,
) -> Tuple[List[dict], dict]:
    """CRPS/NLL per (station[, head]) plus pooled, for one continuous component.

    Returns the rows and a {group: {metric: {seed: mean}}} map. Passing the
    reference system's map back in adds the CRN-paired difference columns.
    """
    by_group: Dict[tuple, dict] = defaultdict(
        lambda: {"crps": defaultdict(list), "nll": defaultdict(list), "n_cens": 0}
    )
    for _, r in df.iterrows():
        p = _parse(r["params"])
        if p is None:
            continue
        try:
            y = float(r["realized"])
        except (ValueError, TypeError):
            continue
        cens = bool(int(r.get("censored", 0)))
        crps, nll = continuous_scores(p["family"], p, y, cens)
        key = (_label(r, "station"), _label(r, "head"))
        seed = int(r["seed"])
        g = by_group[key]
        g["nll"][seed].append(nll)
        if crps is None:
            g["n_cens"] += 1
        else:
            g["crps"][seed].append(crps)
    pooled: dict = {"crps": defaultdict(list), "nll": defaultdict(list)}
    tot_cens = 0
    for g in by_group.values():
        for metric in ("crps", "nll"):
            for seed, vals in g[metric].items():
                pooled[metric][seed].extend(vals)
        tot_cens += g["n_cens"]

    groups = sorted(by_group.items()) + [(("POOLED", ""), {**pooled, "n_cens": tot_cens})]
    seed_map: dict = {}
    rows: List[dict] = []
    for (station, head), g in groups:
        row = {"system": system, "component": component,
               "station": station, "head": head}
        for metric in ("crps", "nll"):
            means, n_obs = _seed_means(g[metric])
            m, se, k = _mean_se(means)
            row[f"{metric}_mean"] = m
            row[f"{metric}_se"] = se
            row[f"n_{metric}"] = n_obs
            row[f"n_seeds_{metric}"] = k
            seed_map[(station, head, metric)] = means
            if reference is not None:
                ref = reference.get((station, head, metric), {})
                dm, dse, dk = _paired_mean_se(means, ref)
                row[f"{metric}_paired_diff"] = dm
                row[f"{metric}_paired_se"] = dse
                row[f"n_seeds_paired_{metric}"] = dk
        row["n_censored_excl_crps"] = g["n_cens"]
        rows.append(row)
    return rows, seed_map


def pit_uniformity(df, *, seed: int = 0) -> dict:
    """KS test of the randomized PIT against uniform, for a continuous component.

    With whole-second realizations the plain PIT u = F(y) takes a handful of
    values and its KS statistic converges to max_k P(Y = k), a property of the
    quantization rather than of the model — it rejects a correct forecast with
    certainty and is blind to sigma. Drawing u uniformly inside the observed
    interval [F(k-1), F(k)] restores exact uniformity under a correct model.
    """
    from scipy.stats import kstest

    rng = np.random.default_rng(seed)
    us: List[float] = []
    for _, r in df.iterrows():
        p = _parse(r["params"])
        if p is None or bool(int(r.get("censored", 0))):
            continue
        try:
            k = float(r["realized"])
        except (ValueError, TypeError):
            continue
        cdf = _family_cdf(p["family"], p)
        lo, hi = float(cdf(max(k - 1.0, 0.0))), float(cdf(k))
        us.append(lo + rng.random() * max(hi - lo, 0.0))
    if len(us) < 2:
        return {"n": len(us), "ks_d": float("nan"), "ks_p": float("nan")}
    res = kstest(us, "uniform")
    return {"n": len(us), "ks_d": float(res.statistic), "ks_p": float(res.pvalue)}


def _row_brier(probs: Dict[str, float], realized: str) -> float:
    """Multiclass Brier for one row: Σ_k (p_k − 1{k=realized})² (+1 if realized ∉ support)."""
    s = sum(v * v for k, v in probs.items() if k != realized)
    if realized in probs:
        s += (probs[realized] - 1.0) ** 2
    else:
        s += 1.0
    return float(s)


def aggregate_categorical(df, system: str, component: str, *, n_bins: int = 15) -> List[dict]:
    """Brier + ECE per (station[, head]) plus pooled, for one categorical component."""
    by_group: Dict[tuple, dict] = defaultdict(lambda: {"brier": [], "conf": [], "correct": []})
    for _, r in df.iterrows():
        p = _parse(r["params"])
        if p is None or "probs" not in p:
            continue
        probs = p["probs"]
        realized = str(r["realized"])
        key = (_label(r, "station"), _label(r, "head"))
        g = by_group[key]
        g["brier"].append(_row_brier(probs, realized))
        if probs:
            top_label, top_p = max(probs.items(), key=lambda kv: kv[1])
            g["conf"].append(float(top_p))
            g["correct"].append(top_label == realized)
    rows: List[dict] = []
    all_brier: List[float] = []
    all_conf: List[float] = []
    all_corr: List[bool] = []
    for (station, head), g in sorted(by_group.items()):
        bm, bse, bn = _mean_se(g["brier"])
        ece = ps.compute_ece(g["conf"], g["correct"], n_bins=n_bins)
        rows.append({"system": system, "component": component, "station": station,
                     "head": head, "brier_mean": bm, "brier_se": bse, "n": bn,
                     "ece": ece["ece"], "top_bin_gap": ece["top_bin_gap"]})
        all_brier += g["brier"]
        all_conf += g["conf"]
        all_corr += g["correct"]
    bm, bse, bn = _mean_se(all_brier)
    ece = ps.compute_ece(all_conf, all_corr, n_bins=n_bins)
    rows.append({"system": system, "component": component, "station": "POOLED",
                 "head": "", "brier_mean": bm, "brier_se": bse, "n": bn,
                 "ece": ece["ece"], "top_bin_gap": ece["top_bin_gap"]})
    return rows


CONTINUOUS = ("processing", "survival", "repair")
CATEGORICAL = ("transition", "arrival")


def aggregate_all(
    shadow_dir: Path, systems, components, *, reference: str = "GroundSim",
) -> Dict[str, List[dict]]:
    """Aggregate all existing <system>__<component>.csv files.

    The reference system is scored first so every other system can carry the
    CRN-paired difference against it.
    """
    import warnings

    import pandas as pd
    from scipy.integrate import IntegrationWarning
    cont: List[dict] = []
    cat: List[dict] = []
    with warnings.catch_warnings():
        # Fail loud: a non-converging quadrature would silently corrupt a score.
        warnings.simplefilter("error", IntegrationWarning)
        ordered = ([reference] + [s for s in systems if s != reference]
                   if reference in systems else list(systems))
        ref_maps: Dict[str, dict] = {}
        for system in ordered:
            for comp in components:
                path = shadow_dir / f"{system}__{comp}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path, dtype={"context_id": str, "realized": str})
                if comp in CONTINUOUS:
                    rows, seed_map = aggregate_continuous(
                        df, system, comp,
                        reference=None if system == reference else ref_maps.get(comp),
                    )
                    cont += rows
                    if system == reference:
                        ref_maps[comp] = seed_map
                elif comp in CATEGORICAL:
                    cat += aggregate_categorical(df, system, comp)
    return {"continuous": cont, "categorical": cat}
