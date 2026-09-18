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



def _parse(x) -> Optional[dict]:
    if x is None or isinstance(x, float) or (isinstance(x, str) and not x.strip()):
        return None
    try:
        return json.loads(x)
    except (ValueError, TypeError):
        return None


def _family_log_cdf_sf(family: str, p: dict):
    """(log_cdf, log_sf) of the family's continuous law, each exact in its tail.

    The families with a closed cumulative hazard H build log_sf = -H and
    log_cdf = log(-expm1(-H)); the normal uses scipy's pair. Working in log
    space is what lets the lattice scores stay exact however far out a
    realization falls.
    """
    if family == "normal":
        mu, sigma = p["mu"], p["sigma"]
        return (lambda x: norm.logcdf(x, mu, sigma), lambda x: norm.logsf(x, mu, sigma))
    if family == "exponential":
        def H(x):
            return np.maximum(np.asarray(x, float), 0.0) / p["scale"]
    elif family == "weibull":
        def H(x):
            return (np.maximum(np.asarray(x, float), 0.0) / p["scale"]) ** p["shape"]
    elif family == "bathtub":
        scale = p["scale"]
        H = np.vectorize(
            lambda x: bathtub.cumulative_hazard_w(max(float(x), 0.0) / scale, p),
            otypes=[float],
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    def log_cdf(x):
        with np.errstate(divide="ignore"):   # H(0) = 0: log F(0) = -inf is the value
            return np.log(-np.expm1(-H(x)))
    return (log_cdf, lambda x: -H(x))


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
    log_cdf, log_sf = _family_log_cdf_sf(family, p)
    if censored:
        return None, ps.survival_nll(log_sf, y)
    nll = ps.interval_nll(log_cdf, log_sf, y)

    def cdf(x):
        return np.exp(log_cdf(x))
    lo, hi = ps._support_bounds(cdf, y, step=_crps_step(family, p))
    return ps.lattice_crps(cdf, y, lo=lo, hi=hi), nll


def _seed_means(by_seed: Dict[int, List[float]]) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Per-seed means and per-seed counts of the usable observations.

    A seed with no usable value is absent from both mappings rather than NaN.
    """
    means: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for seed, vals in by_seed.items():
        a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if a.size:
            means[seed] = float(a.mean())
            counts[seed] = int(a.size)
    return means, counts


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
    seed_means: Dict[int, float], counts: Dict[int, int],
    ref_means: Dict[int, float], ref_counts: Dict[int, int],
) -> Tuple[float, float, int]:
    """Mean paired difference against the reference and its standard error.

    Under common random numbers the systems share a seed, so the per-seed
    difference removes the between-seed variation the two share. This is the
    comparison the design was built for, and it is much sharper than the
    unpaired one. Pairing is only meaningful over the same decisions, so a
    seed on which the two systems scored different numbers of rows is refused.
    """
    shared = sorted(set(seed_means) & set(ref_means))
    if not shared:
        return float("nan"), float("nan"), 0
    unequal = [s for s in shared if counts[s] != ref_counts[s]]
    if unequal:
        raise ValueError(
            f"Fail fast: paired difference over unequal row sets on seeds {unequal} "
            f"({[(counts[s], ref_counts[s]) for s in unequal]})."
        )
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

    Returns the rows and a {(group, metric): ({seed: mean}, {seed: n})} map.
    Passing the reference system's map back in adds the CRN-paired difference
    columns.
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
            means, counts = _seed_means(g[metric])
            m, se, k = _mean_se(means)
            row[f"{metric}_mean"] = m
            row[f"{metric}_se"] = se
            row[f"n_{metric}"] = sum(counts.values())
            row[f"n_seeds_{metric}"] = k
            seed_map[(station, head, metric)] = (means, counts)
            if reference is not None:
                ref_means, ref_counts = reference.get((station, head, metric), ({}, {}))
                dm, dse, dk = _paired_mean_se(means, counts, ref_means, ref_counts)
                row[f"{metric}_paired_diff"] = dm
                row[f"{metric}_paired_se"] = dse
                row[f"n_seeds_paired_{metric}"] = dk
        row["n_censored_excl_crps"] = g["n_cens"]
        rows.append(row)
    return rows, seed_map


def _row_brier(probs: Dict[str, float], realized: str) -> float:
    """Multiclass Brier for one row: Σ_k (p_k − 1{k=realized})² (+1 if realized ∉ support)."""
    s = sum(v * v for k, v in probs.items() if k != realized)
    if realized in probs:
        s += (probs[realized] - 1.0) ** 2
    else:
        s += 1.0
    return float(s)


def aggregate_categorical(df, system: str, component: str, *, n_bins: int = 15) -> List[dict]:
    """Brier + ECE per (station[, head]) plus pooled, for one categorical component.

    The Brier score is averaged per seed and reported with the seed-clustered
    standard error like the continuous scores; the ECE is a property of the
    whole set of decisions and is pooled over all rows of a group.
    """
    by_group: Dict[tuple, dict] = defaultdict(
        lambda: {"brier": defaultdict(list), "conf": [], "correct": []}
    )
    for _, r in df.iterrows():
        p = _parse(r["params"])
        if p is None or "probs" not in p:
            continue
        probs = p["probs"]
        realized = str(r["realized"])
        key = (_label(r, "station"), _label(r, "head"))
        g = by_group[key]
        g["brier"][int(r["seed"])].append(_row_brier(probs, realized))
        if probs:
            top_label, top_p = max(probs.items(), key=lambda kv: kv[1])
            g["conf"].append(float(top_p))
            g["correct"].append(top_label == realized)
    pooled: dict = {"brier": defaultdict(list), "conf": [], "correct": []}
    for g in by_group.values():
        for seed, vals in g["brier"].items():
            pooled["brier"][seed].extend(vals)
        pooled["conf"] += g["conf"]
        pooled["correct"] += g["correct"]

    rows: List[dict] = []
    for (station, head), g in sorted(by_group.items()) + [(("POOLED", ""), pooled)]:
        means, counts = _seed_means(g["brier"])
        bm, bse, k = _mean_se(means)
        ece = ps.compute_ece(g["conf"], g["correct"], n_bins=n_bins)
        rows.append({"system": system, "component": component, "station": station,
                     "head": head, "brier_mean": bm, "brier_se": bse, "n": sum(counts.values()),
                     "n_seeds": k, "ece": ece["ece"], "top_bin_gap": ece["top_bin_gap"]})
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
    import pandas as pd
    cont: List[dict] = []
    cat: List[dict] = []
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
