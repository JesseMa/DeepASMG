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


def continuous_scores(family: str, p: dict, y: float, censored: bool
                      ) -> Tuple[Optional[float], float]:
    """(CRPS|None if censored, NLL). NLL uses the survival term under censoring."""
    if family == "normal":
        return ps.crps_normal(p["mu"], p["sigma"], y), ps.nll_normal(p["mu"], p["sigma"], y)
    if family == "exponential":
        b = p["scale"]
        nll = (y / b) if censored else ps.nll_exponential(b, y)  # −log S = y/β
        crps = None if censored else ps.crps_exponential(b, y)
        return crps, float(nll)
    if family == "weibull":
        k, lam = p["shape"], p["scale"]
        nll = ((max(y, 0.0) / lam) ** k) if censored else ps.nll_weibull(k, lam, y)  # −log S
        crps = None if censored else ps.crps_weibull(k, lam, y)
        return crps, float(nll)
    if family == "bathtub":
        nll = bathtub.bathtub_nll(y, p["scale"], censored, _BATHTUB)
        crps = None if censored else bathtub.bathtub_crps(y, p["scale"], _BATHTUB)
        return crps, float(nll)
    raise ValueError(f"Unknown family: {family}")


def _mean_se(vals: List[float]) -> Tuple[float, float, int]:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, se, n


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


def aggregate_continuous(df, system: str, component: str) -> List[dict]:
    """CRPS/NLL per (station[, head]) plus pooled, for one continuous component."""
    by_group: Dict[tuple, dict] = defaultdict(lambda: {"crps": [], "nll": [], "n_cens": 0})
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
        g = by_group[key]
        g["nll"].append(nll)
        if crps is None:
            g["n_cens"] += 1
        else:
            g["crps"].append(crps)
    rows: List[dict] = []
    all_crps: List[float] = []
    all_nll: List[float] = []
    tot_cens = 0
    for (station, head), g in sorted(by_group.items()):
        cm, cse, cn = _mean_se(g["crps"])
        nm, nse, nn = _mean_se(g["nll"])
        rows.append({"system": system, "component": component, "station": station,
                     "head": head, "crps_mean": cm, "crps_se": cse, "n_crps": cn,
                     "nll_mean": nm, "nll_se": nse, "n_nll": nn,
                     "n_censored_excl_crps": g["n_cens"]})
        all_crps += g["crps"]
        all_nll += g["nll"]
        tot_cens += g["n_cens"]
    cm, cse, cn = _mean_se(all_crps)
    nm, nse, nn = _mean_se(all_nll)
    rows.append({"system": system, "component": component, "station": "POOLED",
                 "head": "", "crps_mean": cm, "crps_se": cse, "n_crps": cn,
                 "nll_mean": nm, "nll_se": nse, "n_nll": nn,
                 "n_censored_excl_crps": tot_cens})
    return rows


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


def aggregate_all(shadow_dir: Path, systems, components) -> Dict[str, List[dict]]:
    """Aggregate all existing <system>__<component>.csv files."""
    import warnings

    import pandas as pd
    from scipy.integrate import IntegrationWarning
    cont: List[dict] = []
    cat: List[dict] = []
    with warnings.catch_warnings():
        # Fail loud: a non-converging quadrature would silently corrupt a score.
        warnings.simplefilter("error", IntegrationWarning)
        for system in systems:
            for comp in components:
                path = shadow_dir / f"{system}__{comp}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path, dtype={"context_id": str, "realized": str})
                if comp in CONTINUOUS:
                    cont += aggregate_continuous(df, system, comp)
                elif comp in CATEGORICAL:
                    cat += aggregate_categorical(df, system, comp)
    return {"continuous": cont, "categorical": cat}
