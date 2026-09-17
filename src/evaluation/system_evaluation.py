"""System level: cycle-time distances and the KPI panel.

All inputs must be index-paired under common random numbers (same seed at the
same index).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance, wilcoxon

from src.evaluation.paired_stats import assert_paired_seeds, fdr_correction

REF_SYSTEM = "GroundSim"


def paired_w1(ref_runs: Sequence[dict], sys_runs: Sequence[dict]) -> np.ndarray:
    assert_paired_seeds(ref_runs, sys_runs)
    return np.array([
        wasserstein_distance(np.asarray(r["ct"], float), np.asarray(s["ct"], float))
        for r, s in zip(ref_runs, sys_runs, strict=True)
    ])


def per_seed_distances(ref_runs: Sequence[dict], sys_runs: Sequence[dict]) -> List[dict]:
    """W1 and KS distance per seed pair against the reference.

    The KS p-value is deliberately not reported: cycle times within a run are
    serially correlated (lag-1 around 0.5), so its i.i.d. null makes it
    strongly anti-conservative. Significance comes from the per-seed W1 with
    the paired Wilcoxon test, which works at the resolution the CRN design
    actually provides.
    """
    assert_paired_seeds(ref_runs, sys_runs)
    rows: List[dict] = []
    for r, s in zip(ref_runs, sys_runs, strict=True):
        rc = np.asarray(r["ct"], float)
        sc = np.asarray(s["ct"], float)
        rows.append({
            "seed": int(r["seed"]),
            "w1": float(wasserstein_distance(rc, sc)),
            "ks_d": float(ks_2samp(rc, sc).statistic),
        })
    return rows


def system_distance_table(
    runs_by_sim: Dict[str, List[dict]], *, ref_name: str = REF_SYSTEM
) -> List[dict]:
    """Long-format W1/KS per (system, seed) against ref_name (ref itself omitted)."""
    ref = runs_by_sim[ref_name]
    out: List[dict] = []
    for name, runs in runs_by_sim.items():
        if name == ref_name:
            continue
        for row in per_seed_distances(ref, runs):
            out.append({"system": name, "ref": ref_name, **row})
    return out


def w1_diff_wilcoxon(
    runs_by_sim: Dict[str, List[dict]],
    *,
    a_name: str,
    b_name: str,
    ref_name: str = REF_SYSTEM,
) -> dict:
    """Two-sided Wilcoxon over (W1_a,i − W1_b,i), both W1 against ref."""
    ref = runs_by_sim[ref_name]
    wa = paired_w1(ref, runs_by_sim[a_name])
    wb = paired_w1(ref, runs_by_sim[b_name])
    diff = wa - wb
    if not np.any(diff):
        return {"a": a_name, "b": b_name, "n": int(len(diff)),
                "statistic": float("nan"), "p_exact": 1.0}
    res = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
    return {"a": a_name, "b": b_name, "n": int(len(diff)),
            "statistic": float(res.statistic), "p_exact": float(res.pvalue)}


def _kpi_keys(runs: Sequence[dict], kpis: Optional[Sequence[str]]) -> List[str]:
    if kpis is not None:
        return list(kpis)
    return sorted(k for k, v in runs[0]["kpis"].items() if isinstance(v, (int, float)))


def kpi_panel(
    runs_by_sim: Dict[str, List[dict]],
    *,
    sys_name: str,
    ref_name: str = REF_SYSTEM,
    kpis: Optional[Sequence[str]] = None,
) -> List[dict]:
    """Relative KPI deltas: mean(r_i), sd(r_i), pooled."""
    ref, sys = runs_by_sim[ref_name], runs_by_sim[sys_name]
    assert_paired_seeds(ref, sys)
    rows: List[dict] = []
    for kpi in _kpi_keys(ref, kpis):
        mg = np.array([r["kpis"][kpi] for r in ref], float)
        ms = np.array([r["kpis"][kpi] for r in sys], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_i = np.where(mg != 0.0, (ms - mg) / mg, np.nan)
        mg_mean = float(mg.mean())
        pooled = (float(ms.mean()) - mg_mean) / mg_mean if mg_mean != 0.0 else float("nan")
        rows.append({
            "system": sys_name, "kpi": kpi,
            "mean_r": float(np.nanmean(r_i)),
            "sd_r": float(np.nanstd(r_i, ddof=1)) if np.sum(np.isfinite(r_i)) > 1 else float("nan"),
            "pooled_rel": float(pooled),
        })
    return rows


def kpi_wilcoxon_bh(
    runs_by_sim: Dict[str, List[dict]],
    *,
    sys_name: str,
    ref_name: str = REF_SYSTEM,
    kpis: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> List[dict]:
    """Per-KPI paired Wilcoxon (sys vs ground) + BH across the panel.

    KPIs with all-zero differences get p=1.0 and stay in the BH family.
    """
    ref, sys = runs_by_sim[ref_name], runs_by_sim[sys_name]
    assert_paired_seeds(ref, sys)
    keys = _kpi_keys(ref, kpis)
    stats: List[dict] = []
    for kpi in keys:
        mg = np.array([r["kpis"][kpi] for r in ref], float)
        ms = np.array([r["kpis"][kpi] for r in sys], float)
        diff = ms - mg
        if not np.any(diff):
            stats.append({"kpi": kpi, "statistic": float("nan"), "p_exact": 1.0})
        else:
            # "auto" lets scipy fall back to its deterministic permutation
            # treatment under ties and zeros; "exact" silently is not exact there.
            res = wilcoxon(ms, mg, alternative="two-sided",
                           zero_method="wilcox", method="auto")
            stats.append({"kpi": kpi, "statistic": float(res.statistic),
                          "p_exact": float(res.pvalue)})
    pvals = np.array([s["p_exact"] for s in stats])
    rejected, q = fdr_correction(pvals, alpha=alpha)
    out: List[dict] = []
    for s, rej, qv in zip(stats, rejected, q, strict=True):
        out.append({"system": sys_name, **s, "q_bh": float(qv),
                    "reject_at_alpha": bool(rej)})
    return out
