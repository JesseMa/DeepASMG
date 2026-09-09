"""Re-score the processing component under TOTAL semantics (consumer-side only).

Processing scoring target = log-native total duration (TOTAL):
  realized_total = logged realized + setup[station]   (setup from the config)
  GroundSim:      mu_total = mu + setup[station], sigma unchanged
  DeepSim/RefSim: parameters taken over unchanged (net)
Shadow driver and bundle CSVs are untouched; output goes to its own directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.evaluation.proper_scores import crps_normal_vec, nll_normal_vec  # noqa: E402
from src.experiments.sim_runner import get_process_config  # noqa: E402
from src.experiments.shadow_evaluation import SYSTEMS  # noqa: E402


def _load(system, bundle):
    df = pd.read_csv(bundle / f"{system}__processing.csv", dtype={"realized": str})
    ex = df["params"].str.extract(r'"mu": ([^,]+), "sigma": ([^}]+)}')
    df["mu"] = pd.to_numeric(ex[0], errors="coerce")
    df["sigma"] = pd.to_numeric(ex[1], errors="coerce")
    df["y"] = pd.to_numeric(df["realized"], errors="coerce")
    kept = df.dropna(subset=["mu", "sigma", "y"])
    n_dropped = len(df) - len(kept)
    if n_dropped:
        print(f"  {system}: {n_dropped}/{len(df)} rows unusable, dropped")
    return kept


def _agg(system, station, g):
    c, n_ = g["crps"].to_numpy(), len(g)
    nl = g["nll"].to_numpy()
    return {"system": system, "component": "processing", "station": station,
            "crps_mean": float(c.mean()), "crps_se": float(c.std(ddof=1) / np.sqrt(n_)) if n_ > 1 else 0.0,
            "nll_mean": float(nl.mean()), "nll_se": float(nl.std(ddof=1) / np.sqrt(n_)) if n_ > 1 else 0.0,
            "n": int(n_)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shadow-dir",
        type=Path,
        default=REPO / "results/verification/shadow",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/verification/processing_rescored",
    )
    args = ap.parse_args()

    if not args.shadow_dir.is_dir():
        raise SystemExit(
            f"Shadow bundle not found: {args.shadow_dir}\n"
            "Run 'python -m scripts.run_shadow_evaluation' first, "
            "or point --shadow-dir at an existing bundle."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pc = get_process_config()
    setup = {s.id: float(s.setup_time) for s in pc.stations}

    rows, pit_lines, resid_lines = [], [], []
    for system in SYSTEMS:
        df = _load(system, args.shadow_dir)
        su = df["station"].map(setup).fillna(0.0).to_numpy()
        y_total = df["y"].to_numpy() + su
        mu = df["mu"].to_numpy() + (su if system == "GroundSim" else 0.0)
        sigma = df["sigma"].to_numpy()
        crps = crps_normal_vec(mu, sigma, y_total)
        nll = nll_normal_vec(mu, sigma, y_total)
        tmp = pd.DataFrame({"station": df["station"].to_numpy(), "crps": crps, "nll": nll})
        for station, g in tmp.groupby("station"):
            rows.append(_agg(system, station, g))
        rows.append(_agg(system, "POOLED", tmp))
        r = y_total - mu
        q = np.percentile(r, [5, 50, 95])
        resid_lines.append({"system": system, "p5": q[0], "p50": q[1], "p95": q[2]})
        if system == "GroundSim":
            u = norm.cdf(y_total, mu, sigma)
            u = np.clip(u, 0, 1)
            ksd, ksp = kstest(u, "uniform")
            pit_lines.append({"scope": "GroundSim processing TOTAL", "n": len(u),
                              "ks": float(ksd), "p": float(ksp)})

    pd.DataFrame(rows).to_csv(args.output_dir / "scores_continuous_processing.csv", index=False)
    pd.DataFrame([{"station": k, "setup_s": v} for k, v in sorted(setup.items())]
                 ).to_csv(args.output_dir / "processing_setup_values.csv", index=False)
    print("=== Re-Scoring Processing (TOTAL) — POOLED per system ===")
    for r in rows:
        if r["station"] == "POOLED":
            print(f"  {r['system']:<11} CRPS={r['crps_mean']:.4f}±{r['crps_se']:.4f} "
                  f"NLL={r['nll_mean']:.4f} n={r['n']}")
    print("\n(a) PIT GroundSim processing (TOTAL):")
    for p in pit_lines:
        print(f"  KS={p['ks']:.4f} p={p['p']:.3g} n={p['n']}")
    print("\n(b) p5/p50/p95 (realized_total − mu) per system:")
    for r in resid_lines:
        print(f"  {r['system']:<11} p5={r['p5']:.3f} p50={r['p50']:.3f} p95={r['p95']:.3f}")
    print("\n(c) setup[station] values:")
    for k, v in sorted(setup.items()):
        if v > 0:
            print(f"  {k}: {v:.0f}s")
    print(f"\nOutput → {args.output_dir}")


if __name__ == "__main__":
    main()
