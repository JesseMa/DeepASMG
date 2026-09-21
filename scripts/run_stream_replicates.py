"""Closed-loop distance of DeepSim and GroundSim-DEC under five independent stream assignments."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config.simulation_config import N_RUNS, SEED_OFFSET_FLOOR, get_seeds  # noqa: E402
from src.evaluation import system_evaluation as system  # noqa: E402
from src.experiments.sim_runner import SimFactorySet, run_replications  # noqa: E402

SYSTEMS = {"GroundSim-DEC": ("ground",) * 5, "DeepSim": ("deep",) * 5}
N_SETS = 5


def compose_seed(seed: int, stream_set: int) -> int:
    # set 1 is the release assignment (offset streams of `seed`); every further set
    # shifts the seed by whole multiples of the floor offset, so its streams are
    # spawned from a seed no other run uses
    return seed + SEED_OFFSET_FLOOR * (stream_set - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--closed-loop-file", type=Path,
                    default=REPO / "results/verification/closed_loop_runs.pkl")
    ap.add_argument("--output-dir", type=Path, default=REPO / "results/verification")
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    args = ap.parse_args()

    with open(args.closed_loop_file, "rb") as file:
        bundle = pickle.load(file)
    target = bundle["runs_by_sim"]["GroundSim"]
    seeds = get_seeds(args.num_seeds)
    fs = SimFactorySet()

    rows: list[dict] = []
    per_set: dict[tuple[str, int], np.ndarray] = {}
    for stream_set in range(1, N_SETS + 1):
        for name, kinds in SYSTEMS.items():
            runs = run_replications(
                f"{name} streams={stream_set}",
                lambda seed, run_id, k=stream_set, kinds=kinds: fs.compose(compose_seed(seed, k), run_id, kinds),
                seeds,
            )
            for run, seed in zip(runs, seeds, strict=True):
                run["seed"] = seed
            if stream_set == 1:
                release = bundle["runs_by_sim"][name]
                for run, ref in zip(runs, release, strict=True):
                    if not np.array_equal(run["ct"], ref["ct"]):
                        raise SystemExit(f"{name}: stream set 1 does not reproduce the release run of seed {run['seed']}")
            w1 = system.paired_w1(target, runs)
            per_set[(name, stream_set)] = w1
            for seed, value in zip(seeds, w1, strict=True):
                rows.append({"stream_set": stream_set, "system": name, "seed": seed, "w1": float(value)})
            print(f"  {name:14} streams={stream_set}: mean W1 {w1.mean():.2f} s", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = args.output_dir / "stream_replicates.csv"
    with open(raw, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["stream_set", "system", "seed", "w1"])
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict] = []
    for name in SYSTEMS:
        means = np.array([per_set[(name, k)].mean() for k in range(1, N_SETS + 1)])
        summary.append({"quantity": f"{name} ten-seed mean", "n": N_SETS, "mean": float(means.mean()),
                        "sd": float(means.std(ddof=1)), "min": float(means.min()), "max": float(means.max()),
                        "se": float(means.std(ddof=1) / np.sqrt(N_SETS)), "p_value": ""})
    diff = np.concatenate([per_set[("DeepSim", k)] - per_set[("GroundSim-DEC", k)] for k in range(1, N_SETS + 1)])
    summary.append({"quantity": "DeepSim minus GroundSim-DEC, paired by seed and stream set", "n": int(diff.size),
                    "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)), "min": float(diff.min()),
                    "max": float(diff.max()), "se": float(diff.std(ddof=1) / np.sqrt(diff.size)),
                    "p_value": float(stats.wilcoxon(diff, zero_method="wilcox", method="auto").pvalue)})
    out = args.output_dir / "stream_replicates_summary.csv"
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nDONE -> {raw} ({len(rows)} rows), {out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
