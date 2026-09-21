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
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild the summary from an existing stream_replicates.csv without simulating")
    args = ap.parse_args()
    if args.summarize_only:
        write_summary(args.output_dir)
        return

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
    out = write_summary(args.output_dir)
    print(f"\nDONE -> {raw} ({len(rows)} rows), {out}")


def write_summary(output_dir: Path) -> Path:
    # the target realization of a seed is shared by all stream sets, so the
    # paired differences are aggregated per stream set and per seed before any
    # standard error or test is computed
    raw = output_dir / "stream_replicates.csv"
    with open(raw, newline="") as file:
        rows = list(csv.DictReader(file))
    sets = sorted({int(r["stream_set"]) for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    w1 = {(r["system"], int(r["stream_set"]), int(r["seed"])): float(r["w1"]) for r in rows}
    diff = np.array([[w1[("DeepSim", k, s)] - w1[("GroundSim-DEC", k, s)] for s in seeds] for k in sets])

    def row(quantity, values, p_value=""):
        values = np.asarray(values, dtype=float)
        return {"quantity": quantity, "n": int(values.size), "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()),
                "se": float(values.std(ddof=1) / np.sqrt(values.size)), "p_value": p_value}

    summary = []
    for name in SYSTEMS:
        summary.append(row(f"{name} ten-seed mean", [np.mean([w1[(name, k, s)] for s in seeds]) for k in sets]))
    set_means = diff.mean(axis=1)
    summary.append(row("DeepSim minus GroundSim-DEC, mean per stream set",
                       set_means, float(stats.ttest_1samp(set_means, 0.0).pvalue)))
    seed_means = diff.mean(axis=0)
    summary.append(row("DeepSim minus GroundSim-DEC, mean per seed over stream sets",
                       seed_means, float(stats.wilcoxon(seed_means, zero_method="wilcox", method="auto").pvalue)))
    summary.append(row("DeepSim minus GroundSim-DEC, all seed-set pairs (not independent)", diff.ravel()))
    out = output_dir / "stream_replicates_summary.csv"
    with open(out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return out


if __name__ == "__main__":
    main()
