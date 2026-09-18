"""Bundled closed-loop run at the system level.

All systems run on the production training state (models/ plus
statistic_params.pkl) that was derived from a 365-day log; the evaluation
window itself is the short horizon from simulation_config, NOT 365 days.
One common-random-numbers replication series per system, stored as a pickle.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.experiments.sim_runner import SimFactorySet, run_replications  # noqa: E402
from src.config.simulation_config import (  # noqa: E402
    N_RUNS, SIMULATION_DAYS, WARMUP_DAYS, get_seeds,
)


def build_systems(fs: SimFactorySet) -> dict:
    """The six evaluated systems, in evaluation order.

    Single source of truth: scripts/run_safeguard_audit.py reuses this so the
    instrumented rerun cannot drift from the authoritative run.
    """
    return {
        "GroundSim": fs.base, "GroundSim-DEC": fs.floor,
        "DeepSim": fs.deep,
        "RefSim-M": fs.refm, "RefSim-V": fs.refv, "RefSim-W": fs.refw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=SIMULATION_DAYS)
    ap.add_argument("--warmup", type=float, default=WARMUP_DAYS)
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    ap.add_argument(
        "--output-file",
        type=Path,
        default=REPO / "results/verification/closed_loop_runs.pkl",
    )
    args = ap.parse_args()

    seeds = get_seeds(args.num_seeds)
    fs = SimFactorySet(duration_days=args.days, warmup_days=args.warmup)
    systems = build_systems(fs)
    print(f"=== Closed-loop run: {len(systems)} systems × {len(seeds)} seeds "
          f"× {args.days}d (+{args.warmup}d warmup) ===")
    print(f"  seeds: {seeds}")

    runs_by_sim: dict[str, list] = {}
    for name, fac in systems.items():
        t0 = time.time()
        runs = run_replications(name, fac, seeds)
        runs_by_sim[name] = runs
        cts = sum(len(r["ct"]) for r in runs)
        print(f"  {name:14} done: {len(runs)} runs, {cts:,} cycle-times ({time.time()-t0:.0f}s)")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "wb") as f:
        pickle.dump({"runs_by_sim": runs_by_sim, "seeds": seeds,
                     "days": args.days, "warmup": args.warmup}, f)
    print(f"\nDONE → {args.output_file}")


if __name__ == "__main__":
    main()
