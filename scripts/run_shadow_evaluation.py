"""Bundled shadow run over all shadow systems and evaluation seeds."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.experiments.shadow_evaluation import (  # noqa: E402
    COMPONENTS,
    SYSTEMS,
    run_shadow_pilot,
)
from src.config.simulation_config import (  # noqa: E402
    N_RUNS, SIMULATION_DAYS, get_seeds,
)


def _concat(out: Path, seed_dirs: list) -> None:
    for system in SYSTEMS:
        for comp in COMPONENTS:
            parts = [d / f"{system}__{comp}.csv" for d in seed_dirs]
            parts = [p for p in parts if p.exists()]
            if not parts:
                continue
            pd.concat([pd.read_csv(p, dtype={"context_id": str}) for p in parts],
                      ignore_index=True).to_csv(out / f"{system}__{comp}.csv", index=False)
    sc = [d / "_context_variant.csv" for d in seed_dirs]
    sc = [p for p in sc if p.exists()]
    if sc:
        pd.concat([pd.read_csv(p, dtype={"context_id": str}) for p in sc],
                  ignore_index=True).to_csv(out / "_context_variant.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=SIMULATION_DAYS)
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/verification/shadow",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = get_seeds(args.num_seeds)
    print(f"=== Shadow bundle: {len(SYSTEMS)} systems × {len(seeds)} seeds × {args.days}d ===")
    seed_dirs = []
    for seed in seeds:
        t0 = time.time()
        d = args.output_dir / f"_seed_{seed}"
        rows = run_shadow_pilot(seed, days=args.days, out_dir=d)
        seed_dirs.append(d)
        _concat(args.output_dir, seed_dirs)
        print(f"  seed {seed}: {rows:,} rows total ({time.time()-t0:.0f}s)")
    print(f"\nDONE → {args.output_dir}")


if __name__ == "__main__":
    main()
