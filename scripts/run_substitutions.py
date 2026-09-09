"""Two-way component substitutions at the system level.

Scores every configuration in ``src.experiments.ablation_runner.CONFIGS`` against
the paired ``Target (GroundSim)`` realization. The ``direction`` column labels
each row: swap_to_one inserts one candidate mechanism into the true core,
swap_to_perfect restores one true mechanism inside a learned or statistical
core, module_selection is the learned core with statistical repair. The
whole-core anchors appear under both substitution directions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.experiments.ablation_runner import CONFIGS, run_ablation_grid  # noqa: E402
from src.config.simulation_config import N_RUNS, get_seeds  # noqa: E402

TARGET = "Target (GroundSim)"

# Component label per mechanism slot.
COMPONENTS = {
    "Orders": "arrival",
    "Process": "processing",
    "Routing": "routing",
    "Survival": "survival",
    "Repair": "repair",
}

ANCHORS = ("DeepSim (All NN)", "RefSim (All Statistical)", "Floor (Full)")


def _label(configuration: str) -> list[tuple[str, str]]:
    """(direction, component) pairs a configuration contributes rows for."""
    if configuration == TARGET:
        return []
    if configuration == "Hybrid (Stat Repair)":
        return [("module_selection", "repair")]
    if configuration in ANCHORS:
        return [("swap_to_one", "all"), ("swap_to_perfect", "all")]
    for slot, component in COMPONENTS.items():
        if f"Perfect {slot}" in configuration:
            return [("swap_to_perfect", component)]
        if f"{slot} only" in configuration or f"{slot} decorrelated" in configuration:
            return [("swap_to_one", component)]
    raise ValueError(f"unlabelled ablation configuration: {configuration!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    ap.add_argument(
        "--output-file",
        type=Path,
        default=REPO / "results/verification/component_substitutions.csv",
    )
    args = ap.parse_args()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    seeds = get_seeds(args.num_seeds)
    print(f"=== Substitutions: {len(CONFIGS)} configurations x {len(seeds)} seeds ===")
    grid = run_ablation_grid(seeds)

    target = {run["seed"]: run["ct"] for run in grid[TARGET]}
    rows = []
    for configuration, runs in grid.items():
        labels = _label(configuration)
        if not labels:
            continue
        for run in runs:
            reference = target[run["seed"]]
            w1 = float(wasserstein_distance(reference, run["ct"]))
            ks_d = float(ks_2samp(reference, run["ct"]).statistic)
            for direction, component in labels:
                rows.append({
                    "direction": direction,
                    "component": component,
                    "configuration": configuration,
                    "seed": run["seed"],
                    "reference_system": TARGET,
                    "w1": w1,
                    "ks_d": ks_d,
                })

    pd.DataFrame(rows).to_csv(args.output_file, index=False)
    print(f"\nDONE → {args.output_file} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
