"""Two-way component substitutions at the system level."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.evaluation import system_evaluation as system  # noqa: E402
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
        paired = system.per_seed_distances(
            [{"seed": r["seed"], "ct": target[r["seed"]]} for r in runs], runs,
        )
        for run, dist in zip(runs, paired, strict=True):
            w1, ks_d = dist["w1"], dist["ks_d"]
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
