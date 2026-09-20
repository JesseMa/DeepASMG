"""Build the sensitivity-configuration tree that run_sensitivity_sweeps consumes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HORIZON_DAYS = (14, 31, 61, 91, 183, 365)
DRAW365_SEEDS = (102, 103, 104, 105)
DRAW31_SEEDS = (101, 102, 103, 104, 105)


def _run(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as log:
        subprocess.run(
            cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def _find_data_dir(parent: Path) -> Path:
    from scripts.train_models import find_newest_data_dir
    return find_newest_data_dir(parent)


def _models_trained(model_dir: Path) -> bool:
    from src.experiments.sim_runner import models_trained
    return models_trained(model_dir)


def _build_config(
    label: str, seed: int, days: int, data_parent: Path, model_dir: Path,
    log_dir: Path, skip_existing: bool, hpo_dir: Path,
) -> None:
    if skip_existing and _models_trained(model_dir):
        print(f"[{label}] exists — skipped")
        return

    t0 = time.time()
    print(f"[{label}] generate: seed={seed} days={days}", flush=True)
    _run(
        [sys.executable, "-m", "scripts.generate_training_data",
         "--seed", str(seed), "--days", str(days),
         "--output-dir", str(data_parent)],
        log_dir / f"{label}_generate.log",
    )
    data_dir = _find_data_dir(data_parent)

    print(f"[{label}] train: {data_dir.name}", flush=True)
    _run(
        [sys.executable, "-m", "scripts.train_models",
         "--data-dir", str(data_dir),
         "--model-dir", str(model_dir),
         "--hpo-dir", str(hpo_dir)],
        log_dir / f"{label}_train.log",
    )
    print(f"[{label}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="Target directory for the sensitivity tree")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Directory for per-step logs (default: <root>/_logs)")
    ap.add_argument("--hpo-dir", type=Path, default=REPO / "models" / "hpo",
                    help="Hyperparameters for every configuration "
                         "(default: the production models/hpo)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip configurations whose model registry already exists")
    ap.add_argument("--only", type=str, default=None,
                    choices=["horizons", "draw365", "draw31"],
                    help="Build only one section of the tree")
    args = ap.parse_args()
    root: Path = args.root
    log_dir = args.log_dir if args.log_dir is not None else root / "_logs"

    jobs: list[tuple[str, int, int, Path, Path]] = []
    if args.only in (None, "horizons"):
        for days in HORIZON_DAYS:
            name = f"days_{days:04d}"
            jobs.append((name, 101, days, root / "data" / name,
                         root / "models" / name))
    if args.only in (None, "draw365"):
        for seed in DRAW365_SEEDS:
            name = f"seed_{seed:04d}"
            jobs.append((name, seed, 365, root / "data" / name,
                         root / "models" / name))
    if args.only in (None, "draw31"):
        for seed in DRAW31_SEEDS:
            name = f"seed_{seed:04d}"
            jobs.append((f"31d_{name}", seed, 31,
                         root / "data" / "31d_seed_variance" / name,
                         root / "models" / "31d_seed_variance" / name))

    print(f"=== Sensitivity tree: {len(jobs)} configurations → {root} ===")
    for label, seed, days, data_parent, model_dir in jobs:
        _build_config(label, seed, days, data_parent, model_dir,
                      log_dir, args.skip_existing, args.hpo_dir)
    print("\nDONE. Next: python -m scripts.run_sensitivity_sweeps "
          f"--sensitivity-root {root} ...")


if __name__ == "__main__":
    main()
