"""Runs GroundSim and saves the resulting process events and order logs as CSVs
for the DeepSim training pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.experiments.sim_runner import get_process_config  # noqa: E402
from src.config.simulation_config import (  # noqa: E402
    INITIAL_ORDERS,
    SECONDS_PER_DAY,
    SIM_START_TIMESTAMP,
    SimulationConfig,
    TRAINING_DAYS,
    WARMUP_DAYS,
)
from src.dynamics.GroundSim.ground_survival import GroundSurvival  # noqa: E402
from src.dynamics.GroundSim.ground_repair import GroundRepair  # noqa: E402
from src.dynamics.GroundSim.ground_process_time import GroundProcessTime  # noqa: E402
from src.dynamics.GroundSim.ground_product import GroundProduct  # noqa: E402
from src.dynamics.GroundSim.ground_transition import GroundTransition  # noqa: E402
from src.recording.results_saver import ResultsSaver  # noqa: E402
from src.simulation.engine import SimulationEngine  # noqa: E402

DEFAULT_DAYS       = TRAINING_DAYS
DEFAULT_SEED       = 101
DEFAULT_OUTPUT     = "data"
DEFAULT_EXPERIMENT = "training"


def generate(
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = DEFAULT_OUTPUT,
    experiment_name: str = DEFAULT_EXPERIMENT,
    order_id_prefix: str | None = None,
) -> Path:
    """Run GroundSim and write process, order and run-metadata logs.

    order_id_prefix prevents order_id collisions in prepare_transition_data
    (which groups by order_id) when several seed runs share one parent
    directory.
    """
    output_dir = Path(output_dir)

    process_config = get_process_config()

    saver = ResultsSaver(
        base_dir=output_dir,
        experiment_name=experiment_name,
        process_name=process_config.name,
        order_id_prefix=order_id_prefix,
    )

    # Training data ends exactly at the eval anchor (SIM_START_TIMESTAMP): the
    # start is backdated by days + warmup so the recording window ('days')
    # abuts the evaluation period seamlessly.
    training_start_timestamp = (
        SIM_START_TIMESTAMP - (days + WARMUP_DAYS) * SECONDS_PER_DAY
    )

    # Independent per-module RNG streams, same stream plan as SimFactorySet
    # on the eval path.
    rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = (
        np.random.default_rng(c) for c in np.random.SeedSequence(seed).spawn(5)
    )
    sim_config = SimulationConfig(
        process_time_strategy=GroundProcessTime(rng_pt),
        transition_strategy=GroundTransition(rng_tr),
        survival_strategy=GroundSurvival(rng_sv),
        repair_strategy=GroundRepair(rng_rt),
        product_strategy=GroundProduct(rng_pr, start_timestamp=training_start_timestamp),
        duration_days=days,
        warmup_days=WARMUP_DAYS,
        seed=seed,
        initial_orders=INITIAL_ORDERS,
        run_id=1,
        start_timestamp=training_start_timestamp,
    )

    engine = SimulationEngine(process_config, sim_config)
    process_log, order_log = engine.run(label=f"groundsim seed={seed} days={days}")

    saver.save_run(run_number=1, process_log=process_log, order_log=order_log)

    completed = sum(1 for o in order_log if o["timestamp_completion"] is not None)

    # Absolute generation anchor for downstream feature preparation: the
    # generator modulates calendar cycles on start_timestamp + t, so surrogate
    # features with week/month periods must be built on the same absolute
    # clock (a purely sim-relative timestamp is phase-shifted by
    # start_timestamp mod period).
    run_metadata = {
        "seed": seed,
        "days": days,
        "warmup_days": WARMUP_DAYS,
        "start_timestamp": training_start_timestamp,
        "sim_start_anchor": SIM_START_TIMESTAMP,
    }
    with open(Path(saver.batch_dir) / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"  Events: {len(process_log):,}  |  Orders completed: {completed:,}")
    print(f"  Output: {saver.batch_dir}")

    return saver.batch_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate GroundSim training data for DeepSim.",
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Simulation days (default: {DEFAULT_DAYS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help=f"Base output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT,
                        help=f"Experiment name (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--order-id-prefix", type=str, default=None,
                        help=("Prefix for all order_ids. Use for "
                              "multi-seed HPO aggregation (e.g. 's101_'); "
                              "prevents order_id collisions when multiple "
                              "seed runs share one parent directory."))
    args = parser.parse_args()

    print(f"Generating GroundSim data: {args.days} days, seed={args.seed}"
          + (f", prefix='{args.order_id_prefix}'" if args.order_id_prefix else ""))
    generate(
        days=args.days,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        order_id_prefix=args.order_id_prefix,
    )
    print("Done.")


if __name__ == "__main__":
    main()
