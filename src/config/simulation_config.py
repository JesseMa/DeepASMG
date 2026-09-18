"""Run configuration and pipeline constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dynamics.foundation_dynamics import (
        ProcessTimeStrategy,
        ProductStrategy,
        RepairStrategy,
        SurvivalStrategy,
        TransitionStrategy,
    )

# Chronological split. TRAIN_RATIO + 2 * HPO_TEST_SIZE must equal 1.0 (val == test).
TRAIN_RATIO = 0.70
HPO_TEST_SIZE = 0.15

SECONDS_PER_DAY = 86400

# Eval anchor. Training data is generated backwards from this point so that
# training and evaluation join without a time gap.
SIM_START_TIMESTAMP = 1775001600
if SIM_START_TIMESTAMP % SECONDS_PER_DAY != 0:
    raise ValueError(
        "Fail fast: SIM_START_TIMESTAMP must be midnight UTC. Shift and daily-phase "
        "features are built from simulation-relative time, which is only a calendar "
        "clock when the anchor is midnight; the training log inherits the anchor."
    )


TRAINING_DAYS    = 365
SIMULATION_DAYS  = 31
WARMUP_DAYS      = 1
N_RUNS           = 10
SEED_BASE        = 1000
SEED_OFFSET_FLOOR = 10_000
TRAIN_SEED = 42
INITIAL_ORDERS   = 10
MAX_EPOCHS       = 100
PATIENCE         = 15


def get_seeds(n: int = N_RUNS) -> list[int]:
    return list(range(SEED_BASE, SEED_BASE + n))


@dataclass
class SimulationConfig:

    process_time_strategy: ProcessTimeStrategy
    transition_strategy: TransitionStrategy
    survival_strategy: SurvivalStrategy
    repair_strategy: RepairStrategy
    product_strategy: ProductStrategy

    duration_days: float = 1.0
    warmup_days: float = 0.0
    seed: int = 42
    initial_orders: int = INITIAL_ORDERS
    run_id: int = 0

    @property
    def max_time(self) -> float:
        return (self.warmup_days + self.duration_days) * SECONDS_PER_DAY

    @property
    def warmup_time(self) -> float:
        return self.warmup_days * SECONDS_PER_DAY

    @property
    def total_steps(self) -> int:
        return int(self.max_time)