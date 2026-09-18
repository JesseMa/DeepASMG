"""Run configuration and pipeline constants. Time unit is seconds, durations in days."""

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
SIM_START_TIMESTAMP = 1775001600  # 2026-04-01 00:00:00 UTC (Wednesday)
if SIM_START_TIMESTAMP % SECONDS_PER_DAY != 0:
    raise ValueError(
        "Fail fast: SIM_START_TIMESTAMP must be midnight UTC. Shift and daily-phase "
        "features are built from simulation-relative time, which is only a calendar "
        "clock when the anchor is midnight; the training log inherits the anchor."
    )

# Pipeline constants. Scripts import these instead of defining local literals.

TRAINING_DAYS    = 365    # GroundSim data-gen + train horizon
SIMULATION_DAYS  = 31     # eval horizon
WARMUP_DAYS      = 1      # Eval warmup before recording
N_RUNS           = 10     # CRN replications per simulator
SEED_BASE        = 1000
SEED_OFFSET_FLOOR = 10_000  # Offset for the decorrelated floor configurations
TRAIN_SEED = 42            # torch/lightning seed for every model training run
INITIAL_ORDERS   = 10
MAX_EPOCHS       = 100
PATIENCE         = 15     # Early-stopping patience


def get_seeds(n: int = N_RUNS) -> list[int]:
    """CRN seed list for `n` replications, anchored at SEED_BASE."""
    return list(range(SEED_BASE, SEED_BASE + n))


@dataclass
class SimulationConfig:
    """Configuration of one simulation run."""

    process_time_strategy: ProcessTimeStrategy
    transition_strategy: TransitionStrategy
    survival_strategy: SurvivalStrategy
    repair_strategy: RepairStrategy
    product_strategy: ProductStrategy

    duration_days: float = 1.0
    warmup_days: float = 0.0            # recording starts only after warmup
    seed: int = 42
    initial_orders: int = INITIAL_ORDERS    # orders pre-loaded at t=0
    run_id: int = 0                         # order IDs become R{run_id}_J{counter:06d}

    @property
    def max_time(self) -> float:
        """Total simulation time in seconds (warmup + main duration)."""
        return (self.warmup_days + self.duration_days) * SECONDS_PER_DAY

    @property
    def warmup_time(self) -> float:
        """Warmup time in seconds."""
        return self.warmup_days * SECONDS_PER_DAY

    @property
    def total_steps(self) -> int:
        """Simulation length in whole one-second steps."""
        return int(self.max_time)