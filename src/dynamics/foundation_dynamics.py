"""
Abstract base classes for the five simulation mechanisms.

    ProcessTimeStrategy  (pt) → process times per station and product
    TransitionStrategy   (tr) → routing decisions
    SurvivalStrategy     (sv) → time to failure, in operating seconds
    RepairStrategy       (rt) → downtime duration, in wall-clock seconds
    ProductStrategy      (pr) → attributes of the next released order

The two-letter keys index the trained-model registry and the ablation runner.

Integer time contract: every duration a strategy returns (process time, repair
duration, time to failure) is whole seconds, rounded exactly once here at the
module boundary. The kernel never rounds again, so the logged duration is the
executed one and a surrogate fitted on the log can reproduce it without a
second discretization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from src.config.schema import StationConfig
    from src.simulation.order import Order

NAMED_PERIODS_HOURS = {
    "shift":   8.0,
    "day":     24.0,
    "week":    168.0,
    "month":   730.0,
    "quarter": 2190.0,
    "year":    8760.0,
}

NONE_TOKEN: str = "<NONE>"
"""Sentinel for 'no predecessor' / 'unknown' in the feature encodings."""

END_TOKEN: str = "End"
"""Sentinel for 'order leaves the system' in the transition logic."""

TWO_PI: float = 2.0 * np.pi

# Seconds since start of day.
_DAY_SECONDS = 86400
_EARLY_START = 6 * 3600
_LATE_START = 14 * 3600
_NIGHT_START = 22 * 3600
N_SHIFTS = 3


def set_onehot(
    x: np.ndarray, offset: int, mapping: Dict[str, int], value: str,
) -> None:
    """Set a one-hot entry in x if value is present in mapping."""
    idx = mapping.get(value)
    if idx is not None:
        x[offset + idx] = 1.0


def detect_shift(timestamp: float) -> int:
    """Shift index: 0=early (06-14h), 1=late (14-22h), 2=night (22-06h)."""
    tod = timestamp % _DAY_SECONDS
    if tod < _EARLY_START or tod >= _NIGHT_START:
        return 2
    if tod < _LATE_START:
        return 0
    return 1


def encode_time_features(absolute_time: float, periods_seconds: List[float]) -> np.ndarray:
    """Encode a timestamp as interleaved sin/cos pairs, length 2 * len(periods_seconds)."""
    n = len(periods_seconds)
    result = np.zeros(2 * n, dtype=np.float32)
    for j, period in enumerate(periods_seconds):
        if period > 0:
            phase = TWO_PI * absolute_time / period
            result[2 * j] = np.sin(phase)
            result[2 * j + 1] = np.cos(phase)
    return result


def normalize_distribution(
    weights_dict: Dict[str, float],
    *,
    label: str = "",
) -> Tuple[List[str], np.ndarray]:
    """Validate a weight dict and normalize it to sum 1; label tags error messages."""
    if not weights_dict:
        raise ValueError(f"Empty distribution{f' in {label}' if label else ''}.")

    values = list(weights_dict.keys())
    weights = np.array(list(weights_dict.values()), dtype=float)

    if np.any(weights < 0):
        raise ValueError(
            f"Negative probability{f' in {label}' if label else ''}."
        )

    total = weights.sum()
    if total <= 0:
        raise ValueError(
            f"Probabilities must have a positive sum"
            f"{f' in {label}' if label else ''}."
        )

    return values, weights / total


def masked_categorical_draw(
    rng: np.random.Generator,
    targets: List[str],
    weights: np.ndarray,
    available_targets: Set[str],
    *,
    label: str = "",
) -> Tuple[str, float, bool, float, np.ndarray]:
    """Draw one target under an admissibility mask; exactly one RNG draw per call.

    Inadmissible targets lose their mass and the rest is renormalized. With zero
    fitted mass on all admissible targets the draw falls back to uniform over
    ``available_targets``, which may contain labels absent from ``targets``.
    Returns (chosen_target, removed_mass, used_fallback, kept_mass, admissible);
    chosen_target is raw and may be END_TOKEN, which the caller maps to None.
    """
    admissible = np.fromiter(
        (t in available_targets for t in targets), dtype=bool, count=len(targets)
    )
    kept_mass = float(weights[admissible].sum())
    removed_mass = float(weights[~admissible].sum())

    if kept_mass <= 0.0:
        avail = sorted(available_targets)
        if not avail:
            raise ValueError(
                f"Routing mask empty for '{label}': available_targets={available_targets}."
            )
        return str(rng.choice(avail)), removed_mass, True, kept_mass, admissible

    kept_targets = [t for t, ok in zip(targets, admissible, strict=True) if ok]
    kept_weights = weights[admissible] / kept_mass
    return str(rng.choice(kept_targets, p=kept_weights)), removed_mass, False, kept_mass, admissible


def masked_categorical_probs(
    targets: List[str],
    weights: "np.ndarray",
    available_targets: Optional[Set[str]],
) -> Dict[str, float]:
    """Post-mask renormalized {label: prob}; no RNG draw.

    Without a mask every target is a key. With a mask only admissible targets
    are keys; zero admissible mass falls back to uniform over
    ``available_targets``, whose keys need not appear in ``targets``.
    """
    if available_targets is None:
        total = float(np.sum(weights))
        return {t: float(w) / total for t, w in zip(targets, weights, strict=True)}
    admissible = [t in available_targets for t in targets]
    kept_mass = float(sum(float(w) for w, a in zip(weights, admissible, strict=True) if a))
    if kept_mass <= 0.0:
        avail = sorted(available_targets)
        if not avail:
            raise ValueError(f"Empty admissibility set: {available_targets}.")
        return {t: 1.0 / len(avail) for t in avail}
    return {
        t: float(w) / kept_mass
        for t, w, a in zip(targets, weights, admissible, strict=True) if a
    }


def compile_offsets(feature_layout: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract {name: offset} from a feature_layout."""
    return {g["name"]: g["offset"] for g in feature_layout}


def load_deep_model(
    model_path: "Any",
    metadata_path: "Any",
) -> "tuple":
    """Load a TorchScript model (eval mode) plus its JSON metadata."""
    import json
    from pathlib import Path
    import torch

    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"Please run training first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run training first."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    return model, metadata


def infer_single(model: Any, x: np.ndarray) -> "torch.Tensor":
    """Run inference for a single feature vector; returns the 1D output tensor."""
    import torch
    with torch.inference_mode():
        return model(torch.from_numpy(x).unsqueeze(0).float()).squeeze(0)


class ProcessTimeStrategy(ABC):
    @abstractmethod
    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float: ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None: ...


class TransitionStrategy(ABC):
    @abstractmethod
    def predict(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,
    ) -> Optional[str]:
        """Returns next-station id, or None for End.

        available_targets: reachable downstream stations (incl. End). DeepSim
        renormalizes its softmax over this set; RefSim masks its fitted table
        the same way when built with apply_admissibility_mask=True (as the
        shipped runners do); GroundSim ignores it because its configured
        transition map already encodes the topology. current_time is unused by
        the shipped strategies; the shadow logger records it as t_sim.
        """
        ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None: ...


class SurvivalStrategy(ABC):
    """Time-to-failure strategy; downtime duration belongs to ``RepairStrategy``.

    TTF is in operating seconds: only time the machine actually works counts;
    idle time (starvation, blocking) does not.
    """

    @abstractmethod
    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,
    ) -> Optional[float]:
        """Operating seconds (> 0) until the next failure, or None if the station cannot fail."""
        ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        ...

    def notify_cycle_end(
        self,
        station_id: str,
        ttf: float,
        n_jobs: int,
        repair_time: float,
    ) -> None:
        """Hook after a completed failure cycle (default: no-op).

        ``repair_time`` comes from the paired ``RepairStrategy``; FailureManager
        calls both hooks once per cycle.
        """


class RepairStrategy(ABC):
    """Downtime duration in wall-clock seconds; survival sampling belongs to ``SurvivalStrategy``."""

    @abstractmethod
    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
        current_time: float = 0.0,
    ) -> float:
        """Predict the downtime duration in seconds (wall clock).

        operating_time_since_last is in operating seconds since the last
        failure; utilization is operating_time / wall_clock_elapsed in the
        current cycle, in [0, 1].
        """
        ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        ...

    def notify_cycle_end(
        self,
        station_id: str,
        ttf: float,
        n_jobs: int,
        repair_time: float,
    ) -> None:
        """Hook after a completed failure cycle (default: no-op)."""


class ProductStrategy(ABC):
    @abstractmethod
    def sample_features(self, current_time: float = 0.0) -> Dict[str, str]:
        """Features for a new order; current_time is seconds since simulation start."""
        ...

    def initialize(
        self,
        product_features: Dict[str, Dict[str, float]],
        temporal_modulation: Optional[Dict[str, Dict[str, Any]]] = None,
        markov_alphas: Optional[Dict[str, float]] = None,
    ) -> None:
        """Bind the product definitions.

        markov_alphas: per-feature lazy-random-walk mixing, α ∈ [0, 1)
        (0 = IID); the stationary distribution is preserved exactly.
        """
        ...