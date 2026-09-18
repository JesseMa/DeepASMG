"""Shared contracts and helpers of the five simulation mechanisms; every returned duration is whole seconds, rounded once here."""

from abc import ABC, abstractmethod
from typing import NamedTuple, Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

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
}

NONE_TOKEN: str = "<NONE>"

END_TOKEN: str = "End"

VISIT_CAP = 4


def visit_token(n: int) -> str:
    return str(min(max(int(n), 1), VISIT_CAP))

TWO_PI: float = 2.0 * np.pi

_DAY_SECONDS = 86400
_EARLY_START = 6 * 3600
_LATE_START = 14 * 3600
_NIGHT_START = 22 * 3600
N_SHIFTS = 3


def set_onehot(
    x: np.ndarray, offset: int, mapping: Dict[str, int], value: str,
) -> None:
    idx = mapping.get(value)
    if idx is not None:
        x[offset + idx] = 1.0


def detect_shift(timestamp: float) -> int:
    tod = timestamp % _DAY_SECONDS
    if tod < _EARLY_START or tod >= _NIGHT_START:
        return 2
    if tod < _LATE_START:
        return 0
    return 1


def encode_time_features(absolute_time: float, periods_seconds: List[float]) -> np.ndarray:
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


def weighted_draw(values, weights, rng: "np.random.Generator"):
    cdf = np.asarray(weights, dtype=float).cumsum()
    idx = int(cdf.searchsorted(rng.random() * cdf[-1], side="right"))
    if isinstance(values, (int, np.integer)):
        return min(idx, int(values) - 1)
    return values[min(idx, len(values) - 1)]


class MaskedTable(NamedTuple):
    targets: list
    weights: Optional[np.ndarray]
    removed_mass: float
    fallback: bool


def masked_categorical_prepare(
    targets: List[str],
    weights: np.ndarray,
    available_targets: Set[str],
    *,
    label: str = "",
) -> MaskedTable:
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
        return MaskedTable(avail, None, removed_mass, True)
    kept_targets = [t for t, ok in zip(targets, admissible, strict=True) if ok]
    return MaskedTable(kept_targets, weights[admissible] / kept_mass, removed_mass, False)


def masked_categorical_draw(rng: np.random.Generator, table: MaskedTable) -> str:
    if table.fallback:
        return str(rng.choice(table.targets))
    return str(weighted_draw(table.targets, table.weights, rng))


def masked_categorical_probs(table: MaskedTable) -> Dict[str, float]:
    if table.fallback:
        u = 1.0 / len(table.targets)
        return {t: u for t in table.targets}
    return {t: float(w) for t, w in zip(table.targets, table.weights, strict=True)}


def compile_offsets(feature_layout: List[Dict[str, Any]]) -> Dict[str, int]:
    return {g["name"]: g["offset"] for g in feature_layout}


def load_deep_model(
    model_path: "Any",
    metadata_path: "Any",
) -> "tuple":
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
    import torch
    with torch.inference_mode():
        return model.forward(torch.from_numpy(x).unsqueeze(0)).squeeze(0)


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
        available_targets: Set[str],
        current_time: float = 0.0,
    ) -> Optional[str]:
        ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None: ...


class SurvivalStrategy(ABC):

    @abstractmethod
    def sample_time_to_failure(
        self, station_id: str, current_time: float = 0.0,
    ) -> Optional[float]:
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
        ...


class RepairStrategy(ABC):

    @abstractmethod
    def predict_repair_time(
        self,
        station_id: str,
        operating_time_since_last: float,
        utilization: float,
        current_time: float = 0.0,
    ) -> float:
        ...

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        ...


class ProductStrategy(ABC):
    @abstractmethod
    def sample_features(self, current_time: float = 0.0) -> Dict[str, str]:
        ...

    def initialize(
        self,
        product_features: Dict[str, Dict[str, float]],
        temporal_modulation: Optional[Dict[str, Dict[str, Any]]] = None,
        markov_alphas: Optional[Dict[str, float]] = None,
    ) -> None:
        ...