"""Schema of the process definition: one station and the whole process."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class StationConfig:
    id: str
    capacity: int
    is_machine: bool = False
    process_times: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    transitions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mttr: float = 0.0
    setup_time: float = 0.0
    transit_time: float = 0.0
    is_start_station: bool = False
    ttf_scale_seconds: float = 0.0
    sequential_routing: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ProcessConfig:
    """Full production process definition.

    temporal_modulation: per feature a period and an amplitude a, scaling that
        feature's base weights by 1 +/- a over the cycle before renormalisation.
        Features without an entry stay static.
    """

    name: str
    description: str
    product_features: Dict[str, Dict[str, float]]
    stations: List[StationConfig]
    temporal_modulation: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    markov_alphas: Dict[str, float] = field(default_factory=dict)
