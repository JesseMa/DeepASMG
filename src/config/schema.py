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

    def __post_init__(self) -> None:
        # Integer time contract: durations the kernel schedules on directly must
        # be whole seconds, otherwise the engine clock would round them a second
        # time. setup_time enters through a strategy that ceils its total, so it
        # is exempt.
        for name in ("transit_time",):
            value = getattr(self, name)
            if float(value) != int(value):
                raise ValueError(
                    f"Fail fast: {self.id}.{name} must be whole seconds "
                    f"(integer time contract), got {value}."
                )
        if self.capacity < 1:
            raise ValueError(f"Fail fast: {self.id}.capacity must be at least 1, got {self.capacity}.")


@dataclass
class ProcessConfig:

    name: str
    description: str
    product_features: Dict[str, Dict[str, float]]
    stations: List[StationConfig]
    temporal_modulation: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    markov_alphas: Dict[str, float] = field(default_factory=dict)
