"""Order – runtime object for a job in the simulation."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config.routing_keys import resolve_key


@dataclass
class Order:
    id: str
    features: Dict[str, str]
    timestamp_creation: float
    timestamp_completion: Optional[float] = None

    visits: Dict[str, int] = field(default_factory=dict)

    def resolve_transition_key(self, transitions: dict, *, station_id: str) -> str:
        return resolve_key(self.features, transitions, wildcard=True,
                           visit=self.visits.get(station_id))

    def resolve_process_key(self, process_times: Dict[str, tuple]) -> str:
        return resolve_key(self.features, process_times, wildcard=False)