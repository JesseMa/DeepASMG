"""Order – runtime object for a job in the simulation."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.config.routing_keys import resolve_key


@dataclass
class Order:
    id: str
    features: Dict[str, str]  # e.g. {'modell': 'A', 'feature_a': 'a.1', 'feature_b': 'b.1'}
    timestamp_creation: float
    timestamp_completion: Optional[float] = None

    visits: Dict[str, int] = field(default_factory=dict)

    def resolve_transition_key(self, transitions: dict, *, station_id: str) -> str:
        """Matching key in transitions; a visit-indexed entry wins if present."""
        return resolve_key(self.features, transitions, wildcard=True,
                           visit=self.visits.get(station_id))

    def resolve_process_key(self, process_times: Dict[str, tuple]) -> str:
        """Find the matching key in process_times (no wildcard)."""
        return resolve_key(self.features, process_times, wildcard=False)