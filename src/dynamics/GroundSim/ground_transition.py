"""GroundTransition — routing from configured probabilities per station/model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, normalize_distribution, masked_categorical_probs, END_TOKEN,
    weighted_draw,
)
from src.config.routing_keys import resolve_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


class GroundTransition(TransitionStrategy):
    """Transitions from configured probabilities (StationConfig.transitions).

    With `sequential_routing` set, matching orders are routed deterministically
    via a counter instead of stochastically; the target switches every `cycle`
    orders.
    """

    # Visit cap for M5: the configured return probabilities (largest at C_b.4)
    # would otherwise give a geometric tail with rare, very high visit counts.
    # The cap lives in the strategy, not in the config probabilities, so the
    # per-visit probabilities keep their domain meaning.
    M5_STATION_ID = "M5"
    M5_MAX_VISITS = 3

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._stations: Dict[str, "StationConfig"] = {}
        self._distributions: Dict[str, Dict[str, tuple[List[str], np.ndarray]]] = {}
        # {station_id: {resolved_key: {'cycle': n, 'targets': [...]}}}
        self._seq_configs: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # {station_id: {resolved_key: order_count_so_far}}
        self._seq_counters: Dict[str, Dict[str, int]] = {}
        # {order_id: M5 visits so far}
        self._m5_visits: Dict[str, int] = {}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._stations = stations
        self._distributions = {}
        self._seq_configs = {}
        self._seq_counters = {}
        self._m5_visits = {}

        for station_id, config in stations.items():
            by_key: Dict[str, tuple[List[str], np.ndarray]] = {}

            for key, transition_map in config.transitions.items():
                if not transition_map:
                    continue
                targets, probs = normalize_distribution(
                    transition_map, label=f"{station_id}:{key}",
                )
                by_key[key] = (targets, probs)

            self._distributions[station_id] = by_key

            if config.sequential_routing:
                self._seq_configs[station_id] = config.sequential_routing
                self._seq_counters[station_id] = {}

    def predict(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        # M5 visit cap: force End after M5_MAX_VISITS visits (checked before the draw).
        if station_id == self.M5_STATION_ID:
            n_visits = self._m5_visits.get(order.id, 0) + 1
            self._m5_visits[order.id] = n_visits
            if n_visits >= self.M5_MAX_VISITS:
                self._m5_visits.pop(order.id, None)
                return None

        config = self._stations[station_id]
        key = order.resolve_transition_key(config.transitions)

        seq_rule = self._resolve_sequential_rule(station_id, order)
        if seq_rule is not None:
            return self._apply_sequential(station_id, order, seq_rule)

        distribution = self._distributions.get(station_id, {}).get(key)
        if distribution is None:
            raise ValueError(
                f"Empty transition map for station '{station_id}' and key '{key}'."
            )

        targets, weights = distribution
        result = weighted_draw(targets, weights, self._rng)
        if result == END_TOKEN:
            self._m5_visits.pop(order.id, None)
            return None
        return result

    def distribution_params(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """True categorical routing distribution (config, post-mask).

        The deterministic overrides (sequential_routing, M5 visit cap) are not
        categorical and are not represented here.
        """
        config = self._stations[station_id]
        key = order.resolve_transition_key(config.transitions)
        dist = self._distributions.get(station_id, {}).get(key)
        if dist is None:
            return {"family": "categorical", "probs": {}}
        targets, weights = dist
        return {"family": "categorical",
                "probs": masked_categorical_probs(targets, weights, available_targets)}

    def _resolve_sequential_rule(
        self, station_id: str, order: "Order"
    ) -> Optional[Dict[str, Any]]:
        """Find a matching sequential_routing rule for this order, if any."""
        seq_conf = self._seq_configs.get(station_id)
        if not seq_conf:
            return None
        key = resolve_key(order.features, seq_conf, wildcard=True, default=None)
        return seq_conf[key] if key is not None else None

    def _apply_sequential(
        self,
        station_id: str,
        order: "Order",
        rule: Dict[str, Any],
    ) -> Optional[str]:
        """Determine the target deterministically via the counter and increment it."""
        # Count under the same key as used in transitions
        key = order.resolve_transition_key(self._stations[station_id].transitions)
        counters = self._seq_counters[station_id]
        count = counters.get(key, 0)

        cycle: int = rule["cycle"]
        targets: List[str] = rule["targets"]
        target = targets[(count // cycle) % len(targets)]
        counters[key] = count + 1

        if target == END_TOKEN:
            return None
        return target