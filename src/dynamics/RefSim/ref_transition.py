"""Statistical routing: a fitted categorical table per station, optionally
refined per product type and per full variant, drawn under the admissibility
mask.

RefSim-M uses the station-marginal table alone; RefSim-V adds the conditioned
cells and falls back through (station, variant) → (station, product type) →
station when a cell is missing. Either way one RNG draw per call.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, normalize_distribution, masked_categorical_prepare,
    masked_categorical_draw, masked_categorical_probs, END_TOKEN,
)
from src.config.routing_keys import full_variant_key, product_type_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order

_logger = logging.getLogger(__name__)

Dist = Tuple[list, np.ndarray]


class RefTransition(TransitionStrategy):
    """Fitted routing table with cell fallback and admissibility mask.

    The fitted table may put mass on targets the topology does not admit from
    a station, so every draw is restricted to available_targets and
    renormalized. The masked, renormalized table for a (cell, target set) is
    built once and reused: both inputs are fixed for the run.
    """

    def __init__(
        self,
        station_marginal_probs: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        *,
        by_producttype: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        by_variant: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> None:
        if not station_marginal_probs:
            raise ValueError("Fail fast: station_marginal_probs must not be empty.")
        self._rng = rng
        self._marg = self._prep_flat(station_marginal_probs)
        self._byp = self._prep_nested(by_producttype or {})
        self._byv = self._prep_nested(by_variant or {})
        self._prepared: Dict[Tuple[str, str, FrozenSet[str]], tuple] = {}

    @staticmethod
    def _prep_flat(probs: Dict[str, Dict[str, float]]) -> Dict[str, Dist]:
        return {sid: normalize_distribution(pd, label=f"Route {sid}")
                for sid, pd in probs.items()}

    @staticmethod
    def _prep_nested(
        probs: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Dist]]:
        return {sid: {key: normalize_distribution(pd, label=f"Route {sid}:{key}")
                      for key, pd in by_key.items()}
                for sid, by_key in probs.items()}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = [sid for sid, sc in stations.items()
                   if sc.transitions and sid not in self._marg]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal transition data for {missing}. "
                f"Simulation cannot start."
            )

    def _resolve(self, station_id: str, order: "Order") -> Tuple[str, Dist]:
        """(cell key, distribution): the most specific fitted cell available."""
        feats = order.features
        key = full_variant_key(feats)
        dist = self._byv.get(station_id, {}).get(key)
        if dist is not None:
            return key, dist
        key = product_type_key(feats)
        dist = self._byp.get(station_id, {}).get(key)
        if dist is not None:
            return key, dist
        dist = self._marg.get(station_id)
        if dist is None:
            raise ValueError(f"Fail fast: no transition data for station '{station_id}'.")
        return "", dist

    def _prepared_for(self, station_id: str, order: "Order",
                      available_targets: Set[str]) -> tuple:
        cell, (targets, weights) = self._resolve(station_id, order)
        key = (station_id, cell, frozenset(available_targets))
        prep = self._prepared.get(key)
        if prep is None:
            prep = masked_categorical_prepare(targets, weights, available_targets,
                                              label=station_id)
            self._prepared[key] = prep
        return prep

    def predict(
        self,
        station_id: str,
        order: "Order",
        available_targets: Set[str],
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        prep = self._prepared_for(station_id, order, available_targets)
        # Read-only instrumentation: no RNG draw, no control-flow effect.
        self._sg_mask_calls = getattr(self, "_sg_mask_calls", 0) + 1
        if prep.removed_mass > 0.0:
            self._sg_mask_effective = getattr(self, "_sg_mask_effective", 0) + 1
        if prep.fallback:
            self._sg_mask_fallback = getattr(self, "_sg_mask_fallback", 0) + 1
            _logger.warning(
                "RefTransition uniform-fallback @ station '%s': fitted mass on "
                "all admissible targets = 0 (admissible=%s).",
                station_id, sorted(available_targets),
            )
        result = masked_categorical_draw(self._rng, prep)
        return None if result == END_TOKEN else result

    def distribution_params(
        self,
        station_id: str,
        order: "Order",
        available_targets: Set[str],
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed categorical distribution (resolved cell, post-mask)."""
        prep = self._prepared_for(station_id, order, available_targets)
        return {"family": "categorical", "probs": masked_categorical_probs(prep)}
