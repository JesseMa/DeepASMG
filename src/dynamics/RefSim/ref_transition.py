"""Statistical routing strategies: station-marginal and variant-conditioned."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, normalize_distribution, masked_categorical_draw,
    masked_categorical_probs, END_TOKEN, weighted_draw,
)
from src.config.routing_keys import full_variant_key, product_type_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order

_logger = logging.getLogger(__name__)


class RefTransition(TransitionStrategy):
    """Marginal router: P(target | station); order.features are ignored.

    The fitted table may put mass on targets the topology does not admit from
    this station, so the draw is restricted to available_targets.
    """

    def __init__(
        self,
        transition_probs: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        *,
        apply_admissibility_mask: bool = True,
    ) -> None:
        if not transition_probs:
            raise ValueError("Fail fast: transition_probs must not be empty.")

        self._rng = rng
        self._apply_mask = bool(apply_admissibility_mask)
        self._distributions: Dict[str, tuple[list[str], np.ndarray]] = {}
        self._prepare_distributions(transition_probs)

        # Per-instance (= per-run) instrumentation; reset in initialize().
        self.mask_calls: int = 0
        self.mask_effective: int = 0   # draws where >=1 fitted target with
                                       # positive mass was inadmissible (mass removed)
        self.mask_fallback: int = 0    # edge-case events (uniform fallback)
        self.mask_max_removed: float = 0.0  # maximum inadmissible probability
                                       # mass removed in a single draw

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self.mask_calls = 0
        self.mask_effective = 0
        self.mask_fallback = 0
        self.mask_max_removed = 0.0
        missing = [
            sid for sid, sc in stations.items()
            if sc.transitions and sid not in self._distributions
        ]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal transition data for {missing}. "
                f"Simulation cannot start."
            )

    def _prepare_distributions(self, transition_probs: Dict[str, Dict[str, float]]) -> None:
        self._distributions = {}
        for station_id, probs_dict in transition_probs.items():
            targets, probs = normalize_distribution(
                probs_dict, label=f"Route {station_id}",
            )
            self._distributions[station_id] = (targets, probs)

    def predict(
        self,
        station_id: str,
        order: "Order",  # noqa: ARG002 — not conditioned on
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        distribution = self._distributions.get(station_id)
        if distribution is None:
            raise ValueError(
                f"Fail fast: no station-marginal transition data for "
                f"station '{station_id}'."
            )

        targets, weights = distribution

        if not self._apply_mask or available_targets is None:
            result = str(weighted_draw(targets, weights, self._rng))
            return None if result == END_TOKEN else result

        result = self._draw_masked(station_id, targets, weights, available_targets)
        return None if result == END_TOKEN else result

    def distribution_params(
        self,
        station_id: str,
        order: "Order",  # noqa: ARG002 — not conditioned on
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed categorical distribution (post-mask, renormalized)."""
        targets, weights = self._distributions[station_id]
        mask = available_targets if self._apply_mask else None
        return {"family": "categorical",
                "probs": masked_categorical_probs(targets, weights, mask)}

    def _draw_masked(
        self,
        station_id: str,
        targets: list[str],
        weights: np.ndarray,
        available_targets: Set[str],
    ) -> str:
        """One RNG draw per call, as in the unmasked path, so RNG consumption matches."""
        self.mask_calls += 1
        result, removed_mass, used_fallback, kept_mass, admissible = masked_categorical_draw(
            self._rng, targets, weights, available_targets, label=station_id,
        )
        if removed_mass > 0.0:
            self.mask_effective += 1
            if removed_mass > self.mask_max_removed:
                self.mask_max_removed = removed_mass
        if used_fallback:
            self.mask_fallback += 1
            _logger.warning(
                "RefTransition uniform-fallback @ station '%s': fitted mass "
                "on all admissible targets = 0 (admissible=%s).",
                station_id, sorted(available_targets),
            )
        return result


class RefTransitionVariant(TransitionStrategy):
    """Per-(station, variant) router with cell fallback and admissibility mask.

    Resolution: (station, full variant) → (station, product type) → station-marginal,
    counted in n_variant/n_producttype/n_station. One RNG draw per call.
    """

    def __init__(
        self,
        station_marginal_probs: Dict[str, Dict[str, float]],
        by_producttype: Dict[str, Dict[str, Dict[str, float]]],
        by_variant: Dict[str, Dict[str, Dict[str, float]]],
        rng: np.random.Generator,
        *,
        apply_admissibility_mask: bool = False,
    ) -> None:
        if not station_marginal_probs:
            raise ValueError("Fail fast: station_marginal_probs must not be empty.")
        self._rng = rng
        self._apply_mask = bool(apply_admissibility_mask)
        self._marg = self._prep_flat(station_marginal_probs)
        self._byp = self._prep_nested(by_producttype)
        self._byv = self._prep_nested(by_variant)

        self.n_variant: int = 0
        self.n_producttype: int = 0
        self.n_station: int = 0
        self.mask_calls: int = 0
        self.mask_effective: int = 0
        self.mask_fallback: int = 0
        self.mask_max_removed: float = 0.0

    @staticmethod
    def _prep_flat(
        probs: Dict[str, Dict[str, float]],
    ) -> Dict[str, tuple[list[str], np.ndarray]]:
        return {
            sid: normalize_distribution(pd, label=f"Route {sid}")
            for sid, pd in probs.items()
        }

    @staticmethod
    def _prep_nested(
        probs: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, tuple[list[str], np.ndarray]]]:
        out: Dict[str, Dict[str, tuple[list[str], np.ndarray]]] = {}
        for sid, by_key in probs.items():
            out[sid] = {
                key: normalize_distribution(pd, label=f"Route {sid}:{key}")
                for key, pd in by_key.items()
            }
        return out

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self.n_variant = self.n_producttype = self.n_station = 0
        self.mask_calls = self.mask_effective = self.mask_fallback = 0
        self.mask_max_removed = 0.0
        missing = [
            sid for sid, sc in stations.items()
            if sc.transitions and sid not in self._marg
        ]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal transition data for {missing}. "
                f"Simulation cannot start."
            )

    def predict(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        dist, level = self._resolve_dist(station_id, order)
        if level == "variant":
            self.n_variant += 1
        elif level == "producttype":
            self.n_producttype += 1
        else:
            self.n_station += 1

        targets, weights = dist

        if not self._apply_mask or available_targets is None:
            result = str(weighted_draw(targets, weights, self._rng))
            return None if result == END_TOKEN else result

        self.mask_calls += 1
        result, removed_mass, used_fallback, _, _ = masked_categorical_draw(
            self._rng, targets, weights, available_targets, label=station_id,
        )
        if removed_mass > 0.0:
            self.mask_effective += 1
            if removed_mass > self.mask_max_removed:
                self.mask_max_removed = removed_mass
        if used_fallback:
            self.mask_fallback += 1
            _logger.warning(
                "RefTransitionVariant uniform-fallback @ station '%s': fitted "
                "mass on all admissible targets = 0 (admissible=%s).",
                station_id, sorted(available_targets),
            )
        return None if result == END_TOKEN else result

    def _resolve_dist(self, station_id: str, order: "Order"):
        """Resolved ((targets, weights), hierarchy level); the single resolution path."""
        feats = order.features
        dist = self._byv.get(station_id, {}).get(full_variant_key(feats))
        if dist is not None:
            return dist, "variant"
        dist = self._byp.get(station_id, {}).get(product_type_key(feats))
        if dist is not None:
            return dist, "producttype"
        dist = self._marg.get(station_id)
        if dist is None:
            raise ValueError(
                f"Fail fast: no transition data for station '{station_id}'."
            )
        return dist, "station"

    def distribution_params(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed categorical distribution (resolved cell, post-mask)."""
        (targets, weights), _ = self._resolve_dist(station_id, order)
        mask = available_targets if self._apply_mask else None
        return {"family": "categorical",
                "probs": masked_categorical_probs(targets, weights, mask)}
