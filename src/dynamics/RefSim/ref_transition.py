"""Statistical routing strategies: station-marginal and variant-conditioned."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, normalize_distribution, masked_categorical_draw,
    masked_categorical_probs, END_TOKEN,
)
from src.config.routing_keys import full_variant_key, product_type_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order

_logger = logging.getLogger(__name__)


class RefTransition(TransitionStrategy):
    """Marginal router: P(target | station); order.features are ignored.

    With apply_admissibility_mask=False (the default) the fitted distribution is
    drawn unmasked even when available_targets is supplied.
    """

    def __init__(
        self,
        transition_probs: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        *,
        apply_admissibility_mask: bool = False,
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
        self.predict_calls: int = 0
        self.unmasked_calls: int = 0
        self._mask_decision_points: Dict[str, Dict[str, object]] = {}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self.mask_calls = 0
        self.mask_effective = 0
        self.mask_fallback = 0
        self.mask_max_removed = 0.0
        self.predict_calls = 0
        self.unmasked_calls = 0
        self._mask_decision_points = {}
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
        self.predict_calls += 1
        distribution = self._distributions.get(station_id)
        if distribution is None:
            raise ValueError(
                f"Fail fast: no station-marginal transition data for "
                f"station '{station_id}'."
            )

        targets, weights = distribution

        if not self._apply_mask or available_targets is None:
            self.unmasked_calls += 1
            result = str(self._rng.choice(targets, p=weights))
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
        pre_nonzero = int(np.count_nonzero(weights > 0.0))
        result, removed_mass, used_fallback, kept_mass, admissible = masked_categorical_draw(
            self._rng, targets, weights, available_targets, label=station_id,
        )
        post_nonzero = (
            len(available_targets)
            if used_fallback
            else int(np.count_nonzero((weights > 0.0) & admissible))
        )
        selected_is_admissible = result in available_targets
        renormalized = removed_mass > 0.0 and not used_fallback
        self._record_mask_decision(
            station_id=station_id,
            admissible_target_count=len(available_targets),
            pre_mask_nonzero_target_count=pre_nonzero,
            post_mask_nonzero_target_count=post_nonzero,
            probability_mass_removed=removed_mass,
            renormalized=renormalized,
            used_fallback=used_fallback,
            selected_is_admissible=selected_is_admissible,
            kept_mass=kept_mass,
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

    def _record_mask_decision(
        self,
        *,
        station_id: str,
        admissible_target_count: int,
        pre_mask_nonzero_target_count: int,
        post_mask_nonzero_target_count: int,
        probability_mass_removed: float,
        renormalized: bool,
        used_fallback: bool,
        selected_is_admissible: bool,
        kept_mass: float,
    ) -> None:
        row = self._mask_decision_points.setdefault(
            station_id,
            {
                "decision_point_identifier": station_id,
                "stationary_decision_count": 0,
                "admissible_target_count_min": admissible_target_count,
                "admissible_target_count_max": admissible_target_count,
                "pre_mask_nonzero_target_count_min": pre_mask_nonzero_target_count,
                "pre_mask_nonzero_target_count_max": pre_mask_nonzero_target_count,
                "post_mask_nonzero_target_count_min": post_mask_nonzero_target_count,
                "post_mask_nonzero_target_count_max": post_mask_nonzero_target_count,
                "probability_mass_removed_sum": 0.0,
                "probability_mass_removed_max": 0.0,
                "kept_probability_mass_min": kept_mass,
                "renormalization_count": 0,
                "zero_mass_fallback_count": 0,
                "selected_target_admissible_count": 0,
                "invalid_selected_target_count_after_masking": 0,
            },
        )
        row["stationary_decision_count"] = int(row["stationary_decision_count"]) + 1
        for prefix, value in (
            ("admissible_target_count", admissible_target_count),
            ("pre_mask_nonzero_target_count", pre_mask_nonzero_target_count),
            ("post_mask_nonzero_target_count", post_mask_nonzero_target_count),
        ):
            row[f"{prefix}_min"] = min(int(row[f"{prefix}_min"]), value)
            row[f"{prefix}_max"] = max(int(row[f"{prefix}_max"]), value)
        row["probability_mass_removed_sum"] = (
            float(row["probability_mass_removed_sum"]) + probability_mass_removed
        )
        row["probability_mass_removed_max"] = max(
            float(row["probability_mass_removed_max"]), probability_mass_removed,
        )
        row["kept_probability_mass_min"] = min(
            float(row["kept_probability_mass_min"]), kept_mass,
        )
        row["renormalization_count"] = int(row["renormalization_count"]) + int(renormalized)
        row["zero_mass_fallback_count"] = int(row["zero_mass_fallback_count"]) + int(used_fallback)
        row["selected_target_admissible_count"] = (
            int(row["selected_target_admissible_count"]) + int(selected_is_admissible)
        )
        row["invalid_selected_target_count_after_masking"] = (
            int(row["invalid_selected_target_count_after_masking"])
            + int(not selected_is_admissible)
        )

    def mask_audit_rows(self) -> list[Dict[str, object]]:
        """Return deterministic per-decision-point aggregates for this run."""
        return [dict(self._mask_decision_points[key]) for key in sorted(self._mask_decision_points)]

    def mask_audit_summary(self) -> Dict[str, object]:
        """Run-level mask counters; recorded per run by the ablation runner."""
        invalid = sum(
            int(row["invalid_selected_target_count_after_masking"])
            for row in self._mask_decision_points.values()
        )
        renormalizations = sum(
            int(row["renormalization_count"])
            for row in self._mask_decision_points.values()
        )
        return {
            "mask_active": self._apply_mask,
            "stationary_predict_count": self.predict_calls,
            "mask_call_count": self.mask_calls,
            "unmasked_call_count": self.unmasked_calls,
            "mask_effective_count": self.mask_effective,
            "renormalization_count": renormalizations,
            "zero_mass_fallback_count": self.mask_fallback,
            "invalid_selected_target_count_after_masking": invalid,
            "maximum_probability_mass_removed": self.mask_max_removed,
            "all_stationary_calls_masked": (
                self._apply_mask
                and self.predict_calls == self.mask_calls
                and self.unmasked_calls == 0
            ),
        }


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
            result = str(self._rng.choice(targets, p=weights))
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
