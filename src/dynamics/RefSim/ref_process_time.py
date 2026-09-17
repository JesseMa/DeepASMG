"""
Statistical process-time strategy (station-marginal, naive baseline).

Uses per-station aggregated (mean, std), marginalized over model/feature;
unseen stations fail fast. Distribution family Normal(mean, std) — family
match with GroundSim.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import ProcessTimeStrategy
from src.config.routing_keys import full_variant_key, product_type_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


class RefProcessTime(ProcessTimeStrategy):
    """Station-marginal process times, N(mean, std) floored at 0.1 s."""

    def __init__(
        self,
        process_times: Dict[str, tuple[float, float]],
        rng: np.random.Generator,
    ) -> None:
        if not process_times:
            raise ValueError("Fail fast: process_times must not be empty.")

        for key, (mean, std) in process_times.items():
            if std < 0:
                raise ValueError(f"Fail fast: extracted standard deviation for '{key}' is negative (std={std}).")
            if mean < 0:
                raise ValueError(f"Fail fast: extracted mean for '{key}' is negative (mean={mean}).")

        self._process_times = process_times
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = [
            sid for sid, sc in stations.items()
            if sc.process_times and sid not in self._process_times
        ]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal process times for machines {missing}. "
                f"Simulation cannot start."
            )

    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float:  # noqa: ARG002
        if station_id not in self._process_times:
            raise ValueError(
                f"Fail fast: no process time available for station '{station_id}'."
            )

        mean, std = self._process_times[station_id]
        draw = self._rng.normal(mean, std)
        self._sg_proc_draws = getattr(self, "_sg_proc_draws", 0) + 1
        if draw < 0.1:
            self._sg_proc_clamp = getattr(self, "_sg_proc_clamp", 0) + 1
        return float(np.ceil(max(0.1, float(draw))))  # integer time contract

    def distribution_params(
        self, station_id: str, order: "Order", current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed Normal params on the total scale (the fitted values include
        setup time); the 0.1 s floor applied in predict() is not represented.
        """
        mean, std = self._process_times[station_id]
        return {"family": "normal", "mu": float(mean), "sigma": float(std)}


class RefProcessTimeVariant(ProcessTimeStrategy):
    """Per-(station, variant) process times with fallback.

    Resolution: (station, full variant) → (station, product type) → station-marginal,
    counted in n_variant/n_producttype/n_station. One RNG draw per call.
    """

    def __init__(
        self,
        station_marginal: Dict[str, tuple[float, float]],
        by_producttype: Dict[str, Dict[str, tuple[float, float]]],
        by_variant: Dict[str, Dict[str, tuple[float, float]]],
        rng: np.random.Generator,
    ) -> None:
        if not station_marginal:
            raise ValueError("Fail fast: station_marginal must not be empty.")
        self._marg = station_marginal
        self._byp = by_producttype
        self._byv = by_variant
        self._rng = rng
        self.n_variant: int = 0
        self.n_producttype: int = 0
        self.n_station: int = 0

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self.n_variant = self.n_producttype = self.n_station = 0
        missing = [
            sid for sid, sc in stations.items()
            if sc.process_times and sid not in self._marg
        ]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal process times for machines {missing}. "
                f"Simulation cannot start."
            )

    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float:  # noqa: ARG002
        cell, level = self._resolve_cell(station_id, order)
        if level == "variant":
            self.n_variant += 1
        elif level == "producttype":
            self.n_producttype += 1
        else:
            self.n_station += 1
        mean, std = cell
        draw = self._rng.normal(mean, std)
        self._sg_proc_draws = getattr(self, "_sg_proc_draws", 0) + 1
        if draw < 0.1:
            self._sg_proc_clamp = getattr(self, "_sg_proc_clamp", 0) + 1
        return float(np.ceil(max(0.1, float(draw))))  # integer time contract

    def _resolve_cell(self, station_id: str, order: "Order"):
        """Resolved ((mean, std), hierarchy level); the single resolution path."""
        feats = order.features
        cell = self._byv.get(station_id, {}).get(full_variant_key(feats))
        if cell is not None:
            return cell, "variant"
        cell = self._byp.get(station_id, {}).get(product_type_key(feats))
        if cell is not None:
            return cell, "producttype"
        cell = self._marg.get(station_id)
        if cell is None:
            raise ValueError(f"Fail fast: no process time for station '{station_id}'.")
        return cell, "station"

    def distribution_params(
        self, station_id: str, order: "Order", current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Deployed Normal params of the resolved (station, variant) cell."""
        (mean, std), _ = self._resolve_cell(station_id, order)
        return {"family": "normal", "mu": float(mean), "sigma": float(std)}
