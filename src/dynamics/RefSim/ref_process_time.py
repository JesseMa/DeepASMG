"""Statistical process times: N(mean, std) per station, optionally refined per product type and per full variant."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import ProcessTimeStrategy
from src.config.routing_keys import full_variant_key, product_type_key

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order

Cell = Tuple[float, float]


class RefProcessTime(ProcessTimeStrategy):

    def __init__(
        self,
        station_marginal: Dict[str, Cell],
        rng: np.random.Generator,
        *,
        by_producttype: Optional[Dict[str, Dict[str, Cell]]] = None,
        by_variant: Optional[Dict[str, Dict[str, Cell]]] = None,
    ) -> None:
        if not station_marginal:
            raise ValueError("Fail fast: station_marginal must not be empty.")
        for table in (station_marginal, *(by_producttype or {}).values(),
                      *(by_variant or {}).values()):
            for key, (mean, std) in table.items():
                if std < 0 or mean < 0:
                    raise ValueError(
                        f"Fail fast: fitted process time for '{key}' is negative "
                        f"(mean={mean}, std={std})."
                    )
        self._marg = station_marginal
        self._byp = by_producttype or {}
        self._byv = by_variant or {}
        self._rng = rng

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        missing = [sid for sid, sc in stations.items()
                   if sc.process_times and sid not in self._marg]
        if missing:
            raise ValueError(
                f"Fail fast: no station-marginal process times for machines {missing}. "
                f"Simulation cannot start."
            )

    def _resolve(self, station_id: str, order: "Order") -> Cell:
        feats = order.features
        cell = self._byv.get(station_id, {}).get(full_variant_key(feats))
        if cell is None:
            cell = self._byp.get(station_id, {}).get(product_type_key(feats))
        if cell is None:
            cell = self._marg.get(station_id)
        if cell is None:
            raise ValueError(f"Fail fast: no process time for station '{station_id}'.")
        return cell

    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float:  # noqa: ARG002
        mean, std = self._resolve(station_id, order)
        draw = self._rng.normal(mean, std)
        self._sg_proc_draws = getattr(self, "_sg_proc_draws", 0) + 1
        if draw < 0.1:
            self._sg_proc_clamp = getattr(self, "_sg_proc_clamp", 0) + 1
        return float(np.ceil(max(0.1, float(draw))))

    def distribution_params(
        self, station_id: str, order: "Order", current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        mean, std = self._resolve(station_id, order)
        return {"family": "normal", "mu": float(mean), "sigma": float(std)}
