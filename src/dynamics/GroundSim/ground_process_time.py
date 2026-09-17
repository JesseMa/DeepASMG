"""GroundProcessTime — normally distributed process times per station/model."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import ProcessTimeStrategy, detect_shift

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order

_NIGHT_FACTOR  = 1.15


def _night_shift_factor(current_time: float) -> float:
    """Night-shift (22:00-06:00) multiplier, else 1.0."""
    return _NIGHT_FACTOR if detect_shift(current_time) == 2 else 1.0


class GroundProcessTime(ProcessTimeStrategy):
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._station_process_times: Dict[str, Dict[str, tuple[float, float]]] = {}
        self._setup_times: Dict[str, float] = {}

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._station_process_times = {}
        self._setup_times = {}

        for station_id, config in stations.items():
            if not config.process_times:
                continue

            for key, (mean, std) in config.process_times.items():
                if std < 0:
                    raise ValueError(
                        f"Fail fast: standard deviation (std) must not be negative in {station_id}:{key} (std={std})."
                    )
                if mean < 0:
                    raise ValueError(
                        f"Fail fast: mean process time must be >= 0 in {station_id}:{key} (mean={mean})."
                    )

            self._station_process_times[station_id] = config.process_times
            self._setup_times[station_id] = config.setup_time

    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float:
        process_times = self._station_process_times[station_id]
        key = order.resolve_process_key(process_times)

        mean, std = process_times[key]
        factor = _night_shift_factor(current_time)

        draw = self._rng.normal(mean * factor, std * factor)
        self._sg_proc_draws = getattr(self, "_sg_proc_draws", 0) + 1
        if draw < 0.1:
            self._sg_proc_clamp = getattr(self, "_sg_proc_clamp", 0) + 1
        net = max(0.1, draw)
        # Integer time contract: durations are whole seconds, rounded exactly
        # once at the module boundary (the kernel clock ticks in 1-s steps).
        return float(np.ceil(net + self._setup_times.get(station_id, 0.0)))

    def distribution_params(
        self, station_id: str, order: "Order", current_time: float = 0.0,
    ) -> Dict[str, object]:
        """Normal params of the TOTAL process time, setup included.

        predict() returns ceil() of this draw (integer time contract), so these
        parameters label the lattice law the station deploys — the same scale
        DeepSim and RefSim report, which is what makes them comparable without
        a reconciliation step.
        """
        process_times = self._station_process_times[station_id]
        mean, std = process_times[order.resolve_process_key(process_times)]
        factor = _night_shift_factor(current_time)
        mean = mean + self._setup_times.get(station_id, 0.0) / factor
        return {"family": "normal", "mu": float(mean * factor), "sigma": float(std * factor)}
