"""Process-time strategy based on a trained neural network."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import math

import numpy as np

from src.dynamics.foundation_dynamics import (
    ProcessTimeStrategy, set_onehot, NONE_TOKEN,
    detect_shift, encode_time_features, load_deep_model, compile_offsets,
    infer_single,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


class DeepProcessTime(ProcessTimeStrategy):

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        rng: np.random.Generator,
    ) -> None:
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._rng = rng

        self._model = None
        self._metadata = None
        self._encoding_maps = None
        self._feature_dim = None
        self._feature_layout = None
        self._none_token = NONE_TOKEN
        self._offsets: Dict[str, int] = {}
        self._time_periods_seconds: list = []
        self._n_time_features: int = 0

        self._prev_on_machine: Dict[str, Dict[str, str]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._model, self._metadata = load_deep_model(self._model_path, self._metadata_path)

        self._encoding_maps = self._metadata["encoding_maps"]
        self._feature_dim = self._encoding_maps["feature_dim"]
        self._feature_layout = self._metadata["feature_layout"]
        self._none_token = self._metadata.get("none_token", NONE_TOKEN)
        self._time_periods_seconds = self._metadata.get("time_periods_seconds", [])
        self._n_time_features = 2 * len(self._time_periods_seconds)
        self._offsets = compile_offsets(self._feature_layout)

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._prev_on_machine.clear()
        self._ensure_loaded()

    def predict(self, station_id: str, order: "Order", current_time: float = 0.0) -> float:
        self._ensure_loaded()

        x = self._encode_single(station_id, order, current_time)
        output = infer_single(self._model, x)
        pred_mean = output[0].item()
        sigma = math.exp(0.5 * output[1].item())
        sample = self._rng.normal(pred_mean, sigma)
        self._sg_proc_draws = getattr(self, "_sg_proc_draws", 0) + 1
        if sample < 0.1:
            self._sg_proc_clamp = getattr(self, "_sg_proc_clamp", 0) + 1

        self._prev_on_machine[station_id] = dict(order.features)

        return float(np.ceil(max(0.1, float(sample))))

    def distribution_params(
        self, station_id: str, order: "Order", current_time: float = 0.0,
    ) -> Dict[str, object]:
        self._ensure_loaded()
        output = infer_single(self._model, self._encode_single(station_id, order, current_time))
        return {"family": "normal", "mu": float(output[0].item()),
                "sigma": math.exp(0.5 * output[1].item())}

    def _encode_single(self, station_id: str, order: "Order", current_time: float = 0.0) -> np.ndarray:
        x = np.zeros(self._feature_dim, dtype=np.float32)
        none = self._none_token

        maps = self._encoding_maps
        features = order.features

        set_onehot(x, self._offsets["modell"], maps["modell"],
                         features.get("modell", none))
        set_onehot(x, self._offsets["feature_a"], maps["feature_a"],
                         features.get("feature_a", none))
        set_onehot(x, self._offsets["feature_b"], maps["feature_b"],
                         features.get("feature_b", none))

        prev = self._prev_on_machine.get(station_id)
        if prev is None:
            prev_modell, prev_fa, prev_fb = none, none, none
        else:
            prev_modell = prev.get("modell", none)
            prev_fa = prev.get("feature_a", none)
            prev_fb = prev.get("feature_b", none)

        set_onehot(x, self._offsets["prev_modell"], maps["modell"], prev_modell)
        set_onehot(x, self._offsets["prev_feature_a"], maps["feature_a"], prev_fa)
        set_onehot(x, self._offsets["prev_feature_b"], maps["feature_b"], prev_fb)

        set_onehot(x, self._offsets["station"], maps["station"], station_id)

        x[self._offsets["shift"] + detect_shift(current_time)] = 1.0

        if self._time_periods_seconds:
            off_time = self._feature_dim - self._n_time_features
            time_enc = encode_time_features(current_time, self._time_periods_seconds)
            x[off_time:off_time + self._n_time_features] = time_enc

        return x

