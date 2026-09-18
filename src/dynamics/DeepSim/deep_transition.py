"""Transition strategy based on a neural network."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, set_onehot, NONE_TOKEN, END_TOKEN, load_deep_model,
    compile_offsets, infer_single, weighted_draw, visit_token,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


class DeepTransition(TransitionStrategy):

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
        self._num_classes = None
        self._class_names = None
        self._end_token = END_TOKEN
        self._none_token = NONE_TOKEN
        self._offsets: Dict[str, int] = {}
        self._n_hist_slots: int = 0

        self._mask_cache: Dict[frozenset, Any] = {}

        self._slot_buffers: Dict[str, Deque[Tuple[str, str]]] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._model, self._metadata = load_deep_model(self._model_path, self._metadata_path)

        self._encoding_maps = self._metadata["encoding_maps"]
        self._feature_dim = self._encoding_maps["feature_dim"]
        self._feature_layout = self._metadata["feature_layout"]
        self._num_classes = self._metadata["num_classes"]
        self._class_names = self._metadata["class_names"]
        self._end_token = self._metadata.get("end_token", END_TOKEN)
        self._none_token = self._metadata.get("none_token", NONE_TOKEN)
        self._offsets = compile_offsets(self._feature_layout)
        if "visit" not in self._offsets:
            raise ValueError(
                "Fail fast: this transition model predates the visit-index "
                "feature (integer time contract / repeat-visit routing). Its "
                "input layout has no 'visit' block, so it cannot represent a "
                "repeat-visit rule. Retrain with the current preparation."
            )

        if "n_hist_slots" not in self._metadata:
            raise ValueError(
                "Incompatible model: metadata.json has no 'n_hist_slots' "
                "(K-slot history), so its feature layout does not match this "
                "class. Retrain with prepare_transition_data.py + train_models.py."
            )
        self._n_hist_slots = int(self._metadata["n_hist_slots"])

        if len(self._class_names) != self._num_classes:
            raise ValueError(
                "Inconsistent metadata: num_classes does not match class_names."
            )

    def initialize(self, stations: Dict[str, "StationConfig"]) -> None:
        self._slot_buffers.clear()
        self._ensure_loaded()

    def _get_or_init_buffer(self, station_id: str) -> Deque[Tuple[str, str]]:
        buf = self._slot_buffers.get(station_id)
        if buf is None:
            buf = deque(
                [(self._none_token, self._none_token)] * self._n_hist_slots,
                maxlen=self._n_hist_slots,
            )
            self._slot_buffers[station_id] = buf
        return buf

    def _compute_probs(
        self,
        station_id: str,
        order: "Order",
        available_targets: Set[str],
    ) -> np.ndarray:
        import torch

        x = self._encode_single(station_id, order)
        logits = infer_single(self._model, x)

        key = frozenset(available_targets)
        inverse = self._mask_cache.get(key)
        if inverse is None:
            mask = torch.tensor(
                [name in available_targets for name in self._class_names],
                dtype=torch.bool,
            )
            if not mask.any():
                raise ValueError(
                    f"Routing mask empty for station '{station_id}': "
                    f"no class_names ∈ available_targets={available_targets}."
                )
            inverse = ~mask
            self._mask_cache[key] = inverse
        logits = logits.masked_fill(inverse, float("-inf"))

        probs = torch.softmax(logits, dim=0).numpy()
        return probs / probs.sum()

    def predict(
        self,
        station_id: str,
        order: "Order",
        available_targets: Set[str],
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        self._ensure_loaded()

        self._sg_mask_calls = getattr(self, "_sg_mask_calls", 0) + 1
        if not all(name in available_targets for name in self._class_names):
            self._sg_mask_effective = getattr(self, "_sg_mask_effective", 0) + 1

        probs = self._compute_probs(station_id, order, available_targets)
        chosen_idx = weighted_draw(self._num_classes, probs, self._rng)
        chosen_station = self._class_names[chosen_idx]

        modell = order.features.get("modell", self._none_token)
        self._get_or_init_buffer(station_id).appendleft((modell, chosen_station))

        if chosen_station == self._end_token:
            return None
        return chosen_station

    def distribution_params(
        self,
        station_id: str,
        order: "Order",
        available_targets: Set[str],
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        self._ensure_loaded()
        probs = self._compute_probs(station_id, order, available_targets)
        return {"family": "categorical",
                "probs": {name: float(p) for name, p in zip(self._class_names, probs, strict=True)
                          if name in available_targets}}

    def _encode_single(self, station_id: str, order: "Order") -> np.ndarray:
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
        set_onehot(x, self._offsets["from_station"], maps["from_station"],
                   station_id)
        set_onehot(x, self._offsets["visit"], maps["visit"],
                   visit_token(order.visits.get(station_id, 1)))

        buf = self._slot_buffers.get(station_id)
        if buf is None:
            for k in range(self._n_hist_slots):
                set_onehot(x, self._offsets[f"hist_modell_{k}"],
                           maps["modell"], none)
                set_onehot(x, self._offsets[f"hist_target_{k}"],
                           maps["slot_target"], none)
        else:
            for k, (m, t) in enumerate(buf):
                set_onehot(x, self._offsets[f"hist_modell_{k}"],
                           maps["modell"], m)
                set_onehot(x, self._offsets[f"hist_target_{k}"],
                           maps["slot_target"], t)

        return x
