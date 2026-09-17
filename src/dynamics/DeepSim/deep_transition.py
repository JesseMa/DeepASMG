"""
Transition strategy based on a neural network.

K-slot history per from_station (autoregressive online; teacher-forced offline
in `prepare_transition_data.py`). K separate slots (modell + target per slot)
resolve deterministic counter patterns such as the InputBuffer cycle. K is
stored in metadata.json.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from src.dynamics.foundation_dynamics import (
    TransitionStrategy, set_onehot, NONE_TOKEN, END_TOKEN, load_deep_model,
    compile_offsets, infer_single,
)

if TYPE_CHECKING:
    from src.config.schema import StationConfig
    from src.simulation.order import Order


class DeepTransition(TransitionStrategy):
    """Transitions from a trained classification NN.

    Model and metadata load on first use (initialize() or predict()), so the
    strategy can be constructed before the model exists.
    """

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("Fail fast: temperature must be > 0.")

        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._rng = rng
        self._temperature = temperature

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

        # Slot 0 = most recent entry (appendleft), slot K-1 = oldest.
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
        available_targets: Optional[Set[str]] = None,
    ) -> np.ndarray:
        """Classes outside available_targets are masked to -inf before the softmax."""
        import torch

        x = self._encode_single(station_id, order)
        logits = infer_single(self._model, x)

        if self._temperature != 1.0:
            logits = logits / self._temperature

        if available_targets is not None:
            # The mask depends only on the target set, which the engine builds
            # once per station, so it is cached rather than rebuilt per event.
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
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Optional[str]:
        """Sample the next station; None if "End" was sampled. Order: encode with
        the CURRENT buffer, predict, then append (modell, chosen_station)."""
        self._ensure_loaded()

        # Read-only instrumentation: no RNG draw, no control-flow effect.
        if available_targets is not None:
            self._sg_mask_calls = getattr(self, "_sg_mask_calls", 0) + 1
            if not all(name in available_targets for name in self._class_names):
                self._sg_mask_effective = getattr(self, "_sg_mask_effective", 0) + 1

        probs = self._compute_probs(station_id, order, available_targets)
        chosen_idx = self._rng.choice(self._num_classes, p=probs)
        chosen_station = self._class_names[chosen_idx]

        modell = order.features.get("modell", self._none_token)
        self._get_or_init_buffer(station_id).appendleft((modell, chosen_station))

        if chosen_station == self._end_token:
            return None
        return chosen_station

    def predict_proba(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Full probability distribution; does NOT update the slot buffer."""
        self._ensure_loaded()

        probs = self._compute_probs(station_id, order, available_targets)
        return {
            name: float(prob)
            for name, prob in zip(self._class_names, probs, strict=True)
        }

    def distribution_params(
        self,
        station_id: str,
        order: "Order",
        available_targets: Optional[Set[str]] = None,
        current_time: float = 0.0,  # noqa: ARG002
    ) -> Dict[str, object]:
        """Categorical distribution over admissible targets; no buffer update."""
        proba = self.predict_proba(station_id, order, available_targets)
        if available_targets is not None:
            proba = {k: v for k, v in proba.items() if k in available_targets}
        total = sum(proba.values())
        if total > 0:
            proba = {k: v / total for k, v in proba.items()}
        return {"family": "categorical", "probs": proba}
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
