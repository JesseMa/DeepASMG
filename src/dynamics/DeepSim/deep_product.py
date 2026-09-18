"""DeepProduct — NN-based product generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.dynamics.foundation_dynamics import (
    ProductStrategy, NONE_TOKEN, TWO_PI, load_deep_model,
    infer_single,
    weighted_draw,
)


class DeepProduct(ProductStrategy):

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        rng: np.random.Generator,
        start_timestamp: float = 0.0,
    ) -> None:
        self._model_path = Path(model_path)
        self._metadata_path = Path(metadata_path)
        self._rng = rng
        self._start_timestamp = start_timestamp

        self._model = None

        self._feature_names: List[str] = []
        self._encoding_maps: Dict[str, Dict[str, int]] = {}
        self._feature_dim: int = 0
        self._feature_layout: List[Dict[str, Any]] = []
        self._head_layout: List[Dict[str, Any]] = []
        self._time_periods_seconds: List[float] = []
        self._none_token: str = NONE_TOKEN

        # Smoothed class frequencies per feature; None until the first update,
        # encoded as uniform until then.
        self._prev_ewma: Optional[Dict[str, np.ndarray]] = None
        self._ewma_alpha: float = 0.3  # default; replaced by metadata.json
        self._n_classes_map: Dict[str, int] = {}
        self._teacher_forced: bool = False  # shadow inference: external EWMA source

        self._load_metadata()

    def initialize(
        self,
        product_features: Dict[str, Any],
        temporal_modulation: Optional[Dict[str, Any]] = None,
        markov_alphas: Optional[Dict[str, float]] = None,
    ) -> None:
        self._prev_ewma = None

    def sample_features(self, current_time: float = 0.0) -> Dict[str, str]:
        self._ensure_loaded()
        absolute_time = self._start_timestamp + current_time
        features = self._sample_from_model(absolute_time)
        if not self._teacher_forced:
            self._update_ewma(features)
        return features

    def observe_external(self, features: Dict[str, str]) -> None:
        self._ensure_loaded()
        self._update_ewma(features)

    def _load_metadata(self) -> None:
        if not self._metadata_path.exists():
            raise FileNotFoundError(
                f"DeepProduct metadata not found: {self._metadata_path}"
            )

        with open(self._metadata_path) as f:
            meta = json.load(f)

        self._feature_names = meta["feature_names"]
        self._encoding_maps = meta["encoding_maps"]
        self._feature_dim = meta["feature_dim"]
        self._feature_layout = meta.get("feature_layout", [])
        self._head_layout = meta.get("head_layout", [])
        self._time_periods_seconds = meta.get("time_periods_seconds", [])
        self._none_token = meta.get("none_token", NONE_TOKEN)
        self._ewma_alpha = meta.get("ewma_alpha", 0.3)
        # n_classes per feature: len(encoding_map) - 1 (NONE excluded)
        self._n_classes_map = {
            feat: len(enc) - 1
            for feat, enc in self._encoding_maps.items()
            if feat in self._feature_names
        }

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        if not self._head_layout:
            raise RuntimeError(
                f"DeepProduct: no head_layout in metadata: {self._metadata_path}"
            )

        try:
            self._model, _ = load_deep_model(self._model_path, self._metadata_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"DeepProduct: {e}\n"
                f"Alternatively, use GroundProduct for rule-based sampling."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"DeepProduct: failed to load model: {self._model_path}\n"
                f"Error: {e}"
            ) from e

    def _sample_from_model(self, absolute_time: float) -> Dict[str, str]:
        x_base = self._encode_input(absolute_time)
        modell_head = self._head_layout[0]
        if modell_head["feature_name"] != "modell":
            raise RuntimeError(
                f"DeepProduct: expected 'modell' as the first head, "
                f"got: {modell_head['feature_name']}"
            )
        cond_dim = int(modell_head["size"])

        features: Dict[str, str] = {}

        x_pass1 = np.concatenate([x_base, np.zeros(cond_dim, dtype=np.float32)])
        logits_pass1 = infer_single(self._model, x_pass1)
        modell_idx = self._sample_head_idx(
            logits_pass1, modell_head["offset"], cond_dim,
        )
        features["modell"] = self._idx_to_value("modell", modell_idx)

        if len(self._head_layout) == 1:
            return features

        cond_onehot = np.zeros(cond_dim, dtype=np.float32)
        cond_onehot[modell_idx] = 1.0
        x_pass2 = np.concatenate([x_base, cond_onehot])
        logits_pass2 = infer_single(self._model, x_pass2)

        for head in self._head_layout[1:]:
            feat_name: str = head["feature_name"]
            chosen_idx = self._sample_head_idx(
                logits_pass2, head["offset"], head["size"],
            )
            features[feat_name] = self._idx_to_value(feat_name, chosen_idx)

        return features

    def _sample_head_idx(self, logits_flat, offset: int, size: int) -> int:
        probs = self._head_probs(logits_flat, offset, size)
        return int(weighted_draw(len(probs), probs, self._rng))

    def _head_values(self, feat_name: str) -> list:
        enc = self._encoding_maps[feat_name]
        return [v for v, _ in sorted(enc.items(), key=lambda kv: kv[1]) if v != self._none_token]

    def _head_probs(self, logits_flat, offset: int, size: int) -> "np.ndarray":
        import torch
        hl = logits_flat[offset: offset + size]
        p = torch.softmax(hl, dim=0).numpy().astype(np.float64)
        return p / p.sum()

    def distribution_params(
        self, current_time: float = 0.0,
        realized_features: Optional[Dict[str, str]] = None,
        prev_features: Optional[Dict[str, str]] = None,  # noqa: ARG002 — EWMA via observe_external
    ) -> Dict[str, Dict[str, float]]:
        self._ensure_loaded()
        x_base = self._encode_input(self._start_timestamp + current_time)
        mh = self._head_layout[0]
        cond_dim = int(mh["size"])
        out: Dict[str, Dict[str, float]] = {}
        p1 = self._head_probs(
            infer_single(self._model, np.concatenate([x_base, np.zeros(cond_dim, dtype=np.float32)])),
            mh["offset"], cond_dim)
        mvals = self._head_values("modell")
        out["type"] = {str(mvals[i]): float(p1[i]) for i in range(len(p1))}
        if len(self._head_layout) == 1:
            return out
        realized_modell = (realized_features or {}).get("modell")
        midx = mvals.index(realized_modell) if realized_modell in mvals else int(np.argmax(p1))
        cond = np.zeros(cond_dim, dtype=np.float32)
        cond[midx] = 1.0
        logits2 = infer_single(self._model, np.concatenate([x_base, cond]))
        for head in self._head_layout[1:]:
            fn = head["feature_name"]
            pr = self._head_probs(logits2, head["offset"], head["size"])
            vals = self._head_values(fn)
            out[fn] = {str(vals[i]): float(pr[i]) for i in range(len(pr))}
        return out

    def _idx_to_value(self, feat_name: str, idx: int) -> str:
        return self._head_values(feat_name)[idx]

    def _encode_input(self, absolute_time: float) -> np.ndarray:
        x = np.zeros(self._feature_dim, dtype=np.float32)

        for period_s, entry_sin, entry_cos in zip(
            self._time_periods_seconds,
            self._feature_layout[0::2][:len(self._time_periods_seconds)],
            self._feature_layout[1::2][:len(self._time_periods_seconds)],
            strict=True,
        ):
            if period_s <= 0:
                continue
            phase = TWO_PI * absolute_time / period_s
            x[entry_sin["offset"]] = np.sin(phase)
            x[entry_cos["offset"]] = np.cos(phase)

        for entry in self._feature_layout:
            name: str = entry["name"]
            if not name.startswith("prev_"):
                continue

            feat_name = name[5:]  # "prev_modell" → "modell"
            offset: int = entry["offset"]
            size: int = entry["size"]

            if self._prev_ewma is None or feat_name not in self._prev_ewma:
                if size > 0:
                    x[offset:offset + size] = 1.0 / size
            else:
                x[offset:offset + size] = self._prev_ewma[feat_name]

        return x

    def _update_ewma(self, features: Dict[str, str]) -> None:
        if self._prev_ewma is None:
            self._prev_ewma = {
                feat_name: np.full(nc, 1.0 / nc, dtype=np.float32)
                for feat_name, nc in self._n_classes_map.items()
                if nc > 0
            }

        for feat_name, value in features.items():
            if feat_name not in self._prev_ewma:
                continue
            enc = self._encoding_maps.get(feat_name)
            if enc is None:
                continue
            val_idx = enc.get(value, 0) - 1  # NONE has index 0 → val_idx = -1 = invalid
            if val_idx < 0:
                continue
            freq = self._prev_ewma[feat_name]
            freq *= (1.0 - self._ewma_alpha)
            freq[val_idx] += self._ewma_alpha

