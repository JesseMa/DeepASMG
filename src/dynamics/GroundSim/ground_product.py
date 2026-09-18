"""Released-order attributes: which order is created, not when."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.dynamics.foundation_dynamics import (
    NAMED_PERIODS_HOURS,
    ProductStrategy,
    TWO_PI,
    normalize_distribution,
    weighted_draw,
)

# Unconditional: (values, probs)
# Conditional:   {modell_key: (values, probs)}
_FlatDist = Tuple[List[str], np.ndarray]
_CondDist = Dict[str, _FlatDist]
_Modulation = Tuple[float, float, np.ndarray]  # (period_s, amplitude, phase_offsets)


def _is_conditional(feature_config: Dict) -> bool:
    return any(isinstance(v, dict) for v in feature_config.values())


class GroundProduct(ProductStrategy):

    def __init__(self, rng: np.random.Generator, start_timestamp: int = 0) -> None:
        self._rng = rng
        self._start_timestamp = start_timestamp

        self._feature_names: List[str] = []

        self._flat_dists: Dict[str, _FlatDist] = {}
        self._cond_dists: Dict[str, _CondDist] = {}

        self._is_cond: Dict[str, bool] = {}

        self._modulations: Dict[str, _Modulation] = {}

        self._markov_alphas: Dict[str, float] = {}
        self._last_features: Dict[str, str] = {}

    def initialize(
        self,
        product_features: Dict[str, Any],
        temporal_modulation: Optional[Dict[str, Dict[str, Any]]] = None,
        markov_alphas: Optional[Dict[str, float]] = None,
    ) -> None:
        self._feature_names = list(product_features.keys())
        self._flat_dists.clear()
        self._cond_dists.clear()
        self._is_cond.clear()
        self._modulations.clear()
        self._markov_alphas.clear()
        self._last_features.clear()

        if markov_alphas:
            for feat_name, alpha in markov_alphas.items():
                alpha_f = float(alpha)
                if not (0.0 <= alpha_f < 1.0):
                    raise ValueError(
                        f"Fail fast: markov_alpha for '{feat_name}' must be in [0, 1) "
                        f"(received: {alpha_f})."
                    )
                if alpha_f > 0.0:
                    self._markov_alphas[feat_name] = alpha_f

        if self._feature_names and self._feature_names[0] != "modell":
            if "modell" in self._feature_names:
                self._feature_names.remove("modell")
                self._feature_names.insert(0, "modell")
            else:
                raise ValueError(
                    "Fail fast: product_features must contain 'modell' as a feature. "
                    "Conditional distributions condition on 'modell'."
                )

        for feat_name in self._feature_names:
            feat_config = product_features[feat_name]

            if _is_conditional(feat_config):
                self._init_conditional_feature(feat_name, feat_config)
            else:
                self._init_flat_feature(feat_name, feat_config)

        if temporal_modulation:
            self._compile_modulations(temporal_modulation)

    def _init_flat_feature(self, feat_name: str, config: Dict[str, float]) -> None:
        values, probs = normalize_distribution(config, label=feat_name)
        self._flat_dists[feat_name] = (values, probs)
        self._is_cond[feat_name] = False

    def _init_conditional_feature(self, feat_name: str, config: Dict[str, Dict[str, float]]) -> None:
        if not config:
            raise ValueError(f"Fail fast: feature '{feat_name}' has no conditional distributions.")

        non_dict = [k for k, v in config.items() if not isinstance(v, dict)]
        if non_dict:
            raise ValueError(
                f"Fail fast: feature '{feat_name}' has mixed configuration "
                f"(both numbers and dicts). Define it either flat or conditional. "
                f"Non-dict keys: {non_dict}"
            )

        cond_dist: _CondDist = {}
        for model_key, sub_config in config.items():
            values, probs = normalize_distribution(
                sub_config, label=f"{feat_name}|{model_key}",
            )
            cond_dist[model_key] = (values, probs)

        if "modell" in self._flat_dists:
            modell_values, _ = self._flat_dists["modell"]
            missing_keys = [k for k in modell_values if k not in cond_dist]
            if missing_keys:
                raise ValueError(
                    f"Fail fast: feature '{feat_name}' has no conditional distributions "
                    f"for model keys {missing_keys}. All model keys must be covered."
                )

        self._cond_dists[feat_name] = cond_dist
        self._is_cond[feat_name] = True

    def _compile_modulations(self, temporal_modulation: Dict[str, Dict[str, Any]]) -> None:
        for feat_name, mod_config in temporal_modulation.items():
            if feat_name not in self._is_cond:
                raise ValueError(
                    f"Fail fast: temporal_modulation references unknown "
                    f"feature '{feat_name}'."
                )

            period_s = self._resolve_period(feat_name, mod_config)
            amplitude = float(mod_config.get("amplitude", 0.0))

            if not (0.0 <= amplitude <= 1.0):
                raise ValueError(
                    f"Fail fast: amplitude must be in [0, 1] for '{feat_name}' "
                    f"(amplitude={amplitude})."
                )

            if amplitude == 0.0:
                continue

            if self._is_cond[feat_name]:
                cond_dist = self._cond_dists[feat_name]
                first_values, _ = next(iter(cond_dist.values()))
                n_values = len(first_values)
            else:
                values, _ = self._flat_dists[feat_name]
                n_values = len(values)

            phase_offsets = np.arange(n_values, dtype=np.float64) * (TWO_PI / n_values)
            self._modulations[feat_name] = (period_s, amplitude, phase_offsets)

    @staticmethod
    def _resolve_period(feat_name: str, mod_config: Dict[str, Any]) -> float:
        period = mod_config.get("period", "day")
        if isinstance(period, str):
            if period not in NAMED_PERIODS_HOURS:
                raise ValueError(
                    f"Fail fast: unknown period '{period}' for '{feat_name}'. "
                    f"Allowed: {list(NAMED_PERIODS_HOURS.keys())} or a number in hours."
                )
            period_hours = NAMED_PERIODS_HOURS[period]
        else:
            period_hours = float(period)

        if period_hours <= 0:
            raise ValueError(f"Fail fast: period must be > 0 for '{feat_name}'.")

        return period_hours * 3600.0

    def sample_features(self, current_time: float = 0.0) -> Dict[str, str]:
        features: Dict[str, str] = {}

        absolute_time = (
            float(self._start_timestamp) + current_time
            if self._modulations else 0.0
        )

        sampled_model: Optional[str] = None
        prev_modell = self._last_features.get("modell")

        for feat_name in self._feature_names:
            if self._is_cond[feat_name]:
                value = self._sample_conditional(
                    feat_name, sampled_model, absolute_time, prev_modell,
                )
            else:
                value = self._sample_flat(feat_name, absolute_time)

            features[feat_name] = value

            if feat_name == "modell":
                sampled_model = value

        self._last_features = dict(features)
        return features

    _HEAD_NAME = {"modell": "type"}

    def distribution_params(
        self,
        current_time: float = 0.0,
        realized_features: Optional[Dict[str, str]] = None,
        prev_features: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        realized_features = realized_features or {}
        prev = prev_features or {}
        abs_time = (float(self._start_timestamp) + current_time) if self._modulations else 0.0
        realized_modell = realized_features.get("modell")
        prev_modell = prev.get("modell")
        out: Dict[str, Dict[str, float]] = {}
        for feat_name in self._feature_names:
            if self._is_cond[feat_name]:
                values, base = self._cond_dists[feat_name][realized_modell]
                modulation = self._modulations.get(feat_name)
                if modulation is None:
                    probs = base
                else:
                    period_s, amplitude, ref_offsets = modulation
                    if len(values) != len(ref_offsets):
                        n = len(values)
                        modulation = (period_s, amplitude,
                                      np.arange(n, dtype=np.float64) * (TWO_PI / n))
                    probs = _modulate_probs(base, abs_time, modulation)
                prev_val = prev.get(feat_name) if prev_modell == realized_modell else None
                probs = self._apply_lazy_walk(feat_name, values, probs, prev_val)
            else:
                values, base = self._flat_dists[feat_name]
                modulation = self._modulations.get(feat_name)
                probs = base if modulation is None else _modulate_probs(base, abs_time, modulation)
                probs = self._apply_lazy_walk(feat_name, values, probs, prev.get(feat_name))
            s = float(np.sum(probs))
            head = self._HEAD_NAME.get(feat_name, feat_name)
            out[head] = {str(v): float(p) / s for v, p in zip(values, probs, strict=True)}
        return out

    def _sample_flat(self, feat_name: str, absolute_time: float) -> str:
        values, base_probs = self._flat_dists[feat_name]
        modulation = self._modulations.get(feat_name)

        if modulation is None:
            probs = base_probs
        else:
            probs = _modulate_probs(base_probs, absolute_time, modulation)

        probs = self._apply_lazy_walk(feat_name, values, probs,
                                      self._last_features.get(feat_name))
        return weighted_draw(values, probs, self._rng)

    def _sample_conditional(
        self,
        feat_name: str,
        sampled_model: Optional[str],
        absolute_time: float,
        prev_modell: Optional[str],
    ) -> str:
        cond_dist = self._cond_dists[feat_name]

        if sampled_model is None or sampled_model not in cond_dist:
            raise KeyError(f"Product configuration incomplete: feature '{feat_name}' "
                        f"has no rules for model '{sampled_model}'.")
        values, base_probs = cond_dist[sampled_model]

        modulation = self._modulations.get(feat_name)

        if modulation is None:
            probs = base_probs
        else:
            period_s, amplitude, ref_offsets = modulation
            if len(values) != len(ref_offsets):
                n = len(values)
                offsets = np.arange(n, dtype=np.float64) * (TWO_PI / n)
                modulation = (period_s, amplitude, offsets)
            probs = _modulate_probs(base_probs, absolute_time, modulation)

        prev_value = (
            self._last_features.get(feat_name)
            if prev_modell == sampled_model
            else None
        )
        probs = self._apply_lazy_walk(feat_name, values, probs, prev_value)
        return weighted_draw(values, probs, self._rng)

    def _apply_lazy_walk(
        self,
        feat_name: str,
        values: List[str],
        probs: np.ndarray,
        prev_value: Optional[str],
    ) -> np.ndarray:
        alpha = self._markov_alphas.get(feat_name, 0.0)
        if alpha <= 0.0 or prev_value is None:
            return probs

        try:
            idx = list(values).index(prev_value)
        except ValueError:
            return probs  # previous value not in current value space (e.g. condition switch)

        mixed = (1.0 - alpha) * probs
        mixed[idx] += alpha
        return mixed


def _modulate_probs(
    base_probs: np.ndarray,
    absolute_time: float,
    modulation: _Modulation,
) -> np.ndarray:
    period_s, amplitude, phase_offsets = modulation

    cycle_phase = TWO_PI * absolute_time / period_s
    shifts = np.sin(cycle_phase + phase_offsets)
    modulated = base_probs * (1.0 + amplitude * shifts)
    np.maximum(modulated, 1e-6, out=modulated)
    modulated /= modulated.sum()
    return modulated