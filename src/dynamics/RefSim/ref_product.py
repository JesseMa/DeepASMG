"""Product strategy: decides WHAT order is created, not when."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.dynamics.foundation_dynamics import ProductStrategy



class RefProduct(ProductStrategy):
    """Features from fitted per-feature marginals; temporal_modulation is ignored."""

    def __init__(
        self,
        product_features: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
    ) -> None:
        if not product_features:
            raise ValueError("RefProduct: product_features must not be empty.")
        
        self._rng = rng
        self._distributions: Dict[str, tuple[list[str], np.ndarray]] = {}

        self._prepare_distributions(product_features)

    def initialize(self, product_features: Dict[str, Any], temporal_modulation: Optional[Dict] = None, markov_alphas: Optional[Dict[str, float]] = None) -> None:
        """No-op: the distributions are fixed in __init__."""

    def _prepare_distributions(self, product_features: Dict[str, Any]) -> None:
        self._distributions = {}

        for feature_name, content in product_features.items():
            if not isinstance(content, dict) or not content:
                raise ValueError(
                    f"Fail fast: feature '{feature_name}' has no valid "
                    f"probability distribution (empty or not a dict)."
                )
                
            keys = list(content.keys())
            probs = np.asarray([float(content[k]) for k in keys], dtype=float)

            if probs.sum() <= 0:
                raise ValueError(
                    f"Fail fast: feature '{feature_name}' has a probability sum "
                    f"<= 0. Simulation cannot be started."
                )

            probs /= probs.sum()
            self._distributions[feature_name] = (keys, probs)

    def sample_features(self, current_time: float = 0.0) -> Dict[str, str]:  # noqa: ARG002
        features: Dict[str, str] = {}
        for feature_name, (values, probs) in self._distributions.items():
            features[feature_name] = self._rng.choice(values, p=probs)
        return features

    _HEAD_NAME = {"modell": "type", "feature_a": "feature_a", "feature_b": "feature_b"}

    def distribution_params(
        self, current_time: float = 0.0,  # noqa: ARG002
        realized_features: Optional[Dict[str, str]] = None,  # noqa: ARG002
        prev_features: Optional[Dict[str, str]] = None,  # noqa: ARG002
    ) -> Dict[str, Dict[str, float]]:
        """Deployed categorical per attribute head; context-free (unconditional marginals)."""
        out: Dict[str, Dict[str, float]] = {}
        for feature_name, (values, probs) in self._distributions.items():
            head = self._HEAD_NAME.get(feature_name, feature_name)
            out[head] = {str(v): float(p) for v, p in zip(values, probs, strict=True)}
        return out