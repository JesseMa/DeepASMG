"""Routing-key resolution."""

from __future__ import annotations

from typing import Any, Dict, List

_MISSING = object()


def full_variant_key(features: Dict[str, str]) -> str:
    if "modell" not in features:
        raise KeyError(
            f"Feature 'modell' missing in features={features}. "
            f"Every order must have a 'modell' feature."
        )
    return (
        f"{features['modell']}_"
        f"{features.get('feature_a', '')}_"
        f"{features.get('feature_b', '')}"
    )


def product_type_key(features: Dict[str, str]) -> str:
    if "modell" not in features:
        raise KeyError(
            f"Feature 'modell' missing in features={features}. "
            f"Every order must have a 'modell' feature."
        )
    return features["modell"]


def build_candidate_keys(
    features: Dict[str, str], *, wildcard: bool, visit: int | None = None,
) -> List[str]:
    if "modell" not in features:
        raise KeyError(
            f"Feature 'modell' missing in features={features}. "
            f"Every order must have a 'modell' feature."
        )
    candidates: List[str] = []
    if visit is not None:
        candidates.append(f"visit_{visit}")
    model = features["modell"]
    fa = features.get("feature_a", "")
    fb = features.get("feature_b", "")
    if fa and fb:
        candidates.append(f"{model}_{fa}_{fb}")
    if fa:
        candidates.append(f"{model}_{fa}")
    if fb:
        candidates.append(f"{model}_{fb}")
    candidates.append(model)
    if wildcard:
        candidates.append("*")
    return candidates


def resolve_key(
    features: Dict[str, str],
    lookup: dict,
    *,
    wildcard: bool,
    default: Any = _MISSING,
    visit: int | None = None,
) -> Any:
    for key in build_candidate_keys(features, wildcard=wildcard, visit=visit):
        if key in lookup:
            return key
    if default is _MISSING:
        raise KeyError(
            f"No matching key for features={features} "
            f"in {list(lookup.keys())}"
        )
    return default


def station_admissible_targets(station_config) -> set:
    targets: set = set()
    for transition_map in station_config.transitions.values():
        if transition_map:
            targets.update(transition_map.keys())
    for rule in station_config.sequential_routing.values():
        targets.update(rule.get("targets", []))
    return targets
