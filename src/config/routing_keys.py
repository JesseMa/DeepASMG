"""Routing-key resolution.

Order features resolve against transitions/process_times dicts in fixed
specificity order: "visit_{n}" when a visit index is supplied, then
"{model}_{fa}_{fb}", "{model}_{fa}", "{model}_{fb}", "{model}", and finally
"*" if wildcards are enabled.

A station that routes differently on a repeat visit declares that as a
"visit_{n}" entry in its transition table, so the rule stays a lookup like
every other and remains expressible as a categorical distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List

_MISSING = object()


def full_variant_key(features: Dict[str, str]) -> str:
    """Variant key "modell_feature_a_feature_b", missing parts become empty strings."""
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
    """Product type = modell (coarser fallback level between variant and station)."""
    if "modell" not in features:
        raise KeyError(
            f"Feature 'modell' missing in features={features}. "
            f"Every order must have a 'modell' feature."
        )
    return features["modell"]


def build_candidate_keys(
    features: Dict[str, str], *, wildcard: bool, visit: int | None = None,
) -> List[str]:
    """Build the candidate key list from most specific to most general."""
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
    """First candidate key present in `lookup`. Raises KeyError on no match
    unless `default` is given.
    """
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
    """Union of all transition targets and all sequential-routing targets."""
    targets: set = set()
    for transition_map in station_config.transitions.values():
        if transition_map:
            targets.update(transition_map.keys())
    for rule in station_config.sequential_routing.values():
        targets.update(rule.get("targets", []))
    return targets
