"""
Shared helpers for chronological train-only vocabulary fitting.

The encoder vocabulary is fit on the TRAIN slice only; categories first seen in
val/test collapse to all-zero one-hot at inference (the OOV pathway). The cut
index approximates the train end of three_way_split to within two samples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set


from src.config.simulation_config import TRAIN_RATIO as DEFAULT_TRAIN_RATIO


def chronological_train_end_idx(n: int, train_ratio: float = DEFAULT_TRAIN_RATIO) -> int:
    """Exclusive end index of the train slice: samples[:end] fit the vocabulary."""
    if n <= 0:
        raise ValueError(f"Fail-fast: n_samples={n}, cannot form a train cut.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"Fail-fast: train_ratio={train_ratio} must lie in (0, 1).")
    return max(1, int(n * train_ratio))


def count_oov(
    samples: List[Any],
    holdout_start_idx: int,
    feature_extractors: Dict[str, Callable[[Any], str]],
    train_vocabs: Dict[str, Set[str]],
) -> Dict[str, int]:
    """OOV hits per feature in samples[holdout_start_idx:]; samples must already
    be in chronological order."""
    holdout = samples[holdout_start_idx:]
    out: Dict[str, int] = {name: 0 for name in feature_extractors}
    for s in holdout:
        for name, extractor in feature_extractors.items():
            value = extractor(s)
            if value not in train_vocabs[name]:
                out[name] += 1
    return out


def summarize_oov(
    oov_counts: Dict[str, int],
    n_holdout: int,
) -> Dict[str, Dict[str, float]]:
    """Aggregates OOV counts into count + ratio."""
    if n_holdout <= 0:
        return {name: {"count": int(c), "ratio": 0.0} for name, c in oov_counts.items()}
    return {
        name: {"count": int(c), "ratio": float(c) / float(n_holdout)}
        for name, c in oov_counts.items()
    }
