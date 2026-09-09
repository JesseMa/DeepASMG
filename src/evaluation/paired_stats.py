"""Statistics helpers for paired sim-to-sim comparisons (CRN design)."""

from __future__ import annotations

import numpy as np
from scipy.stats import (
    false_discovery_control,
)


def fdr_correction(
    pvals, alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction, returning (rejected, q_values).

    Apply per family (e.g. per KPI group); BH over all tests at once is
    over-conservative.
    """
    p = np.asarray(pvals, dtype=float)
    q = false_discovery_control(p, method="bh")
    return q < alpha, q


def assert_paired_seeds(runs_a, runs_b) -> None:
    """Assert CRN pairing: equal length and equal seed at each index."""
    assert len(runs_a) == len(runs_b), (
        f"CRN pairing violated: len mismatch {len(runs_a)} vs {len(runs_b)}"
    )
    for i, (a, b) in enumerate(zip(runs_a, runs_b, strict=True)):
        assert a["seed"] == b["seed"], (
            f"CRN pairing violated @ idx {i}: "
            f"seed {a['seed']} vs {b['seed']}"
        )
