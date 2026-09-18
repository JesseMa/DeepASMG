"""The substitution grid: every system as one module kind per slot, CRN-paired or decorrelated."""

from __future__ import annotations

from src.experiments.sim_runner import MODULES, SimFactorySet, run_replications

G, D, S = "ground", "deep", "stat"
_ALL = frozenset(MODULES)

def _floor(*decorrelated: str) -> tuple:
    return ((G, G, G, G, G), frozenset(decorrelated))

CONFIGS: dict[str, tuple] = {
    "Target (GroundSim)":            ((G, G, G, G, G), frozenset()),
    "DeepSim (All NN)":              ((D, D, D, D, D), frozenset()),
    # Swap-to-perfect: 4 DeepSim + 1 GroundSim module
    "Hybrid (Perfect Orders)":       ((D, D, D, D, G), frozenset()),
    "Hybrid (Perfect Routing)":      ((D, G, D, D, D), frozenset()),
    "Hybrid (Perfect Process)":      ((G, D, D, D, D), frozenset()),
    "Hybrid (Perfect Survival)":     ((D, D, G, D, D), frozenset()),
    "Hybrid (Perfect Repair)":       ((D, D, D, G, D), frozenset()),
    # Swap-to-one-surrogate: 4 Ground + 1 DeepSim module
    "Hybrid (DeepSim Process only)":  ((D, G, G, G, G), frozenset()),
    "Hybrid (DeepSim Routing only)":  ((G, D, G, G, G), frozenset()),
    "Hybrid (DeepSim Orders only)":   ((G, G, G, G, D), frozenset()),
    "Hybrid (DeepSim Survival only)": ((G, G, D, G, G), frozenset()),
    "Hybrid (DeepSim Repair only)":   ((G, G, G, D, G), frozenset()),
    # Floor (swap-to-perfect): the "perfect" module CRN-paired, the other 4 decorrelated
    "Floor (Full)":              (( G, G, G, G, G), _ALL),
    "Floor (Perfect Orders)":    _floor("pt", "tr", "sv", "rt"),
    "Floor (Perfect Routing)":   _floor("pt", "sv", "rt", "pr"),
    "Floor (Perfect Process)":   _floor("tr", "sv", "rt", "pr"),
    "Floor (Perfect Survival)":  _floor("pt", "tr", "rt", "pr"),
    "Floor (Perfect Repair)":    _floor("pt", "tr", "sv", "pr"),
    # Floor (swap-to-one): one module decorrelated, the others CRN-paired
    "Floor (Process decorrelated)":  _floor("pt"),
    "Floor (Routing decorrelated)":  _floor("tr"),
    "Floor (Survival decorrelated)": _floor("sv"),
    "Floor (Repair decorrelated)":   _floor("rt"),
    "Floor (Orders decorrelated)":   _floor("pr"),
    # RefSim: statistical strategies from training-frequency extraction.
    "RefSim (All Statistical)":           ((S, S, S, S, S), frozenset()),
    "RefSim Hybrid (Perfect Orders)":     ((S, S, S, S, G), frozenset()),
    "RefSim Hybrid (Perfect Routing)":    ((S, G, S, S, S), frozenset()),
    "RefSim Hybrid (Perfect Process)":    ((G, S, S, S, S), frozenset()),
    "RefSim Hybrid (Perfect Survival)":   ((S, S, G, S, S), frozenset()),
    "RefSim Hybrid (Perfect Repair)":     ((S, S, S, G, S), frozenset()),
    "RefSim Hybrid (Stat Process only)":  ((S, G, G, G, G), frozenset()),
    "RefSim Hybrid (Stat Routing only)":  ((G, S, G, G, G), frozenset()),
    "RefSim Hybrid (Stat Orders only)":   ((G, G, G, G, S), frozenset()),
    "RefSim Hybrid (Stat Survival only)": ((G, G, S, G, G), frozenset()),
    "RefSim Hybrid (Stat Repair only)":   ((G, G, G, S, G), frozenset()),
    # Module selection: the learned core with the statistical repair module.
    "Hybrid (Stat Repair)":               ((D, D, D, S, D), frozenset()),
}


def run_ablation_grid(seeds: list[int]) -> dict[str, list[dict]]:
    fs = SimFactorySet()
    return {
        name: run_replications(
            name, lambda s, r, k=kinds, d=dec: fs.compose(s, r, k, decorrelated=d), seeds,
        )
        for name, (kinds, dec) in CONFIGS.items()
    }
