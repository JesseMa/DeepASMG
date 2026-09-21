"""The substitution grid: every system as one module kind per slot, with the slots that share the target's streams."""

from __future__ import annotations

from src.experiments.sim_runner import MODULES, SimFactorySet, run_replications

G, D, S = "ground", "deep", "stat"
_ALL = frozenset(MODULES)
_TRUE = (G, G, G, G, G)


def _true_core(kinds: tuple) -> tuple:
    # the true core keeps the target's streams, the inserted module draws independently
    return (kinds, frozenset(slot for kind, slot in zip(kinds, MODULES, strict=True) if kind == G))


def _independent(kinds: tuple) -> tuple:
    # a candidate core, with or without a true module inside, draws independently
    return (kinds, frozenset())


def _floor(*independent: str) -> tuple:
    return (_TRUE, _ALL - frozenset(independent))


CONFIGS: dict[str, tuple] = {
    "Target (GroundSim)":            (_TRUE, _ALL),
    "DeepSim (All NN)":              _independent((D, D, D, D, D)),
    "Hybrid (Perfect Orders)":       _independent((D, D, D, D, G)),
    "Hybrid (Perfect Routing)":      _independent((D, G, D, D, D)),
    "Hybrid (Perfect Process)":      _independent((G, D, D, D, D)),
    "Hybrid (Perfect Survival)":     _independent((D, D, G, D, D)),
    "Hybrid (Perfect Repair)":       _independent((D, D, D, G, D)),
    "Hybrid (DeepSim Process only)":  _true_core((D, G, G, G, G)),
    "Hybrid (DeepSim Routing only)":  _true_core((G, D, G, G, G)),
    "Hybrid (DeepSim Orders only)":   _true_core((G, G, G, G, D)),
    "Hybrid (DeepSim Survival only)": _true_core((G, G, D, G, G)),
    "Hybrid (DeepSim Repair only)":   _true_core((G, G, G, D, G)),
    "Floor (Full)":              (_TRUE, frozenset()),
    "Floor (Perfect Orders)":    _floor("pt", "tr", "sv", "rt"),
    "Floor (Perfect Routing)":   _floor("pt", "sv", "rt", "pr"),
    "Floor (Perfect Process)":   _floor("tr", "sv", "rt", "pr"),
    "Floor (Perfect Survival)":  _floor("pt", "tr", "rt", "pr"),
    "Floor (Perfect Repair)":    _floor("pt", "tr", "sv", "pr"),
    "Floor (Process decorrelated)":  _floor("pt"),
    "Floor (Routing decorrelated)":  _floor("tr"),
    "Floor (Survival decorrelated)": _floor("sv"),
    "Floor (Repair decorrelated)":   _floor("rt"),
    "Floor (Orders decorrelated)":   _floor("pr"),
    "RefSim (All Statistical)":           _independent((S, S, S, S, S)),
    "RefSim Hybrid (Perfect Orders)":     _independent((S, S, S, S, G)),
    "RefSim Hybrid (Perfect Routing)":    _independent((S, G, S, S, S)),
    "RefSim Hybrid (Perfect Process)":    _independent((G, S, S, S, S)),
    "RefSim Hybrid (Perfect Survival)":   _independent((S, S, G, S, S)),
    "RefSim Hybrid (Perfect Repair)":     _independent((S, S, S, G, S)),
    "RefSim Hybrid (Stat Process only)":  _true_core((S, G, G, G, G)),
    "RefSim Hybrid (Stat Routing only)":  _true_core((G, S, G, G, G)),
    "RefSim Hybrid (Stat Orders only)":   _true_core((G, G, G, G, S)),
    "RefSim Hybrid (Stat Survival only)": _true_core((G, G, S, G, G)),
    "RefSim Hybrid (Stat Repair only)":   _true_core((G, G, G, S, G)),
    "Hybrid (Stat Repair)":               _independent((D, D, D, S, D)),
}


def run_ablation_grid(seeds: list[int]) -> dict[str, list[dict]]:
    fs = SimFactorySet()
    return {
        name: run_replications(
            name, lambda s, r, k=kinds, c=coupled: fs.compose(s, r, k, coupled=c), seeds,
        )
        for name, (kinds, coupled) in CONFIGS.items()
    }
