"""
Ablation runner: a grid of configurations that mix GroundSim, DeepSim and RefSim
strategies module by module. Module keys are pt (process time), tr (transition),
sv (survival), rt (repair) and pr (released-order attributes).

Each run draws five independent RNG streams, one per module, so swapping one
module leaves the others' streams untouched. Floor configs derive a subset of
those streams from seed + SEED_OFFSET_FLOOR, decorrelating them from the target.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np

from src.evaluation.kpi_report import analyze, extract_cycle_times
from src.experiments.sim_runner import (
    get_num_machines,
    get_process_config,
    load_model_paths,
    load_stats_data,
    REWORK_STATION_DEFAULT,
)
from src.config.simulation_config import (
    INITIAL_ORDERS,
    SECONDS_PER_DAY,
    SIM_START_TIMESTAMP,
    SIMULATION_DAYS,
    SimulationConfig,
    WARMUP_DAYS,
    SEED_OFFSET_FLOOR,
)
from src.dynamics.GroundSim.ground_survival import GroundSurvival
from src.dynamics.GroundSim.ground_repair import GroundRepair
from src.dynamics.GroundSim.ground_process_time import GroundProcessTime
from src.dynamics.GroundSim.ground_product import GroundProduct
from src.dynamics.GroundSim.ground_transition import GroundTransition
from src.dynamics.DeepSim.deep_survival import DeepSurvival
from src.dynamics.DeepSim.deep_repair import DeepRepair
from src.dynamics.DeepSim.deep_process_time import DeepProcessTime
from src.dynamics.DeepSim.deep_product import DeepProduct
from src.dynamics.DeepSim.deep_transition import DeepTransition
from src.dynamics.RefSim.ref_survival import RefSurvival
from src.dynamics.RefSim.ref_repair import RefRepair
from src.dynamics.RefSim.ref_process_time import RefProcessTime
from src.dynamics.RefSim.ref_product import RefProduct
from src.dynamics.RefSim.ref_transition import RefTransition
from src.simulation.engine import SimulationEngine



def _deep_pt(rng):
    mp = load_model_paths()
    return DeepProcessTime(
        model_path=mp["pt_model_path"],
        metadata_path=mp["pt_metadata_path"],
        rng=rng,
    )


def _deep_tr(rng):
    mp = load_model_paths()
    return DeepTransition(
        model_path=mp["tr_model_path"],
        metadata_path=mp["tr_metadata_path"],
        rng=rng,
        temperature=1.0,
    )


def _deep_pr(rng):
    mp = load_model_paths()
    return DeepProduct(
        model_path=mp["pr_model_path"],
        metadata_path=mp["pr_metadata_path"],
        rng=rng,
        temperature=1.0,
        start_timestamp=SIM_START_TIMESTAMP,
    )


def _deep_sv(rng):
    mp = load_model_paths()
    return DeepSurvival(
        survival_model_path=mp["sv_model_path"],
        survival_metadata_path=mp["sv_metadata_path"],
        rng=rng,
    )


def _deep_rt(rng):
    mp = load_model_paths()
    return DeepRepair(
        regressor_model_path=mp["rt_model_path"],
        regressor_metadata_path=mp["rt_metadata_path"],
        rng=rng,
    )


def _base_pr(rng):
    return GroundProduct(rng, start_timestamp=SIM_START_TIMESTAMP)


def _base_pt(rng):
    return GroundProcessTime(rng)


def _base_tr(rng):
    return GroundTransition(rng)


def _base_sv(rng):
    return GroundSurvival(rng)


def _base_rt(rng):
    return GroundRepair(rng)


def _stat_pt(rng):
    sd = load_stats_data()
    return RefProcessTime(sd["process_times"], rng)


def _stat_tr(rng):
    sd = load_stats_data()
    return RefTransition(
        sd["transition_probs"], rng, apply_admissibility_mask=True,
    )


def _stat_sv(rng):
    sd = load_stats_data()
    return RefSurvival(sd["mttf_data"], rng)


def _stat_rt(rng):
    sd = load_stats_data()
    return RefRepair(sd["repair_times"], rng)


def _stat_pr(rng):
    sd = load_stats_data()
    return RefProduct(sd["product_features"], rng)



# Module slots in consistent order; used for SeedSequence.spawn() indexing
_MODULES = ("pt", "tr", "sv", "rt", "pr")


# Tuple: (pt_fn, tr_fn, sv_fn, rt_fn, pr_fn, offset_modules).
# offset_modules: frozenset of module slots whose RNG derives from
# seed+SEED_OFFSET_FLOOR (decorrelated from the target). Empty = all CRN-paired;
# all 5 = full noise floor.
CONFIGS: dict[str, tuple] = {
    "Target (GroundSim)":           (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset()),
    "DeepSim (All NN)":             (_deep_pt, _deep_tr, _deep_sv, _deep_rt, _deep_pr, frozenset()),
    # Swap-to-perfect: 4 DeepSim + 1 GroundSim module
    "Hybrid (Perfect Orders)":      (_deep_pt, _deep_tr, _deep_sv, _deep_rt, _base_pr, frozenset()),
    "Hybrid (Perfect Routing)":     (_deep_pt, _base_tr, _deep_sv, _deep_rt, _deep_pr, frozenset()),
    "Hybrid (Perfect Process)":     (_base_pt, _deep_tr, _deep_sv, _deep_rt, _deep_pr, frozenset()),
    "Hybrid (Perfect Survival)":    (_deep_pt, _deep_tr, _base_sv, _deep_rt, _deep_pr, frozenset()),
    "Hybrid (Perfect Repair)":      (_deep_pt, _deep_tr, _deep_sv, _base_rt, _deep_pr, frozenset()),
    # Swap-to-one-surrogate: 4 Ground + 1 DeepSim module
    "Hybrid (DeepSim Process only)":  (_deep_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset()),
    "Hybrid (DeepSim Routing only)":  (_base_pt, _deep_tr, _base_sv, _base_rt, _base_pr, frozenset()),
    "Hybrid (DeepSim Orders only)":   (_base_pt, _base_tr, _base_sv, _base_rt, _deep_pr, frozenset()),
    "Hybrid (DeepSim Survival only)": (_base_pt, _base_tr, _deep_sv, _base_rt, _base_pr, frozenset()),
    "Hybrid (DeepSim Repair only)":   (_base_pt, _base_tr, _base_sv, _deep_rt, _base_pr, frozenset()),
    # Floor (swap-to-perfect): all Ground; the "perfect" module is CRN-paired, the other 4 decorrelated
    "Floor (Full)":              (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(_MODULES)),
    "Floor (Perfect Orders)":    (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pt", "tr", "sv", "rt"))),
    "Floor (Perfect Routing)":   (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pt", "sv", "rt", "pr"))),
    "Floor (Perfect Process)":   (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("tr", "sv", "rt", "pr"))),
    "Floor (Perfect Survival)":  (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pt", "tr", "rt", "pr"))),
    "Floor (Perfect Repair)":    (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pt", "tr", "sv", "pr"))),
    # Floor (swap-to-one): one module decorrelated, others CRN-paired
    "Floor (Process decorrelated)":   (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pt",))),
    "Floor (Routing decorrelated)":   (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("tr",))),
    "Floor (Survival decorrelated)":  (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("sv",))),
    "Floor (Repair decorrelated)":    (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("rt",))),
    "Floor (Orders decorrelated)":    (_base_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset(("pr",))),
    # RefSim: statistical strategies from training-frequency extraction.
    "RefSim (All Statistical)":              (_stat_pt, _stat_tr, _stat_sv, _stat_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Perfect Orders)":        (_stat_pt, _stat_tr, _stat_sv, _stat_rt, _base_pr, frozenset()),
    "RefSim Hybrid (Perfect Routing)":       (_stat_pt, _base_tr, _stat_sv, _stat_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Perfect Process)":       (_base_pt, _stat_tr, _stat_sv, _stat_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Perfect Survival)":      (_stat_pt, _stat_tr, _base_sv, _stat_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Perfect Repair)":        (_stat_pt, _stat_tr, _stat_sv, _base_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Stat Process only)":     (_stat_pt, _base_tr, _base_sv, _base_rt, _base_pr, frozenset()),
    "RefSim Hybrid (Stat Routing only)":     (_base_pt, _stat_tr, _base_sv, _base_rt, _base_pr, frozenset()),
    "RefSim Hybrid (Stat Orders only)":      (_base_pt, _base_tr, _base_sv, _base_rt, _stat_pr, frozenset()),
    "RefSim Hybrid (Stat Survival only)":    (_base_pt, _base_tr, _stat_sv, _base_rt, _base_pr, frozenset()),
    "RefSim Hybrid (Stat Repair only)":      (_base_pt, _base_tr, _base_sv, _stat_rt, _base_pr, frozenset()),
    # Module selection: the learned core with the statistical repair module.
    "Hybrid (Stat Repair)":                  (_deep_pt, _deep_tr, _deep_sv, _stat_rt, _deep_pr, frozenset()),
}


def _make_per_module_rngs(seed: int, offset_modules: frozenset) -> dict:
    """One RNG stream per module slot: modules in ``offset_modules`` derive from
    seed+SEED_OFFSET_FLOOR, all others from ``seed``. Both roots go through
    SeedSequence.spawn(5) for clean independent per-slot streams.
    """
    ss_target = np.random.SeedSequence(seed)
    ss_offset = np.random.SeedSequence(seed + SEED_OFFSET_FLOOR)
    target_children = dict(zip(_MODULES, ss_target.spawn(len(_MODULES)), strict=True))
    offset_children = dict(zip(_MODULES, ss_offset.spawn(len(_MODULES)), strict=True))
    rngs: dict[str, np.random.Generator] = {}
    for m in _MODULES:
        child = offset_children[m] if m in offset_modules else target_children[m]
        rngs[m] = np.random.default_rng(child)
    return rngs


def _make_config(
    seed: int, run_id: int,
    pt_fn, tr_fn, sv_fn, rt_fn, pr_fn,
    offset_modules: frozenset,
) -> SimulationConfig:
    rngs = _make_per_module_rngs(seed, offset_modules)
    return SimulationConfig(
        process_time_strategy=pt_fn(rngs["pt"]),
        transition_strategy=tr_fn(rngs["tr"]),
        survival_strategy=sv_fn(rngs["sv"]),
        repair_strategy=rt_fn(rngs["rt"]),
        product_strategy=pr_fn(rngs["pr"]),
        duration_days=SIMULATION_DAYS,
        warmup_days=WARMUP_DAYS,
        seed=seed,
        initial_orders=INITIAL_ORDERS,
        run_id=run_id,
        start_timestamp=SIM_START_TIMESTAMP,
    )


def run_ablation_grid(seeds: list[int]) -> dict[str, list[dict]]:
    """Run all CONFIGS × len(seeds) simulations."""
    process_config = get_process_config()
    num_machines = get_num_machines()
    sim_duration = SIMULATION_DAYS * SECONDS_PER_DAY

    results: dict[str, list[dict]] = {}
    for name, (pt_fn, tr_fn, sv_fn, rt_fn, pr_fn, offset_modules) in CONFIGS.items():
        records: list[dict] = []
        for run_idx, seed in enumerate(seeds, start=1):
            started = time.perf_counter()
            cfg = _make_config(
                seed, run_idx, pt_fn, tr_fn, sv_fn, rt_fn, pr_fn, offset_modules,
            )
            engine = SimulationEngine(process_config, cfg)
            process_log, order_log = engine.run(label=f"{name} run={run_idx}")

            with contextlib.redirect_stdout(io.StringIO()):
                kpis = analyze(
                    process_log=process_log,
                    order_log=order_log,
                    label=f"{name}_r{run_idx}",
                    sim_duration=sim_duration,
                    num_machines=num_machines,
                    rework_station_id=REWORK_STATION_DEFAULT,
                )

            ct = extract_cycle_times(process_log, order_log)
            transition = cfg.transition_strategy
            if isinstance(transition, RefTransition):
                routing_mask_rows = transition.mask_audit_rows()
                routing_mask_summary = transition.mask_audit_summary()
            else:
                routing_mask_rows = []
                routing_mask_summary = {
                    "mask_active": None,
                    "stationary_predict_count": 0,
                    "mask_call_count": 0,
                    "unmasked_call_count": 0,
                    "mask_effective_count": 0,
                    "renormalization_count": 0,
                    "zero_mass_fallback_count": 0,
                    "invalid_selected_target_count_after_masking": 0,
                    "maximum_probability_mass_removed": 0.0,
                    "all_stationary_calls_masked": None,
                }
            records.append({
                "config": name,
                "seed": seed,
                "run_id": run_idx,
                "kpis": kpis,
                "ct": ct,
                "completed": True,
                "aborted": False,
                "runtime_seconds": time.perf_counter() - started,
                "routing_mask_rows": routing_mask_rows,
                "routing_mask_summary": routing_mask_summary,
            })
        results[name] = records
    return results
