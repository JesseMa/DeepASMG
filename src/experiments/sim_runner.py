"""
Sim runner — canonical replication series for GroundSim/RefSim/DeepSim.

Invariants:
  C1 — the model registry and the statistics blob are read once per model_dir
       (lru_cache); DeepSim TorchScript models load lazily per strategy
       instance and are re-read for every replication.
  C2 — strict CRN pairing: every run entry carries the seed it was called
       with, which for floor is the un-offset seed (see C3).
  C3 — fresh streams per run: every factory method spawns five independent
       module generators from SeedSequence(seed), floor from
       SeedSequence(seed + SEED_OFFSET_FLOOR); never a shared self._rng.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.config import topology_4stage
from src.config.schema import ProcessConfig
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
from src.dynamics.RefSim.ref_survival import RefSurvival, RefSurvivalWeibull
from src.dynamics.RefSim.ref_repair import RefRepair
from src.dynamics.RefSim.ref_process_time import RefProcessTime, RefProcessTimeVariant
from src.dynamics.RefSim.ref_product import RefProduct
from src.dynamics.RefSim.ref_transition import RefTransition, RefTransitionVariant
from src.evaluation.kpi_report import analyze, extract_cycle_times
from src.simulation.engine import SimulationEngine

REWORK_STATION_DEFAULT  = "M5"
REPO_ROOT               = Path(__file__).resolve().parents[2]
MODEL_DIR_DEFAULT       = REPO_ROOT / "models"


@functools.lru_cache(maxsize=1)
def get_process_config() -> ProcessConfig:
    return ProcessConfig(
        name="4-stage-crossover-rework",
        description=topology_4stage.DESCRIPTION,
        product_features=topology_4stage.PRODUCT_FEATURES,
        stations=topology_4stage.PROCESS_STATIONS,
        temporal_modulation=topology_4stage.TEMPORAL_MODULATION,
        markov_alphas=topology_4stage.PRODUCT_MARKOV_ALPHAS,
    )


@functools.lru_cache(maxsize=1)
def get_num_machines() -> int:
    return topology_4stage.NUM_MACHINES


def _resolve_registry_paths(registry: dict, model_dir: Path) -> dict:
    """Resolve registry paths given relative to the repository root
    (``models/model.pt``, as shipped) or to the model directory. Returned paths
    are absolute, so simulation commands do not depend on the caller's cwd.
    """
    model_dir = model_dir.expanduser().resolve()
    resolved = dict(registry)
    for key, raw_path in registry.items():
        if not (isinstance(raw_path, str) and key.endswith(("_path", "_meta"))):
            continue
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            resolved[key] = str(path)
            continue

        candidates = []
        if path.parts and path.parts[0] == model_dir.name:
            candidates.append(model_dir.joinpath(*path.parts[1:]))
        candidates.extend((model_dir / path, REPO_ROOT / path, Path.cwd() / path))
        match = next((candidate for candidate in candidates if candidate.exists()), None)
        resolved[key] = str((match or candidates[0]).resolve())
    return resolved


@functools.lru_cache(maxsize=4)
def load_model_paths(model_dir: Path = MODEL_DIR_DEFAULT) -> dict[str, str]:
    model_dir = Path(model_dir).expanduser().resolve()
    with open(model_dir / "trained_model_paths.json") as f:
        return _resolve_registry_paths(json.load(f), model_dir)


@functools.lru_cache(maxsize=4)
def load_stats_data(model_dir: Path = MODEL_DIR_DEFAULT) -> dict:
    model_dir = Path(model_dir).expanduser().resolve()
    with open(model_dir / "statistic_params.pkl", "rb") as f:
        data = pickle.load(f)
    return data


class SimFactorySet:
    """Holds ONE copy of model_paths / stats_data / process_config (C1); the
    factory methods create a fresh RNG per call (C3).
    """

    def __init__(
        self,
        *,
        model_paths: dict | None = None,
        stats_data: dict | None = None,
        duration_days: float = SIMULATION_DAYS,
        warmup_days: float = WARMUP_DAYS,
        initial_orders: int = INITIAL_ORDERS,
    ) -> None:
        self.model_paths    = model_paths    if model_paths    is not None else load_model_paths()
        self.stats_data     = stats_data     if stats_data     is not None else load_stats_data()
        self.process_config = get_process_config()
        self.duration_days  = duration_days
        self.warmup_days    = warmup_days
        self.initial_orders = initial_orders

    def _spawn_module_rngs(self, seed: int) -> tuple:
        """Per-module RNG streams via SeedSequence.spawn(5): swapping one module
        does not shift the RNG state of the others (common random numbers).
        """
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(5)
        return (
            np.random.default_rng(children[0]),  # pt
            np.random.default_rng(children[1]),  # tr
            np.random.default_rng(children[2]),  # sv
            np.random.default_rng(children[3]),  # rt
            np.random.default_rng(children[4]),  # pr
        )

    def base(self, seed: int, run_id: int) -> SimulationConfig:
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed)
        return SimulationConfig(
            process_time_strategy=GroundProcessTime(rng_pt),
            transition_strategy=GroundTransition(rng_tr),
            survival_strategy=GroundSurvival(rng_sv),
            repair_strategy=GroundRepair(rng_rt),
            product_strategy=GroundProduct(rng_pr, start_timestamp=SIM_START_TIMESTAMP),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )

    def refm(self, seed: int, run_id: int) -> SimulationConfig:
        """RefSim-M: station-marginal reference with the topological
        admissibility mask applied in the router.
        """
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed)
        sd  = self.stats_data                    # C1
        return SimulationConfig(
            process_time_strategy=RefProcessTime(sd["process_times"], rng_pt),
            transition_strategy=RefTransition(
                sd["transition_probs"], rng_tr, apply_admissibility_mask=True,
            ),
            survival_strategy=RefSurvival(sd["mttf_data"], rng_sv),
            repair_strategy=RefRepair(sd["repair_times"], rng_rt),
            product_strategy=RefProduct(sd["product_features"], rng_pr),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )

    def refv(self, seed: int, run_id: int) -> SimulationConfig:
        """RefSim-V: variant-conditioned reference (processing and routing per
        (station, variant) with fallback), routing masked as in ``refm``.
        """
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed)
        sd = self.stats_data                     # C1
        ptv = sd["process_times_variant"]
        trv = sd["transition_probs_variant"]
        return SimulationConfig(
            process_time_strategy=RefProcessTimeVariant(
                sd["process_times"], ptv["by_producttype"], ptv["by_variant"], rng_pt,
            ),
            transition_strategy=RefTransitionVariant(
                sd["transition_probs"], trv["by_producttype"], trv["by_variant"],
                rng_tr, apply_admissibility_mask=True,
            ),
            survival_strategy=RefSurvival(sd["mttf_data"], rng_sv),
            repair_strategy=RefRepair(sd["repair_times"], rng_rt),
            product_strategy=RefProduct(sd["product_features"], rng_pr),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )

    def refw(self, seed: int, run_id: int) -> SimulationConfig:
        """RefSim-W: ``refm`` with per-station Weibull TTF instead of exponential."""
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed)
        sd = self.stats_data                     # C1
        return SimulationConfig(
            process_time_strategy=RefProcessTime(sd["process_times"], rng_pt),
            transition_strategy=RefTransition(
                sd["transition_probs"], rng_tr, apply_admissibility_mask=True,
            ),
            survival_strategy=RefSurvivalWeibull(sd["weibull_ttf"]["params"], rng_sv),
            repair_strategy=RefRepair(sd["repair_times"], rng_rt),
            product_strategy=RefProduct(sd["product_features"], rng_pr),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )

    def deep(self, seed: int, run_id: int) -> SimulationConfig:
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed)
        mp  = self.model_paths                   # C1
        return SimulationConfig(
            process_time_strategy=DeepProcessTime(
                model_path=mp["pt_model_path"],
                metadata_path=mp["pt_metadata_path"],
                rng=rng_pt,
            ),
            transition_strategy=DeepTransition(
                model_path=mp["tr_model_path"],
                metadata_path=mp["tr_metadata_path"],
                rng=rng_tr,
                temperature=1.0,
            ),
            survival_strategy=DeepSurvival(
                survival_model_path=mp["sv_model_path"],
                survival_metadata_path=mp["sv_metadata_path"],
                rng=rng_sv,
            ),
            repair_strategy=DeepRepair(
                regressor_model_path=mp["rt_model_path"],
                regressor_metadata_path=mp["rt_metadata_path"],
                rng=rng_rt,
            ),
            product_strategy=DeepProduct(
                model_path=mp["pr_model_path"],
                metadata_path=mp["pr_metadata_path"],
                rng=rng_pr,
                temperature=1.0,
                start_timestamp=SIM_START_TIMESTAMP,
            ),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )

    def floor(self, seed: int, run_id: int) -> SimulationConfig:
        """Full GroundSim on a decorrelated stream (seed + SEED_OFFSET_FLOOR on
        all 5 module spawns). Bit-identical to the ablation config
        ``Floor (Full)``: same SeedSequence root, same spawn order
        (pt, tr, sv, rt, pr).
        """
        rng_pt, rng_tr, rng_sv, rng_rt, rng_pr = self._spawn_module_rngs(seed + SEED_OFFSET_FLOOR)
        return SimulationConfig(
            process_time_strategy=GroundProcessTime(rng_pt),
            transition_strategy=GroundTransition(rng_tr),
            survival_strategy=GroundSurvival(rng_sv),
            repair_strategy=GroundRepair(rng_rt),
            product_strategy=GroundProduct(rng_pr, start_timestamp=SIM_START_TIMESTAMP),
            duration_days=self.duration_days,
            warmup_days=self.warmup_days,
            seed=seed,
            initial_orders=self.initial_orders,
            run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )


def _hash_run(sim_name: str, entry: dict) -> str:
    """Deterministic 16-hex hash of a run entry for bit-diff validation."""
    h = hashlib.sha256()
    h.update(sim_name.encode())
    h.update(b"|")
    h.update(str(entry.get("seed", "")).encode())
    h.update(b"|")
    h.update(str(entry.get("run_id", "")).encode())
    h.update(b"|")
    kpis = entry.get("kpis")
    if kpis is not None:
        for k, v in sorted(kpis.items()):
            h.update(f"{k}={v!r};".encode())
    h.update(b"|")
    ct = entry.get("ct")
    if ct is not None:
        h.update(np.ascontiguousarray(ct, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _emit_hash(sim_name: str, entry: dict) -> None:
    """Write a hash line to stderr when SIM_HASH_LOG=1."""
    if os.environ.get("SIM_HASH_LOG") == "1":
        sys.stderr.write(
            f"[HASH] {sim_name} seed={entry.get('seed')} "
            f"run_id={entry.get('run_id')} {_hash_run(sim_name, entry)}\n"
        )
        sys.stderr.flush()


# Counters set lazily by the dynamics (`_sg_*`): reading them post-run consumes no
# RNG; a strategy that never triggered a given path simply lacks the attribute → 0.
_SG_ATTRS = {
    "repair": ("_sg_stress_calls", "_sg_stress_floor", "_sg_repair_draws",
               "_sg_repair_clamp", "_sg_repair_logu_draws", "_sg_repair_logu_guard"),
    "processing": ("_sg_proc_draws", "_sg_proc_clamp",
                   "_sg_proc_nnclip_calls", "_sg_proc_nnclip"),
    "survival": ("_sg_ttf_draws", "_sg_ttf_clamp", "_sg_ttf_logu_draws",
                 "_sg_ttf_logu_guard", "_sg_ttf_nnclip_calls", "_sg_ttf_nnclip"),
    "routing": ("_sg_mask_calls", "_sg_mask_effective"),
}


def _collect_safeguards(cfg: SimulationConfig, engine: "SimulationEngine") -> dict:
    slot = {
        "repair": cfg.repair_strategy,
        "processing": cfg.process_time_strategy,
        "survival": cfg.survival_strategy,
        "routing": cfg.transition_strategy,
    }
    out: dict = {}
    for module, attrs in _SG_ATTRS.items():
        strat = slot[module]
        out[module] = {a[4:]: int(getattr(strat, a, 0)) for a in attrs}
    dm = getattr(engine, "_deadlock", None)
    out["deadlock"] = {
        "deadlock_count": int(getattr(dm, "deadlock_count", 0)),
    }
    return out


def run_replications(
    sim_name: str,
    factory: Callable[[int, int], SimulationConfig],
    seeds: list[int],
    *,
    return_kpis: bool = True,
    return_ct: bool = True,
    rework_station: str = REWORK_STATION_DEFAULT,
) -> list[dict]:
    """One replication series for a sim variant.

    CRN guarantee: with an identical ``seeds`` list across sim variants,
    runs[i] is paired — the five per-module RNG streams are identical, so
    differences between variants come from the strategies, not the draws.
    """
    pc = get_process_config()
    out: list[dict] = []

    for run_idx, seed in enumerate(seeds, start=1):
        cfg = factory(seed, run_idx)
        sim_duration = cfg.duration_days * SECONDS_PER_DAY
        engine = SimulationEngine(pc, cfg)
        process_log, order_log = engine.run(label=f"{sim_name} run={run_idx}")

        entry: dict = {"seed": seed, "run_id": run_idx}
        entry["meta"] = {
            "sim_name":        sim_name,
            "duration_days":   cfg.duration_days,
            "warmup_days":     cfg.warmup_days,
            "start_timestamp": cfg.start_timestamp,
            "seed":            seed,
            "run_id":          run_idx,
        }
        entry["meta"]["safeguards"] = _collect_safeguards(cfg, engine)
        _tr = cfg.transition_strategy
        if hasattr(_tr, "mask_calls"):
            entry["meta"]["routing_mask"] = {
                "mask_calls":       int(_tr.mask_calls),
                "mask_effective":   int(_tr.mask_effective),
                "mask_fallback":    int(_tr.mask_fallback),
                "mask_max_removed": float(_tr.mask_max_removed),
            }
        if hasattr(_tr, "n_variant"):
            entry["meta"]["routing_levels"] = {
                "n_variant":     int(_tr.n_variant),
                "n_producttype": int(_tr.n_producttype),
                "n_station":     int(_tr.n_station),
            }
        _pt = cfg.process_time_strategy
        if hasattr(_pt, "n_variant"):
            entry["meta"]["process_levels"] = {
                "n_variant":     int(_pt.n_variant),
                "n_producttype": int(_pt.n_producttype),
                "n_station":     int(_pt.n_station),
            }
        if return_kpis:
            with contextlib.redirect_stdout(io.StringIO()):
                entry["kpis"] = analyze(
                    process_log=process_log,
                    order_log=order_log,
                    label=f"{sim_name}_run{run_idx}",
                    sim_duration=sim_duration,
                    num_machines=get_num_machines(),
                    rework_station_id=rework_station,
                )
        if return_ct:
            entry["ct"] = extract_cycle_times(process_log, order_log)
        _emit_hash(sim_name, entry)
        out.append(entry)
    return out


