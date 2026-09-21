"""Sim runner — canonical replication series for GroundSim/RefSim/DeepSim."""

from __future__ import annotations

import functools
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

import numpy as np

from src.config import topology_4stage
from src.config.schema import ProcessConfig
from src.config.simulation_config import (
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
from src.dynamics.RefSim.ref_process_time import RefProcessTime
from src.dynamics.RefSim.ref_product import RefProduct
from src.dynamics.RefSim.ref_transition import RefTransition
from src.evaluation.kpi_report import analyze, extract_cycle_times
from src.simulation.engine import SimulationEngine

REWORK_STATION_DEFAULT  = "M5"
REPO_ROOT               = Path(__file__).resolve().parents[2]
MODEL_DIR_DEFAULT       = REPO_ROOT / "models"

MODULES = ("pt", "tr", "sv", "rt", "pr")
Kinds = Tuple[str, str, str, str, str]

MODEL_FILES: Dict[str, Tuple[str, str]] = {
    "pt": ("process_time_model.pt", "process_time_data/metadata.json"),
    "tr": ("transition_model.pt", "transition_data/metadata.json"),
    "sv": ("survival_weibull_model.pt", "survival_data/metadata.json"),
    "rt": ("repair_time_model.pt", "repair_time_data/metadata.json"),
    "pr": ("product_model.pt", "product_data/metadata.json"),
}


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


def model_paths(model_dir: Path = MODEL_DIR_DEFAULT) -> Dict[str, str]:
    model_dir = Path(model_dir).expanduser().resolve()
    out: Dict[str, str] = {}
    for slot, (model_file, metadata_file) in MODEL_FILES.items():
        out[f"{slot}_model_path"] = str(model_dir / model_file)
        out[f"{slot}_metadata_path"] = str(model_dir / metadata_file)
    return out


def models_trained(model_dir: Path) -> bool:
    return all(Path(p).exists() for p in model_paths(model_dir).values())


@functools.lru_cache(maxsize=4)
def load_stats_data(model_dir: Path = MODEL_DIR_DEFAULT) -> dict:
    model_dir = Path(model_dir).expanduser().resolve()
    with open(model_dir / "statistic_params.pkl", "rb") as f:
        data = pickle.load(f)
    return data


class SimFactorySet:
    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR_DEFAULT,
        stats_data: dict | None = None,
        duration_days: float = SIMULATION_DAYS,
        warmup_days: float = WARMUP_DAYS,
    ) -> None:
        self.model_paths    = model_paths(model_dir)
        self.stats_data     = stats_data if stats_data is not None else load_stats_data(model_dir)
        self.process_config = get_process_config()
        self.duration_days  = duration_days
        self.warmup_days    = warmup_days

    def module(self, kind: str, slot: str, rng: np.random.Generator):
        mp, sd = self.model_paths, self.stats_data
        if kind == "ground":
            return {
                "pt": lambda: GroundProcessTime(rng),
                "tr": lambda: GroundTransition(rng),
                "sv": lambda: GroundSurvival(rng),
                "rt": lambda: GroundRepair(rng),
                "pr": lambda: GroundProduct(rng, start_timestamp=SIM_START_TIMESTAMP),
            }[slot]()
        if kind == "deep":
            return {
                "pt": lambda: DeepProcessTime(
                    model_path=mp["pt_model_path"], metadata_path=mp["pt_metadata_path"], rng=rng),
                "tr": lambda: DeepTransition(
                    model_path=mp["tr_model_path"], metadata_path=mp["tr_metadata_path"], rng=rng),
                "sv": lambda: DeepSurvival(
                    survival_model_path=mp["sv_model_path"],
                    survival_metadata_path=mp["sv_metadata_path"], rng=rng),
                "rt": lambda: DeepRepair(
                    regressor_model_path=mp["rt_model_path"],
                    regressor_metadata_path=mp["rt_metadata_path"], rng=rng),
                "pr": lambda: DeepProduct(
                    model_path=mp["pr_model_path"], metadata_path=mp["pr_metadata_path"],
                    rng=rng, start_timestamp=SIM_START_TIMESTAMP),
            }[slot]()
        if kind == "stat":
            return {
                "pt": lambda: RefProcessTime(sd["process_times"], rng),
                "tr": lambda: RefTransition(sd["transition_probs"], rng),
                "sv": lambda: RefSurvival(sd["mttf_data"], rng, sd["mttf_pooled"]),
                "rt": lambda: RefRepair(sd["repair_times"], rng, sd["repair_pooled"]),
                "pr": lambda: RefProduct(sd["product_features"], rng),
            }[slot]()
        if kind == "statv":
            ptv, trv = sd["process_times_variant"], sd["transition_probs_variant"]
            return {
                "pt": lambda: RefProcessTime(
                    sd["process_times"], rng,
                    by_producttype=ptv["by_producttype"], by_variant=ptv["by_variant"]),
                "tr": lambda: RefTransition(
                    sd["transition_probs"], rng,
                    by_producttype=trv["by_producttype"], by_variant=trv["by_variant"]),
            }[slot]()
        if kind == "statw":
            return {
                "sv": lambda: RefSurvivalWeibull(
                    sd["weibull_ttf"]["params"], rng, sd["weibull_ttf"]["pooled"]),
            }[slot]()
        raise ValueError(f"Unknown module kind '{kind}'.")


    @staticmethod
    @staticmethod
    def _module_rngs(seed: int, coupled: FrozenSet[str]) -> Dict[str, np.random.Generator]:
        # the target realization owns the streams of `seed`; every other draw comes
        # from the offset streams, which all evaluated systems share among themselves
        target = dict(zip(MODULES, np.random.SeedSequence(seed).spawn(len(MODULES)), strict=True))
        offset = dict(zip(MODULES, np.random.SeedSequence(seed + SEED_OFFSET_FLOOR).spawn(len(MODULES)), strict=True))
        return {m: np.random.default_rng(target[m] if m in coupled else offset[m])
                for m in MODULES}

    @staticmethod
    def coupled_slots(kinds: Kinds, *, true_core: bool) -> FrozenSet[str]:
        # only a true module inside the true core may share the target's streams
        if not true_core:
            return frozenset()
        return frozenset(slot for kind, slot in zip(kinds, MODULES, strict=True) if kind == "ground")

    def compose(
        self, seed: int, run_id: int, kinds: Kinds, *,
        coupled: FrozenSet[str] = frozenset(),
    ) -> SimulationConfig:
        unknown = coupled - set(MODULES)
        if unknown:
            raise ValueError(f"unknown slots in coupled set: {sorted(unknown)}")
        for kind, slot in zip(kinds, MODULES, strict=True):
            if slot in coupled and kind != "ground":
                raise ValueError(f"slot {slot!r} of kind {kind!r} cannot share the target's stream")
        rngs = self._module_rngs(seed, coupled)
        pt, tr, sv, rt, pr = (self.module(kind, slot, rngs[slot])
                              for kind, slot in zip(kinds, MODULES, strict=True))
        return SimulationConfig(
            process_time_strategy=pt, transition_strategy=tr, survival_strategy=sv,
            repair_strategy=rt, product_strategy=pr,
            duration_days=self.duration_days, warmup_days=self.warmup_days,
            seed=seed, run_id=run_id,
        )


    def base(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("ground",) * 5, coupled=frozenset(MODULES))

    def floor(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("ground",) * 5)

    def deep(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("deep",) * 5)

    def refm(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("stat",) * 5)

    def refv(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("statv", "statv", "stat", "stat", "stat"))

    def refw(self, seed: int, run_id: int) -> SimulationConfig:
        return self.compose(seed, run_id, ("stat", "stat", "statw", "stat", "stat"))


_SG_ATTRS = {
    "repair": ("_sg_stress_calls", "_sg_stress_floor", "_sg_repair_draws",
               "_sg_repair_clamp", "_sg_repair_logu_draws", "_sg_repair_logu_guard"),
    "processing": ("_sg_proc_draws", "_sg_proc_clamp"),
    "survival": ("_sg_ttf_draws", "_sg_ttf_clamp", "_sg_ttf_logu_draws",
                 "_sg_ttf_logu_guard"),
    "routing": ("_sg_mask_calls", "_sg_mask_effective", "_sg_mask_fallback"),
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
    out["deadlock"] = {"deadlock_count": int(engine._deadlock.deadlock_count)}
    return out


def run_replications(
    sim_name: str,
    factory: Callable[[int, int], SimulationConfig],
    seeds: list[int],
) -> list[dict]:
    pc = get_process_config()
    out: list[dict] = []

    for run_idx, seed in enumerate(seeds, start=1):
        cfg = factory(seed, run_idx)
        engine = SimulationEngine(pc, cfg)
        process_log, order_log = engine.run(label=f"{sim_name} run={run_idx}")

        out.append({
            "seed": seed, "run_id": run_idx,
            "meta": {
                "sim_name":      sim_name,
                "duration_days": cfg.duration_days,
                "warmup_days":   cfg.warmup_days,
                "seed":          seed,
                "run_id":        run_idx,
                "safeguards":    _collect_safeguards(cfg, engine),
            },
            "kpis": analyze(
                process_log, order_log,
                sim_duration=cfg.duration_days * SECONDS_PER_DAY,
                num_machines=topology_4stage.NUM_MACHINES,
                rework_station_id=REWORK_STATION_DEFAULT,
            ),
            "ct": extract_cycle_times(process_log, order_log),
        })
    return out
