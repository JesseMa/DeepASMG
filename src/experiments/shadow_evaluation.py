"""N-system shadow-parameter driver.

One GroundSim trajectory per seed advances the state; at every strategy call all
systems are queried on the identical context for their deployed
distribution_params, one logged row per (system, component). DeepSim process,
transition and product state is teacher-forced to the Ground realization after
every call.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.experiments.sim_runner import SimFactorySet
from src.simulation.engine import SimulationEngine
from src.dynamics.foundation_dynamics import END_TOKEN
from src.config.routing_keys import full_variant_key
from src.config.simulation_config import SimulationConfig

SYSTEMS = ("GroundSim", "DeepSim", "RefSim-M", "RefSim-V", "RefSim-W")
COMPONENTS = ("processing", "transition", "survival", "repair", "arrival")
CSV_COLUMNS = ("seed", "context_id", "t_sim", "station", "component", "head",
               "family", "params", "realized", "censored")


class ShadowLog:
    def __init__(self, seed: int, warmup_time: float) -> None:
        self.seed = seed
        self.warmup_time = float(warmup_time)
        self.rows: Dict[tuple, List[dict]] = defaultdict(list)
        self._counter: Dict[str, int] = defaultdict(int)
        # Sidecar context_id → (variant, visit); keeps the score CSVs feature-free.
        self.context_variant: Dict[str, Tuple[str, int]] = {}

    def next_ctx(self, component: str) -> str:
        idx = self._counter[component]
        self._counter[component] = idx + 1
        return f"{self.seed}:{component}:{idx}"

    def record(self, system: str, component: str, ctx_id: str, station: str,
               t_sim: float, head: str, family: str, params: Optional[dict],
               realized: Any, censored: int) -> None:
        if t_sim < self.warmup_time:
            return  # the evaluation window starts after warm-up, as in the closed loop
        self.rows[(system, component)].append({
            "seed": self.seed, "context_id": ctx_id, "t_sim": float(t_sim),
            "station": station, "component": component, "head": head,
            "family": family,
            "params": "" if params is None else json.dumps(params, sort_keys=True),
            "realized": realized, "censored": int(censored),
        })

    def write(self, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for (system, component), rows in self.rows.items():
            path = out_dir / f"{system}__{component}.csv"
            import csv
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({c: _fmt(r.get(c, "")) for c in CSV_COLUMNS})
            written.append(path)
        if self.context_variant:
            import csv
            sidecar = out_dir / "_context_variant.csv"
            with sidecar.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["context_id", "variant", "visit"])
                for ctx, (var, visit) in self.context_variant.items():
                    w.writerow([ctx, var, visit])
            written.append(sidecar)
        return written


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)  # full float precision
    return v


def _is_deep(name: str) -> bool:
    return name.startswith("DeepSim")


class _ShadowStations:
    """Mixin: forward initialize(stations) to the ground and every system strategy."""

    def initialize(self, stations):
        self._g.initialize(stations)
        for s in self._sys.values():
            s.initialize(stations)


class _ShadowPT(_ShadowStations):
    def __init__(self, ground, systems, log):
        self._g, self._sys, self._log = ground, systems, log

    def predict(self, station_id, order, current_time=0.0):
        ctx = self._log.next_ctx("processing")
        realized = float(self._g.predict(station_id, order, current_time))
        for name, strat in self._sys.items():
            p = strat.distribution_params(station_id, order, current_time)
            self._log.record(name, "processing", ctx, station_id, current_time,
                             "", p["family"], p, realized, 0)
            if _is_deep(name):
                strat._prev_on_machine[station_id] = dict(order.features)
        return realized


class _ShadowTR(_ShadowStations):
    def __init__(self, ground, systems, log):
        self._g, self._sys, self._log = ground, systems, log

    def predict(self, station_id, order, available_targets=None, current_time=0.0):
        ctx = self._log.next_ctx("transition")
        self._log.context_variant[ctx] = (
            full_variant_key(order.features), order.visits.get(station_id, 1))
        # Collect params BEFORE the steering draw (deep buffer = pre-call state).
        pending = {}
        for name, strat in self._sys.items():
            pending[name] = strat.distribution_params(
                station_id, order, available_targets, current_time)
        target = self._g.predict(station_id, order, available_targets, current_time=current_time)
        realized = target if target is not None else END_TOKEN
        for name, strat in self._sys.items():
            self._log.record(name, "transition", ctx, station_id, current_time,
                             "", pending[name]["family"], pending[name], realized, 0)
            if _is_deep(name):  # teacher forcing: push the Ground target into the slot buffer
                modell = order.features.get("modell", strat._none_token)
                strat._get_or_init_buffer(station_id).appendleft((modell, realized))
        return target


class _ShadowSV(_ShadowStations):
    """Lifecycle matching: every sample_time_to_failure opens a cycle (at most
    one open per station); notify_cycle_end (ttf = accumulated_op_time =
    realized operating duration) closes it uncensored. At sim end, a cycle whose
    breakdown already occurred (repair running past the horizon) is emitted as
    an observed failure; only cycles without a breakdown are right-censored.
    """

    def __init__(self, ground, systems, log):
        self._g, self._sys, self._log = ground, systems, log
        self._open: Dict[str, List[dict]] = defaultdict(list)  # station → open cycles (FIFO)
        # Breakdown happened but repair outlives the horizon → observed, not censored.
        self._breakdown_pending: set = set()

    def sample_time_to_failure(self, station_id, current_time=0.0):
        ttf = self._g.sample_time_to_failure(station_id, current_time=current_time)
        if ttf is None:
            return None  # non-failing station (buffer): no survival log
        ctx = self._log.next_ctx("survival")
        cyc = {"ctx": ctx, "station": station_id, "t_sim": float(current_time),
               "params": {n: s.distribution_params(station_id, current_time)
                          for n, s in self._sys.items()}}
        self._open[station_id].append(cyc)
        return ttf

    def notify_cycle_end(self, station_id, ttf, n_jobs, repair_time):
        self._g.notify_cycle_end(station_id=station_id, ttf=ttf, n_jobs=n_jobs, repair_time=repair_time)
        for name, s in self._sys.items():
            if name != "GroundSim":
                s.notify_cycle_end(station_id=station_id, ttf=ttf, n_jobs=n_jobs, repair_time=repair_time)
        self._breakdown_pending.discard(station_id)
        if self._open[station_id]:
            self._emit(self._open[station_id].pop(0), float(ttf), 0)

    def notify_breakdown(self, station_id: str) -> None:
        """Called at breakdown time (via the repair wrapper)."""
        self._breakdown_pending.add(station_id)

    def _emit(self, cyc, realized, censored):
        for name, p in cyc["params"].items():
            if p is None:
                raise RuntimeError(
                    f"Fail fast: {name} returned no survival law for {cyc['station']} "
                    "although GroundSim drew a TTF for it."
                )
            fam = p["family"]
            self._log.record(name, "survival", cyc["ctx"], cyc["station"],
                             cyc["t_sim"], "", fam, p, realized, censored)

    def finalize(self, final_accum: Dict[str, float]) -> None:
        """Close cycles still open at sim end; only the oldest open cycle can
        carry an observed breakdown, the rest are right-censored.
        """
        for station, cycles in self._open.items():
            for index, cyc in enumerate(cycles):
                observed = index == 0 and station in self._breakdown_pending
                self._emit(cyc, final_accum.get(station, ""), 0 if observed else 1)


class _ShadowRT(_ShadowStations):
    def __init__(self, ground, systems, log, survival_wrapper):
        self._g, self._sys, self._log = ground, systems, log
        self._sv = survival_wrapper

    def predict_repair_time(self, station_id, operating_time_since_last, utilization, current_time=0.0):
        self._sv.notify_breakdown(station_id)
        ctx = self._log.next_ctx("repair")
        realized = float(self._g.predict_repair_time(
            station_id, operating_time_since_last, utilization, current_time=current_time))
        for name, strat in self._sys.items():
            p = strat.distribution_params(station_id, operating_time_since_last, utilization, current_time)
            self._log.record(name, "repair", ctx, station_id, current_time,
                             "", p["family"], p, realized, 0)
        return realized


class _ShadowPR:
    """Arrival: GroundProduct steers; all systems are queried on the realized
    context per attribute head for their categorical distribution. Deep
    products are teacher-forced (observe_external = Ground history). One row
    per head with its own context_id.
    """

    HEAD_FEAT = {"type": "modell", "feature_a": "feature_a", "feature_b": "feature_b"}

    def __init__(self, ground, systems, log):
        self._g, self._sys, self._log = ground, systems, log

    def initialize(self, product_features, temporal_modulation=None, markov_alphas=None):
        self._g.initialize(product_features, temporal_modulation, markov_alphas)
        for name, s in self._sys.items():
            s.initialize(product_features, temporal_modulation, markov_alphas)
            if _is_deep(name):
                s._teacher_forced = True

    def sample_features(self, current_time=0.0):
        prev = dict(getattr(self._g, "_last_features", {}) or {})
        realized = self._g.sample_features(current_time=current_time)
        heads = {name: s.distribution_params(current_time, realized_features=realized, prev_features=prev)
                 for name, s in self._sys.items()}
        for name, s in self._sys.items():
            if _is_deep(name):
                s.observe_external(realized)
        for head, feat in self.HEAD_FEAT.items():
            ctx = self._log.next_ctx("arrival")
            rv = realized.get(feat, "")
            for name in self._sys:
                params = {"family": "categorical", "probs": heads[name].get(head, {})}
                self._log.record(name, "arrival", ctx, "", current_time, head,
                                 "categorical", params, rv, 0)
        return realized


def _build_systems(fs: SimFactorySet, seed: int) -> Dict[str, SimulationConfig]:
    return {
        "GroundSim": fs.base(seed, 1), "DeepSim": fs.deep(seed, 1),
        "RefSim-M": fs.refm(seed, 1),
        "RefSim-V": fs.refv(seed, 1), "RefSim-W": fs.refw(seed, 1),
    }


def run_shadow_pilot(
    seed: int, *, days: int, out_dir: Path, warmup_days: float = 1.0,
) -> int:
    """Run one shadow run for a seed, write the CSVs and return the row count."""
    fs = SimFactorySet(duration_days=days, warmup_days=warmup_days)
    pc = fs.process_config
    cfgs = _build_systems(fs, seed)
    log = ShadowLog(seed, warmup_time=warmup_days * 86400.0)

    systems_pt = {n: c.process_time_strategy for n, c in cfgs.items()}
    systems_tr = {n: c.transition_strategy for n, c in cfgs.items()}
    systems_sv = {n: c.survival_strategy for n, c in cfgs.items()}
    systems_rt = {n: c.repair_strategy for n, c in cfgs.items()}
    systems_pr = {n: c.product_strategy for n, c in cfgs.items()}

    g = cfgs["GroundSim"]
    sv_wrap = _ShadowSV(g.survival_strategy, systems_sv, log)
    steer = SimulationConfig(
        process_time_strategy=_ShadowPT(g.process_time_strategy, systems_pt, log),
        transition_strategy=_ShadowTR(g.transition_strategy, systems_tr, log),
        survival_strategy=sv_wrap,
        repair_strategy=_ShadowRT(g.repair_strategy, systems_rt, log, survival_wrapper=sv_wrap),
        product_strategy=_ShadowPR(g.product_strategy, systems_pr, log),
        duration_days=days, warmup_days=warmup_days, seed=seed, run_id=1,
    )
    engine = SimulationEngine(pc, steer)
    engine.run(label=f"verification-shadow seed={seed}")
    sv_wrap.finalize({sid: float(st.accumulated_op_time) for sid, st in engine._stations.items()})

    log.write(out_dir)
    return sum(len(rows) for rows in log.rows.values())
