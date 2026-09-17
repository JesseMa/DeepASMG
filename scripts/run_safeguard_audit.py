"""Side-effect-free instrumented closed-loop safeguard audit.

Reruns the authoritative closed loop on the same seeds, horizon and
common-random-number plan with the read-only `_sg_*` counters captured. The
instrumentation consumes no random numbers and changes no control flow.

Hard gate: every run must reproduce the authoritative bundle bit-identically in
cycle times, completion counts and KPIs, otherwise the counters are not
integrated.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.experiments.sim_runner import SimFactorySet, run_replications  # noqa: E402
from src.config.simulation_config import (  # noqa: E402
    N_RUNS, SIMULATION_DAYS, WARMUP_DAYS, get_seeds,
)
from scripts.run_closed_loop import build_systems  # noqa: E402

AUTH_PKL = REPO / "results/verification/closed_loop_runs.pkl"
AUTH_DIST = REPO / "results/verification/system_distances.csv"
OUT_PKL = REPO / "results/verification/closed_loop_runs_safeguards.pkl"
AUDIT_DIR = REPO / "results/execution_audit"

EVAL_DAYS = SIMULATION_DAYS


def run_instrumented(seeds, out_pkl=OUT_PKL):
    fs = SimFactorySet(duration_days=EVAL_DAYS, warmup_days=WARMUP_DAYS)
    systems = build_systems(fs)
    runs_by_sim = {}
    for name, fac in systems.items():
        t0 = time.time()
        runs_by_sim[name] = run_replications(name, fac, seeds, return_kpis=True, return_ct=True)
        print(f"  {name:14} {len(runs_by_sim[name])} runs ({time.time()-t0:.0f}s)", flush=True)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump({"runs_by_sim": runs_by_sim, "seeds": seeds,
                     "days": EVAL_DAYS, "warmup": WARMUP_DAYS}, f)
    return runs_by_sim


def gate_no_behavior_change(runs_by_sim, auth_pkl=AUTH_PKL, auth_dist=AUTH_DIST,
                            audit_dir=AUDIT_DIR):
    auth = pickle.load(auth_pkl.open("rb"))["runs_by_sim"]
    authw = {}
    with auth_dist.open() as f:
        for row in csv.DictReader(f):
            authw[(row["system"], int(row["seed"]))] = float(row["w1"])
    gnd = {r["seed"]: r["ct"] for r in runs_by_sim["GroundSim"]}
    rows, ok_all = [], True
    for sys_name, runs in runs_by_sim.items():
        for r in runs:
            a = next(x for x in auth[sys_name] if x["seed"] == r["seed"])
            ct_ok = bool(np.array_equal(r["ct"], a["ct"]))
            nv_ok = r["kpis"]["n_valid"] == a["kpis"]["n_valid"]
            kpi_ok = all(abs(float(r["kpis"][k]) - float(a["kpis"][k])) == 0.0
                         for k in a["kpis"] if isinstance(a["kpis"][k], (int, float)))
            w1_ok = True
            if sys_name != "GroundSim" and (sys_name, r["seed"]) in authw:
                w1 = float(wasserstein_distance(gnd[r["seed"]], r["ct"]))
                w1_ok = abs(w1 - authw[(sys_name, r["seed"])]) < 1e-8
            row_ok = ct_ok and nv_ok and kpi_ok and w1_ok
            ok_all = ok_all and row_ok
            rows.append({"system": sys_name, "seed": r["seed"],
                         "ct_bit_identical": int(ct_ok), "n_valid_match": int(nv_ok),
                         "kpi_exact": int(kpi_ok), "w1_lt_1e-8": int(w1_ok), "pass": int(row_ok)})
    with (audit_dir / "m6_no_behavior_change.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "seed", "ct_bit_identical",
                                          "n_valid_match", "kpi_exact", "w1_lt_1e-8", "pass"])
        w.writeheader()
        w.writerows(rows)
    return ok_all, rows


def aggregate(runs_by_sim, audit_dir=AUDIT_DIR):
    """Flatten per-(system,seed) safeguard counters and separate the two categories."""
    by_seed = []
    for sys_name, runs in runs_by_sim.items():
        for r in runs:
            sg = r["meta"].get("safeguards", {})
            rm = r["meta"].get("routing_mask")
            row = {"system": sys_name, "seed": r["seed"]}
            for mod, d in sg.items():
                for k, v in d.items():
                    row[f"{mod}.{k}"] = v
            row["ref_mask_calls"] = int(rm["mask_calls"]) if rm else 0
            row["ref_mask_effective"] = int(rm["mask_effective"]) if rm else 0
            row["ref_mask_fallback"] = int(rm["mask_fallback"]) if rm else 0
            by_seed.append(row)

    cols = ["system", "seed"] + sorted({k for r in by_seed for k in r if k not in ("system", "seed")})
    with (audit_dir / "m6_safeguards_by_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in by_seed:
            w.writerow({c: r.get(c, 0) for c in cols})

    # ---- category tables (summed over seeds, per system) ----
    def s(system, key):
        return sum(r.get(key, 0) for r in by_seed if r["system"] == system)

    systems = list(runs_by_sim.keys())

    # Renorm counts mix two definitions: Ref removed_mass>0, Deep any-inadmissible.
    routine_rows = []
    for sysn in systems:
        ref_calls, ref_eff = s(sysn, "ref_mask_calls"), s(sysn, "ref_mask_effective")
        deep_calls, deep_eff = s(sysn, "routing.mask_calls"), s(sysn, "routing.mask_effective")
        mask_calls = ref_calls + deep_calls
        mask_eff = ref_eff + deep_eff
        proc_clip = s(sysn, "processing.proc_nnclip")
        proc_clip_calls = s(sysn, "processing.proc_nnclip_calls")
        ttf_clip = s(sysn, "survival.ttf_nnclip")
        ttf_clip_calls = s(sysn, "survival.ttf_nnclip_calls")
        stress_floor = s(sysn, "repair.stress_floor")
        stress_calls = s(sysn, "repair.stress_calls")
        routine_rows.append({
            "system": sysn,
            "routing_mask_calls": mask_calls, "routing_renormalizations": mask_eff,
            "routing_renorm_rate": (mask_eff / mask_calls) if mask_calls else "NOT_APPLICABLE",
            "nn_param_clip_processing": proc_clip, "nn_param_clip_processing_calls": proc_clip_calls,
            "nn_param_clip_survival": ttf_clip, "nn_param_clip_survival_calls": ttf_clip_calls,
            "repair_stress_floor_binds": stress_floor, "repair_stress_calls": stress_calls,
        })
    if not routine_rows:
        raise SystemExit("No systems in the instrumented rerun: routine-constraint "
                         "table would be empty.")
    with (audit_dir / "m6_routine_constraint_enforcement.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(routine_rows[0].keys()))
        w.writeheader()
        w.writerows(routine_rows)

    corr_rows = []
    for sysn in systems:
        corr_rows.append({
            "system": sysn,
            "repair_duration_floor_binds": s(sysn, "repair.repair_clamp"),
            "repair_draws": s(sysn, "repair.repair_draws"),
            "processing_floor_binds": s(sysn, "processing.proc_clamp"),
            "processing_draws": s(sysn, "processing.proc_draws"),
            "ttf_floor_binds": s(sysn, "survival.ttf_clamp"),
            "ttf_draws": s(sysn, "survival.ttf_draws"),
            "repair_log0_guard": s(sysn, "repair.repair_logu_guard"),
            "ttf_log0_guard": s(sysn, "survival.ttf_logu_guard"),
            "zero_mass_routing_fallback": s(sysn, "ref_mask_fallback"),
            "deadlock_detections_recoveries": s(sysn, "deadlock.deadlock_count"),
        })
    if not corr_rows:
        raise SystemExit("No systems in the instrumented rerun: corrective-safeguard "
                         "table would be empty.")
    with (audit_dir / "m6_corrective_safeguards.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        w.writerows(corr_rows)

    return routine_rows, corr_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    ap.add_argument("--closed-loop-file", type=Path, default=AUTH_PKL)
    ap.add_argument("--system-distances", type=Path, default=AUTH_DIST)
    ap.add_argument("--output-dir", type=Path, default=AUDIT_DIR)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The default leaves the bulky intermediate next to the bundle it is diffed against.
    out_pkl = (OUT_PKL if args.output_dir == AUDIT_DIR
               else args.output_dir / OUT_PKL.name)

    seeds = get_seeds(args.num_seeds)
    if not args.closed_loop_file.exists():
        raise SystemExit(
            f"Authoritative closed-loop bundle not found: {args.closed_loop_file}\n"
            "Run 'python -m scripts.run_closed_loop' first."
        )
    print(f"[safeguards] instrumented rerun: {len(seeds)} seeds x {EVAL_DAYS}d")
    runs_by_sim = run_instrumented(seeds, out_pkl)
    ok, nbc_rows = gate_no_behavior_change(
        runs_by_sim, args.closed_loop_file, args.system_distances, args.output_dir,
    )
    routine_rows, corr_rows = aggregate(runs_by_sim, args.output_dir)

    total_routine_renorm = sum(r["routing_renormalizations"] for r in routine_rows)
    total_routing_calls = sum(r["routing_mask_calls"] for r in routine_rows)
    total_corrective = sum(
        r["repair_duration_floor_binds"] + r["processing_floor_binds"] + r["ttf_floor_binds"]
        + r["repair_log0_guard"] + r["ttf_log0_guard"] + r["zero_mass_routing_fallback"]
        + r["deadlock_detections_recoveries"] for r in corr_rows)

    summary = {
        "no_behavior_change_pass": bool(ok),
        "n_runs": len(nbc_rows),
        "n_runs_bit_identical": sum(r["ct_bit_identical"] for r in nbc_rows),
        "routine_total_routing_decisions": total_routing_calls,
        "routine_total_renormalizations": total_routine_renorm,
        "corrective_total_activations": total_corrective,
        "corrective_by_type": {
            "repair_duration_floor": sum(r["repair_duration_floor_binds"] for r in corr_rows),
            "processing_floor": sum(r["processing_floor_binds"] for r in corr_rows),
            "ttf_floor": sum(r["ttf_floor_binds"] for r in corr_rows),
            "repair_log0_guard": sum(r["repair_log0_guard"] for r in corr_rows),
            "ttf_log0_guard": sum(r["ttf_log0_guard"] for r in corr_rows),
            "zero_mass_routing_fallback": sum(r["zero_mass_routing_fallback"] for r in corr_rows),
            "deadlock_detections_recoveries": sum(r["deadlock_detections_recoveries"] for r in corr_rows),
        },
    }
    (args.output_dir / "m6_safeguard_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    print("\n===== Safeguard audit summary =====")
    print(f"no-behavior-change: {'PASS' if ok else 'FAIL'} "
          f"({summary['n_runs_bit_identical']}/{summary['n_runs']} bit-identical)")
    print(f"routine renormalizations: {total_routine_renorm:,} / {total_routing_calls:,} routing decisions")
    print(f"corrective activations (total): {total_corrective:,}")
    for k, v in summary["corrective_by_type"].items():
        print(f"    {k}: {v:,}")


if __name__ == "__main__":
    main()
