"""Bundled closed-loop sweep runs for the data-regime analysis.

Build the configuration tree first: python -m scripts.build_sensitivity_tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.experiments.sim_runner import SimFactorySet, load_model_paths, run_replications  # noqa: E402
from src.config.simulation_config import (  # noqa: E402
    TRAIN_RATIO,
    SIM_START_TIMESTAMP,
    SimulationConfig,
    get_seeds, N_RUNS,
)
from src.fitting.ref_data_preparation.ref_analyzer import RefSimAnalyzer  # noqa: E402

REFERENCE_FACTORIES = {
    "RefSim-M": "refm",
    "RefSim-V": "refv",
    "RefSim-W": "refw",
}


def _run_config(
    name: str, data_dir: Path, model_dir: Path, ref_variants, seeds,
    shared: dict, *, with_dec: bool = False,
) -> tuple[dict, "SimFactorySet"]:
    """Run one sweep configuration; returns (runs, factory set).

    The factory set is returned so callers that need more replications on the
    same fit do not rebuild the analyzer, and it is deliberately NOT part of
    the runs dict so it can never reach the pickle.
    """
    print(f"\n[{name}] refit RefSim (extended analyzer) on {data_dir.name} …")
    stats = RefSimAnalyzer(data_dir=data_dir, train_ratio=TRAIN_RATIO).extract_all()
    fs = SimFactorySet(model_paths=load_model_paths(model_dir), stats_data=stats)
    out: dict = {"provenance": {}}
    out["GroundSim"] = shared["GroundSim"]
    out["provenance"]["GroundSim"] = "shared across configurations (generator-only, CRN)"
    if with_dec:
        out["GroundSim-DEC"] = shared["GroundSim-DEC"]
        out["provenance"]["GroundSim-DEC"] = "shared across configurations (generator-only, CRN)"
    t0 = time.time()
    out["DeepSim"] = run_replications("DeepSim", fs.deep, seeds,
                                      return_kpis=True, return_ct=True)
    out["provenance"]["DeepSim"] = "recomputed (this configuration's model set)"
    print(f"  DeepSim    recomputed ({time.time()-t0:.0f}s)")
    for rv in ref_variants:
        sysname = [k for k, v in REFERENCE_FACTORIES.items() if v == rv][0]
        t0 = time.time()
        out[sysname] = run_replications(sysname, getattr(fs, rv), seeds,
                                        return_kpis=True, return_ct=True)
        out["provenance"][sysname] = "recomputed (RefSim refit)"
        print(f"  {sysname:10} recomputed ({time.time()-t0:.0f}s)")
    return out, fs


def _hybrid_factory(fs: SimFactorySet, *, stat_survival: bool):
    """Learned core with the statistical repair module (module selection), and
    optionally the statistical survival module as the exposure-only contrast."""
    def make(seed: int, run_id: int) -> SimulationConfig:
        deep = fs.deep(seed, run_id)
        ref = fs.refm(seed, run_id)
        return SimulationConfig(
            process_time_strategy=deep.process_time_strategy,
            transition_strategy=deep.transition_strategy,
            survival_strategy=ref.survival_strategy if stat_survival else deep.survival_strategy,
            repair_strategy=ref.repair_strategy,
            product_strategy=deep.product_strategy,
            duration_days=deep.duration_days, warmup_days=deep.warmup_days,
            seed=seed, initial_orders=deep.initial_orders, run_id=run_id,
            start_timestamp=SIM_START_TIMESTAMP,
        )
    return make


def _first_data_dir(parent: Path) -> Path | None:
    dirs = sorted(parent.rglob("data_4-stage-crossover-rework_*"))
    return dirs[0] if dirs else (parent if any(parent.rglob("*_events_*.csv")) else None)


DATA_COMPONENTS = (
    ("released_order", "product_data"),
    ("processing", "process_time_data"),
    ("routing", "transition_data"),
    ("survival", "survival_data"),
    ("repair", "repair_time_data"),
)


def _observation_counts(section: str, config: str, model_dir: Path) -> list[dict]:
    """Per-component observation counts of one derivation.

    The training logs are excluded from the release; exporting the counts per
    derivation horizon keeps the sample sizes verifiable without them.
    """
    rows = []
    for component, data_dir in DATA_COMPONENTS:
        meta_path = model_dir / data_dir / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        rows.append({
            "section": section,
            "config": config,
            "component": component,
            "n_total": meta.get("n_total"),
            "n_events": meta.get("n_events"),
            "n_censored": meta.get("n_censored"),
        })
    return rows


# Number of 365-day derivation draws the data-regime analysis uses.
N_DRAW365_CONFIGS = 5


def _sweep_horizon(args: argparse.Namespace, seeds, shared: dict, result: dict) -> None:
    for hd in sorted((args.sensitivity_root / "data").glob("days_*")):
        model_dir = args.sensitivity_root / "models" / hd.name
        data_dir = _first_data_dir(hd)
        if data_dir is None or not (model_dir / "trained_model_paths.json").exists():
            print(f"[skip horizon {hd.name}] data/model missing")
            continue
        result["horizon"][hd.name], _ = _run_config(
            f"horizon/{hd.name}", data_dir, model_dir, ["refm"], seeds, shared)
        _dump(args.output_file, result)


def _sweep_draw365(args: argparse.Namespace, seeds, shared: dict, result: dict) -> None:
    draws = []
    if args.production_data_dir is not None:
        draws.append(
            ("production", args.production_data_dir, args.production_model_dir)
        )
    for sd in sorted((args.sensitivity_root / "data").glob("seed_*")):
        model_dir = args.sensitivity_root / "models" / sd.name
        data_dir = _first_data_dir(sd)
        if data_dir is not None and (model_dir / "trained_model_paths.json").exists():
            draws.append((sd.name, data_dir, model_dir))
    for name, data_dir, model_dir in draws[:N_DRAW365_CONFIGS]:
        result["draw365"][name], _ = _run_config(
            f"draw365/{name}", data_dir, model_dir, ["refm", "refv", "refw"],
            seeds, shared, with_dec=True)
        _dump(args.output_file, result)


def _sweep_draw31(args: argparse.Namespace, seeds, shared: dict, result: dict) -> None:
    d31 = args.sensitivity_root / "data" / "31d_seed_variance"
    if not d31.exists():
        print("[draw31] 31d_seed_variance structure not present as per-draw dirs — skipped")
        return
    for sub in sorted(p for p in d31.iterdir() if p.is_dir()):
        model_dir = (
            args.sensitivity_root / "models" / "31d_seed_variance" / sub.name
        )
        data_dir = _first_data_dir(sub)
        if data_dir is None or not (model_dir / "trained_model_paths.json").exists():
            continue
        cfg_runs, fs31 = _run_config(
            f"draw31/{sub.name}", data_dir, model_dir, ["refm"],
            seeds, shared, with_dec=True)
        for key, sv in (("DeepSim-StatRepair", False), ("DeepSim-StatRepair-StatSurvival", True)):
            t0 = time.time()
            cfg_runs[key] = run_replications(
                f"{sub.name}/{key}", _hybrid_factory(fs31, stat_survival=sv),
                seeds, return_kpis=True, return_ct=True)
            cfg_runs["provenance"][key] = "recomputed (module selection)"
            print(f"  {key:32} recomputed ({time.time()-t0:.0f}s)")
        result["draw31"][sub.name] = cfg_runs
        _dump(args.output_file, result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-file",
        type=Path,
        default=REPO / "results/verification/sensitivity_sweeps.pkl",
    )
    ap.add_argument(
        "--sensitivity-root",
        type=Path,
        required=True,
        help="Directory containing data/ and models/ for the sensitivity configurations",
    )
    ap.add_argument(
        "--production-data-dir",
        type=Path,
        help="Optional production training-data directory to include as a 365-day draw",
    )
    ap.add_argument(
        "--production-model-dir",
        type=Path,
        default=REPO / "models",
        help="Model registry paired with --production-data-dir (default: models/)",
    )
    ap.add_argument("--num-seeds", type=int, default=N_RUNS)
    ap.add_argument("--only", choices=["horizon", "draw365", "draw31"], default=None,
                    help="Recompute only one section; other sections are kept "
                         "from an existing output pickle.")
    args = ap.parse_args()
    seeds = get_seeds(args.num_seeds)
    result: dict = {"seeds": seeds, "horizon": {}, "draw365": {}, "draw31": {}}
    if args.only is not None and args.output_file.exists():
        with open(args.output_file, "rb") as f:
            previous = pickle.load(f)
        for section in ("horizon", "draw365", "draw31"):
            if section != args.only:
                result[section] = previous.get(section, {})

    # Paired GroundSim + decorrelated reference: generator-only, independent of
    # any training data, therefore computed once and shared across configs
    # (common random numbers make them bit-identical per seed either way).
    fs_shared = SimFactorySet()
    shared = {}
    t0 = time.time()
    shared["GroundSim"] = run_replications(
        "GroundSim", fs_shared.base, seeds, return_kpis=True, return_ct=True)
    print(f"[shared] GroundSim ({time.time()-t0:.0f}s)")
    t0 = time.time()
    shared["GroundSim-DEC"] = run_replications(
        "GroundSim-DEC", fs_shared.floor, seeds, return_kpis=True, return_ct=True)
    print(f"[shared] GroundSim-DEC ({time.time()-t0:.0f}s)")

    # (a) Horizon sweep — RefSim-M
    if args.only in (None, "horizon"):
        _sweep_horizon(args, seeds, shared, result)

    # (b) Five 365-day draw configs — RefSim-M/V/W
    if args.only in (None, "draw365"):
        _sweep_draw365(args, seeds, shared, result)

    # (c) 31-day draw configs — RefSim-M
    if args.only in (None, "draw31"):
        _sweep_draw31(args, seeds, shared, result)

    _dump(args.output_file, result)

    # Observation counts are metadata reads only and must cover every section
    # regardless of --only, or a partial rerun would truncate the table.
    counts = []
    for hd in sorted((args.sensitivity_root / "data").glob("days_*")):
        counts += _observation_counts("horizon", hd.name,
                                      args.sensitivity_root / "models" / hd.name)
    if args.production_data_dir is not None:
        counts += _observation_counts("draw365", "production", args.production_model_dir)
    for sd in sorted((args.sensitivity_root / "data").glob("seed_*")):
        counts += _observation_counts("draw365", sd.name,
                                      args.sensitivity_root / "models" / sd.name)
    d31_counts_dir = args.sensitivity_root / "data" / "31d_seed_variance"
    if d31_counts_dir.exists():
        for sub in sorted(q for q in d31_counts_dir.iterdir() if q.is_dir()):
            counts += _observation_counts(
                "draw31", sub.name,
                args.sensitivity_root / "models" / "31d_seed_variance" / sub.name)

    if counts:
        counts_path = args.output_file.parent / "observation_counts.csv"
        with open(counts_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)
        print(f"Observation counts → {counts_path}")

    print(f"\nDONE → {args.output_file}")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    main()
