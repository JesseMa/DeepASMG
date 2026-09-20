"""Producer: writes all result CSVs plus a directory manifest."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.evaluation import routing_evaluation as routing  # noqa: E402
from src.evaluation import score_aggregation as scores  # noqa: E402
from src.evaluation import system_evaluation as system  # noqa: E402
from src.evaluation import bathtub_hazard as hazard  # noqa: E402
from src.experiments.sim_runner import SimFactorySet, get_process_config, load_stats_data  # noqa: E402
from src.experiments.shadow_evaluation import COMPONENTS, SYSTEMS  # noqa: E402

TABLE_MANIFEST = {
    "tab:component_summary": {
        "desc": "Pooled component-level verification scores: Brier and ECE "
                "(released-order), CRPS and NLL (processing, survival, repair), "
                "routing L1 per (decision point, variant, visit)",
        "sources": ["scores_categorical.csv", "scores_continuous.csv", "routing_compare.csv"],
    },
    "tab:system_summary": {
        "desc": "System-level throughput time: W1 mean/SD and D_KS mean/SD over the "
                "ten evaluation seeds, significant-KPI counts after BH adjustment",
        "sources": ["w1_summary.csv", "system_distances.csv", "kpi_bh.csv"],
    },
    "fig:system_fidelity": {
        "desc": "Per-seed W1 per system and the paired DeepSim minus GroundSim-DEC "
                "difference with t-based CI and Wilcoxon signed-rank test",
        "sources": ["system_distances.csv", "w1_wilcoxon.csv"],
    },
    "fig:kpi_deviation_matrix": {
        "desc": "Relative KPI deviation per system with BH-adjusted paired Wilcoxon tests",
        "sources": ["kpi_panel.csv", "kpi_bh.csv"],
    },
    "fig:component_ablation": {
        "desc": "Two-way component substitutions and module selection: per-seed W1 "
                "per configuration (written by run_substitutions)",
        "sources": ["component_substitutions.csv"],
    },
    "fig:data_regime": {
        "desc": "Data-regime characterization: derivation-horizon sweep and the "
                "repeated 31-day and 365-day derivations, with observation counts",
        "sources": ["sweep_horizon.csv", "sweep_draw31.csv", "sweep_draw365.csv",
                    "observation_counts.csv"],
    },
}

PAPER_OUTPUTS = {
    "component_scores": [
        "scores_continuous.csv",
        "scores_categorical.csv",
    ],
    "routing_comparison": ["routing_compare.csv"],
    "system_comparison": [
        "system_distances.csv",
        "w1_summary.csv",
        "w1_wilcoxon.csv",
        "kpi_panel.csv",
        "kpi_bh.csv",
        "kpi_absolute.csv",
    ],
    "component_substitutions": ["component_substitutions.csv"],
    "data_sensitivity": [
        "sweep_horizon.csv",
        "sweep_draw31.csv",
        "sweep_draw365.csv",
        "observation_counts.csv",
    ],
}

DIAGNOSTICS = {
    "hazard": ["hazard_true_grid.csv", "hazard_params.csv"],
}


def _write(df_rows, path: Path) -> int:
    pd.DataFrame(df_rows).to_csv(path, index=False)
    return len(df_rows)


def _produce_scores(shadow_dir: Path, output_dir: Path) -> dict[str, int]:
    if not any((shadow_dir / f"{name}__transition.csv").exists() for name in SYSTEMS):
        print(f"[skip] component scores: no shadow bundle under {shadow_dir}")
        return {}
    aggregated = scores.aggregate_all(shadow_dir, SYSTEMS, COMPONENTS)
    return {
        "scores_continuous.csv": _write(
            aggregated["continuous"], output_dir / "scores_continuous.csv"
        ),
        "scores_categorical.csv": _write(
            aggregated["categorical"], output_dir / "scores_categorical.csv"
        ),
    }


def _produce_routing(shadow_dir: Path, output_dir: Path) -> dict[str, int]:
    context_file = shadow_dir / "_context_variant.csv"
    if not context_file.exists():
        print(f"[skip] routing comparison: {context_file} missing")
        return {}

    process_config = get_process_config()
    deep_vectors = routing.deep_shadow_vectors(shadow_dir, "DeepSim")
    rows = routing.compare_rows(process_config, deep_vectors, "DeepSim")
    realized = set(deep_vectors)

    fs = SimFactorySet()
    rng = np.random.default_rng(0)
    marginal_fitted = routing.ref_fitted_vectors(fs.module("stat", "tr", rng), process_config, realized)
    for name, fitted in (
        ("RefSim-M", marginal_fitted),
        ("RefSim-W", marginal_fitted),
        ("RefSim-V", routing.ref_fitted_vectors(fs.module("statv", "tr", rng), process_config, realized)),
    ):
        rows += routing.compare_rows(process_config, fitted, name)
    return {
        "routing_compare.csv": _write(rows, output_dir / "routing_compare.csv")
    }


def _produce_hazard(output_dir: Path) -> dict[str, int]:
    process_config = get_process_config()
    stats_data = load_stats_data()
    grid_rows, param_rows = [], []
    for station in process_config.stations:
        if not (
            station.is_machine
            and station.mttr > 0
            and station.ttf_scale_seconds > 0
        ):
            continue
        scale = station.ttf_scale_seconds
        weibull = stats_data.get("weibull_ttf", {}).get("params", {}).get(station.id)
        t_max = weibull[1] * 2.0 if weibull else scale * 2.0
        time_grid, true_hazard = hazard.true_hazard_grid(scale, t_max)
        grid_rows.extend(
            {
                "station": station.id,
                "t_op_s": float(time_value),
                "h_true": float(hazard_value),
            }
            for time_value, hazard_value in zip(time_grid, true_hazard, strict=True)
        )
        param_rows.append(
            {
                "station": station.id,
                "ttf_scale_seconds": float(scale),
                "refm_mttf": stats_data["mttf_data"].get(station.id),
                "refw_shape": weibull[0] if weibull else None,
                "refw_scale": weibull[1] if weibull else None,
            }
        )
    return {
        "hazard_true_grid.csv": _write(
            grid_rows, output_dir / "hazard_true_grid.csv"
        ),
        "hazard_params.csv": _write(param_rows, output_dir / "hazard_params.csv"),
    }


def _normalise_seeds(values) -> list[int] | None:
    if values is None:
        return None
    return [int(value) for value in values]


def _observed_seed_order(runs: dict[str, list[dict]]) -> list[int] | None:
    orders = {
        name: [int(run["seed"]) for run in system_runs if "seed" in run]
        for name, system_runs in runs.items()
    }
    nonempty = {name: order for name, order in orders.items() if order}
    if not nonempty:
        return None
    first_name, first_order = next(iter(nonempty.items()))
    for name, order in nonempty.items():
        if order != first_order:
            raise ValueError(
                "Closed-loop run seed order differs across systems: "
                f"{first_name}={first_order}, {name}={order}"
            )
    return first_order


def _observed_run_meta_value(runs: dict[str, list[dict]], field: str):
    observed = {
        run["meta"][field]
        for system_runs in runs.values()
        for run in system_runs
        if isinstance(run.get("meta"), dict) and field in run["meta"]
    }
    if len(observed) > 1:
        raise ValueError(
            f"Closed-loop runs disagree on meta.{field}: {sorted(observed)}"
        )
    return next(iter(observed)) if observed else None


def _cross_checked_bundle_value(
    bundle: dict,
    bundle_field: str,
    observed_value,
    observed_field: str,
):
    declared = bundle.get(bundle_field)
    if declared is not None and observed_value is not None and declared != observed_value:
        raise ValueError(
            f"Closed-loop bundle {bundle_field}={declared} disagrees with "
            f"runs_by_sim[*].meta.{observed_field}={observed_value}"
        )
    if declared is not None and observed_value is not None:
        source = (
            f"bundle.{bundle_field}, validated against "
            f"runs_by_sim[*].meta.{observed_field}"
        )
    elif declared is not None:
        source = f"bundle.{bundle_field}"
    elif observed_value is not None:
        source = f"derived from runs_by_sim[*].meta.{observed_field}"
    else:
        source = None
    return declared if declared is not None else observed_value, source


def _closed_loop_metadata(bundle: dict, source_file: Path) -> dict:
    runs = bundle["runs_by_sim"]
    declared_seeds = _normalise_seeds(bundle.get("seeds"))
    run_seeds = _observed_seed_order(runs)
    if declared_seeds is not None and run_seeds is not None:
        if declared_seeds != run_seeds:
            raise ValueError(
                "Closed-loop bundle seed metadata does not match runs_by_sim: "
                f"declared={declared_seeds}, observed={run_seeds}"
            )

    duration_days, duration_source = _cross_checked_bundle_value(
        bundle,
        "days",
        _observed_run_meta_value(runs, "duration_days"),
        "duration_days",
    )
    warmup_days, warmup_source = _cross_checked_bundle_value(
        bundle,
        "warmup",
        _observed_run_meta_value(runs, "warmup_days"),
        "warmup_days",
    )

    seeds = declared_seeds if declared_seeds is not None else run_seeds
    seed_source = None
    if declared_seeds is not None and run_seeds is not None:
        seed_source = "bundle.seeds, validated against runs_by_sim[*].seed"
    elif declared_seeds is not None:
        seed_source = "bundle.seeds"
    elif run_seeds is not None:
        seed_source = "derived from runs_by_sim[*].seed"

    return {
        "source_bundle": source_file.name,
        "bundle_read": True,
        "duration_days": duration_days,
        "warmup_days": warmup_days,
        "seeds": seeds,
        "systems": sorted(runs),
        "runs_per_system": {
            name: len(system_runs) for name, system_runs in sorted(runs.items())
        },
        "field_sources": {
            "duration_days": duration_source,
            "warmup_days": warmup_source,
            "seeds": seed_source,
        },
    }


def _missing_closed_loop_metadata(source_file: Path) -> dict:
    return {
        "source_bundle": source_file.name,
        "bundle_read": False,
        "duration_days": None,
        "warmup_days": None,
        "seeds": None,
        "systems": None,
        "runs_per_system": None,
        "field_sources": {
            "duration_days": None,
            "warmup_days": None,
            "seeds": None,
        },
    }


def _produce_system(
    closed_loop_file: Path, output_dir: Path
) -> tuple[dict[str, int], dict]:
    if not closed_loop_file.exists():
        print(f"[skip] system-level tables: {closed_loop_file.name} missing")
        return {}, _missing_closed_loop_metadata(closed_loop_file)
    with open(closed_loop_file, "rb") as file:
        bundle = pickle.load(file)
    runs = bundle["runs_by_sim"]
    metadata = _closed_loop_metadata(bundle, closed_loop_file)

    panel, bh = [], []
    for name in runs:
        if name == system.REF_SYSTEM:
            continue
        panel += system.kpi_panel(runs, sys_name=name)
        bh += system.kpi_wilcoxon_bh(runs, sys_name=name)
    w1 = system.w1_paired_difference(
        runs, a_name="DeepSim", b_name="GroundSim-DEC"
    )
    absolute = [
        {"system": name, "seed": run["seed"], "kpi": kpi, "value": float(value)}
        for name, seed_runs in runs.items()
        for run in seed_runs
        for kpi, value in sorted(run["kpis"].items())
    ]
    return (
        {
            "system_distances.csv": _write(
                system.system_distance_table(runs),
                output_dir / "system_distances.csv",
            ),
            "w1_summary.csv": _write(
                system.w1_summary(runs), output_dir / "w1_summary.csv"
            ),
            "w1_wilcoxon.csv": _write([w1], output_dir / "w1_wilcoxon.csv"),
            "kpi_panel.csv": _write(panel, output_dir / "kpi_panel.csv"),
            "kpi_bh.csv": _write(bh, output_dir / "kpi_bh.csv"),
            "kpi_absolute.csv": _write(absolute, output_dir / "kpi_absolute.csv"),
        },
        metadata,
    )


def _system_w1(runs: dict, system_name: str) -> list[float]:
    return [float(x) for x in system.paired_w1(
        runs[system.REF_SYSTEM], runs[system_name]
    )]


def _horizon_sensitivity_rows(sweeps: dict) -> list[dict]:
    day_map = {
        "days_0014": 14,
        "days_0031": 31,
        "days_0061": 61,
        "days_0091": 91,
        "days_0183": 183,
        "days_0365": 365,
    }
    rows = []
    for config, runs in sweeps.get("horizon", {}).items():
        days = day_map[config]
        deep_w1 = _system_w1(runs, "DeepSim")
        refm_w1 = _system_w1(runs, "RefSim-M")
        rows.extend(
            {
                "config": config,
                "days": days,
                "seed": run["seed"],
                "deep_w1": deep_w1[index],
                "refm_w1": refm_w1[index],
            }
            for index, run in enumerate(runs["RefSim-M"])
        )
    return rows


def _draw365_sensitivity_rows(sweeps: dict) -> list[dict]:
    rows = []
    for config, runs in sweeps.get("draw365", {}).items():
        deep_w1 = _system_w1(runs, "DeepSim")
        refm_w1 = _system_w1(runs, "RefSim-M")
        refv_w1 = _system_w1(runs, "RefSim-V")
        refw_w1 = _system_w1(runs, "RefSim-W")
        dec_w1 = _system_w1(runs, "GroundSim-DEC")
        rows.extend(
            {
                "config": config,
                "seed": run["seed"],
                "deep_w1": deep_w1[index],
                "refm_w1": refm_w1[index],
                "refv_w1": refv_w1[index],
                "refw_w1": refw_w1[index],
                "groundsim_dec_w1": dec_w1[index],
            }
            for index, run in enumerate(runs["RefSim-M"])
        )
    return rows


def _draw31_sensitivity_rows(sweeps: dict) -> list[dict]:
    rows = []
    for config, runs in sweeps.get("draw31", {}).items():
        deep_w1 = _system_w1(runs, "DeepSim")
        refm_w1 = _system_w1(runs, "RefSim-M")
        dec_w1 = _system_w1(runs, "GroundSim-DEC")
        hybrid = (_system_w1(runs, "DeepSim-StatRepair")
                  if "DeepSim-StatRepair" in runs else None)
        hybrid_sv = (_system_w1(runs, "DeepSim-StatRepair-StatSurvival")
                     if "DeepSim-StatRepair-StatSurvival" in runs else None)
        for index, run in enumerate(runs["RefSim-M"]):
            row = {
                "config": config,
                "seed": run["seed"],
                "deep_w1": deep_w1[index],
                "refm_w1": refm_w1[index],
                "groundsim_dec_w1": dec_w1[index],
            }
            if hybrid is not None:
                row["hybrid_statrepair_w1"] = hybrid[index]
            if hybrid_sv is not None:
                row["hybrid_statrepair_statsurvival_w1"] = hybrid_sv[index]
            rows.append(row)
    return rows


def _sweep_tables(output_dir: Path) -> dict[str, list[dict]] | None:
    sweep_file = output_dir / "sensitivity_sweeps.pkl"
    if not sweep_file.exists():
        return None
    with open(sweep_file, "rb") as file:
        sweeps = pickle.load(file)
    return {
        "sweep_horizon.csv": _horizon_sensitivity_rows(sweeps),
        "sweep_draw365.csv": _draw365_sensitivity_rows(sweeps),
        "sweep_draw31.csv": _draw31_sensitivity_rows(sweeps),
    }


def _require_sweep_configurations(output_dir: Path) -> None:
    # checked before any table is rewritten, so an empty bundle aborts cleanly
    tables = _sweep_tables(output_dir)
    if tables is None:
        return
    empty = [name for name, rows in tables.items() if not rows]
    if empty:
        raise SystemExit(
            f"sensitivity_sweeps.pkl holds no configurations for {', '.join(empty)}; "
            "no table was rewritten"
        )


def _produce_sensitivity(output_dir: Path) -> dict[str, int]:
    tables = _sweep_tables(output_dir)
    if tables is None:
        print(f"[skip] sweep tables: {output_dir / 'sensitivity_sweeps.pkl'} missing")
        return {}
    return {name: _write(rows, output_dir / name) for name, rows in tables.items()}


def _shadow_metadata(shadow_dir: Path) -> dict:
    csv_files = sorted(shadow_dir.glob("*__*.csv")) if shadow_dir.is_dir() else []
    system_components = {
        tuple(path.stem.split("__", maxsplit=1))
        for path in csv_files
        if "__" in path.stem
    }

    seeds = (
        sorted(
            {
                int(path.name.removeprefix("_seed_"))
                for path in shadow_dir.glob("_seed_*")
                if path.is_dir()
                and path.name.removeprefix("_seed_").lstrip("-").isdigit()
            }
        )
        if shadow_dir.is_dir()
        else []
    )
    seed_source = "derived from _seed_<seed> directory names" if seeds else None

    if not seeds:
        for path in csv_files:
            try:
                observed = set()
                for chunk in pd.read_csv(path, usecols=["seed"], chunksize=250_000):
                    observed.update(int(value) for value in chunk["seed"].dropna())
                if observed:
                    seeds = sorted(observed)
                    seed_source = f"derived from {path.name}:seed"
                    break
            except (ValueError, KeyError):
                continue

    return {
        "source_bundle": shadow_dir.name,
        "bundle_read": bool(csv_files),
        "duration_days": None,
        "warmup_days": None,
        "seeds": seeds or None,
        "systems": sorted({system for system, _component in system_components})
        or None,
        "components": sorted(
            {component for _system, component in system_components}
        )
        or None,
        "field_sources": {
            "duration_days": None,
            "warmup_days": None,
            "seeds": seed_source,
        },
        "metadata_note": (
            "The shadow CSV format records seed and simulated timestamps but "
            "does not serialize the requested evaluation and warm-up durations; "
            "those fields therefore remain null."
        ),
    }


def _csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as file:
        rows = csv.reader(file)
        next(rows, None)
        return sum(1 for _row in rows)


def _csv_inventory(output_dir: Path, shadow_dir: Path) -> dict[str, int]:
    shadow_root = shadow_dir.resolve()
    inventory = {}
    for path in sorted(output_dir.rglob("*.csv")):
        if path.resolve().is_relative_to(shadow_root):
            continue
        relative = path.relative_to(output_dir).as_posix()
        inventory[relative] = _csv_row_count(path)
    return inventory


def _enrich_closed_loop_metadata(metadata: dict, output_dir: Path) -> dict:
    if metadata["seeds"] is not None:
        return metadata
    distances_path = output_dir / "system_distances.csv"
    if not distances_path.exists():
        return metadata

    distances = pd.read_csv(distances_path, usecols=["system", "seed"])
    enriched = dict(metadata)
    enriched["seeds"] = sorted(int(seed) for seed in distances["seed"].unique())
    enriched["systems"] = sorted(str(name) for name in distances["system"].unique())
    enriched["runs_per_system"] = {
        str(name): int(group["seed"].nunique())
        for name, group in distances.groupby("system")
    }
    enriched["field_sources"] = {
        **metadata["field_sources"],
        "seeds": "derived from system_distances.csv:seed",
    }
    return enriched


def _write_manifest(
    output_dir: Path,
    shadow_dir: Path,
    produced: dict[str, int],
    closed_loop_metadata: dict,
) -> None:
    files = _csv_inventory(output_dir, shadow_dir)
    closed_loop_metadata = _enrich_closed_loop_metadata(
        closed_loop_metadata, output_dir
    )
    manifest = {
        "schema_version": 2,
        "description": "Derived verification tables supporting the DeepASMG article",
        "evaluation": {
            "closed_loop": closed_loop_metadata,
            "shadow": _shadow_metadata(shadow_dir),
        },
        "files": files,
        "paper_tables": TABLE_MANIFEST,
        "paper_outputs": PAPER_OUTPUTS,
        "diagnostics": DIAGNOSTICS,
        "last_production_run": {
            "produced_csvs": {
                name: files[name] for name in sorted(produced) if name in files
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow-dir",
        type=Path,
        default=REPO / "results/verification/shadow",
    )
    parser.add_argument(
        "--closed-loop-file",
        type=Path,
        default=REPO / "results/verification/closed_loop_runs.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/verification",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, int] = {}
    _require_sweep_configurations(args.output_dir)
    produced.update(_produce_scores(args.shadow_dir, args.output_dir))
    produced.update(_produce_routing(args.shadow_dir, args.output_dir))
    produced.update(_produce_hazard(args.output_dir))
    system_files, closed_loop_metadata = _produce_system(
        args.closed_loop_file, args.output_dir
    )
    produced.update(system_files)
    produced.update(_produce_sensitivity(args.output_dir))

    _write_manifest(
        args.output_dir,
        args.shadow_dir,
        produced,
        closed_loop_metadata,
    )
    print("=== Result producer ===")
    for filename, row_count in produced.items():
        print(f"  {filename}: {row_count} rows")
    print(f"  manifest.json (paper labels {', '.join(TABLE_MANIFEST)})")


if __name__ == "__main__":
    main()
