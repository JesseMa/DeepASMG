"""Trains all five DeepSim models and extracts RefSim parameters from the same
training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


from src.config.simulation_config import (  # noqa: E402
    HPO_TEST_SIZE, MAX_EPOCHS, PATIENCE, TRAIN_RATIO, TRAIN_SEED,
)
from src.fitting.deep_training.foundation_training import (  # noqa: E402
    validate_deployment_eligibility,
)
from src.fitting.deep_training.train_process_time import train as train_pt  # noqa: E402
from src.fitting.deep_training.train_transition import train as train_tr  # noqa: E402
from src.fitting.deep_training.train_survival import train as train_sv  # noqa: E402
from src.fitting.deep_training.train_repair_time import train as train_rt  # noqa: E402
from src.fitting.deep_training.train_product import train as train_pr  # noqa: E402
from src.fitting.ref_data_preparation.ref_analyzer import (  # noqa: E402
    RefSimAnalyzer,
)

# Fallback hyperparameters: used only when <hpo_dir>/<model>_best_params.json
# is missing (HPO skipped); otherwise _get_params() loads the HPO bests.

# Display name and train() entry point per model, in training order.
TRAINERS = (
    ("process_time", "Process Time", train_pt),
    ("transition",   "Transition",   train_tr),
    ("survival",     "Survival",     train_sv),
    ("repair_time",  "Repair Time",  train_rt),
    ("product",      "Product",      train_pr),
)


# Prose fields of the training manifest: the only part a machine cannot derive.
MANIFEST_PROSE = {
    "schema_version": 2,
    "description": "Training configuration of the DeepASMG release model set, "
                   "emitted by scripts/train_models.py at training time.",
    "release_model_set": "DeepASMG v1.0.0",
    "path_base": "repository_root",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    model_dir: Path, data_dir: Path, params: Dict[str, Any],
    results: Dict[str, Dict[str, Any]], max_epochs: int, patience: int,
    test_size: float,
) -> Path:
    """Emit production_training_manifest.json from what actually ran.

    Everything here is read off the run itself — the data directory's
    run_metadata, the hyperparameters that were loaded, the training constants
    and the resulting artifacts — so the manifest cannot drift from the models
    it describes.
    """
    run_meta_path = data_dir / "run_metadata.json"
    run_meta = json.loads(run_meta_path.read_text()) if run_meta_path.exists() else {}

    logs = {}
    for label, pattern in (("event_log", "*_events_*.csv"), ("order_log", "*_orders_*.csv")):
        found = sorted(data_dir.glob(pattern))
        if found:
            logs[label] = {
                "excluded_release_path": str(found[0].relative_to(REPO))
                if found[0].is_relative_to(REPO) else str(found[0]),
                "sha256": _sha256(found[0]),
            }

    artifacts = {}
    for name, _, _ in TRAINERS:
        model_path = Path(results[name]["model_path"])
        if model_path.exists():
            artifacts[name] = {
                "model": str(model_path.relative_to(REPO))
                if model_path.is_relative_to(REPO) else str(model_path),
                "sha256": _sha256(model_path),
                "best_val_loss": float(results[name]["best_val_loss"]),
                "epochs_trained": int(results[name]["epochs_trained"]),
            }

    manifest = {
        **MANIFEST_PROSE,
        "environment": {
            "python": platform.python_version(),
            "requirements_file": "requirements.txt",
        },
        "training_data": {
            "generator": "GroundSim",
            "directory": data_dir.name,
            **{k: v for k, v in run_meta.items()},
            **logs,
        },
        "split": {
            "method": "chronological",
            "train_fraction": TRAIN_RATIO,
            "validation_fraction": test_size,
            "test_fraction": test_size,
            "refsim_train_fraction": TRAIN_RATIO,
        },
        "training": {
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "optimizer": "Adam",
            "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5},
            "export_format": "TorchScript",
            "seed": TRAIN_SEED,
            "hyperparameters": {k: dict(v) for k, v in params.items()},
        },
        "artifacts": artifacts,
    }

    out = model_dir / "production_training_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return out


def _get_params(model_name: str, hpo_dir: Path) -> Dict[str, Any]:
    """Load the HPO best params; their absence is an error, not a fallback.

    A silent default would let the sensitivity tree train on one set of
    hyperparameters while the production models use another, confounding data
    volume with hyperparameter mismatch.
    """
    from scripts.optimize_hyperparameters import load_best_params
    try:
        params = load_best_params(model_name, hpo_dir=hpo_dir)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Fail fast: no HPO best params for '{model_name}' in {hpo_dir}. "
            f"Run scripts.optimize_hyperparameters first, or point --hpo-dir "
            f"at a completed study."
        ) from exc
    print(f"  {model_name:<16} <- HPO best params")
    return params


def train_all(
    data_dir: Path,
    model_dir: Path,
    hpo_dir: Path,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    test_size: float = HPO_TEST_SIZE,
) -> Dict[str, Dict[str, Any]]:
    """Train all 5 DeepSim models and save results."""
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hyperparameters...")
    params = {name: _get_params(name, hpo_dir) for name, _, _ in TRAINERS}

    results = {}

    for i, (name, label, train_fn) in enumerate(TRAINERS, 1):
        print(f"\n[{i}/{len(TRAINERS)}] Training {label}...")
        results[name] = train_fn(
            data_dir=data_dir, model_dir=model_dir,
            max_epochs=max_epochs, test_size=test_size,
            patience=patience,
            **params[name],
        )
        print(f"  val_loss={results[name]['best_val_loss']:.6f}  "
              f"epochs={results[name]['epochs_trained']}")
        validate_deployment_eligibility(name, model_dir)

    model_paths = {
        "pt_model_path":     results["process_time"]["model_path"],
        "pt_metadata_path":  results["process_time"]["metadata_path"],
        "pt_ckpt_path":      results["process_time"]["best_ckpt_path"],
        "tr_model_path":     results["transition"]["model_path"],
        "tr_metadata_path":  results["transition"]["metadata_path"],
        "tr_ckpt_path":      results["transition"]["best_ckpt_path"],
        "sv_model_path":     results["survival"]["model_path"],
        "sv_metadata_path":  results["survival"]["metadata_path"],
        "sv_ckpt_path":      results["survival"]["best_ckpt_path"],
        "rt_model_path":     results["repair_time"]["model_path"],
        "rt_metadata_path":  results["repair_time"]["metadata_path"],
        "rt_ckpt_path":      results["repair_time"]["best_ckpt_path"],
        "pr_model_path":     results["product"]["model_path"],
        "pr_metadata_path":  results["product"]["metadata_path"],
        "pr_ckpt_path":      results["product"]["best_ckpt_path"],
    }

    paths_file = model_dir / "trained_model_paths.json"
    with open(paths_file, "w") as f:
        json.dump(model_paths, f, indent=2)
    print(f"\nModel paths saved to: {paths_file}")

    manifest_file = _write_manifest(
        model_dir, data_dir, params, results, max_epochs, patience, test_size,
    )
    print(f"Training manifest saved to: {manifest_file}")

    print("\nExtracting RefSim parameters (chronological 70% train cut per run)...")
    stat_analyzer = RefSimAnalyzer(data_dir=data_dir, train_ratio=TRAIN_RATIO)
    stats_data = stat_analyzer.extract_all()
    for cm in stats_data.get("train_cut_metadata", []):
        print(
            f"  {cm['run_key']}: kept {cm['n_train_events']:,}/{cm['n_total_events']:,} events"
            f" (cut at t={cm['train_cut_timestamp']})"
        )

    stats_file = model_dir / "statistic_params.pkl"
    with open(stats_file, "wb") as f:
        pickle.dump(stats_data, f)
    print(f"RefSim params saved to: {stats_file}")

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<18} {'Val Loss':>12} {'Epochs':>8}  Model File")
    print("-" * 70)
    for key, label, _ in TRAINERS:
        r = results[key]
        print(f"{label:<18} {r['best_val_loss']:>12.6f} {r['epochs_trained']:>8}  "
              f"{Path(r['model_path']).name}")
    print("=" * 70)

    return results


def find_newest_data_dir(parent: Path) -> Path:
    """Newest generated data directory under `parent` (single selection rule,
    shared with scripts.build_sensitivity_tree)."""
    dirs = sorted(parent.rglob("data_4-stage-crossover-rework_*"))
    if not dirs:
        raise FileNotFoundError(f"no generated data directory under {parent}")
    return dirs[-1]


def _find_data_dir(data_dir_arg: str | None) -> Path:
    if data_dir_arg:
        p = Path(data_dir_arg)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Data directory not found: {p}")

    parent_path = Path("data/training")
    if parent_path.exists():
        try:
            found = find_newest_data_dir(parent_path)
        except FileNotFoundError:
            pass
        else:
            print(f"Auto-detected data: {found}")
            return found

    raise FileNotFoundError(
        "No training data found. Run generate_training_data.py first, "
        "or specify --data-dir explicitly."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train all 5 DeepSim models + RefSim parameters.",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to training data directory with CSVs")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Output directory for models (default: models)")
    parser.add_argument("--hpo-dir", type=str, default="models/hpo",
                        help="Directory with HPO best params (default: models/hpo)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help=f"Maximum training epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early stopping patience (default: {PATIENCE})")
    args = parser.parse_args()

    data_dir = _find_data_dir(args.data_dir)

    print(f"Data:      {data_dir}")
    print(f"Models:    {args.model_dir}")
    print(f"HPO:       {args.hpo_dir}")
    print(f"Epochs:    {args.max_epochs}  |  Patience: {args.patience}")
    print()

    train_all(
        data_dir=data_dir,
        model_dir=Path(args.model_dir),
        hpo_dir=Path(args.hpo_dir),
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
