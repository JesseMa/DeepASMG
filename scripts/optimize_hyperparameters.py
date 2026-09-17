"""DeepSim hyperparameter optimization (Optuna)."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config.simulation_config import HPO_TEST_SIZE  # noqa: E402

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_objective(train_module: str, suggest_fn, *, train_kwargs: Dict[str, Any]):
    """Factory for HPO objective functions.

    ``train_module`` is the fully qualified module path of a train() function.
    """
    def objective(trial, data_dir: Path, model_dir: Path,
                  train_kwargs_override: Optional[Dict[str, Any]] = None) -> float:
        mod = importlib.import_module(train_module)
        train = mod.train

        params = suggest_fn(trial)

        trial_dir = model_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        shared_prep_dir = model_dir / "_shared_prep"

        effective_kwargs = dict(train_kwargs)
        if train_kwargs_override:
            effective_kwargs.update(train_kwargs_override)

        # Forward only extra params the respective train() signature accepts.
        extra = {k: v for k, v in params.items() if k not in
                 ("hidden_dims", "learning_rate", "dropout_rate", "batch_size")}
        results = train(
            data_dir      = data_dir,
            model_dir     = trial_dir,
            hidden_dims   = ast.literal_eval(params["hidden_dims"]),
            learning_rate = params["learning_rate"],
            dropout_rate  = params["dropout_rate"],
            batch_size    = params["batch_size"],
            _prep_dir     = shared_prep_dir,
            **extra,
            **effective_kwargs,
        )
        return results["best_val_loss"]

    return objective


# Broad bounds: minimal a-priori restriction so Optuna locates the optimal
# region itself.

def _suggest_process_time(trial) -> dict:
    return dict(
        hidden_dims   = trial.suggest_categorical("hidden_dims",
                            ["(512, 256)", "(1024, 512)", "(1024, 1024, 512)",
                             "(2048, 1024, 512)"]),
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 1e-2, log=True),
        dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.30),
        batch_size    = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
    )


def _suggest_transition(trial) -> dict:
    # Softmax saturation can occur even with the K=10 slot-history features.
    return dict(
        hidden_dims     = trial.suggest_categorical("hidden_dims",
                              ["(256, 128)", "(512, 256, 128)", "(512, 256, 128, 64)",
                               "(1024, 512, 256, 128)"]),
        learning_rate   = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
        dropout_rate    = trial.suggest_float("dropout_rate", 0.10, 0.50),
        batch_size      = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
    )


def _suggest_survival(trial) -> dict:
    return dict(
        hidden_dims   = trial.suggest_categorical("hidden_dims",
                            ["(128, 128)", "(256, 256)", "(256, 256, 128)",
                             "(256, 256, 256)", "(512, 256, 256)",
                             "(512, 512, 256)", "(256, 256, 256, 128)"]),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.30),
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
    )


def _suggest_repair_time(trial) -> dict:
    return dict(
        hidden_dims   = trial.suggest_categorical("hidden_dims",
                            ["(64, 32)", "(128, 64)", "(128, 128)", "(256, 128)",
                             "(256, 256)", "(512, 256)"]),
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.50),
        batch_size    = trial.suggest_categorical("batch_size", [2, 4, 8, 16, 32, 64, 128]),
    )


def _suggest_product(trial) -> dict:
    return dict(
        hidden_dims   = trial.suggest_categorical("hidden_dims",
                            ["(256, 256)", "(512, 256)", "(512, 256, 128)",
                             "(1024, 512, 256)", "(1024, 512, 256, 128)"]),
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
        dropout_rate  = trial.suggest_float("dropout_rate", 0.0, 0.30),
        batch_size    = trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
    )


# Per-model study version (default "v1"); bump on search-space drift so Optuna
# does not reuse an incompatible sampler prior.
# Study identity is bound to the training-data generation: bumping the
# version here starts a FRESH Optuna study, so load_if_exists can never let a
# trial scored on pre-integer-contract data win against new trials.
STUDY_VERSIONS: Dict[str, str] = {
    "process_time": "v2_int",
    "transition": "v3_int_history10",
    "survival": "v2_int",
    "repair_time": "v2_int",
    "product": "v2_int",
}

# Same chronological split as production training so the train-only vocabulary
# fit stays consistent.
OBJECTIVES = {
    "process_time": _make_objective(
        "src.fitting.deep_training.train_process_time",
        _suggest_process_time,
        train_kwargs=dict(max_epochs=20, test_size=HPO_TEST_SIZE, patience=10),
    ),
    "transition": _make_objective(
        "src.fitting.deep_training.train_transition",
        _suggest_transition,
        train_kwargs=dict(max_epochs=20, test_size=HPO_TEST_SIZE, patience=8),
    ),
    "survival": _make_objective(
        "src.fitting.deep_training.train_survival",
        _suggest_survival,
        train_kwargs=dict(max_epochs=25, test_size=HPO_TEST_SIZE, patience=12),
    ),
    "repair_time": _make_objective(
        "src.fitting.deep_training.train_repair_time",
        _suggest_repair_time,
        train_kwargs=dict(max_epochs=25, test_size=HPO_TEST_SIZE, patience=12),
    ),
    "product": _make_objective(
        "src.fitting.deep_training.train_product",
        _suggest_product,
        train_kwargs=dict(max_epochs=20, test_size=HPO_TEST_SIZE, patience=10),
    ),
}


TRIAL_BUDGET = {
    "process_time": 20,
    "transition":   20,
    "survival":     30,
    "repair_time":  30,
    "product":      20,
}


def run_hpo(
    model:      str,
    n_trials:   int,
    data_dir:   Union[str, Path],
    hpo_dir:    Union[str, Path] = REPO / "models/hpo",
    db_path:    Optional[Union[str, Path]] = REPO / "models/hpo/hpo_studies.db",
    n_jobs:     int = 1,
    max_epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run HPO for a single DeepSim model.

    Parameters
    ----------
    db_path   : SQLite path for study persistence (None = in-memory)
    n_jobs    : parallel trials (1 = sequential, safe default)
    """
    import optuna

    if model not in OBJECTIVES:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(OBJECTIVES)}")

    data_dir  = Path(data_dir)
    hpo_dir   = Path(hpo_dir)
    model_dir = hpo_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    storage = None
    if db_path is not None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study_name = f"deepsim_{model}_{STUDY_VERSIONS.get(model, 'v1')}"
    study = optuna.create_study(
        study_name     = study_name,
        direction      = "minimize",
        storage        = storage,
        load_if_exists = True,
    )

    objective_fn = OBJECTIVES[model]

    logger.info(f"{'='*60}")
    logger.info(f"  HPO: {model}  |  {n_trials} trials  |  data: {data_dir.name}")
    logger.info(f"{'='*60}")

    override = {"max_epochs": max_epochs_override} if max_epochs_override else None

    def _wrapped_objective(trial):
        try:
            val = objective_fn(trial, data_dir, model_dir,
                               train_kwargs_override=override)
            logger.info(
                f"  Trial {trial.number:>3}  |  "
                f"val_loss={val:.6f}  |  "
                f"params={trial.params}"
            )
            return val
        except Exception as e:
            logger.warning(f"  Trial {trial.number} failed: {e}")
            return float("inf")

    study.optimize(_wrapped_objective, n_trials=n_trials, n_jobs=n_jobs)

    best = study.best_trial
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Best trial:    #{best.number}")
    logger.info(f"  Best val_loss: {best.value:.6f}")
    logger.info("  Best params:")
    for k, v in best.params.items():
        logger.info(f"    {k:<20} {v}")
    logger.info(f"{'─'*60}")

    results_path = hpo_dir / f"{model}_best_params.json"
    payload: Dict[str, Any] = {
        "model":      model,
        "best_value": best.value,
        "best_trial": best.number,
        "params":     best.params,
        "n_trials":   n_trials,
        "data_dir":   str(data_dir),
    }

    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"  Saved: {results_path}")

    return {
        "best_params":  best.params,
        "best_value":   best.value,
        "study":        study,
        "results_path": results_path,
    }


def run_hpo_all(
    n_trials:  Optional[int],
    data_dir:  Union[str, Path],
    hpo_dir:   Union[str, Path] = REPO / "models/hpo",
    db_path:   Optional[Union[str, Path]] = REPO / "models/hpo/hpo_studies.db",
    max_epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Run HPO for all five models sequentially.

    If n_trials is None, uses per-model defaults from TRIAL_BUDGET.
    """
    results = {}
    for model in OBJECTIVES:
        trials_for_model = n_trials if n_trials is not None else TRIAL_BUDGET[model]
        logger.info(f"\n{'#'*60}")
        logger.info(f"  Starting HPO for: {model}  (n_trials={trials_for_model})")
        logger.info(f"{'#'*60}")
        results[model] = run_hpo(
            model    = model,
            n_trials = trials_for_model,
            data_dir = data_dir,
            hpo_dir  = hpo_dir,
            db_path  = db_path,
            max_epochs_override = max_epochs_override,
        )

    logger.info(f"\n{'='*60}")
    logger.info("  HPO COMPLETE – BEST PARAMETERS SUMMARY")
    logger.info(f"{'='*60}")
    for model, res in results.items():
        logger.info(f"\n  {model.upper()}")
        logger.info(f"  {'─'*40}")
        logger.info(f"  Best val_loss: {res['best_value']:.6f}")
        for k, v in res["best_params"].items():
            logger.info(f"  {k:<20} {v}")

    return results


def load_best_params(
    model:    str,
    hpo_dir:  Union[str, Path] = REPO / "models/hpo",
) -> Dict[str, Any]:
    """Load the best HPO params as kwargs for the corresponding train() function."""
    path = Path(hpo_dir) / f"{model}_best_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No HPO results found at {path}. Run run_hpo('{model}', ...) first."
        )
    with open(path) as f:
        data = json.load(f)

    params = data["params"].copy()

    if "hidden_dims" in params:
        params["hidden_dims"] = ast.literal_eval(params["hidden_dims"])

    logger.info(f"Loaded best params for '{model}' (val_loss={data['best_value']:.6f}):")
    for k, v in params.items():
        logger.info(f"  {k:<20} {v}")

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSim HPO with Optuna")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all"] + list(OBJECTIVES.keys()),
        help="Which model to optimize (default: all)",
    )
    parser.add_argument("--num-trials", type=int, default=None,
                        help="Trials per model (default: TRIAL_BUDGET per model)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--hpo-dir", type=str, default="models/hpo",
                        help=("Output directory for _best_params.json + trial artifacts. "
                              "For HPO-val on a second seed, set e.g. 'models/hpo_val'."))
    parser.add_argument("--database-path", type=str, default=None,
                        help="SQLite path (default: <hpo_dir>/hpo_studies.db)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max_epochs in HPO train_kwargs (default: "
                             "model-specific in OBJECTIVES).")
    args = parser.parse_args()

    if args.database_path is None:
        args.database_path = str(Path(args.hpo_dir) / "hpo_studies.db")

    if args.model == "all":
        run_hpo_all(
            n_trials = args.num_trials,
            data_dir = args.data_dir,
            hpo_dir  = args.hpo_dir,
            db_path  = args.database_path,
            max_epochs_override = args.max_epochs,
        )
    else:
        n_trials = args.num_trials if args.num_trials is not None else TRIAL_BUDGET[args.model]
        run_hpo(
            model    = args.model,
            n_trials = n_trials,
            data_dir = args.data_dir,
            hpo_dir  = args.hpo_dir,
            db_path  = args.database_path,
            max_epochs_override = args.max_epochs,
        )


if __name__ == "__main__":
    main()
