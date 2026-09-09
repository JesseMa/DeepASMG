"""
Downtime-duration training, conditional exponential regression.

The output is the log_scale of an exponential, matching the GroundRepair
family. Entry point: train().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import pytorch_lightning as pl

from src.fitting.deep_training.foundation_training import (
    three_way_split, make_loaders, train_lightning_model, cached_prepare,
    ExponentialNLLModule, save_eval_artifact,
)

TRAIN_SEED: int = 42


class RepairTimeRegressorModule(ExponentialNLLModule, pl.LightningModule):
    """
    MLP for downtime-duration prediction (conditional exponential).

    Output: log_scale (1 channel). Sampling y = -log(u) * exp(log_scale),
            u ~ Uniform(0, 1); mean = exp(log_scale).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float,
        dropout_rate: float,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._init_base(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1,
            learning_rate=learning_rate, dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            output_activation="none",  # log_scale in R
        )


def prepare_data(
    data_dir: Path,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare downtime-duration training data (cached)."""
    from src.fitting.deep_data_preparation.prepare_repair_time_data import (
        prepare_repair_time_data,
    )

    def _validate_cache(metadata: Dict) -> bool:
        return not metadata.get("features_normalized_in_prepare", True)

    return cached_prepare(data_dir, output_dir, prepare_repair_time_data,
                          csv_pattern="events_only", cache_validator=_validate_cache)


def _evaluate_per_station(
    model: pl.LightningModule,
    test_loader,
    station_map: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Evaluate MAE/RMSE per station."""
    all_preds, all_targets, all_stations = [], [], []
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            # Exp: pred=log_scale. Mean = exp(log_scale).
            pred_mean = torch.exp(pred[:, 0])
            all_preds.append(pred_mean.numpy())
            all_targets.append(y.numpy())
            # Station one-hot occupies the first n_stations columns
            all_stations.append(x[:, :len(station_map)].argmax(dim=1).numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    stations = np.concatenate(all_stations)

    idx_to_name = {v: k for k, v in station_map.items()}
    per_station = {}

    for idx in sorted(idx_to_name.keys()):
        mask = stations == idx
        if mask.sum() == 0:
            continue
        name = idx_to_name[idx]
        errors = preds[mask] - targets[mask]
        per_station[name] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "count": int(mask.sum()),
            "mean_actual": float(np.mean(targets[mask])),
            "mean_predicted": float(np.mean(preds[mask])),
        }

    return per_station


def train(
    *,
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    batch_size: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 0.0023150938044562237,
    hidden_dims: Union[Tuple[int, ...], List[int]] = (256, 128),
    dropout_rate: float = 0.0011248523716306177,
    test_size: float = 0.15,
    weight_decay: float = 1e-4,
    patience: int = 15,
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Train the downtime-duration regressor.

    Args:
        _prep_dir: shared directory for prepared data; enables caching across
            HPO trials (CSVs are loaded only once).

    Returns:
        Dict with model_path, metadata_path, best_val_loss, evaluation, history.
    """
    # Must run before model init and loader construction.
    pl.seed_everything(TRAIN_SEED, workers=True)

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "repair_time_data"
    hidden_dims = list(hidden_dims)

    # Prepare raw (unnormalized) data
    X, y, metadata = prepare_data(data_dir, prep_dir)

    splits = three_way_split(X, y, test_size=test_size)

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])
    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")
    print(f"  Target: mean={metadata['y_mean']:.2f}s, std={metadata['y_std']:.2f}s")
    print(f"  Features: {metadata['feature_dim']} "
          f"({metadata['feature_layout'][0]['size']} station + 3 continuous: "
          f"op_time, utilization, wear_ratio)")

    # Compute normalization parameters from training data only (no test leakage)
    n_stations = len(metadata["encoding_maps"]["station"])
    X_tr, _ = splits["train"]

    def _fit_stats(col: int) -> tuple[float, float]:
        v = X_tr[:, col]
        mean = float(np.mean(v))
        std = float(np.std(v)) if len(v) > 1 else 1.0
        return mean, (std if std >= 1e-6 else 1.0)

    op_mean, op_std = _fit_stats(n_stations)
    util_mean, util_std = _fit_stats(n_stations + 1)
    wear_mean, wear_std = _fit_stats(n_stations + 2)

    print(f"  Normalization (from training data): "
          f"op_time mean={op_mean:.2f}s std={op_std:.2f}s | "
          f"utilization mean={util_mean:.3f} std={util_std:.3f} | "
          f"wear_ratio mean={wear_mean:.3f} std={wear_std:.3f}")

    def _normalize(X_s: np.ndarray, y_s: np.ndarray):
        X_s = X_s.copy()
        X_s[:, n_stations] = (X_s[:, n_stations] - op_mean) / op_std
        X_s[:, n_stations + 1] = (X_s[:, n_stations + 1] - util_mean) / util_std
        X_s[:, n_stations + 2] = (X_s[:, n_stations + 2] - wear_mean) / wear_std
        return X_s, y_s

    splits = {name: _normalize(Xs, ys) for name, (Xs, ys) in splits.items()}

    # Store normalization parameters in metadata.json for inference
    import json as _json
    metadata["operating_time_mean"] = op_mean
    metadata["operating_time_std"]  = op_std
    metadata["utilization_mean"] = util_mean
    metadata["utilization_std"]  = util_std
    metadata["wear_ratio_mean"] = wear_mean
    metadata["wear_ratio_std"]  = wear_std
    with open(prep_dir / "metadata.json", "w") as _f:
        _json.dump(metadata, _f, indent=2)

    loaders = make_loaders(splits, batch_size=batch_size, task="regression")

    input_dim = X.shape[1]
    module = RepairTimeRegressorModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="repair_time",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    # module already holds the best weights from train_lightning_model
    station_map = metadata["encoding_maps"]["station"]
    module.eval()
    per_station = _evaluate_per_station(module, loaders["test"], station_map)

    results["metadata_path"] = str(prep_dir / "metadata.json")
    results["evaluation"] = {
        "per_station": per_station,
        **results.get("test_metrics", {}),
    }
    results["input_dim"] = input_dim
    results["hidden_dims"] = hidden_dims

    eval_artifact_path = save_eval_artifact(
        model_name="repair_time",
        model_dir=model_dir,
        evaluation=results["evaluation"],
        data_source=data_dir,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )
    results["eval_artifact_path"] = str(eval_artifact_path)

    print(f"\n{'─' * 40}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (NLL): {results['best_val_loss']:.6f}")
    if results.get("test_metrics"):
        print(f"  Test Metrics: {results['test_metrics']}")
    if per_station:
        print("\n  Per Station:")
        for name, stats in sorted(per_station.items()):
            print(f"    {name:<12} MAE={stats['mae']:.2f}s  RMSE={stats['rmse']:.2f}s  "
                  f"n={stats['count']}  actual={stats['mean_actual']:.2f}s  "
                  f"pred={stats['mean_predicted']:.2f}s")
    print(f"  Eval artifact: {eval_artifact_path}")
    print(f"{'─' * 40}")

    return results
