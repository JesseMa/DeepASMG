"""
Process-time prediction training (heteroscedastic regression).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    three_way_split, make_loaders, train_lightning_model, cached_prepare,
    GaussianNLLModule, save_eval_artifact,
)



class ProcessTimeLightningModule(GaussianNLLModule, pl.LightningModule):
    """
    MLP for process-time prediction (heteroscedastic regression).

    Input:  one-hot [product_features + prev_features + station]
    Output: (mean, log_var) of the process time in seconds; the simulation
            samples N(mean, exp(0.5 * log_var)).
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
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2,
            learning_rate=learning_rate, dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            output_activation="softplus_first",  # structural mean > 0
        )


def prepare_data(
    data_dir: Path,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare process-time training data (cached)."""
    from src.fitting.deep_data_preparation.prepare_process_time_data import (
        prepare_training_data,
    )
    return cached_prepare(data_dir, output_dir, prepare_training_data, csv_pattern="events")


def _evaluate_per_station(
    model: pl.LightningModule,
    test_loader,
    metadata: Dict,
) -> Dict[str, Dict[str, float]]:
    """Evaluate MAE/RMSE per station."""
    station_map = metadata["encoding_maps"]["station"]
    station_offset = None
    for g in metadata["feature_layout"]:
        if g["name"] == "station":
            station_offset = g["offset"]
            break

    if station_offset is None:
        return {}

    all_preds, all_targets, all_stations = [], [], []
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            all_preds.append(pred[:, 0].numpy())  # mean channel only
            all_targets.append(y.numpy())
            st_oh = x[:, station_offset:station_offset + len(station_map)]
            all_stations.append(st_oh.argmax(dim=1).numpy())

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
        }

    return per_station


def train(
    *,
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    hidden_dims: Union[Tuple[int, ...], List[int]],
    dropout_rate: float,
    test_size: float,
    weight_decay: float = 1e-4,
    patience: int = 15,
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Train the process-time model.

    Args:
        _prep_dir: shared directory for prepared data; enables caching across
            HPO trials (CSVs are loaded only once).

    Returns:
        Dict with samples_trained, best_val_loss, model_path, metadata_path,
        history, test_metrics, evaluation, epochs_trained, input_dim.
    """
    # Must run before model init and loader construction.
    pl.seed_everything(TRAIN_SEED, workers=True)

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "process_time_data"
    hidden_dims = list(hidden_dims)

    X, y, metadata = prepare_data(data_dir, prep_dir)

    splits = three_way_split(X, y, test_size=test_size)
    loaders = make_loaders(splits, batch_size=batch_size, task="regression")

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])
    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")

    input_dim = X.shape[1]
    module = ProcessTimeLightningModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="process_time",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    # module already holds the best weights from train_lightning_model
    module.eval()
    per_station = _evaluate_per_station(module, loaders["test"], metadata)

    results["metadata_path"] = str(prep_dir / "metadata.json")
    results["evaluation"] = {
        "per_station": per_station,
        **results.get("test_metrics", {}),
    }
    results["input_dim"] = input_dim
    results["hidden_dims"] = hidden_dims

    eval_artifact_path = save_eval_artifact(
        model_name="process_time",
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
    print(f"  Eval artifact: {eval_artifact_path}")
    print(f"{'─' * 40}")

    return results