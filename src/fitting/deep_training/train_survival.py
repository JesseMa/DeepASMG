"""
Weibull survival model training (time to failure in normalized operating time).

Learns per-station Weibull parameters (shape k, scale λ) so that
TTF ~ Weibull(k, λ) in normalized units. Training data are divided by the
mean operating time (duration_scale); inference converts back via
TTF_seconds = TTF_normalized * duration_scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    three_way_split, train_lightning_model, cached_prepare, make_loaders,
    BaseTrainingModule, save_eval_artifact,
)



def weibull_nll_loss(
    log_shape: torch.Tensor,
    log_scale: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """
    Mean Weibull negative log-likelihood with right-censoring.

    duration is normalized operating time; event is 1.0 for an observed
    breakdown and 0.0 for a right-censored cycle.
    """
    k = torch.exp(torch.clamp(log_shape, -5.0, 5.0))
    lam = torch.exp(torch.clamp(log_scale, -5.0, 15.0))

    t = torch.clamp(duration, min=1e-6)

    t_over_lam = t / lam
    t_over_lam_k = torch.pow(t_over_lam, k)

    # Uncensored: log f(t) = log(k) - log(λ) + (k-1)·log(t/λ) - (t/λ)^k
    # Censored:   log S(t) = -(t/λ)^k
    log_hazard = torch.log(k) - torch.log(lam) + (k - 1.0) * torch.log(t_over_lam)
    log_survival = -t_over_lam_k

    nll = -(event * log_hazard + log_survival)

    return nll.mean()


class WeibullSurvivalModule(BaseTrainingModule, pl.LightningModule):
    """
    MLP for Weibull survival modelling.

    Input:  [station_onehot, prev_ttf, prev_n_jobs, mean_ttf, prev_repair_time]  (n_stations + 4)
    Output: [log_shape, log_scale] — Weibull parameters in normalized units.
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
            weight_decay=weight_decay, scheduler_patience=15,
        )

    def _compute_loss(self, batch, stage: str):
        x, y = batch
        duration = y[:, 0]
        event = y[:, 1]

        output = self(x)
        log_shape = output[:, 0]
        log_scale = output[:, 1]

        loss = weibull_nll_loss(log_shape, log_scale, duration, event)

        with torch.no_grad():
            shape = torch.exp(torch.clamp(log_shape, -5.0, 5.0))
            scale = torch.exp(torch.clamp(log_scale, -5.0, 15.0))
            # Weibull mean (normalized units)
            mean_ttf = scale * torch.exp(torch.lgamma(1.0 + 1.0 / shape))

        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage in ("val", "test"):
            self.log(f"{stage}_mean_shape", shape.mean(), prog_bar=False)
            self.log(f"{stage}_mean_scale", scale.mean(), prog_bar=False)
            self.log(f"{stage}_mean_ttf", mean_ttf.mean(), prog_bar=True)

        return loss



def prepare_data(
    data_dir: Path,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare survival training data (cached)."""
    from src.fitting.deep_data_preparation.prepare_survival_data import (
        prepare_survival_data,
    )

    def _validate_cache(metadata: Dict) -> bool:
        return not metadata.get("features_normalized_in_prepare", True)

    return cached_prepare(data_dir, output_dir, prepare_survival_data,
                          csv_pattern="events_only", cache_validator=_validate_cache)


def _evaluate_survival(
    model: pl.LightningModule,
    test_loader,
    station_map: Dict[str, int],
    duration_scale: float,
) -> Dict[str, Any]:
    """
    TTF values are converted back to operating seconds via duration_scale.
    """
    all_shapes, all_scales = [], []
    all_durations, all_events, all_stations = [], [], []

    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            log_shape = output[:, 0]
            log_scale = output[:, 1]

            shapes = torch.exp(torch.clamp(log_shape, -5.0, 5.0))
            scales = torch.exp(torch.clamp(log_scale, -5.0, 15.0))

            all_shapes.append(shapes.numpy())
            all_scales.append(scales.numpy())
            all_durations.append(y[:, 0].numpy())
            all_events.append(y[:, 1].numpy())

            st_oh = x[:, :len(station_map)]
            all_stations.append(st_oh.argmax(dim=1).numpy())

    shapes = np.concatenate(all_shapes)
    scales = np.concatenate(all_scales)
    durations_norm = np.concatenate(all_durations)
    events = np.concatenate(all_events)
    stations = np.concatenate(all_stations)

    from scipy.special import gamma as gamma_fn

    # Normalized TTFs -> operating seconds
    mean_ttfs_norm = scales * gamma_fn(1.0 + 1.0 / np.clip(shapes, 0.01, None))
    mean_ttfs_sec = mean_ttfs_norm * duration_scale
    durations_sec = durations_norm * duration_scale

    idx_to_name = {v: k for k, v in station_map.items()}
    per_station = {}
    for idx in sorted(idx_to_name.keys()):
        mask = stations == idx
        if mask.sum() == 0:
            continue
        name = idx_to_name[idx]
        s_shapes = shapes[mask]
        s_scales = scales[mask]
        s_durations_sec = durations_sec[mask]
        s_events = events[mask]
        s_mean_ttf_sec = mean_ttfs_sec[mask]

        per_station[name] = {
            "n_samples": int(mask.sum()),
            "n_uncensored": int(s_events.sum()),
            "mean_shape": float(np.mean(s_shapes)),
            "mean_scale_normalized": float(np.mean(s_scales)),
            "mean_scale_seconds": float(np.mean(s_scales) * duration_scale),
            "mean_predicted_ttf": float(np.mean(s_mean_ttf_sec)),
            "mean_observed_operating_time": float(np.mean(s_durations_sec)),
            "median_observed_operating_time": float(np.median(s_durations_sec)),
        }

    return {
        "duration_scale": duration_scale,
        "n_samples": len(durations_norm),
        "n_uncensored": int(events.sum()),
        "mean_shape": float(np.mean(shapes)),
        "mean_scale_normalized": float(np.mean(scales)),
        "mean_scale_seconds": float(np.mean(scales) * duration_scale),
        "mean_predicted_ttf": float(np.mean(mean_ttfs_sec)),
        "mean_observed_operating_time": float(np.mean(durations_sec)),
        "per_station": per_station,
    }


def train(
    *,
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    batch_size: int = 64,
    max_epochs: int = 200,
    learning_rate: float = 0.001,
    hidden_dims: Union[Tuple[int, ...], List[int]] = (128, 64),
    dropout_rate: float = 0.1,
    test_size: float = 0.15,
    weight_decay: float = 1e-4,
    patience: int = 20,
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Train the Weibull survival model (normalized operating time).

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
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "survival_data"
    hidden_dims = list(hidden_dims)

    # Prepare raw (unnormalized) data
    X, y, metadata = prepare_data(data_dir, prep_dir)

    splits = three_way_split(X, y, test_size=test_size)

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])
    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")

    # Compute normalization parameters from training data only (no test leakage)
    n_stations = len(metadata["encoding_maps"]["station"])
    X_tr, y_tr = splits["train"]

    duration_scale = float(np.mean(y_tr[:, 0]))
    if duration_scale < 1.0:
        duration_scale = 1.0

    n_jobs_scale = float(np.mean(X_tr[:, n_stations + 1]))
    if n_jobs_scale < 1.0:
        n_jobs_scale = 1.0

    repair_vals_tr = X_tr[:, n_stations + 3]
    nonzero_repairs_tr = repair_vals_tr[repair_vals_tr > 0]
    repair_time_scale = float(np.mean(nonzero_repairs_tr)) if len(nonzero_repairs_tr) > 0 else 1.0

    print("  Normalization parameters (from training data):")
    print(f"    duration_scale={duration_scale:.1f}s  "
          f"n_jobs_scale={n_jobs_scale:.1f}  "
          f"repair_time_scale={repair_time_scale:.1f}s")

    def _normalize(X_s: np.ndarray, y_s: np.ndarray):
        X_s = X_s.copy()
        y_s = y_s.copy()
        X_s[:, n_stations]     /= duration_scale
        X_s[:, n_stations + 1] /= n_jobs_scale
        X_s[:, n_stations + 2] /= duration_scale
        X_s[:, n_stations + 3] /= repair_time_scale
        y_s[:, 0]              /= duration_scale
        return X_s, y_s

    splits = {name: _normalize(Xs, ys) for name, (Xs, ys) in splits.items()}

    # Store scaling parameters in metadata.json for inference
    import json as _json
    metadata["duration_scale"]    = duration_scale
    metadata["n_jobs_scale"]      = n_jobs_scale
    metadata["repair_time_scale"] = repair_time_scale
    with open(prep_dir / "metadata.json", "w") as _f:
        _json.dump(metadata, _f, indent=2)

    n_uncensored = int(np.sum(splits["train"][1][:, 1] == 1))
    print(f"  Uncensored in Train: {n_uncensored:,} / {n_train:,}")
    print(f"  Duration Scale: {duration_scale:.1f}s (y normalized to ~1.0)")

    loaders = make_loaders(splits, batch_size=batch_size, task="regression")

    input_dim = X.shape[1]
    module = WeibullSurvivalModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="survival_weibull",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    # module already holds the best weights from train_lightning_model
    station_map = metadata["encoding_maps"]["station"]
    module.eval()
    evaluation = _evaluate_survival(module, loaders["test"], station_map, duration_scale)

    results["metadata_path"] = str(prep_dir / "metadata.json")
    results["evaluation"] = evaluation
    results["input_dim"] = input_dim
    results["hidden_dims"] = hidden_dims

    eval_artifact_path = save_eval_artifact(
        model_name="survival",
        model_dir=model_dir,
        evaluation=results["evaluation"],
        data_source=data_dir,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )
    results["eval_artifact_path"] = str(eval_artifact_path)

    print(f"\n{'─' * 55}")
    print("  WEIBULL SURVIVAL MODEL (Operating-Time)")
    print(f"{'─' * 55}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (NLL): {results['best_val_loss']:.4f}")
    print(f"  Duration Scale: {duration_scale:.1f}s")
    if evaluation:
        print(f"  Mean Predicted TTF: {evaluation['mean_predicted_ttf']:.0f}s (Operating)")
        print(f"  Mean Observed OpTime: {evaluation['mean_observed_operating_time']:.0f}s")
        print(f"  Mean Shape (k): {evaluation['mean_shape']:.3f}")
        print(f"  Mean Scale (λ, norm): {evaluation['mean_scale_normalized']:.3f}")
        print(f"  Mean Scale (λ, sec):  {evaluation['mean_scale_seconds']:.0f}s")
        if evaluation.get("per_station"):
            print("\n  Per Station:")
            for name, stats in sorted(evaluation["per_station"].items()):
                print(
                    f"    {name:<12} "
                    f"k={stats['mean_shape']:.2f}  "
                    f"λ={stats['mean_scale_seconds']:.0f}s  "
                    f"E[TTF]={stats['mean_predicted_ttf']:.0f}s  "
                    f"obs={stats['mean_observed_operating_time']:.0f}s  "
                    f"BD={stats['n_uncensored']}/{stats['n_samples']}"
                )
    print(f"{'─' * 55}")

    return results