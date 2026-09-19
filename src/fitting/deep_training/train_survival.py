"""Weibull survival model training (time to failure in normalized operating time)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    init_output_at_marginal,
    three_way_split, train_lightning_model, cached_prepare, make_loaders,
    BaseTrainingModule,
)


def weibull_nll_loss(
    log_shape: torch.Tensor,
    log_scale: torch.Tensor,
    duration: torch.Tensor,
    event: torch.Tensor,
    bin_width: float,
) -> torch.Tensor:
    k = torch.exp(log_shape)

    t_hi = torch.clamp(duration, min=1e-6)
    t_lo = torch.clamp(duration - bin_width, min=1e-6)
    # log of the cumulative hazard at both bin edges, soft-capped so that a law
    # calling the observation impossible yields a huge finite loss with a
    # gradient instead of inf and NaN weights
    log_u_hi = _softcap(k * (torch.log(t_hi) - log_scale))
    log_u_lo = _softcap(k * (torch.log(t_lo) - log_scale))
    log_d = _softcap(log_u_hi + torch.log(-torch.expm1(k * torch.log(t_lo / t_hi))))

    log_s_hi = -torch.exp(log_u_hi)
    log_p_event = -torch.exp(log_u_lo) + torch.log(-torch.expm1(-torch.exp(log_d)))

    nll = -(event * log_p_event + (1.0 - event) * log_s_hi)
    return nll.mean()


_LOG_U_CAP = 60.0


def _softcap(x: torch.Tensor, cap: float = _LOG_U_CAP) -> torch.Tensor:
    return torch.where(x < cap, x, cap + torch.log1p(torch.clamp(x - cap, min=0.0)))


class WeibullSurvivalModule(BaseTrainingModule, pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float,
        dropout_rate: float,
        bin_width: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._bin_width = float(bin_width)
        self._init_base(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2,
            learning_rate=learning_rate, dropout_rate=dropout_rate,
            scheduler_patience=15,
        )

    @staticmethod
    def marginal_bias(y_train: np.ndarray) -> List[float]:
        return [0.0, float(np.log(np.asarray(y_train, dtype=float)[:, 0].mean()))]

    def _compute_loss(self, batch, stage: str):
        x, y = batch
        duration = y[:, 0]
        event = y[:, 1]

        output = self(x)
        log_shape = output[:, 0]
        log_scale = output[:, 1]

        loss = weibull_nll_loss(log_shape, log_scale, duration, event, self._bin_width)

        with torch.no_grad():
            shape = torch.exp(log_shape)
            scale = torch.exp(log_scale)
            mean_ttf = scale * torch.exp(torch.lgamma(1.0 + 1.0 / shape))

        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage in ("val", "test"):
            self.log(f"{stage}_mean_shape", shape.mean(), prog_bar=False)
            self.log(f"{stage}_mean_scale", scale.mean(), prog_bar=False)
            self.log(f"{stage}_mean_ttf", mean_ttf.mean(), prog_bar=True)

        return loss


def prepare_data(data_dir: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from src.fitting.deep_data_preparation.prepare_survival_data import (
        prepare_survival_data,
    )
    return cached_prepare(data_dir, output_dir, prepare_survival_data, csv_pattern="events_only")


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
    patience: int = 15,
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    pl.seed_everything(TRAIN_SEED, workers=True)

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "survival_data"
    hidden_dims = list(hidden_dims)

    X, y, metadata = prepare_data(data_dir, prep_dir)

    splits = three_way_split(X, y, test_size=test_size)

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])
    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")

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
        # exponential tail: log1p keeps a 20x outlier within the range the head can absorb
        X_s[:, n_stations + 3] = np.log1p(X_s[:, n_stations + 3] / repair_time_scale)
        y_s[:, 0]              /= duration_scale
        return X_s, y_s

    splits = {name: _normalize(Xs, ys) for name, (Xs, ys) in splits.items()}

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
        bin_width=1.0 / duration_scale,
    )
    init_output_at_marginal(module.net, module.marginal_bias(splits["train"][1]))

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="survival_weibull",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    results["metadata_path"] = str(prep_dir / "metadata.json")
    print(f"\n{'─' * 40}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (NLL): {results['best_val_loss']:.6f}")
    print(f"  Duration Scale: {duration_scale:.1f}s")
    print(f"  Test Metrics: {results['test_metrics']}")
    print(f"{'─' * 40}")

    return results