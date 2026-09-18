"""Process-time prediction training (heteroscedastic regression)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    init_output_at_marginal,
    three_way_split, make_loaders, train_lightning_model, cached_prepare,
    GaussianNLLModule,
)


class ProcessTimeLightningModule(GaussianNLLModule, pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        learning_rate: float,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._init_base(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2,
            learning_rate=learning_rate, dropout_rate=dropout_rate,
            output_activation="softplus_first",
        )


def prepare_data(
    data_dir: Path,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from src.fitting.deep_data_preparation.prepare_process_time_data import (
        prepare_training_data,
    )
    return cached_prepare(data_dir, output_dir, prepare_training_data, csv_pattern="events")


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
    )
    init_output_at_marginal(module.net, module.marginal_bias(splits["train"][1]))

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="process_time",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    results["metadata_path"] = str(prep_dir / "metadata.json")
    print(f"\n{'─' * 40}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (NLL): {results['best_val_loss']:.6f}")
    print(f"  Test Metrics: {results['test_metrics']}")
    print(f"{'─' * 40}")

    return results