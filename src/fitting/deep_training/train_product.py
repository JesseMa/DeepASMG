"""Multi-head MLP training for product-feature prediction (DeepProduct)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    three_way_split, train_lightning_model, cached_prepare, make_loaders,
    build_mlp_layers, BaseTrainingModule,
)


class MultiHeadProductNet(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        head_sizes: List[int],
        dropout_rate: float,
        conditioning_dim: int = 0,
    ) -> None:
        super().__init__()

        self._conditioning_dim = conditioning_dim
        self._base_dim = input_dim - conditioning_dim

        self.full_input_dim = input_dim

        self.trunk = build_mlp_layers(self._base_dim, hidden_dims, dropout_rate=dropout_rate)
        trunk_out = hidden_dims[-1]

        self.modell_head = nn.Linear(trunk_out, head_sizes[0])

        if conditioning_dim > 0:
            self.cond_heads = nn.ModuleList([
                nn.Linear(trunk_out + conditioning_dim, size)
                for size in head_sizes[1:]
            ])
        else:
            self.cond_heads = nn.ModuleList([
                nn.Linear(trunk_out, size) for size in head_sizes[1:]
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._conditioning_dim > 0:
            x_base = x[:, :self._base_dim]
            modell_cond = x[:, self._base_dim:]
        else:
            x_base = x
            modell_cond = None

        shared = self.trunk(x_base)
        modell_logits = self.modell_head(shared)

        if modell_cond is not None:
            conditioned = torch.cat([shared, modell_cond], dim=-1)
        else:
            conditioned = shared

        cond_outputs = [head(conditioned) for head in self.cond_heads]
        return torch.cat([modell_logits] + cond_outputs, dim=-1)


class ProductMLPModule(BaseTrainingModule, pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        head_sizes: List[int],
        feature_names: List[str],
        conditioning_dim: int,
        learning_rate: float,
        dropout_rate: float,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()

        self._conditioning_dim = conditioning_dim

        self.net = MultiHeadProductNet(
            input_dim=input_dim + conditioning_dim,
            hidden_dims=hidden_dims,
            head_sizes=head_sizes,
            dropout_rate=dropout_rate,
            conditioning_dim=conditioning_dim,
        )
        self.lr = learning_rate
        self._scheduler_patience = 10

        self._head_sizes = head_sizes
        self._feature_names = feature_names

        self._head_offsets = [0]
        for size in head_sizes[:-1]:
            self._head_offsets.append(self._head_offsets[-1] + size)

    def _compute_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str,
    ) -> torch.Tensor:
        x, y = batch

        if self._conditioning_dim > 0:
            modell_targets = y[:, 0]
            modell_onehot = nn.functional.one_hot(
                modell_targets, num_classes=self._conditioning_dim,
            ).float()
            x_cond = torch.cat([x, modell_onehot], dim=-1)
        else:
            x_cond = x

        logits_flat = self(x_cond)

        total_loss = torch.tensor(0.0, device=x.device)

        for i, (feat_name, size) in enumerate(
            zip(self._feature_names, self._head_sizes, strict=True)
        ):
            offset = self._head_offsets[i]
            head_logits = logits_flat[:, offset: offset + size]
            head_targets = y[:, i]

            head_loss = nn.functional.cross_entropy(head_logits, head_targets)
            total_loss = total_loss + head_loss

            if stage in ("val", "test"):
                preds = head_logits.argmax(dim=-1)
                acc = (preds == head_targets).float().mean()
                self.log(
                    f"{stage}_acc_{feat_name}", acc,
                    prog_bar=(i == 0),
                    on_epoch=True, on_step=False,
                )
                self.log(
                    f"{stage}_loss_{feat_name}", head_loss,
                    prog_bar=False,
                    on_epoch=True, on_step=False,
                )

        self.log(f"{stage}_loss", total_loss, prog_bar=True,
                 on_epoch=True, on_step=(stage == "train"))
        return total_loss


def prepare_data(data_dir: Path, output_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from src.fitting.deep_data_preparation.prepare_product_data import (
        prepare_product_data,
    )
    return cached_prepare(data_dir, output_dir, prepare_product_data, csv_pattern="orders")


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
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "product_data"
    hidden_dims = list(hidden_dims)

    X, y, metadata = prepare_data(data_dir, prep_dir)

    feature_names_loaded: List[str] = metadata["feature_names"]
    head_layout = metadata["head_layout"]
    head_sizes = [h["size"] for h in head_layout]

    splits = three_way_split(X, y, test_size=test_size)
    loaders = make_loaders(splits, batch_size=batch_size, task="classification")

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])

    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")
    print(f"  Features: {feature_names_loaded}")
    print(f"  Heads:     {list(zip(feature_names_loaded, head_sizes, strict=True))}")
    print(f"  Conditioning: modell -> {feature_names_loaded[1:]} (dim={head_sizes[0]})")

    input_dim = X.shape[1]
    conditioning_dim = head_sizes[0]
    module = ProductMLPModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        head_sizes=head_sizes,
        feature_names=feature_names_loaded,
        conditioning_dim=conditioning_dim,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="product",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    results["metadata_path"] = str(prep_dir / "metadata.json")
    print(f"\n{'─' * 40}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (CE): {results['best_val_loss']:.6f}")
    print(f"  Test Metrics: {results['test_metrics']}")
    print(f"{'─' * 40}")

    return results