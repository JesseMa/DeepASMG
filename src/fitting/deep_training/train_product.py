"""
Multi-head MLP training for product-feature prediction (DeepProduct).

TorchScript export: the traced model emits a single flat logit vector;
DeepProduct splits it via head_layout from metadata.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.config.simulation_config import TRAIN_SEED
from src.fitting.deep_training.foundation_training import (
    three_way_split, train_lightning_model, cached_prepare, make_loaders,
    build_mlp_layers, BaseTrainingModule, save_eval_artifact,
)



class MultiHeadProductNet(nn.Module):
    """
    Autoregressive multi-head MLP with model conditioning.

    Input layout: [base_features | modell_onehot]. The trunk sees only the
    base features (time encoding + EWMA); the modell head predicts
    P(modell | t, ewma) unconditionally, while the remaining heads consume
    cat(trunk_output, modell_onehot) to learn P(feature | t, ewma, modell).

    Training uses teacher forcing (ground-truth one-hot).
    TorchScript-compatible: a single forward(x) with fixed input shape.
    """

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

        # Full input dimension for the TorchScript trace example.
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
    """
    Lightning wrapper for MultiHeadProductNet with teacher forcing.

    Loss: weighted sum of per-head cross-entropy. input_dim is the base size
    WITHOUT conditioning; the net is built with input_dim + conditioning_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        head_sizes: List[int],
        feature_names: List[str],
        conditioning_dim: int = 0,
        head_weights: Optional[List[float]] = None,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> None:
        # pl.LightningModule.__init__ directly; the net is built differently than _init_base
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
        self.weight_decay = weight_decay
        self._scheduler_patience = 10

        self._head_sizes = head_sizes
        self._feature_names = feature_names
        self._n_heads = len(head_sizes)

        self._head_offsets = [0]
        for size in head_sizes[:-1]:
            self._head_offsets.append(self._head_offsets[-1] + size)

        if head_weights is None:
            self._head_weights = [1.0] * self._n_heads
        else:
            if len(head_weights) != self._n_heads:
                raise ValueError(
                    f"head_weights must have {self._n_heads} entries "
                    f"(one per feature head)."
                )
            total = sum(head_weights)
            self._head_weights = [w / total * self._n_heads for w in head_weights]

    def _compute_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str,
    ) -> torch.Tensor:
        x, y = batch

        # Teacher forcing: append the ground-truth modell one-hot to the input.
        if self._conditioning_dim > 0:
            modell_targets = y[:, 0]  # first feature is always 'modell'
            modell_onehot = nn.functional.one_hot(
                modell_targets, num_classes=self._conditioning_dim,
            ).float()
            x_cond = torch.cat([x, modell_onehot], dim=-1)
        else:
            x_cond = x

        logits_flat = self(x_cond)

        total_loss = torch.tensor(0.0, device=x.device)

        for i, (feat_name, size, weight) in enumerate(
            zip(self._feature_names, self._head_sizes, self._head_weights, strict=True)
        ):
            offset = self._head_offsets[i]
            head_logits = logits_flat[:, offset: offset + size]
            head_targets = y[:, i]

            head_loss = nn.functional.cross_entropy(head_logits, head_targets)
            total_loss = total_loss + weight * head_loss

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



def prepare_data(
    data_dir: Path,
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    ewma_alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prepare product training data (cached)."""
    from src.fitting.deep_data_preparation.prepare_product_data import (
        prepare_product_data,
    )

    def _validate_cache(metadata: Dict) -> bool:
        cached_alpha = metadata.get("ewma_alpha", 0.3)
        cached_features = metadata.get("feature_names", [])
        if abs(cached_alpha - ewma_alpha) > 1e-9:
            print(f"  [Cache] ewma_alpha changed ({cached_alpha} → {ewma_alpha}), rebuilding cache.")
            return False
        if feature_names is not None and sorted(cached_features) != sorted(feature_names):
            print(f"  [Cache] feature_names changed ({cached_features} → {feature_names}), rebuilding cache.")
            return False
        return True

    return cached_prepare(
        data_dir, output_dir, prepare_product_data,
        csv_pattern="orders",
        cache_validator=_validate_cache,
        feature_names=feature_names,
        ewma_alpha=ewma_alpha,
    )


def _evaluate_per_feature(
    model: pl.LightningModule,
    test_loader,
    feature_names: List[str],
    head_sizes: List[int],
    head_offsets: List[int],
    conditioning_dim: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate top-1/top-2 accuracy and cross-entropy per feature head."""
    all_logits, all_targets = [], []
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            # Teacher forcing: condition on the ground-truth modell.
            if conditioning_dim > 0:
                modell_onehot = nn.functional.one_hot(
                    y[:, 0], num_classes=conditioning_dim,
                ).float()
                x = torch.cat([x, modell_onehot], dim=-1)
            logits = model(x)
            all_logits.append(logits.numpy())
            all_targets.append(y.numpy())

    logits_all = np.concatenate(all_logits, axis=0)
    targets_all = np.concatenate(all_targets, axis=0)

    per_feature: Dict[str, Dict[str, float]] = {}

    for i, (feat_name, size) in enumerate(zip(feature_names, head_sizes, strict=True)):
        offset = head_offsets[i]
        head_logits = logits_all[:, offset: offset + size]
        head_targets = targets_all[:, i]

        top1_preds = head_logits.argmax(axis=-1)
        top1_acc = float(np.mean(top1_preds == head_targets))

        if size >= 2:
            top2_preds = np.argsort(head_logits, axis=-1)[:, -2:]
            top2_acc = float(np.mean(
                np.any(top2_preds == head_targets[:, None], axis=-1)
            ))
        else:
            top2_acc = top1_acc

        logits_t = torch.from_numpy(head_logits).float()
        targets_t = torch.from_numpy(head_targets).long()
        ce = float(nn.functional.cross_entropy(logits_t, targets_t).item())

        class_counts = {
            int(cls): int(np.sum(head_targets == cls))
            for cls in range(size)
        }

        per_feature[feat_name] = {
            "top1_accuracy": top1_acc,
            "top2_accuracy": top2_acc,
            "cross_entropy": ce,
            "n_classes": size,
            "n_samples": len(head_targets),
            "class_counts": class_counts,
        }

    return per_feature


def train(
    *,
    data_dir: Union[str, Path],
    model_dir: Union[str, Path],
    batch_size: int = 256,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    hidden_dims: Union[Tuple[int, ...], List[int]] = (128, 64),
    dropout_rate: float = 0.1,
    head_weights: Optional[List[float]] = None,
    test_size: float = 0.15,
    weight_decay: float = 1e-4,
    patience: int = 15,
    feature_names: Optional[List[str]] = None,
    ewma_alpha: float = 0.3,
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    Train the multi-head product-feature MLP.

    Args:
        head_weights: optional per-head loss weights, e.g. [2.0, 1.0, 1.0]
            to upweight 'modell'.
        ewma_alpha: EWMA smoothing for the sequence context; stored in
            metadata.json, cache is rebuilt on change.
        _prep_dir: shared directory for prepared data; enables caching
            across HPO trials.

    Returns:
        Dict with model_path, metadata_path, best_val_loss, evaluation, history.
    """
    # Reproducibility seed before anything else (model init, shuffling,
    # DataLoader workers). Together with Trainer(deterministic=True) in
    # foundation_training this yields bit-identical models for an identical
    # training dataset.
    pl.seed_everything(TRAIN_SEED, workers=True)

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / "product_data"
    hidden_dims = list(hidden_dims)

    X, y, metadata = prepare_data(data_dir, prep_dir, feature_names, ewma_alpha=ewma_alpha)

    feature_names_loaded: List[str] = metadata["feature_names"]
    head_layout = metadata["head_layout"]
    head_sizes = [h["size"] for h in head_layout]
    head_offsets = [h["offset"] for h in head_layout]

    splits = three_way_split(X, y, test_size=test_size)
    loaders = make_loaders(splits, batch_size=batch_size, task="classification")

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])

    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")
    print(f"  Features: {feature_names_loaded}")
    print(f"  Heads:     {list(zip(feature_names_loaded, head_sizes, strict=True))}")
    print(f"  Conditioning: modell -> {feature_names_loaded[1:]} (dim={head_sizes[0]})")

    # Condition on the first feature (modell).
    input_dim = X.shape[1]
    conditioning_dim = head_sizes[0]
    module = ProductMLPModule(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        head_sizes=head_sizes,
        feature_names=feature_names_loaded,
        conditioning_dim=conditioning_dim,
        head_weights=head_weights,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name="product",
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    # module already holds the best weights from train_lightning_model
    module.eval()
    per_feature = _evaluate_per_feature(
        module, loaders["test"],
        feature_names_loaded, head_sizes, head_offsets,
        conditioning_dim=conditioning_dim,
    )

    # TorchScript export already happened in train_lightning_model.
    results["metadata_path"] = str(prep_dir / "metadata.json")
    results["evaluation"] = {"per_feature": per_feature}
    results["input_dim"] = input_dim
    results["hidden_dims"] = hidden_dims
    results["head_sizes"] = head_sizes
    results["conditioning_dim"] = conditioning_dim

    eval_artifact_path = save_eval_artifact(
        model_name="product",
        model_dir=model_dir,
        evaluation=results["evaluation"],
        data_source=data_dir,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )
    results["eval_artifact_path"] = str(eval_artifact_path)

    print(f"\n{'─' * 55}")
    print("  PRODUCT MLP (Multi-Head Classification)")
    print(f"{'─' * 55}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    if per_feature:
        print("\n  Per Feature (Test-Set):")
        for feat_name, stats in per_feature.items():
            print(
                f"    {feat_name:<14} "
                f"Top-1={stats['top1_accuracy']*100:5.1f}%  "
                f"Top-2={stats['top2_accuracy']*100:5.1f}%  "
                f"CE={stats['cross_entropy']:.4f}  "
                f"n_cls={stats['n_classes']}"
            )
    print(f"{'─' * 55}")

    return results