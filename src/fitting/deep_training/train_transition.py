"""Training: Transition Prediction (Classification)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.fitting.deep_training.foundation_training import (
    three_way_split, make_loaders, train_lightning_model, cached_prepare,
    BaseTrainingModule, save_eval_artifact,
)


TRAIN_SEED: int = 42


class TransitionLightningModule(BaseTrainingModule, pl.LightningModule):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int],
        learning_rate: float,
        dropout_rate: float,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._init_base(
            input_dim=input_dim, hidden_dims=hidden_dims, output_dim=num_classes,
            learning_rate=learning_rate, dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )
        self.num_classes = num_classes

    def _compute_loss(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        # Smoothing regularizes TRAINING only; val/test (and thus early
        # stopping, checkpointing and the Optuna objective) score plain CE.
        smoothing = self.hparams.label_smoothing if stage == "train" else 0.0
        loss = nn.functional.cross_entropy(
            logits, y, label_smoothing=smoothing,
        )
        acc = (logits.argmax(-1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss


def prepare_data(
    data_dir: Path,
    output_dir: Path,
    n_hist_slots: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    from src.fitting.deep_data_preparation.prepare_transition_data import (
        prepare_transition_data,
    )
    # None keeps the preparation default K; a non-default K only accepts a
    # cache whose metadata.json records the same K.
    kwargs: Dict[str, Any] = {}
    validator = None
    if n_hist_slots is not None:
        kwargs["n_hist_slots"] = n_hist_slots
        validator = lambda md: md.get("n_hist_slots") == n_hist_slots  # noqa: E731
    return cached_prepare(
        data_dir, output_dir, prepare_transition_data,
        csv_pattern="events", cache_validator=validator, **kwargs,
    )


def _evaluate_per_class(
    model: pl.LightningModule,
    test_loader,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    all_preds, all_targets = [], []
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    per_class = {}
    for idx, name in enumerate(class_names):
        tp = int(np.sum((preds == idx) & (targets == idx)))
        fp = int(np.sum((preds == idx) & (targets != idx)))
        fn = int(np.sum((preds != idx) & (targets == idx)))
        support = int(np.sum(targets == idx))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class[name] = {"precision": p, "recall": r, "f1": f1, "support": support}

    return per_class


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
    label_smoothing: float = 0.0,
    patience: int = 15,
    n_hist_slots: int | None = None,
    model_name: str = "transition",
    _prep_dir: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    # Must run before model init and loader construction.
    pl.seed_everything(TRAIN_SEED, workers=True)

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    prep_dir = Path(_prep_dir) if _prep_dir is not None else model_dir / f"{model_name}_data"
    hidden_dims = list(hidden_dims)

    X, y, metadata = prepare_data(data_dir, prep_dir, n_hist_slots=n_hist_slots)

    num_classes = metadata["num_classes"]
    class_names = metadata["class_names"]

    splits = three_way_split(X, y, test_size=test_size)
    loaders = make_loaders(splits, batch_size=batch_size, task="classification")

    n_train = len(splits["train"][0])
    n_val = len(splits["val"][0])
    n_test = len(splits["test"][0])
    print(f"\n  Split: Train={n_train:,} / Val={n_val:,} / Test={n_test:,}")
    print(f"  Classes: {num_classes} → {class_names}")

    print("\n  Class distribution (actual training split):")
    y_train = splits["train"][1]
    for idx, name in enumerate(class_names):
        count = int(np.sum(y_train == idx))
        print(f"    {name:<10} count={count:>6}")

    input_dim = X.shape[1]
    module = TransitionLightningModule(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )

    results = train_lightning_model(
        lightning_module=module,
        loaders=loaders,
        model_name=model_name,
        model_dir=model_dir,
        max_epochs=max_epochs,
        patience=patience,
    )

    module.eval()
    per_class = _evaluate_per_class(module, loaders["test"], class_names)
    total_support = sum(v["support"] for v in per_class.values())
    accuracy = float(sum(
        v["recall"] * v["support"] for v in per_class.values()
    ) / total_support) if total_support > 0 else 0.0

    results["metadata_path"] = str(prep_dir / "metadata.json")
    results["evaluation"] = {
        "accuracy": accuracy,
        "per_class": per_class,
        **results.get("test_metrics", {}),
    }
    results["class_names"] = class_names
    results["num_classes"] = num_classes
    results["input_dim"] = input_dim
    results["hidden_dims"] = hidden_dims

    eval_artifact_path = save_eval_artifact(
        model_name=model_name,
        model_dir=model_dir,
        evaluation=results["evaluation"],
        data_source=data_dir,
        n_train=n_train, n_val=n_val, n_test=n_test,
    )
    results["eval_artifact_path"] = str(eval_artifact_path)

    print(f"\n{'─' * 40}")
    print(f"  Model: {results['model_path']}")
    print(f"  Best Val Loss (CE): {results['best_val_loss']:.6f}")
    if results.get("test_metrics"):
        print(f"  Test Metrics: {results['test_metrics']}")
    print(f"  Eval artifact: {eval_artifact_path}")
    print(f"{'─' * 40}")

    return results
