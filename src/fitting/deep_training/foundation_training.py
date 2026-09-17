"""
Shared Lightning training infrastructure: split, training loop, TorchScript
export and the component-level evaluation artifact.

torch and pytorch_lightning are imported lazily so data preparation works
without the GPU packages installed.
"""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

# Bump when the eval-artifact fields change incompatibly.
EVAL_ARTIFACT_SCHEMA_VERSION = 1


def build_mlp_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    output_activation: str = "none",
) -> "nn.Sequential":
    """
    Build an MLP as nn.Sequential: (Linear -> ReLU -> Dropout) x N [+ Linear [+ act]].

    output_dim=None omits the final Linear (shared trunks). output_activation
    "softplus_first" applies SoftPlus to channel 0 only, structurally forcing
    mean > 0 for heteroscedastic regression.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _SoftplusFirst(nn.Module):
        """SoftPlus on channel 0, identity on the remaining channels.

        The sampling-side floor in deep_process_time catches only the rare
        deep negative tail of the Gaussian sample; the positive mean is the
        primary constraint.
        """

        def forward(self, x):
            first = F.softplus(x[..., 0:1])
            rest = x[..., 1:]
            return torch.cat([first, rest], dim=-1)

    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)])
        prev = h
    if output_dim is not None:
        layers.append(nn.Linear(prev, output_dim))

    if output_activation == "softplus_first":
        if output_dim is None or output_dim < 2:
            raise ValueError(
                f"Fail Fast: output_activation='softplus_first' requires "
                f"output_dim>=2 (mean+log_var); "
                f"got output_dim={output_dim}."
            )
        layers.append(_SoftplusFirst())
    elif output_activation == "softplus":
        layers.append(nn.Softplus())
    elif output_activation != "none":
        raise ValueError(
            f"Fail Fast: unknown output_activation='{output_activation}'. "
            f"Allowed: 'none', 'softplus_first', 'softplus'."
        )

    return nn.Sequential(*layers)


_CACHE_FILES = ("data.npz", "metadata.json")
_CACHE_SOURCE_FILE = "_source.txt"


def _cache_is_valid(output_dir: Path, source_data_dir: Path) -> bool:
    """True if data.npz + metadata.json are both present and were produced from
    source_data_dir."""
    if not all(
        (output_dir / f).exists() and (output_dir / f).stat().st_size > 0
        for f in _CACHE_FILES
    ):
        return False
    source_file = output_dir / _CACHE_SOURCE_FILE
    if not source_file.exists():
        return False
    cached_source = source_file.read_text().strip()
    if cached_source != str(source_data_dir.resolve()):
        print(
            f"  [Cache] Invalid - data source changed:\n"
            f"    Cache:   {cached_source}\n"
            f"    Current: {source_data_dir.resolve()}\n"
            f"    Rebuilding cache."
        )
        return False
    return True


def write_cache_source(output_dir: Path, source_data_dir: Path) -> None:
    """Record the data source in _source.txt after successful preparation."""
    (output_dir / _CACHE_SOURCE_FILE).write_text(str(source_data_dir.resolve()))


def load_prepared_data(output_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(output_dir / "data.npz")
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)
    print(f"  [Cache] {output_dir.name}: {len(data['y']):,} Samples, Feature-Dim={data['X'].shape[1]}")
    return data["X"], data["y"], metadata


def cached_prepare(
    data_dir: Path,
    output_dir: Path,
    prepare_fn: Any,
    *,
    csv_pattern: str = "events",
    cache_validator: Any = None,
    **prepare_kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load cached data if present, otherwise call prepare_fn.

    csv_pattern selects which logs are globbed: 'events' (events+orders),
    'orders', or 'events_only'. A cache_validator returning False discards
    an otherwise valid cache.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if _cache_is_valid(output_dir, source_data_dir=data_dir):
        X, y, metadata = load_prepared_data(output_dir)
        if cache_validator is None or cache_validator(metadata):
            return X, y, metadata

    if csv_pattern == "events":
        events = sorted(data_dir.rglob("*_events_*.csv"))
        orders = sorted(data_dir.rglob("*_orders_*.csv"))
        if not events:
            raise FileNotFoundError(f"No *_events_*.csv in {data_dir}")
        if not orders:
            raise FileNotFoundError(f"No *_orders_*.csv in {data_dir}")
        prepare_fn(events_paths=events, orders_paths=orders,
                    output_dir=output_dir, **prepare_kwargs)
    elif csv_pattern == "orders":
        orders = sorted(data_dir.rglob("*orders*.csv"))
        if not orders:
            raise FileNotFoundError(f"No *orders*.csv in {data_dir}")
        prepare_fn(orders_paths=orders, output_dir=output_dir, **prepare_kwargs)
    elif csv_pattern == "events_only":
        events = sorted(data_dir.rglob("*_events_*.csv"))
        if not events:
            raise FileNotFoundError(f"No *_events_*.csv in {data_dir}")
        prepare_fn(events_paths=events, output_dir=output_dir, **prepare_kwargs)
    else:
        raise ValueError(f"Unknown csv_pattern: {csv_pattern}")

    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    write_cache_source(output_dir, data_dir)
    X_data = np.load(output_dir / "data.npz")
    return X_data["X"], X_data["y"], metadata


def _create_history_callback():
    import pytorch_lightning as pl

    class _HistoryCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self.history: Dict[str, list] = {
                "train_loss": [],
                "val_loss": [],
            }

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            # PL 2.x logs this as train_loss or train_loss_epoch
            loss_val = metrics.get("train_loss") or metrics.get("train_loss_epoch")
            if loss_val is not None:
                self.history["train_loss"].append(float(loss_val))

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if "val_loss" in metrics:
                self.history["val_loss"].append(float(metrics["val_loss"]))
            for key, val in metrics.items():
                if key.startswith("val_") and key != "val_loss":
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(float(val))

    return _HistoryCallback()


def _has_finite_numbers(obj: Any) -> bool:
    if isinstance(obj, dict):
        return all(_has_finite_numbers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_has_finite_numbers(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def save_eval_artifact(
    *,
    model_name: str,
    model_dir: Path,
    evaluation: Dict[str, Any],
    data_source: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    split: str = "test",
) -> Path:
    """Persist the evaluation as `<model_dir>/evaluations/<model_name>_eval.json`."""
    model_dir = Path(model_dir)
    eval_dir = model_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "schema_version": EVAL_ARTIFACT_SCHEMA_VERSION,
        "model_name": model_name,
        "split": split,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "evaluation": evaluation,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": str(Path(data_source).resolve()),
    }

    out_path = eval_dir / f"{model_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    return out_path


def validate_deployment_eligibility(
    model_name: str,
    model_dir: Path,
) -> bool:
    """
    Load the eval artifact and run minimal checks.

    Logs a warning on violation but does not block: the artifact backs the
    deployment-gate statement; the final release is a reviewer decision.
    """
    artifact_path = Path(model_dir) / "evaluations" / f"{model_name}_eval.json"
    if not artifact_path.exists():
        print(f"  [Eligibility] WARN {model_name}: no artifact at {artifact_path}")
        return False

    with open(artifact_path) as f:
        artifact = json.load(f)

    issues: List[str] = []
    schema_v = artifact.get("schema_version")
    if schema_v != EVAL_ARTIFACT_SCHEMA_VERSION:
        issues.append(f"schema_version={schema_v} (expected {EVAL_ARTIFACT_SCHEMA_VERSION})")
    if int(artifact.get("n_test", 0)) <= 0:
        issues.append(f"n_test={artifact.get('n_test')} <= 0")
    if not artifact.get("evaluation"):
        issues.append("evaluation empty")
    if not _has_finite_numbers(artifact.get("evaluation", {})):
        issues.append("evaluation contains NaN/Inf")

    if issues:
        print(f"  [Eligibility] WARN {model_name}: {'; '.join(issues)}")
        return False
    print(f"  [Eligibility] OK   {model_name}: artifact={artifact_path.name}")
    return True


def three_way_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Chronological train/val/test split (data order; no shuffle, no random_state)."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got: {X.shape}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D, got: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] == 0:
        raise ValueError("Empty dataset.")

    n = X.shape[0]
    indices = np.arange(n)

    n_test = max(1, int(n * test_size))
    remaining = n - n_test
    n_val = max(1, int(remaining * test_size / (1.0 - test_size)))
    n_train = remaining - n_val

    if n_train < 1:
        raise ValueError(f"test_size={test_size} too large.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }


def make_loaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    task: str = "regression",
) -> Dict[str, "DataLoader"]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loaders = {}
    for split_name, (X, y) in splits.items():
        x_tensor = torch.from_numpy(X).float()
        if task == "classification":
            y_tensor = torch.from_numpy(y).long()
        else:
            y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(x_tensor, y_tensor)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=0,
        )

    return loaders


def train_lightning_model(
    lightning_module,
    loaders: Dict[str, Any],
    *,
    model_name: str,
    model_dir: Path,
    max_epochs: int,
    patience: int = 15,
) -> Dict[str, Any]:
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    history_cb = _create_history_callback()
    checkpoint_dir = model_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=str(model_dir / "tensorboard"),
        name=model_name,
        default_hp_metric=False,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_loss", mode="min", save_top_k=1,
        ),
        history_cb,
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        enable_progress_bar=True,
        log_every_n_steps=max(1, len(loaders["train"]) // 4),
        deterministic=True,
        enable_checkpointing=True,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(lightning_module, loaders["train"], loaders["val"])

    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_val_loss = float(trainer.checkpoint_callback.best_model_score or float("inf"))

    # Restore the best weights BEFORE testing, so test metrics, the exported
    # TorchScript model and the eval artifact all describe the same weights.
    if best_ckpt:
        ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        lightning_module.load_state_dict(ckpt_data["state_dict"])
    best_module = lightning_module
    best_module.eval()

    test_metrics = {}
    if "test" in loaders:
        test_results = trainer.test(lightning_module, loaders["test"], verbose=False)
        if test_results:
            test_metrics = test_results[0]

    model_path = model_dir / f"{model_name}_model.pt"
    # Trace-example input dim: full_input_dim takes precedence (autoregressive
    # models whose trunk sees only base_dim but whose forward() expects the
    # full input including conditioning).
    if hasattr(best_module.net, "full_input_dim"):
        input_dim = best_module.net.full_input_dim
    else:
        try:
            first_param = next(best_module.net.parameters())
            input_dim = first_param.shape[1]
        except (StopIteration, IndexError) as e:
            raise RuntimeError(
                f"TorchScript export failed: could not derive input_dim from the "
                f"first parameter of '{model_name}'. "
                f"Is the first layer an nn.Linear? Error: {e}"
            ) from e
    # Trace on CPU: saved models must be device-independent (MPS tensors
    # are not serializable).
    cpu_module = best_module.net.cpu()
    example = torch.zeros(1, input_dim)
    scripted = torch.jit.trace(cpu_module, example)
    scripted.save(str(model_path))

    return {
        "best_val_loss":   best_val_loss,
        "best_ckpt_path":  best_ckpt,
        "epochs_trained":  trainer.current_epoch,
        "history":         history_cb.history,
        "test_metrics":    test_metrics,
        "model_path":      str(model_path),
        "tb_log_dir":      tb_logger.log_dir,
        "samples_trained": len(loaders["train"].dataset),
        "samples_val":     len(loaders["val"].dataset),
        "samples_test":    len(loaders.get("test", loaders["val"]).dataset),
    }


class BaseTrainingModule:
    """Mixed with pl.LightningModule only in subclasses, so this module stays
    importable without pytorch_lightning installed."""

    def _init_base(
        self,
        *,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float,
        dropout_rate: float,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        output_activation: str = "none",
    ) -> None:
        self.net = build_mlp_layers(
            input_dim, hidden_dims, output_dim, dropout_rate,
            output_activation=output_activation,
        )
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self._scheduler_patience = scheduler_patience

    def forward(self, x):
        return self.net(x)

    @abstractmethod
    def _compute_loss(self, batch, stage: str):
        ...

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._compute_loss(batch, "test")

    def configure_optimizers(self):
        import torch
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self._scheduler_patience, factor=0.5,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


class GaussianNLLModule(BaseTrainingModule):
    """Heteroscedastic regression: output (mean, log_var), interval NLL.

    Targets are whole seconds (integer time contract): an observation k means
    the latent duration lay in (k-1, k]. The loss is the interval likelihood
    -log(Phi((k-mu)/sigma) - Phi((k-1-mu)/sigma)), evaluated in log space via
    log_ndtr, so the module learns the LATENT continuous density; sampling
    N(mu, sigma) and ceiling once at inference then reproduces the observed
    integer distribution without the +0.5 s double-discretization bias.
    """

    def _nll_loss(self, pred, y):
        import torch
        mean = pred[:, 0]
        log_var = pred[:, 1].clamp(-6.0, 6.0)
        sigma = torch.exp(0.5 * log_var)
        zu = (y - mean) / sigma          # upper edge k
        zl = (y - 1.0 - mean) / sigma    # lower edge k-1
        log_fu = torch.special.log_ndtr(zu)
        log_fl = torch.special.log_ndtr(zl)
        # log(F(k) - F(k-1)) = log_fu + log1p(-exp(log_fl - log_fu)); the
        # difference is < 0 by construction, clamped against underflow.
        log_p = log_fu + torch.log1p(
            -torch.exp((log_fl - log_fu).clamp(max=-1e-12))
        )
        return -log_p.clamp(min=-30.0).mean()

    def _compute_loss(self, batch, stage: str):
        import torch
        import torch.nn as nn
        x, y = batch
        pred = self(x)
        loss = self._nll_loss(pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage in ("val", "test"):
            mae = nn.functional.l1_loss(pred[:, 0], y)
            self.log(f"{stage}_mae", mae, prog_bar=(stage == "val"))
            if stage == "test":
                self.log("test_rmse", torch.sqrt(nn.functional.mse_loss(pred[:, 0], y)))
        return loss

class ExponentialNLLModule(BaseTrainingModule):
    """
    Conditional exponential regression: output (log_scale,), interval NLL.

    Targets are whole seconds (integer time contract): observation k means the
    latent Exp(scale) duration lay in (k-1, k], so
    P(ceil(X) = k) = e^{-(k-1)/s} - e^{-k/s} and
    NLL = (k-1)/s - log(1 - e^{-1/s})  with s = exp(log_scale).
    log_scale is deliberately left unclamped.

    Inference (inversion sampling + one ceil): y = ceil(-log(u) * scale).
    """

    def _compute_loss(self, batch, stage: str):
        import torch
        import torch.nn as nn
        x, y = batch
        pred = self(x)
        log_scale = pred[:, 0]
        inv_s = torch.exp(-log_scale)
        # -log(1 - e^{-1/s}) via expm1 for stability at large s
        loss = torch.mean((y - 1.0) * inv_s - torch.log(-torch.expm1(-inv_s)))
        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage in ("val", "test"):
            # mean(Exp(scale)) = scale = exp(log_scale)
            pred_mean = torch.exp(log_scale)
            mae = nn.functional.l1_loss(pred_mean, y)
            self.log(f"{stage}_mae", mae, prog_bar=(stage == "val"))
            if stage == "test":
                self.log("test_rmse", torch.sqrt(nn.functional.mse_loss(pred_mean, y)))
        return loss

