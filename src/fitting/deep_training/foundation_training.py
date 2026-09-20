"""Shared Lightning training infrastructure: split, training loop, TorchScript export and the component-level evaluation artifact."""

from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 200

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader


def strip_debug_info(model_path: Path) -> None:
    # the .debug_pkl entries hold source ranges with absolute local paths and
    # are not needed for inference
    model_path = Path(model_path)
    tmp = model_path.with_suffix(model_path.suffix + ".tmp")
    with zipfile.ZipFile(model_path) as zin, zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            if not info.filename.endswith(".debug_pkl"):
                zout.writestr(info, zin.read(info.filename))
    tmp.replace(model_path)


def build_mlp_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: Optional[int] = None,
    dropout_rate: float = 0.0,
    output_activation: str = "none",
) -> "nn.Sequential":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _SoftplusFirst(nn.Module):

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
    elif output_activation != "none":
        raise ValueError(
            f"Fail Fast: unknown output_activation='{output_activation}'. "
            f"Allowed: 'none', 'softplus_first', 'softplus'."
        )

    return nn.Sequential(*layers)


_CACHE_FILES = ("data.npz", "metadata.json")
_CACHE_SOURCE_FILE = "_source.txt"


def _cache_signature(prepare_fn: Any, source_data_dir: Path) -> str:
    source = Path(inspect.getsourcefile(prepare_fn)).read_bytes()
    return f"{source_data_dir.resolve()}\n{hashlib.sha256(source).hexdigest()}"


def _cache_is_valid(output_dir: Path, signature: str) -> bool:
    if not all(
        (output_dir / f).exists() and (output_dir / f).stat().st_size > 0
        for f in _CACHE_FILES
    ):
        return False
    source_file = output_dir / _CACHE_SOURCE_FILE
    if not source_file.exists() or source_file.read_text().strip() != signature:
        print(f"  [Cache] {output_dir.name}: data source or preparation changed, rebuilding.")
        return False
    return True


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
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    signature = _cache_signature(prepare_fn, data_dir)

    if _cache_is_valid(output_dir, signature):
        return load_prepared_data(output_dir)

    if csv_pattern == "events":
        events = sorted(data_dir.rglob("*_events_*.csv"))
        orders = sorted(data_dir.rglob("*_orders_*.csv"))
        if not events:
            raise FileNotFoundError(f"No *_events_*.csv in {data_dir}")
        if not orders:
            raise FileNotFoundError(f"No *_orders_*.csv in {data_dir}")
        prepare_fn(events_paths=events, orders_paths=orders, output_dir=output_dir)
    elif csv_pattern == "orders":
        orders = sorted(data_dir.rglob("*orders*.csv"))
        if not orders:
            raise FileNotFoundError(f"No *orders*.csv in {data_dir}")
        prepare_fn(orders_paths=orders, output_dir=output_dir)
    elif csv_pattern == "events_only":
        events = sorted(data_dir.rglob("*_events_*.csv"))
        if not events:
            raise FileNotFoundError(f"No *_events_*.csv in {data_dir}")
        prepare_fn(events_paths=events, output_dir=output_dir)
    else:
        raise ValueError(f"Unknown csv_pattern: {csv_pattern}")

    (output_dir / _CACHE_SOURCE_FILE).write_text(signature)
    return load_prepared_data(output_dir)


def three_way_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got: {X.shape}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1D or 2D, got: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape[0]} vs {y.shape[0]}")
    if X.shape[0] == 0:
        raise ValueError("Empty dataset.")

    n = X.shape[0]

    n_test = max(1, int(n * test_size))
    remaining = n - n_test
    n_val = max(1, int(remaining * test_size / (1.0 - test_size)))
    n_train = remaining - n_val

    if n_train < 1:
        raise ValueError(f"test_size={test_size} too large.")

    # chronological blocks: slices are views, so the split does not double the
    # footprint of a matrix that already fills a third of the machine
    a, b = n_train, n_train + n_val
    return {
        "train": (X[:a], y[:a]),
        "val": (X[a:b], y[a:b]),
        "test": (X[b:], y[b:]),
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

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = model_dir / "checkpoints" / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_loss", mode="min", save_top_k=1,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=False,
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

    # Restore the best weights BEFORE testing, so the test metrics and the
    # exported TorchScript model describe the same weights.
    if best_ckpt:
        ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        lightning_module.load_state_dict(ckpt_data["state_dict"])
    best_module = lightning_module
    best_module.eval()

    test_metrics = trainer.test(lightning_module, loaders["test"], verbose=False)[0]

    model_path = model_dir / f"{model_name}_model.pt"
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
    strip_debug_info(model_path)

    return {
        "best_val_loss":  best_val_loss,
        "test_metrics":   test_metrics,
        "epochs_trained": trainer.current_epoch,
        "model_path":     str(model_path),
    }


def init_output_at_marginal(net, bias: List[float]) -> None:
    import torch
    import torch.nn as nn
    last = [m for m in net if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype))


class BaseTrainingModule:

    def _init_base(
        self,
        *,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        learning_rate: float,
        dropout_rate: float,
        scheduler_patience: int = 10,
        output_activation: str = "none",
    ) -> None:
        self.net = build_mlp_layers(
            input_dim, hidden_dims, output_dim, dropout_rate,
            output_activation=output_activation,
        )
        self.lr = learning_rate
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
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self._scheduler_patience, factor=0.5,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # Adam's first steps move every parameter by ~lr regardless of gradient
        # scale; on a heteroscedastic head that can collapse sigma before the
        # second-moment estimate exists. A linear warm-up bounds those steps.
        if self.trainer.global_step < WARMUP_STEPS:
            scale = (self.trainer.global_step + 1) / WARMUP_STEPS
            for group in optimizer.param_groups:
                group["lr"] = self.lr * scale
        optimizer.step(closure=optimizer_closure)


class GaussianNLLModule(BaseTrainingModule):

    _LOG_HALF = -0.6931471805599453
    _ROOT_TWO = 1.4142135623730951

    @classmethod
    def _log_sf(cls, z):
        import torch
        return cls._LOG_HALF + torch.log(torch.special.erfcx(z / cls._ROOT_TWO)) - 0.5 * z * z

    @classmethod
    def _log_interval_prob(cls, zl, zu):
        import torch
        near = torch.minimum(zl.abs(), zu.abs())
        far = torch.maximum(zl.abs(), zu.abs())
        log_near = cls._log_sf(near)
        ratio = torch.exp(cls._log_sf(far) - log_near).clamp(max=0.999999)
        one_sided = log_near + torch.log1p(-ratio)
        p_center = 1.0 - torch.exp(cls._log_sf(zu.clamp(min=0.0))) \
            - torch.exp(cls._log_sf((-zl).clamp(min=0.0)))
        centered = torch.log(p_center.clamp(min=1e-30))
        return torch.where((zl < 0.0) & (zu > 0.0), centered, one_sided)

    @staticmethod
    def marginal_bias(y_train: np.ndarray) -> List[float]:
        y = np.asarray(y_train, dtype=float)
        return [float(np.log(np.expm1(y.mean()))), float(np.log(y.var()))]

    def _nll_loss(self, pred, y):
        import torch
        mean = pred[:, 0]
        sigma = torch.exp(0.5 * pred[:, 1])
        zu = (y - mean) / sigma
        zl = (y - 1.0 - mean) / sigma
        return -self._log_interval_prob(zl, zu).mean()

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

    @staticmethod
    def marginal_bias(y_train: np.ndarray) -> List[float]:
        return [float(np.log(np.asarray(y_train, dtype=float).mean()))]

    def _compute_loss(self, batch, stage: str):
        import torch
        import torch.nn as nn
        x, y = batch
        pred = self(x)
        log_scale = pred[:, 0]
        inv_s = torch.exp(-log_scale)
        loss = torch.mean((y - 1.0) * inv_s - torch.log(-torch.expm1(-inv_s)))
        self.log(f"{stage}_loss", loss, prog_bar=True)
        if stage in ("val", "test"):
            pred_mean = torch.exp(log_scale)
            mae = nn.functional.l1_loss(pred_mean, y)
            self.log(f"{stage}_mae", mae, prog_bar=(stage == "val"))
            if stage == "test":
                self.log("test_rmse", torch.sqrt(nn.functional.mse_loss(pred_mean, y)))
        return loss

