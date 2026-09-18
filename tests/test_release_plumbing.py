"""The seams between the scripts: what the log writer stores, where the
runner expects a model set, what the trainer refuses without a search result,
and what a shipped model must carry to be loadable.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.experiments.sim_runner import MODEL_FILES, MODULES, model_paths, models_trained
from src.recording.recorder import PROCESS_LOG_DTYPE
from src.recording.results_saver import ResultsSaver


def test_saver_writes_durations_verbatim(tmp_path):
    """logged ≡ executed: the writer must not round a second time."""
    saver = ResultsSaver(tmp_path, "exp", "proc")
    row = np.zeros(1, dtype=PROCESS_LOG_DTYPE)
    row["order_id"], row["station"], row["station_type"] = "R1_J000001", "M1", "machine"
    row["timestamp_event_start"], row["time_processing"] = 12.0, 7.25
    row["net_process_time"], row["repair_time"] = 7.25, 0.0
    saver.save_run(1, row, [{"order_id": "R1_J000001", "timestamp_creation": 0.0,
                             "timestamp_completion": 20.0}])
    events = next(saver.batch_dir.glob("run1_events_*.csv"))
    with events.open() as f:
        rec = next(csv.DictReader(f))
    assert rec["time_processing"] == "7.250000" and rec["order_id"] == "R1_J000001"


def test_model_paths_follow_the_trainers_file_names(tmp_path):
    paths = model_paths(tmp_path)
    assert set(paths) == {f"{s}_{k}" for s in MODULES for k in ("model_path", "metadata_path")}
    assert not models_trained(tmp_path)
    for model_file, metadata_file in MODEL_FILES.values():
        (tmp_path / model_file).write_bytes(b"")
        (tmp_path / metadata_file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / metadata_file).write_text("{}")
    assert models_trained(tmp_path)


def test_training_refuses_to_run_without_a_search_result(tmp_path):
    from scripts.train_models import _get_params
    with pytest.raises(SystemExit, match="HPO"):
        _get_params("process_time", tmp_path)
    (tmp_path / "process_time_best_params.json").write_text(json.dumps({
        "params": {"hidden_dims": "(64, 32)", "learning_rate": 1e-3,
                   "dropout_rate": 0.1, "batch_size": 64}, "best_value": 1.0}))
    params = _get_params("process_time", tmp_path)
    assert params["hidden_dims"] == (64, 32) and params["batch_size"] == 64


def test_manifest_is_derived_from_the_run(tmp_path):
    from scripts.train_models import TRAINERS, _write_manifest
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "run_metadata.json").write_text(json.dumps({"seed": 101, "days": 14}))
    params, results = {}, {}
    for name, _, _ in TRAINERS:
        params[name] = {"hidden_dims": (8,), "learning_rate": 1e-3, "dropout_rate": 0.0, "batch_size": 8}
        model = tmp_path / f"{name}_model.pt"
        model.write_bytes(name.encode())
        results[name] = {"model_path": str(model), "best_val_loss": 0.5, "epochs_trained": 3,
                         "metadata_path": str(tmp_path / f"{name}_data/metadata.json")}
    out = _write_manifest(tmp_path, data_dir, params, results, max_epochs=9, patience=2, test_size=0.15)
    manifest = json.loads(out.read_text())
    assert manifest["artifacts"]["process_time"]["epochs_trained"] == 3
    assert manifest["training"]["hyperparameters"]["process_time"]["hidden_dims"] == [8]
    assert manifest["training"]["max_epochs"] == 9
    assert manifest["training_data"]["seed"] == 101


def test_a_transition_model_without_the_visit_block_is_refused(tmp_path, monkeypatch):
    from src.dynamics.DeepSim import deep_transition
    meta = {"encoding_maps": {"feature_dim": 3}, "num_classes": 2, "class_names": ["End", "B5"],
            "feature_layout": [{"name": "modell", "offset": 0, "size": 3}], "n_hist_slots": 1}
    monkeypatch.setattr(deep_transition, "load_deep_model", lambda m, md: (object(), meta))
    tr = deep_transition.DeepTransition(tmp_path / "m.pt", tmp_path / "meta.json", np.random.default_rng(0))
    with pytest.raises(ValueError, match="predates the visit-index"):
        tr.initialize({})


def test_checksum_manifest_lists_only_release_artifacts(tmp_path, monkeypatch):
    from scripts import write_checksums
    for rel in ("models/x_model.pt", "models/x_data/metadata.json", "models/hpo/x_best_params.json",
                "models/hpo/x/trial_0/model.pt", "models/checkpoints/x.ckpt", "models/x_data/data.npz",
                "results/verification/a.csv", "results/verification/shadow/big.csv",
                "results/execution_audit/b.json"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setattr(write_checksums, "REPO", tmp_path)
    got = sorted(str(p.relative_to(tmp_path)) for p in write_checksums.collect())
    assert got == ["models/hpo/x_best_params.json", "models/x_data/metadata.json", "models/x_model.pt",
                   "results/execution_audit/b.json", "results/verification/a.csv"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
