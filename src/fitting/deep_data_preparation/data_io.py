"""Shared I/O utilities for DeepSim data preparation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class RawEvent:
    order_id: str
    timestamp_event_start: float
    station: str
    station_type: str
    time_processing: float
    is_breakdown: bool
    net_process_time: float
    repair_time: float


def load_events(paths: Sequence[Path]) -> List[RawEvent]:
    events: List[RawEvent] = []
    for path in paths:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                events.append(RawEvent(
                    order_id=r["order_id"],
                    timestamp_event_start=float(r["timestamp_event_start"]),
                    station=r["station"],
                    station_type=r.get("station_type", "machine"),
                    time_processing=float(r["time_processing"]),
                    is_breakdown=r["is_breakdown"] == "True",
                    net_process_time=float(r["net_process_time"]),
                    repair_time=float(r["repair_time"]),
                ))
    return events


def load_orders(
    paths: Sequence[Path],
    *,
    with_completions: bool = False,
) -> "Dict[str, Dict[str, str]] | Tuple[Dict[str, Dict[str, str]], Dict[str, Optional[float]]]":
    features: Dict[str, Dict[str, str]] = {}
    completions: Dict[str, Optional[float]] = {}
    meta_cols = {"order_id", "timestamp_creation", "timestamp_completion"}

    for path in paths:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                oid = r["order_id"]
                features[oid] = {k: v for k, v in r.items() if k not in meta_cols}
                if with_completions:
                    tc = r.get("timestamp_completion", "")
                    completions[oid] = float(tc) if tc and tc != "None" else None

    if with_completions:
        return features, completions
    return features


def build_feature_layout(
    groups: List[Tuple[str, Dict[str, int]]],
) -> Tuple[List[Dict[str, Any]], int]:
    layout: List[Dict[str, Any]] = []
    offset = 0

    for name, mapping in groups:
        layout.append({"name": name, "offset": offset, "size": len(mapping)})
        offset += len(mapping)

    return layout, offset


def save_prepared_data(
    X: np.ndarray,
    y: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "data.npz", X=X, y=y)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to: {output_dir}")
    print(f"  Samples: {len(y):,}")


def find_csv_files(base_dir):
    events = sorted(base_dir.rglob("*_events_*.csv"))
    orders = sorted(base_dir.rglob("*_orders_*.csv"))
    return events, orders
