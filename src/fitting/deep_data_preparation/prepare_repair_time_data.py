"""Training-data preparation for NN-based downtime-duration prediction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.fitting.deep_data_preparation.data_io import (
    RawEvent, load_events, save_prepared_data,
)
from src.fitting.deep_data_preparation.split_helpers import (
    DEFAULT_TRAIN_RATIO, chronological_train_end_idx, count_oov, summarize_oov,
)


@dataclass
class RepairTimeSample:
    station: str
    operating_time_since_last: float
    utilization: float
    wear_ratio: float
    repair_time: float


@dataclass
class RepairTimeEncodingMaps:
    station: Dict[str, int] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return len(self.station) + 3

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "feature_dim": self.feature_dim,
        }


def extract_repair_samples(
    events: List[RawEvent],
) -> List[RepairTimeSample]:
    machine_events = [e for e in events if e.station_type == "machine"]

    events_by_station: Dict[str, List[RawEvent]] = defaultdict(list)
    for e in machine_events:
        events_by_station[e.station].append(e)

    for station in events_by_station:
        events_by_station[station].sort(key=lambda e: e.timestamp_event_start)

    timed_samples: List[Tuple[float, RepairTimeSample]] = []
    skipped_zero_rt = 0
    skipped_truncated = 0
    skipped_no_ops = 0

    for station, station_events in events_by_station.items():
        cycle_start_wall: float = 0.0
        cycle_operating: float = 0.0
        in_cycle: bool = False
        first_cycle_truncated: bool = True

        for event in station_events:
            is_downtime = (
                event.is_breakdown
                or event.order_id == "BREAKDOWN"
            )

            if is_downtime:
                if event.repair_time <= 0:
                    skipped_zero_rt += 1
                    cycle_start_wall = event.timestamp_event_start
                    cycle_operating = 0.0
                    in_cycle = True
                    continue

                if cycle_operating <= 0:
                    skipped_no_ops += 1
                    cycle_start_wall = event.timestamp_event_start + event.repair_time
                    cycle_operating = 0.0
                    in_cycle = True
                    continue

                if first_cycle_truncated:
                    first_cycle_truncated = False
                    skipped_truncated += 1
                    cycle_start_wall = event.timestamp_event_start + event.repair_time
                    cycle_operating = 0.0
                    in_cycle = True
                    continue

                wall_clock = event.timestamp_event_start - cycle_start_wall
                if wall_clock > 0:
                    utilization = min(1.0, cycle_operating / wall_clock)
                else:
                    utilization = 1.0

                timed_samples.append((
                    float(event.timestamp_event_start),
                    RepairTimeSample(
                        station=station,
                        operating_time_since_last=cycle_operating,
                        utilization=utilization,
                        wear_ratio=0.0,
                        repair_time=event.repair_time,
                    ),
                ))

                cycle_start_wall = event.timestamp_event_start + event.repair_time
                cycle_operating = 0.0
                in_cycle = True
                continue

            if not in_cycle:
                cycle_start_wall = event.timestamp_event_start
                in_cycle = True

            cycle_operating += event.net_process_time

    timed_samples.sort(key=lambda ts: ts[0])
    samples: List[RepairTimeSample] = [s for _, s in timed_samples]

    print(f"  Machines: {len(events_by_station)}")
    print(f"  Downtime samples: {len(samples):,}")
    print(f"  Skipped (repair_time <= 0): {skipped_zero_rt:,}")
    print(f"  Skipped (no operating_time): {skipped_no_ops:,}")
    print(f"    Dropped (left-truncated first cycle): {skipped_truncated:,}")

    return samples


def compute_median_ttf(
    samples: List[RepairTimeSample],
) -> Dict[str, float]:
    per_station: Dict[str, List[float]] = defaultdict(list)
    for s in samples:
        per_station[s.station].append(s.operating_time_since_last)
    return {
        sid: float(np.median(v)) if len(v) >= 2 else 0.0
        for sid, v in per_station.items()
    }


def apply_wear_ratio(
    samples: List[RepairTimeSample],
    median_ttf_per_station: Dict[str, float],
) -> None:
    for s in samples:
        denom = median_ttf_per_station.get(s.station, 0.0)
        s.wear_ratio = s.operating_time_since_last / max(denom, 1.0) if denom > 0 else 0.0


def build_encoding_maps(
    samples: List[RepairTimeSample],
) -> RepairTimeEncodingMaps:
    all_stations = sorted({s.station for s in samples})
    return RepairTimeEncodingMaps(
        station={v: i for i, v in enumerate(all_stations)},
    )


def encode_samples(
    samples: List[RepairTimeSample],
    maps: RepairTimeEncodingMaps,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(samples)
    dim = maps.feature_dim
    n_stations = len(maps.station)
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i, s in enumerate(samples):
        idx = maps.station.get(s.station)
        if idx is not None:
            X[i, idx] = 1.0

        X[i, n_stations] = s.operating_time_since_last
        X[i, n_stations + 1] = s.utilization
        X[i, n_stations + 2] = s.wear_ratio

        y[i] = s.repair_time

    return X, y


def save(
    X: np.ndarray,
    y: np.ndarray,
    maps: RepairTimeEncodingMaps,
    output_dir: Path,
    *,
    train_ratio: float,
    n_train_vocab_fit: int,
    oov_stats: Dict[str, Dict[str, float]],
    median_ttf_per_station: Dict[str, float],
) -> None:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    n_stations = len(maps.station)

    save_prepared_data(X, y, {
        "encoding_maps": maps.to_dict(),
        "feature_dim": maps.feature_dim,
        "feature_layout": [
            {"name": "station",        "offset": 0,             "size": n_stations, "type": "one_hot"},
            {"name": "operating_time", "offset": n_stations,     "size": 1,          "type": "continuous"},
            {"name": "utilization",    "offset": n_stations + 1, "size": 1,          "type": "continuous"},
            {"name": "wear_ratio",     "offset": n_stations + 2, "size": 1,          "type": "continuous"},
        ],
        "median_ttf_per_station": median_ttf_per_station,
        "n_total": len(y),
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "vocab_fit_scope": "train_only",
        "vocab_fit_train_ratio": train_ratio,
        "vocab_fit_n_train": n_train_vocab_fit,
        "oov_stats": oov_stats,
    }, output_dir)
    print(f"  Target (repair_time): mean={y_mean:.2f}s, std={y_std:.2f}s")


def print_statistics(
    samples: List[RepairTimeSample],
    maps: RepairTimeEncodingMaps,
) -> None:
    print(f"\n{'─' * 60}")
    print("DATASET STATISTICS (downtime-duration regressor)")
    print(f"{'─' * 60}")
    print(f"\n  Samples total: {len(samples):,}")

    by_station: Dict[str, List[RepairTimeSample]] = defaultdict(list)
    for s in samples:
        by_station[s.station].append(s)

    print(f"\n  {'Station':<12} {'Count':>8} {'Mean RT':>10} {'Std RT':>10} "
          f"{'Mean OpT':>10} {'Mean Util':>10} {'Mean WR':>10} {'Max WR':>10}")
    print(f"  {'─' * 82}")
    for station in sorted(by_station.keys()):
        ss = by_station[station]
        rts = np.array([s.repair_time for s in ss])
        ops = np.array([s.operating_time_since_last for s in ss])
        utils = np.array([s.utilization for s in ss])
        wrs = np.array([s.wear_ratio for s in ss])
        print(f"  {station:<12} {len(ss):>8} {np.mean(rts):>10.2f} "
              f"{np.std(rts):>10.2f} {np.mean(ops):>10.1f} {np.mean(utils):>10.3f} "
              f"{np.mean(wrs):>10.3f} {np.max(wrs):>10.3f}")

    all_rt = np.array([s.repair_time for s in samples])
    all_op = np.array([s.operating_time_since_last for s in samples])
    all_util = np.array([s.utilization for s in samples])
    all_wr = np.array([s.wear_ratio for s in samples])

    print("\n  Total:")
    print(f"    Repair-Time: mean={np.mean(all_rt):.2f}s, std={np.std(all_rt):.2f}s, "
          f"median={np.median(all_rt):.2f}s")
    print(f"    Operating-Time: mean={np.mean(all_op):.1f}s, std={np.std(all_op):.1f}s")
    print(f"    Utilization: mean={np.mean(all_util):.3f}, std={np.std(all_util):.3f}")
    print(f"    Wear-Ratio: mean={np.mean(all_wr):.3f}, std={np.std(all_wr):.3f}, "
          f"max={np.max(all_wr):.3f}")

    print("\n  Encoding:")
    print(f"    Stations: {list(maps.station.keys())}")
    print(f"    Feature dim: {maps.feature_dim} "
          f"({len(maps.station)} station + 3 continuous: op_time, utilization, "
          f"wear_ratio)")


def prepare_repair_time_data(
    events_paths: Sequence[Path],
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, RepairTimeEncodingMaps]:
    print("=" * 60)
    print("DOWNTIME-DURATION REGRESSOR DATA PREPARATION")
    print("=" * 60)

    print("\n1. Loading CSVs...")
    print(f"   Events: {[p.name for p in events_paths]}")
    events = load_events(events_paths)
    print(f"   → {len(events):,} events loaded")

    print("\n2. Extracting downtime durations with context...")
    samples = extract_repair_samples(events)

    if not samples:
        raise ValueError(
            "No downtime events with repair_time > 0 found."
        )

    print("\n3. Encoding (vocabulary fit on TRAIN-only)...")
    n_train = chronological_train_end_idx(len(samples), train_ratio=train_ratio)
    train_samples = samples[:n_train]
    print(f"   → Train slice: {n_train:,} / {len(samples):,} "
          f"(train_ratio={train_ratio:.2f})")
    maps = build_encoding_maps(train_samples)
    median_ttf_per_station = compute_median_ttf(train_samples)
    apply_wear_ratio(samples, median_ttf_per_station)
    X, y = encode_samples(samples, maps)

    train_vocabs = {"station": set(maps.station.keys())}
    extractors = {"station": lambda s: s.station}
    oov_counts = count_oov(samples, n_train, extractors, train_vocabs)
    n_holdout = len(samples) - n_train
    oov_stats = summarize_oov(oov_counts, n_holdout)
    if oov_stats["station"]["count"]:
        print(f"   → OOV (station, val+test): "
              f"count={oov_stats['station']['count']}  ratio={oov_stats['station']['ratio']*100:.2f}%")

    print_statistics(samples, maps)

    print("\n4. Saving...")
    save(X, y, maps, output_dir,
         train_ratio=train_ratio, n_train_vocab_fit=n_train, oov_stats=oov_stats,
         median_ttf_per_station=median_ttf_per_station)

    return X, y, maps


def main():
    parser = argparse.ArgumentParser(
        description="Prepares training data for downtime-duration prediction.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path)
    group.add_argument("--events", type=Path, nargs="+")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/repair_time_training"))

    args = parser.parse_args()

    if args.data_dir:
        events_paths = sorted(args.data_dir.rglob("*_events_*.csv"))
        if not events_paths:
            parser.error(f"No *_events_*.csv in {args.data_dir}")
    else:
        events_paths = args.events

    prepare_repair_time_data(
        events_paths=events_paths,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()