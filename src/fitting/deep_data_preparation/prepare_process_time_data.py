"""Training-data preparation for NN-based process-time prediction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.dynamics.foundation_dynamics import (
    NONE_TOKEN, N_SHIFTS, detect_shift, encode_time_features, set_onehot,
)
from src.fitting.deep_data_preparation.split_helpers import (
    DEFAULT_TRAIN_RATIO, chronological_train_end_idx, count_oov, summarize_oov,
)
from src.fitting.deep_data_preparation.data_io import (
    find_csv_files,
    RawEvent, load_events, load_orders, build_feature_layout, save_prepared_data,
)

# Time-encoding period: 24 h (daily rhythm)
TIME_PERIODS_SECONDS: list = [86400.0]


@dataclass
class TrainingSample:
    modell: str
    feature_a: str
    feature_b: str

    # Predecessor product features on the same machine
    prev_modell: str
    prev_feature_a: str
    prev_feature_b: str

    station: str
    timestamp_event_start: float
    net_process_time: float


@dataclass
class EncodingMaps:
    modell: Dict[str, int] = field(default_factory=dict)
    feature_a: Dict[str, int] = field(default_factory=dict)
    feature_b: Dict[str, int] = field(default_factory=dict)
    station: Dict[str, int] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        return (
            len(self.modell)       # current modell
            + len(self.feature_a)  # current feature_a
            + len(self.feature_b)  # current feature_b
            + len(self.modell)     # prev modell
            + len(self.feature_a)  # prev feature_a
            + len(self.feature_b)  # prev feature_b
            + len(self.station)    # station
            + N_SHIFTS            # shift one-hot (early/late/night)
            + 2 * len(TIME_PERIODS_SECONDS)  # sin/cos per period
        )

    def to_dict(self) -> dict:
        return {
            "modell": self.modell,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "station": self.station,
            "feature_dim": self.feature_dim,
        }


def filter_and_join(
    events: List[RawEvent],
    orders: Dict[str, Dict[str, str]],
) -> List[Tuple[RawEvent, Dict[str, str]]]:
    result = []
    skipped_no_order = 0
    skipped_system = 0

    for event in events:
        if event.station_type != "machine" or event.is_breakdown or event.order_id == "BREAKDOWN":
            skipped_system += 1
            continue

        features = orders.get(event.order_id)
        if features is None:
            skipped_no_order += 1
            continue

        result.append((event, features))

    result.sort(key=lambda x: x[0].timestamp_event_start)

    print(f"  Events total: {len(events)}")
    print(f"  Non-operation rows skipped (buffers, breakdowns, displacements): {skipped_system}")
    print(f"  Events without order skipped: {skipped_no_order}")
    print(f"  Productive events with features: {len(result)}")

    return result


def build_training_samples(
    joined: List[Tuple[RawEvent, Dict[str, str]]],
) -> List[TrainingSample]:
    prev_on_machine: Dict[str, Dict[str, str]] = {}

    samples = []
    for event, features in joined:
        station = event.station

        prev = prev_on_machine.get(station)
        if prev is None:
            prev_modell = NONE_TOKEN
            prev_feature_a = NONE_TOKEN
            prev_feature_b = NONE_TOKEN
        else:
            prev_modell = prev.get("modell", NONE_TOKEN)
            prev_feature_a = prev.get("feature_a", NONE_TOKEN)
            prev_feature_b = prev.get("feature_b", NONE_TOKEN)

        samples.append(TrainingSample(
            modell=features.get("modell", NONE_TOKEN),
            feature_a=features.get("feature_a", NONE_TOKEN),
            feature_b=features.get("feature_b", NONE_TOKEN),
            prev_modell=prev_modell,
            prev_feature_a=prev_feature_a,
            prev_feature_b=prev_feature_b,
            station=station,
            timestamp_event_start=event.timestamp_event_start,
            net_process_time=event.net_process_time,
        ))

        prev_on_machine[station] = dict(features)

    return samples


def build_encoding_maps(samples: List[TrainingSample]) -> EncodingMaps:
    modell_vals = sorted({s.modell for s in samples} | {s.prev_modell for s in samples})
    fa_vals = sorted({s.feature_a for s in samples} | {s.prev_feature_a for s in samples})
    fb_vals = sorted({s.feature_b for s in samples} | {s.prev_feature_b for s in samples})
    station_vals = sorted({s.station for s in samples})

    return EncodingMaps(
        modell={v: i for i, v in enumerate(modell_vals)},
        feature_a={v: i for i, v in enumerate(fa_vals)},
        feature_b={v: i for i, v in enumerate(fb_vals)},
        station={v: i for i, v in enumerate(station_vals)},
    )


def encode_samples(
    samples: List[TrainingSample],
    maps: EncodingMaps,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(samples)
    dim = maps.feature_dim
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    off_modell = 0
    off_fa = off_modell + len(maps.modell)
    off_fb = off_fa + len(maps.feature_a)
    off_prev_modell = off_fb + len(maps.feature_b)
    off_prev_fa = off_prev_modell + len(maps.modell)
    off_prev_fb = off_prev_fa + len(maps.feature_a)
    off_station = off_prev_fb + len(maps.feature_b)
    off_shift = off_station + len(maps.station)

    for i, s in enumerate(samples):
        set_onehot(X[i], off_modell, maps.modell, s.modell)
        set_onehot(X[i], off_fa, maps.feature_a, s.feature_a)
        set_onehot(X[i], off_fb, maps.feature_b, s.feature_b)

        set_onehot(X[i], off_prev_modell, maps.modell, s.prev_modell)
        set_onehot(X[i], off_prev_fa, maps.feature_a, s.prev_feature_a)
        set_onehot(X[i], off_prev_fb, maps.feature_b, s.prev_feature_b)

        set_onehot(X[i], off_station, maps.station, s.station)

        X[i, off_shift + detect_shift(s.timestamp_event_start)] = 1.0

        off_time = off_shift + N_SHIFTS
        X[i, off_time:off_time + 2 * len(TIME_PERIODS_SECONDS)] = encode_time_features(
            s.timestamp_event_start, TIME_PERIODS_SECONDS,
        )

        y[i] = s.net_process_time

    return X, y


def save(
    X: np.ndarray,
    y: np.ndarray,
    maps: EncodingMaps,
    output_dir: Path,
    *,
    train_ratio: float,
    n_train_vocab_fit: int,
    oov_stats: Dict[str, Dict[str, float]],
) -> None:
    layout, offset = build_feature_layout([
        ("modell", maps.modell),
        ("feature_a", maps.feature_a),
        ("feature_b", maps.feature_b),
        ("prev_modell", maps.modell),
        ("prev_feature_a", maps.feature_a),
        ("prev_feature_b", maps.feature_b),
        ("station", maps.station),
    ])

    # Append shift + time-encoding entries manually (not part of build_feature_layout)
    layout.append({"name": "shift", "offset": offset, "size": N_SHIFTS})
    offset += N_SHIFTS

    for j, period in enumerate(TIME_PERIODS_SECONDS):
        layout.append({"name": f"sin_time_{j}", "offset": offset,     "size": 1})
        layout.append({"name": f"cos_time_{j}", "offset": offset + 1, "size": 1})
        offset += 2

    save_prepared_data(X, y, {
        "encoding_maps": maps.to_dict(),
        "feature_layout": layout,
        "n_total": len(y),
        "none_token": NONE_TOKEN,
        "time_periods_seconds": TIME_PERIODS_SECONDS,
        "vocab_fit_scope": "train_only",
        "vocab_fit_train_ratio": train_ratio,
        "vocab_fit_n_train": n_train_vocab_fit,
        "oov_stats": oov_stats,
    }, output_dir)
    print(f"  Feature dim: {maps.feature_dim}")


def print_statistics(
    samples: List[TrainingSample],
    maps: EncodingMaps,
) -> None:
    print(f"\n{'─' * 60}")
    print("DATASET STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Samples total: {len(samples):,}")

    station_stats = defaultdict(list)
    for s in samples:
        station_stats[s.station].append(s.net_process_time)

    print(f"\n  {'Station':<10} {'Count':>8} {'Mean PT':>10} {'Std PT':>10} {'Min':>8} {'Max':>8}")
    print(f"  {'─' * 58}")
    for station in sorted(station_stats.keys()):
        times = np.array(station_stats[station])
        print(f"  {station:<10} {len(times):>8} {np.mean(times):>10.2f} "
              f"{np.std(times):>10.2f} {np.min(times):>8.2f} {np.max(times):>8.2f}")

    model_stats = defaultdict(list)
    for s in samples:
        model_stats[s.modell].append(s.net_process_time)

    print(f"\n  {'Model':<10} {'Count':>8} {'Mean PT':>10} {'Std PT':>10}")
    print(f"  {'─' * 40}")
    for model in sorted(model_stats.keys()):
        times = np.array(model_stats[model])
        print(f"  {model:<10} {len(times):>8} {np.mean(times):>10.2f} {np.std(times):>10.2f}")

    none_count = sum(1 for s in samples if s.prev_modell == NONE_TOKEN)
    print(f"\n  Samples without predecessor (<NONE>): {none_count} ({none_count/len(samples)*100:.1f}%)")

    print("\n  Feature groups:")
    print(f"    modell:    {list(maps.modell.keys())}")
    print(f"    feature_a: {list(maps.feature_a.keys())}")
    print(f"    feature_b: {list(maps.feature_b.keys())}")
    print(f"    station:   {list(maps.station.keys())}")
    print(f"    Total dim: {maps.feature_dim}")


def prepare_training_data(
    events_paths: Sequence[Path],
    orders_paths: Sequence[Path],
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, EncodingMaps]:
    print("=" * 60)
    print("TRAINING-DATA PREPARATION")
    print("=" * 60)

    print("\n1. Loading CSVs...")
    print(f"   Events: {[p.name for p in events_paths]}")
    print(f"   Orders: {[p.name for p in orders_paths]}")
    events = load_events(events_paths)
    orders = load_orders(orders_paths)
    print(f"   → {len(events):,} events, {len(orders):,} orders loaded")

    print("\n2. Filtering & joining...")
    joined = filter_and_join(events, orders)

    print("\n3. Determining predecessor features...")
    samples = build_training_samples(joined)
    print(f"   → {len(samples):,} training samples generated")

    print("\n4. One-hot encoding (vocabulary fit on TRAIN-only)...")
    n_train = chronological_train_end_idx(len(samples), train_ratio=train_ratio)
    print(f"   → Train slice: {n_train:,} / {len(samples):,} "
          f"(train_ratio={train_ratio:.2f})")
    maps = build_encoding_maps(samples[:n_train])
    X, y = encode_samples(samples, maps)

    train_vocabs = {
        "modell": set(maps.modell), "feature_a": set(maps.feature_a),
        "feature_b": set(maps.feature_b), "station": set(maps.station),
        "prev_modell": set(maps.modell), "prev_feature_a": set(maps.feature_a),
        "prev_feature_b": set(maps.feature_b),
    }
    extractors = {
        "modell": lambda s: s.modell, "feature_a": lambda s: s.feature_a,
        "feature_b": lambda s: s.feature_b, "station": lambda s: s.station,
        "prev_modell": lambda s: s.prev_modell,
        "prev_feature_a": lambda s: s.prev_feature_a,
        "prev_feature_b": lambda s: s.prev_feature_b,
    }
    oov_stats = summarize_oov(
        count_oov(samples, n_train, extractors, train_vocabs), len(samples) - n_train,
    )
    for name, st in oov_stats.items():
        if st["count"]:
            print(f"   → OOV ({name}, val+test): count={st['count']} "
                  f"ratio={st['ratio']*100:.2f}%")

    print_statistics(samples, maps)

    print("\n5. Saving...")
    save(X, y, maps, output_dir, train_ratio=train_ratio,
         n_train_vocab_fit=n_train, oov_stats=oov_stats)

    return X, y, maps


def main():
    parser = argparse.ArgumentParser(
        description="Prepares training data for NN process-time prediction.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-dir",
        type=Path,
        help="Directory with simulation CSVs (auto-discovers *_events_*.csv / *_orders_*.csv)",
    )
    group.add_argument(
        "--events",
        type=Path, nargs="+",
        help="Events CSV files (explicit)",
    )
    parser.add_argument(
        "--orders",
        type=Path, nargs="+",
        help="Orders CSV files (explicit, only with --events)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/training"),
        help="Output directory (default: data/training)",
    )
    args = parser.parse_args()

    if args.data_dir:
        events_paths, orders_paths = find_csv_files(args.data_dir)
        if not events_paths:
            parser.error(f"No *_events_*.csv found in {args.data_dir}.")
        if not orders_paths:
            parser.error(f"No *_orders_*.csv found in {args.data_dir}.")
    else:
        events_paths = args.events
        orders_paths = args.orders
        if not orders_paths:
            parser.error("--orders is required when --events is used")

    prepare_training_data(
        events_paths=events_paths,
        orders_paths=orders_paths,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()