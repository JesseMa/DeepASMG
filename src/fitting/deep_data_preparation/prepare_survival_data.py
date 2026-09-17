"""
Training-data preparation for NN-based survival analysis (TTF).

Target (y), two columns:
    duration        → cumulative operating time (Σ net_process_time) per cycle
    event           → 1 = breakdown, 0 = right-censored (cycle open at sim end)
"""

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
class SurvivalSample:
    """One cycle as a survival data point."""
    station: str
    operating_time: float   # cumulative operating time (Σ net_process_time)
    event: bool             # True = breakdown
    n_jobs: int             # productive jobs in this cycle
    wall_clock_time: float  # elapsed wall-clock time
    prev_ttf: float = 0.0           # TTF of the previous cycle (operating seconds)
    prev_n_jobs: float = 0.0        # n_jobs of the previous cycle
    mean_ttf: float = 0.0           # running mean TTF over all previous cycles
    prev_repair_time: float = 0.0   # repair duration of the last failure


@dataclass
class SurvivalEncodingMaps:
    """Categorical value → index mappings."""
    station: Dict[str, int] = field(default_factory=dict)

    @property
    def feature_dim(self) -> int:
        """Station one-hot block size only; the authoritative model input dim
        (station + 4 continuous features) is the top-level feature_dim from save()."""
        return len(self.station)

    def to_dict(self) -> dict:
        return {
            "station": self.station,
            "feature_dim": self.feature_dim,
        }


def extract_survival_samples(
    events: List[RawEvent],
) -> List[SurvivalSample]:
    """Extract survival data points from the event sequence.

    Chronological per machine; duration = cumulative operating time
    (Σ net_process_time). A breakdown closes a cycle as event=1; only cycles
    still open at sim end are censored (event=0). The sample list is sorted
    globally by cycle end (wall clock) — prerequisite for a meaningful temporal
    cut in chronological_train_end_idx.
    """
    machine_events = [e for e in events if e.station_type == "machine"]

    events_by_station: Dict[str, List[RawEvent]] = defaultdict(list)
    for e in machine_events:
        events_by_station[e.station].append(e)

    for station in events_by_station:
        events_by_station[station].sort(key=lambda e: e.timestamp_event_start)

    timed_samples: List[Tuple[float, SurvivalSample]] = []
    total_cycles = 0
    total_events = 0
    total_censored = 0
    total_truncated = 0

    for station, station_events in events_by_station.items():
        cycle_start_wall: float = 0.0
        cycle_operating: float = 0.0
        cycle_jobs: int = 0
        in_cycle: bool = False
        # The first observed cycle per station began BEFORE the warmup cutoff:
        # its operating time is left-truncated, so it must not become a sample
        # (emitting it as event=True biases lifetimes short). It still seeds
        # the history features once it closes.
        first_cycle_truncated: bool = True

        hist_prev_ttf: float = 0.0
        hist_prev_n_jobs: float = 0.0
        hist_mean_ttf: float = 0.0
        hist_n_cycles: int = 0
        hist_prev_repair: float = 0.0

        for event in station_events:
            is_downtime = (
                event.is_breakdown
                or event.order_id == "BREAKDOWN"
            )

            if is_downtime:
                if in_cycle and cycle_jobs > 0 and cycle_operating > 0:
                    if first_cycle_truncated:
                        total_truncated += 1
                    else:
                        wall_time = event.timestamp_event_start - cycle_start_wall
                        timed_samples.append((
                            float(event.timestamp_event_start),
                            SurvivalSample(
                                station=station,
                                operating_time=cycle_operating,
                                event=True,
                                n_jobs=cycle_jobs,
                                wall_clock_time=max(0.0, wall_time),
                                prev_ttf=hist_prev_ttf,
                                prev_n_jobs=hist_prev_n_jobs,
                                mean_ttf=hist_mean_ttf,
                                prev_repair_time=hist_prev_repair,
                            ),
                        ))
                        total_events += 1
                        total_cycles += 1
                    first_cycle_truncated = False

                    hist_mean_ttf = (hist_mean_ttf * hist_n_cycles + cycle_operating) / (hist_n_cycles + 1)
                    hist_n_cycles += 1
                    hist_prev_ttf = cycle_operating
                    hist_prev_n_jobs = float(cycle_jobs)
                    hist_prev_repair = event.repair_time

                # Start new cycle (after downtime + repair)
                cycle_start_wall = event.timestamp_event_start + event.repair_time
                cycle_operating = 0.0
                cycle_jobs = 0
                in_cycle = True
                continue

            if not in_cycle:
                cycle_start_wall = event.timestamp_event_start
                in_cycle = True

            cycle_operating += event.net_process_time
            cycle_jobs += 1

        # Censor the cycle still open at sim end
        if (in_cycle and cycle_jobs > 0 and cycle_operating > 0
                and not first_cycle_truncated):
            last_event = station_events[-1]
            end_time = last_event.timestamp_event_start + last_event.time_processing
            wall_time = end_time - cycle_start_wall
            timed_samples.append((
                float(end_time),
                SurvivalSample(
                    station=station,
                    operating_time=cycle_operating,
                    event=False,
                    n_jobs=cycle_jobs,
                    wall_clock_time=max(0.0, wall_time),
                    prev_ttf=hist_prev_ttf,
                    prev_n_jobs=hist_prev_n_jobs,
                    mean_ttf=hist_mean_ttf,
                    prev_repair_time=hist_prev_repair,
                ),
            ))
            total_censored += 1
            total_cycles += 1

    timed_samples.sort(key=lambda ts: ts[0])
    samples: List[SurvivalSample] = [s for _, s in timed_samples]

    print(f"  Machines: {len(events_by_station)}")
    print(f"  Cycles total: {total_cycles:,}")
    print(f"    Events (downtime): {total_events:,}")
    print(f"    Censored (sim end): {total_censored:,}")
    print(f"    Dropped (left-truncated first cycle): {total_truncated:,}")
    print(f"  → Survival samples: {len(samples):,}")

    return samples


def build_encoding_maps(
    samples: List[SurvivalSample],
) -> SurvivalEncodingMaps:
    all_stations = sorted({s.station for s in samples})
    return SurvivalEncodingMaps(
        station={v: i for i, v in enumerate(all_stations)},
    )


def encode_samples(
    samples: List[SurvivalSample],
    maps: SurvivalEncodingMaps,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature matrix X and target matrix y for survival training.

    X: [station_onehot, prev_ttf, prev_n_jobs, mean_ttf, prev_repair_time]  (n, n_stations+4)
    y: [operating_time, event]  (n, 2)

    Continuous features stay unnormalized — normalization is derived after the
    split from the training set (train_survival.py) to avoid data leakage.
    """
    n = len(samples)
    n_stations = len(maps.station)
    total_dim = n_stations + 4  # station_onehot + prev_ttf + prev_n_jobs + mean_ttf + prev_repair_time
    X = np.zeros((n, total_dim), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)

    for i, s in enumerate(samples):
        idx = maps.station.get(s.station)
        if idx is not None:
            X[i, idx] = 1.0

        X[i, n_stations]     = s.prev_ttf
        X[i, n_stations + 1] = s.prev_n_jobs
        X[i, n_stations + 2] = s.mean_ttf
        X[i, n_stations + 3] = s.prev_repair_time

        # Target (raw, seconds)
        y[i, 0] = s.operating_time
        y[i, 1] = 1.0 if s.event else 0.0

    return X, y


def save(
    X: np.ndarray,
    y: np.ndarray,
    maps: SurvivalEncodingMaps,
    output_dir: Path,
    *,
    train_ratio: float,
    n_train_vocab_fit: int,
    oov_stats: Dict[str, Dict[str, float]],
) -> None:
    """Save all samples as data.npz + metadata.json (no split, no normalization)."""
    n_events = int(np.sum(y[:, 1] == 1))
    n_censored = int(np.sum(y[:, 1] == 0))
    n_stations = len(maps.station)
    total_feature_dim = n_stations + 4

    save_prepared_data(X, y, {
        "encoding_maps": maps.to_dict(),
        "feature_dim": total_feature_dim,
        "target_semantics": "raw_operating_time_seconds",
        "features_normalized_in_prepare": False,
        "feature_layout": [
            {"name": "station",          "offset": 0,              "size": n_stations, "type": "one_hot"},
            {"name": "prev_ttf",         "offset": n_stations,     "size": 1,          "type": "continuous"},
            {"name": "prev_n_jobs",      "offset": n_stations + 1, "size": 1,          "type": "continuous"},
            {"name": "mean_ttf",         "offset": n_stations + 2, "size": 1,          "type": "continuous"},
            {"name": "prev_repair_time", "offset": n_stations + 3, "size": 1,          "type": "continuous"},
        ],
        "n_total": len(y),
        "n_events": n_events,
        "n_censored": n_censored,
        "censoring_rate": float(n_censored / len(y)) if len(y) > 0 else 0.0,
        "vocab_fit_scope": "train_only",
        "vocab_fit_train_ratio": train_ratio,
        "vocab_fit_n_train": n_train_vocab_fit,
        "oov_stats": oov_stats,
    }, output_dir)
    print(f"  Events (downtime): {n_events:,}")
    print(f"  Censored: {n_censored:,}")


def print_statistics(
    samples: List[SurvivalSample],
    maps: SurvivalEncodingMaps,
) -> None:
    print(f"\n{'─' * 60}")
    print("DATASET STATISTICS (survival, operating time)")
    print(f"{'─' * 60}")

    n_total = len(samples)
    n_events = sum(1 for s in samples if s.event)
    n_censored = n_total - n_events

    print(f"\n  Samples total: {n_total:,}")
    print(f"    Events (downtime): {n_events:,} ({n_events / n_total * 100:.2f}%)")
    print(f"    Censored:            {n_censored:,} ({n_censored / n_total * 100:.2f}%)")

    op_times = np.array([s.operating_time for s in samples])
    wall_times = np.array([s.wall_clock_time for s in samples])

    print("\n  Operating time (Σ process_time per cycle):")
    print(f"    Mean: {np.mean(op_times):.1f}s, Median: {np.median(op_times):.1f}s")
    print(f"    Std:  {np.std(op_times):.1f}s")
    print(f"    Min:  {np.min(op_times):.1f}s, Max: {np.max(op_times):.1f}s")

    utilization = op_times / np.maximum(wall_times, 1.0)
    print("\n  Utilization (operating / wall clock):")
    print(f"    Mean: {np.mean(utilization) * 100:.1f}%, Median: {np.median(utilization) * 100:.1f}%")

    if n_events > 0:
        ev_op = np.array([s.operating_time for s in samples if s.event])
        print("\n  Downtimes only:")
        print(f"    Operating-Time: Mean={np.mean(ev_op):.1f}s, Median={np.median(ev_op):.1f}s")

    station_stats: Dict[str, Dict] = defaultdict(
        lambda: {"total": 0, "events": 0, "op_times": [], "wall_times": []}
    )
    for s in samples:
        station_stats[s.station]["total"] += 1
        if s.event:
            station_stats[s.station]["events"] += 1
        station_stats[s.station]["op_times"].append(s.operating_time)
        station_stats[s.station]["wall_times"].append(s.wall_clock_time)

    print(f"\n  {'Station':<12} {'Cycles':>8} {'Events':>7} {'Med OpTime':>12} {'Med Wall':>10} {'Util':>7}")
    print(f"  {'─' * 58}")
    for station in sorted(station_stats.keys()):
        st = station_stats[station]
        med_op = np.median(st["op_times"])
        med_wall = np.median(st["wall_times"])
        util = np.median(np.array(st["op_times"]) / np.maximum(np.array(st["wall_times"]), 1.0))
        print(f"  {station:<12} {st['total']:>8} {st['events']:>7} {med_op:>11.0f}s {med_wall:>9.0f}s {util * 100:>6.1f}%")

    print("\n  Encoding:")
    print(f"    Stations: {list(maps.station.keys())}")
    print(f"    Feature dim: {maps.feature_dim + 4}  ({maps.feature_dim} station + 4 continuous)")


def prepare_survival_data(
    events_paths: Sequence[Path],
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, SurvivalEncodingMaps]:
    """Full pipeline: CSVs → training-ready survival arrays.

    y[:, 0] = raw operating_time (seconds), y[:, 1] = event. Normalization
    happens after the split (train_survival.py), not here.

    The station vocabulary is fitted on the first train_ratio fraction
    only; OOV stations in the val/test slice are reported via oov_stats in
    metadata. Returns (X, y, encoding_maps).
    """
    print("=" * 60)
    print("SURVIVAL-ANALYSIS DATA PREPARATION (operating time)")
    print("=" * 60)

    print("\n1. Loading CSVs...")
    print(f"   Events: {[p.name for p in events_paths]}")
    events = load_events(events_paths)
    print(f"   → {len(events):,} events loaded")

    print("\n2. Extracting cycles (operating time)...")
    samples = extract_survival_samples(events)

    if not samples:
        raise ValueError("No cycles found.")

    print("\n3. Encoding (vocabulary fit on TRAIN-only)...")
    n_train = chronological_train_end_idx(len(samples), train_ratio=train_ratio)
    train_samples = samples[:n_train]
    print(f"   → Train slice: {n_train:,} / {len(samples):,} "
          f"(train_ratio={train_ratio:.2f})")
    maps = build_encoding_maps(train_samples)
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
         train_ratio=train_ratio, n_train_vocab_fit=n_train, oov_stats=oov_stats)

    return X, y, maps


def main():
    parser = argparse.ArgumentParser(
        description="Prepares survival training data (operating-time based).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path)
    group.add_argument("--events", type=Path, nargs="+")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/survival_training"))

    args = parser.parse_args()

    if args.data_dir:
        events_paths = sorted(args.data_dir.rglob("*_events_*.csv"))
        if not events_paths:
            parser.error(f"No *_events_*.csv in {args.data_dir}")
    else:
        events_paths = args.events

    prepare_survival_data(
        events_paths=events_paths,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()