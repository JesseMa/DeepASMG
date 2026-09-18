"""
Training-data preparation for the transition surrogate.

Teacher forcing: training fills the per-station slot buffer with the TRUE
to_station, while online inference fills it autoregressively with its own
prediction. Under shadow inference GroundSim drives, so true targets feed the
buffer there as well.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.dynamics.foundation_dynamics import NONE_TOKEN, END_TOKEN, set_onehot
from src.fitting.deep_data_preparation.data_io import (
    find_csv_files,
    load_events, load_orders, build_feature_layout, save_prepared_data,
)
from src.fitting.deep_data_preparation.split_helpers import (
    DEFAULT_TRAIN_RATIO, chronological_train_end_idx, count_oov, summarize_oov,
)


# K history slots per from_station, calibrated empirically on counter patterns
# (e.g. InputBuffer cycle). Changing it requires retraining all DeepTransition
# models (metadata incompatibility).
MAX_HIST_SLOTS = 10


# Arrivals at the same station by the same order, capped so the vocabulary
# stays finite; VISIT_CAP means "this many or more". Derived from the log, not
# from the generator: it is the count of prior rows with the same order_id and
# station. A station whose transition table carries a visit-indexed row routes
# differently on a repeat visit, and this is what lets a surrogate see it.
VISIT_CAP = 4


def visit_token(n: int) -> str:
    """Categorical token for the n-th arrival (1-based), capped at VISIT_CAP."""
    return str(min(max(n, 1), VISIT_CAP))


@dataclass
class TransitionSample:
    modell: str
    feature_a: str
    feature_b: str

    from_station: str

    # Arrival index of this order at from_station (see visit_token).
    visit: str

    # K-slot history per from_station; slot 0 = most recent, K-1 = oldest.
    hist_modells: Tuple[str, ...]
    hist_targets: Tuple[str, ...]

    to_station: str  # station ID or "End"


@dataclass
class TransitionEncodingMaps:
    """Categorical value → index mappings."""
    modell: Dict[str, int] = field(default_factory=dict)
    feature_a: Dict[str, int] = field(default_factory=dict)
    feature_b: Dict[str, int] = field(default_factory=dict)
    from_station: Dict[str, int] = field(default_factory=dict)
    visit: Dict[str, int] = field(default_factory=dict)
    # Slot-target vocabulary: NONE_TOKEN + the targets seen in the train slice.
    # Separate from to_station, which has no NONE and is fit on the full dataset.
    slot_target: Dict[str, int] = field(default_factory=dict)

    to_station: Dict[str, int] = field(default_factory=dict)

    n_hist_slots: int = MAX_HIST_SLOTS

    @property
    def feature_dim(self) -> int:
        per_slot = len(self.modell) + len(self.slot_target)
        return (
            len(self.modell)
            + len(self.feature_a)
            + len(self.feature_b)
            + len(self.from_station)
            + len(self.visit)
            + self.n_hist_slots * per_slot
        )

    @property
    def num_classes(self) -> int:
        """Number of possible target stations (incl. End)."""
        return len(self.to_station)

    def to_dict(self) -> dict:
        return {
            "modell": self.modell,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "from_station": self.from_station,
            "visit": self.visit,
            "slot_target": self.slot_target,
            "to_station": self.to_station,
            "n_hist_slots": self.n_hist_slots,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }


def reconstruct_transitions(
    events,
    order_features: Dict[str, Dict[str, str]],
    order_completions: Dict[str, Optional[float]],
) -> List[Tuple[str, Dict[str, str], str, str]]:
    """Reconstruct transitions from the event sequence.

    Per order, the chronological station sequence is derived; each pair
    (station_i → station_{i+1}) becomes one transition, and the final step of a
    completed order maps to 'End'.

    The list is sorted globally by completion time (timestamp_event_start +
    time_processing of the source station) — the instant the engine calls
    predict() at PROCESS_COMPLETE. At stations with capacity > 1 (notably M5,
    capacity=3) orders can reorder relative to their start order, so the slot
    history must follow completion order (train/inference parity). This also
    enables a true temporal cut in chronological_train_end_idx.
    """
    productive = [
        e for e in events
        if not e.is_breakdown
        and e.order_id != "BREAKDOWN"
        and e.station_type != "overflow"
        and e.order_id in order_features
    ]

    events_by_order: Dict[str, list] = defaultdict(list)
    for e in productive:
        events_by_order[e.order_id].append(e)

    for oid in events_by_order:
        events_by_order[oid].sort(key=lambda e: e.timestamp_event_start)

    timed_transitions: List[Tuple[float, Tuple[str, Dict[str, str], str, str]]] = []
    skipped_incomplete = 0

    for oid, order_events in events_by_order.items():
        features = order_features[oid]
        is_completed = order_completions.get(oid) is not None

        for i in range(len(order_events) - 1):
            ts = float(order_events[i].timestamp_event_start + order_events[i].time_processing)
            from_station = order_events[i].station
            to_station = order_events[i + 1].station
            timed_transitions.append((ts, (oid, features, from_station, to_station)))

        if is_completed and order_events:
            ts = float(order_events[-1].timestamp_event_start + order_events[-1].time_processing)
            last_station = order_events[-1].station
            timed_transitions.append((ts, (oid, features, last_station, END_TOKEN)))
        elif not is_completed:
            skipped_incomplete += 1

    timed_transitions.sort(key=lambda tt: tt[0])
    transitions: List[Tuple[str, Dict[str, str], str, str]] = [t[1] for t in timed_transitions]

    print(f"  Productive events: {len(productive):,}")
    print(f"  Orders with events: {len(events_by_order):,}")
    print(f"  Transitions reconstructed: {len(transitions):,}")
    print(f"  Incomplete orders (no End): {skipped_incomplete}")

    return transitions


def build_transition_samples(
    transitions: List[Tuple[str, Dict[str, str], str, str]],
    n_hist_slots: int = MAX_HIST_SLOTS,
) -> List[TransitionSample]:
    """Build training samples with a K-slot history per from_station.

    Slot 0 = most recent, NONE-padded before availability. Teacher forcing:
    after each sample the buffer is updated with the TRUE to_station.
    """
    slot_buffers: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
        lambda: deque(
            [(NONE_TOKEN, NONE_TOKEN)] * n_hist_slots,
            maxlen=n_hist_slots,
        )
    )

    visit_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    samples = []
    for oid, features, from_station, to_station in transitions:
        buf = slot_buffers[from_station]
        visit_counts[(oid, from_station)] += 1

        hist_modells = tuple(m for m, _ in buf)
        hist_targets = tuple(t for _, t in buf)

        modell = features.get("modell", NONE_TOKEN)
        samples.append(TransitionSample(
            modell=modell,
            feature_a=features.get("feature_a", NONE_TOKEN),
            feature_b=features.get("feature_b", NONE_TOKEN),
            from_station=from_station,
            visit=visit_token(visit_counts[(oid, from_station)]),
            hist_modells=hist_modells,
            hist_targets=hist_targets,
            to_station=to_station,
        ))

        buf.appendleft((modell, to_station))

    return samples


def build_encoding_maps(
    train_samples: List[TransitionSample],
    all_samples: List[TransitionSample],
    n_hist_slots: int = MAX_HIST_SLOTS,
) -> TransitionEncodingMaps:
    """Build encoding maps.

    Feature vocabularies are fitted on the TRAIN slice only; values first
    seen in val/test yield all-zero one-hot blocks (OOV). The slot_target
    vocabulary comes from the train slot targets plus NONE_TOKEN (padding). The
    target vocabulary (to_station) is built from the FULL dataset because label
    encoding for y has no all-zero path.
    """
    modell_train_vals = (
        {s.modell for s in train_samples}
        | {m for s in train_samples for m in s.hist_modells}
    )
    modell_train_vals.discard(NONE_TOKEN)
    modell_vals = [NONE_TOKEN] + sorted(modell_train_vals)

    fa_vals = sorted({s.feature_a for s in train_samples})
    fb_vals = sorted({s.feature_b for s in train_samples})
    from_vals = sorted({s.from_station for s in train_samples})
    visit_vals = [visit_token(i) for i in range(1, VISIT_CAP + 1)]

    slot_target_train_vals = {t for s in train_samples for t in s.hist_targets}
    slot_target_train_vals.discard(NONE_TOKEN)
    slot_target_vals = [NONE_TOKEN] + sorted(slot_target_train_vals)

    to_vals = sorted({s.to_station for s in all_samples})

    return TransitionEncodingMaps(
        modell={v: i for i, v in enumerate(modell_vals)},
        feature_a={v: i for i, v in enumerate(fa_vals)},
        feature_b={v: i for i, v in enumerate(fb_vals)},
        from_station={v: i for i, v in enumerate(from_vals)},
        visit={v: i for i, v in enumerate(visit_vals)},
        slot_target={v: i for i, v in enumerate(slot_target_vals)},
        to_station={v: i for i, v in enumerate(to_vals)},
        n_hist_slots=n_hist_slots,
    )


def encode_samples(
    samples: List[TransitionSample],
    maps: TransitionEncodingMaps,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-hot feature matrix and integer label vector.

    Feature-vector layout (cumulative offsets):
        [modell | feature_a | feature_b | from_station |
         hist_modell_0 | hist_target_0 | ... | hist_modell_{K-1} | hist_target_{K-1}]
    """
    n = len(samples)
    dim = maps.feature_dim
    K = maps.n_hist_slots
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    n_mod = len(maps.modell)
    n_slot_tgt = len(maps.slot_target)

    off_modell = 0
    off_fa = off_modell + n_mod
    off_fb = off_fa + len(maps.feature_a)
    off_from = off_fb + len(maps.feature_b)
    off_visit = off_from + len(maps.from_station)
    off_hist_start = off_visit + len(maps.visit)

    def _slot_offsets(k: int) -> Tuple[int, int]:
        base = off_hist_start + k * (n_mod + n_slot_tgt)
        return base, base + n_mod

    for i, s in enumerate(samples):
        set_onehot(X[i], off_modell, maps.modell, s.modell)
        set_onehot(X[i], off_fa, maps.feature_a, s.feature_a)
        set_onehot(X[i], off_fb, maps.feature_b, s.feature_b)
        set_onehot(X[i], off_from, maps.from_station, s.from_station)
        set_onehot(X[i], off_visit, maps.visit, s.visit)
        for k in range(K):
            off_m, off_t = _slot_offsets(k)
            set_onehot(X[i], off_m, maps.modell, s.hist_modells[k])
            set_onehot(X[i], off_t, maps.slot_target, s.hist_targets[k])
        y[i] = maps.to_station[s.to_station]

    return X, y


def save(
    X: np.ndarray,
    y: np.ndarray,
    maps: TransitionEncodingMaps,
    output_dir: Path,
    *,
    train_ratio: float,
    n_train_vocab_fit: int,
    oov_stats: Dict[str, Dict[str, float]],
) -> None:
    """Save all samples as data.npz + metadata.json (no split)."""
    groups: List[Tuple[str, Dict[str, int]]] = [
        ("modell", maps.modell),
        ("feature_a", maps.feature_a),
        ("feature_b", maps.feature_b),
        ("from_station", maps.from_station),
        ("visit", maps.visit),
    ]
    for k in range(maps.n_hist_slots):
        groups.append((f"hist_modell_{k}", maps.modell))
        groups.append((f"hist_target_{k}", maps.slot_target))

    layout, _ = build_feature_layout(groups)

    to_station_inv = {v: k for k, v in maps.to_station.items()}
    save_prepared_data(X, y, {
        "encoding_maps": maps.to_dict(),
        "feature_layout": layout,
        "n_total": len(y),
        "num_classes": maps.num_classes,
        "class_names": [to_station_inv[i] for i in range(maps.num_classes)],
        "none_token": NONE_TOKEN,
        "end_token": END_TOKEN,
        "n_hist_slots": maps.n_hist_slots,
        "vocab_fit_scope": "train_only_features",
        "vocab_fit_train_ratio": train_ratio,
        "vocab_fit_n_train": n_train_vocab_fit,
        "oov_stats": oov_stats,
    }, output_dir)
    print(f"  Feature dim: {maps.feature_dim}")
    print(f"  Classes: {maps.num_classes}")
    print(f"  Slot history K={maps.n_hist_slots}  "
          f"(modell vocab {len(maps.modell)}, slot_target vocab {len(maps.slot_target)})")


def print_statistics(
    samples: List[TransitionSample],
    maps: TransitionEncodingMaps,
) -> None:
    print(f"\n{'─' * 60}")
    print("DATASET STATISTICS")
    print(f"{'─' * 60}")
    print(f"  Samples total: {len(samples):,}")

    transition_counts = defaultdict(lambda: defaultdict(int))
    for s in samples:
        transition_counts[s.from_station][s.to_station] += 1

    print("\n  TRANSITION MATRIX:")
    all_to = sorted({s.to_station for s in samples})
    header = f"  {'From':<10}" + "".join(f"{t:>10}" for t in all_to) + f"{'Total':>10}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for from_s in sorted(transition_counts.keys()):
        row = f"  {from_s:<10}"
        total = 0
        for to_s in all_to:
            count = transition_counts[from_s].get(to_s, 0)
            total += count
            if count > 0:
                row += f"{count:>10}"
            else:
                row += f"{'·':>10}"
        row += f"{total:>10}"
        print(row)

    slot0_none = sum(1 for s in samples if s.hist_modells[0] == NONE_TOKEN)
    print(f"\n  Samples with empty slot 0 (sequence start per from_station): "
          f"{slot0_none} ({slot0_none/len(samples)*100:.2f}%)")

    print("\n  Feature groups:")
    print(f"    modell:       {list(maps.modell.keys())}")
    print(f"    feature_a:    {list(maps.feature_a.keys())}")
    print(f"    feature_b:    {list(maps.feature_b.keys())}")
    print(f"    from_station: {list(maps.from_station.keys())}")
    print(f"    slot_target:  {list(maps.slot_target.keys())}")
    print(f"    K (slots):    {maps.n_hist_slots}")
    print(f"    Total dim:    {maps.feature_dim}")
    print(f"\n  Target classes: {list(maps.to_station.keys())}")


def prepare_transition_data(
    events_paths: Sequence[Path],
    orders_paths: Sequence[Path],
    output_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_hist_slots: int = MAX_HIST_SLOTS,
) -> Tuple[np.ndarray, np.ndarray, TransitionEncodingMaps]:
    """Full pipeline: CSVs → training-ready arrays.

    Feature vocabulary from the first train_ratio fraction only; target
    classes from the full dataset (topology). Returns (X, y, encoding_maps).
    """
    print("=" * 60)
    print(f"TRANSITION TRAINING-DATA PREPARATION (K={n_hist_slots})")
    print("=" * 60)

    print("\n1. Loading CSVs...")
    events = load_events(events_paths)
    order_features, order_completions = load_orders(orders_paths, with_completions=True)
    print(f"   → {len(events):,} events, {len(order_features):,} orders loaded")

    print("\n2. Reconstructing transitions...")
    transitions = reconstruct_transitions(events, order_features, order_completions)

    print("\n3. Building K-slot history per from_station (teacher-forced)...")
    samples = build_transition_samples(transitions, n_hist_slots=n_hist_slots)
    print(f"   → {len(samples):,} training samples generated")

    print("\n4. Encoding (feature vocabulary fit on TRAIN-only)...")
    n_train = chronological_train_end_idx(len(samples), train_ratio=train_ratio)
    train_samples = samples[:n_train]
    print(f"   → Train slice: {n_train:,} / {len(samples):,} "
          f"(train_ratio={train_ratio:.2f})")
    maps = build_encoding_maps(train_samples, samples, n_hist_slots=n_hist_slots)
    X, y = encode_samples(samples, maps)

    # OOV reporting: current features plus slot-aggregated (avoids K*2 separate reports).
    train_vocabs = {
        "modell":       set(maps.modell.keys()),
        "feature_a":    set(maps.feature_a.keys()),
        "feature_b":    set(maps.feature_b.keys()),
        "from_station": set(maps.from_station.keys()),
        "slot_modell_any": set(maps.modell.keys()),
        "slot_target_any": set(maps.slot_target.keys()),
    }
    extractors = {
        "modell":       lambda s: s.modell,
        "feature_a":    lambda s: s.feature_a,
        "feature_b":    lambda s: s.feature_b,
        "from_station": lambda s: s.from_station,
        # Slot-aggregated: a sample counts as OOV if ANY of its K slots holds an
        # unknown modell/target.
        "slot_modell_any": lambda s: next(
            (m for m in s.hist_modells if m not in maps.modell), s.hist_modells[0]
        ),
        "slot_target_any": lambda s: next(
            (t for t in s.hist_targets if t not in maps.slot_target), s.hist_targets[0]
        ),
    }
    oov_counts = count_oov(samples, n_train, extractors, train_vocabs)
    n_holdout = len(samples) - n_train
    oov_stats = summarize_oov(oov_counts, n_holdout)
    for name, stats in oov_stats.items():
        if stats["count"]:
            print(f"     {name:<18} count={stats['count']:>5}  ratio={stats['ratio']*100:>5.2f}%")

    print_statistics(samples, maps)

    print("\n5. Saving...")
    save(X, y, maps, output_dir,
         train_ratio=train_ratio, n_train_vocab_fit=n_train,
         oov_stats=oov_stats)

    return X, y, maps


def main():
    parser = argparse.ArgumentParser(
        description="Prepares training data for NN transition prediction.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path)
    group.add_argument("--events", type=Path, nargs="+")
    parser.add_argument("--orders", type=Path, nargs="+")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/transition_training"))
    parser.add_argument("--n-hist-slots", type=int, default=MAX_HIST_SLOTS)

    args = parser.parse_args()

    if args.data_dir:
        events_paths, orders_paths = find_csv_files(args.data_dir)
        if not events_paths:
            parser.error(f"No *_events_*.csv in {args.data_dir}")
        if not orders_paths:
            parser.error(f"No *_orders_*.csv in {args.data_dir}")
    else:
        events_paths = args.events
        orders_paths = args.orders
        if not orders_paths:
            parser.error("--orders required with --events")

    prepare_transition_data(
        events_paths=events_paths,
        orders_paths=orders_paths,
        output_dir=args.output,
        n_hist_slots=args.n_hist_slots,
    )


if __name__ == "__main__":
    main()
