"""
Training-data preparation for the released-order surrogate.

Timestamps are absolute epoch values (generation anchor + timestamp_creation).
Week and month periods are not multiples of a day, so a sim-relative clock
would be phase-shifted by start_timestamp mod period against both the generator
and the deployment query.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.dynamics.foundation_dynamics import (
    NAMED_PERIODS_HOURS, NONE_TOKEN, encode_time_features,
)
from src.fitting.deep_data_preparation.split_helpers import (
    DEFAULT_TRAIN_RATIO,
    chronological_train_end_idx,
    count_oov,
    summarize_oov,
)

NAMED_PERIODS_SECONDS: Dict[str, float] = {
    name: hours * 3600.0 for name, hours in NAMED_PERIODS_HOURS.items()
}

DEFAULT_PERIODS = ["shift", "day", "week", "month"]

META_COLUMNS = {
    "order_id", "run_id", "timestamp_creation", "timestamp_completion",
    "seed", "simulation_days", "warmup_days",
}


@dataclass
class RawOrder:
    timestamp_creation: float
    run_id: str
    features: Dict[str, str]


@dataclass
class ProductSample:
    """Training sample: timestamp + previous order → current order."""
    timestamp: float
    prev_features: Optional[Dict[str, str]]
    curr_features: Dict[str, str]


@dataclass
class ProductEncodingMaps:
    feature_names: List[str] = field(default_factory=list)
    maps: Dict[str, Dict[str, int]] = field(default_factory=dict)
    time_periods: List[str] = field(default_factory=list)

    @property
    def n_time_features(self) -> int:
        """Two values (sin + cos) per period."""
        return len(self.time_periods) * 2

    @property
    def prev_order_dim(self) -> int:
        """Size of the prev_order EWMA block (n_classes per feature, no NONE)."""
        return sum(self.n_classes(f) for f in self.feature_names)

    @property
    def feature_dim(self) -> int:
        return self.n_time_features + self.prev_order_dim

    def n_classes(self, feat_name: str) -> int:
        """Number of real classes (excluding NONE token) for a feature head."""
        return len(self.maps[feat_name]) - 1

    def feature_layout(self) -> List[Dict]:
        """Offset table for the input vector."""
        layout = []
        offset = 0

        for period_name in self.time_periods:
            layout.append({"name": f"sin_{period_name}", "offset": offset, "size": 1})
            offset += 1
            layout.append({"name": f"cos_{period_name}", "offset": offset, "size": 1})
            offset += 1

        for feat_name in self.feature_names:
            size = self.n_classes(feat_name)
            layout.append({"name": f"prev_{feat_name}", "offset": offset, "size": size})
            offset += size

        return layout

    def head_layout(self) -> List[Dict]:
        """Offset table for the output vector (no NONE)."""
        layout = []
        offset = 0
        for feat_name in self.feature_names:
            size = self.n_classes(feat_name)
            layout.append({"feature_name": feat_name, "offset": offset, "size": size})
            offset += size
        return layout

    def to_dict(self, ewma_alpha: float = 0.3) -> dict:
        head_layout = self.head_layout()
        # conditioning_dim = number of classes of the first feature (modell)
        conditioning_dim = head_layout[0]["size"] if head_layout else 0
        return {
            "feature_names": self.feature_names,
            "encoding_maps": self.maps,
            "feature_dim": self.feature_dim,
            "n_time_features": self.n_time_features,
            "prev_order_dim": self.prev_order_dim,
            "time_periods": self.time_periods,
            "time_periods_seconds": [
                NAMED_PERIODS_SECONDS[p] for p in self.time_periods
            ],
            "ewma_alpha": ewma_alpha,
            "feature_layout": self.feature_layout(),
            "head_layout": head_layout,
            "conditioning_dim": conditioning_dim,
        }


def _generation_anchor(path: Path) -> float:
    """Absolute generation start of the run this order log belongs to.

    Fail-fast: without the anchor the week/month phases cannot be aligned
    with the generator and deployment clocks.
    """
    import json as _json
    for candidate in (path.parent / "run_metadata.json",
                      path.parent.parent / "run_metadata.json"):
        if candidate.exists():
            return float(_json.loads(candidate.read_text())["start_timestamp"])
    raise FileNotFoundError(
        f"run_metadata.json not found next to {path}. Regenerate the training "
        "log with scripts/generate_training_data.py, which records the "
        "absolute generation anchor required for phase-true time encoding."
    )


def load_orders(paths: Sequence[Path]) -> List[RawOrder]:
    """Load orders from order-log CSVs; feature columns are auto-detected.

    timestamp_creation is shifted to the absolute generation clock so that
    encode_time sees the same phases the generator modulated on.
    """
    orders = []

    for path in paths:
        file_run_id = path.stem
        anchor = _generation_anchor(path)

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            feature_cols = [
                col for col in fieldnames
                if col.lower() not in META_COLUMNS
            ]

            for row in reader:
                completion = row.get("timestamp_completion", "")
                if not completion or completion in ("None", "nan", ""):
                    continue

                features = {
                    col: row[col]
                    for col in feature_cols
                    if row.get(col) and row[col] not in ("None", "nan", "")
                }
                if not features:
                    continue

                orders.append(RawOrder(
                    timestamp_creation=anchor + float(row.get("timestamp_creation", 0.0)),
                    run_id=row.get("run_id", file_run_id),
                    features=features,
                ))

    return orders


def extract_sequence_samples(
    orders: List[RawOrder],
    feature_names: List[str],
) -> List[ProductSample]:
    """Extract (prev → curr) sequence pairs with timestamps; run boundaries preserved (no cross-run pairs)."""
    by_run: Dict[str, List[RawOrder]] = defaultdict(list)
    for order in orders:
        by_run[order.run_id].append(order)

    samples: List[ProductSample] = []
    skipped_incomplete = 0

    for run_id in sorted(by_run.keys()):
        run_orders = sorted(by_run[run_id], key=lambda o: o.timestamp_creation)
        prev_features: Optional[Dict[str, str]] = None

        for order in run_orders:
            if not all(f in order.features for f in feature_names):
                skipped_incomplete += 1
                prev_features = None
                continue

            samples.append(ProductSample(
                timestamp=order.timestamp_creation,
                prev_features=prev_features,
                curr_features={f: order.features[f] for f in feature_names},
            ))
            prev_features = order.features

    if skipped_incomplete > 0:
        print(f"  Skipped (incomplete features): {skipped_incomplete:,}")

    return samples


def build_encoding_maps(
    samples: List[ProductSample],
    feature_names: List[str],
    time_periods: List[str],
) -> ProductEncodingMaps:
    """NONE token always index 0; real values from index 1 (lexicographic)."""
    maps = ProductEncodingMaps(
        feature_names=feature_names,
        time_periods=time_periods,
    )
    for feat_name in feature_names:
        values = sorted({s.curr_features[feat_name] for s in samples})
        enc = {NONE_TOKEN: 0}
        for i, v in enumerate(values, start=1):
            enc[v] = i
        maps.maps[feat_name] = enc

    return maps


def encode_time(timestamp: float, time_periods: List[str]) -> np.ndarray:
    """Encode a timestamp as a sin/cos vector, one (sin, cos) pair per period."""
    return encode_time_features(
        timestamp, [NAMED_PERIODS_SECONDS[p] for p in time_periods],
    )


def encode_samples(
    samples: List[ProductSample],
    maps: ProductEncodingMaps,
    ewma_alpha: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode samples into numeric arrays with EWMA frequency context.

    X: [sin/cos time encodings | prev_order_ewma], shape (n, feature_dim).
       prev_order_ewma: smoothed frequency vector of recent orders.
           - run start: uniform [1/n_classes, ...]
           - update after order v: freq = (1-alpha)*freq; freq[v] += alpha  (Σ freq = 1)
    y: class indices (0-based), shape (n, n_features).

    Samples must be in run order; run boundaries are detected via
    sample.prev_features == None (EWMA reset).
    """
    n = len(samples)
    n_features = len(maps.feature_names)

    X = np.zeros((n, maps.feature_dim), dtype=np.float32)
    y = np.zeros((n, n_features), dtype=np.int64)

    n_classes_map = {f: maps.n_classes(f) for f in maps.feature_names}
    prev_offsets = {
        entry["name"][5:]: entry["offset"]
        for entry in maps.feature_layout()
        if entry["name"].startswith("prev_")
    }

    ewma_state: Dict[str, np.ndarray] = {}

    def _reset_ewma() -> None:
        for f in maps.feature_names:
            nc = n_classes_map[f]
            ewma_state[f] = np.full(nc, 1.0 / nc, dtype=np.float32)

    def _update_ewma(curr_features: Dict[str, str]) -> None:
        for f, val in curr_features.items():
            if f not in ewma_state:
                continue
            enc = maps.maps[f]
            val_idx = enc.get(val, 0) - 1  # NONE → 0 (idx -1 = invalid)
            if val_idx < 0:
                continue
            freq = ewma_state[f]
            freq *= (1.0 - ewma_alpha)
            freq[val_idx] += ewma_alpha

    _reset_ewma()

    for i, sample in enumerate(samples):

        if sample.prev_features is None:
            _reset_ewma()

        X[i, :maps.n_time_features] = encode_time(
            sample.timestamp, maps.time_periods
        )

        for feat_name in maps.feature_names:
            offset = prev_offsets[feat_name]
            nc = n_classes_map[feat_name]
            X[i, offset:offset + nc] = ewma_state[feat_name]

        for j, feat_name in enumerate(maps.feature_names):
            enc = maps.maps[feat_name]
            raw_idx = enc[sample.curr_features[feat_name]]  # 1-based
            y[i, j] = raw_idx - 1                           # 0-based for CrossEntropy

        # Update EWMA with the current order AFTER encoding the sample
        _update_ewma(sample.curr_features)

    return X, y


def save(
    X: np.ndarray,
    y: np.ndarray,
    maps: ProductEncodingMaps,
    output_dir: Path,
    ewma_alpha: float = 0.3,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_train_vocab_fit: Optional[int] = None,
    oov_stats: Optional[Dict[str, Dict[str, float]]] = None,
    vocab_fit_scope: str = "full_dataset",
) -> None:
    """Save all samples as data.npz + metadata.json (no split)."""
    from src.fitting.deep_data_preparation.data_io import save_prepared_data

    metadata = maps.to_dict(ewma_alpha=ewma_alpha)
    metadata["n_total"] = len(y)
    metadata["vocab_fit_scope"] = vocab_fit_scope
    metadata["vocab_fit_train_ratio"] = float(train_ratio)
    if n_train_vocab_fit is not None:
        metadata["vocab_fit_n_train"] = int(n_train_vocab_fit)
    if oov_stats is not None:
        metadata["oov_stats"] = oov_stats

    save_prepared_data(X, y, metadata, output_dir)
    print(f"  Input dim total:    {maps.feature_dim}")
    print(f"    Time encodings:   {maps.n_time_features}  "
          f"({len(maps.time_periods)} periods × 2)")
    print(f"    prev_order EWMA:   {maps.prev_order_dim}  (alpha={ewma_alpha})")
    print(f"  Periods: {maps.time_periods}")
    for feat_name in maps.feature_names:
        print(f"    {feat_name:<14} → {maps.n_classes(feat_name)} classes")


def print_statistics(
    samples: List[ProductSample],
    maps: ProductEncodingMaps,
) -> None:
    print(f"\n{'─' * 60}")
    print("DATASET STATISTICS (product sequences with time encoding)")
    print(f"{'─' * 60}")
    print(f"\n  Samples total: {len(samples):,}")

    n_none = sum(1 for s in samples if s.prev_features is None)
    print(f"  Of which with NONE predecessor (run starts): {n_none:,}")

    timestamps = [s.timestamp for s in samples]
    t_min, t_max = min(timestamps), max(timestamps)
    span_days = (t_max - t_min) / 86400
    print(f"\n  Time range: {t_min:.0f}s – {t_max:.0f}s  ({span_days:.1f} days)")

    for feat_name in maps.feature_names:
        enc = maps.maps[feat_name]
        values = [v for v in enc if v != NONE_TOKEN]
        counts = defaultdict(int)
        for s in samples:
            counts[s.curr_features[feat_name]] += 1

        print(f"\n  {feat_name}:")
        for v in sorted(values):
            pct = counts[v] / len(samples) * 100
            print(f"    {v:<12} {counts[v]:>7,}  ({pct:5.1f}%)")

    if "week" in maps.time_periods and maps.feature_names:
        feat_name = maps.feature_names[0]
        enc = maps.maps[feat_name]
        values = [v for v in enc if v != NONE_TOKEN]
        period_s = NAMED_PERIODS_SECONDS["week"]
        n_bins = 7

        print(f"\n  Temporal distribution '{feat_name}' over the week "
              f"(7 bins, epoch-week aligned, normalized per value):")

        bin_counts: Dict[str, List[int]] = {v: [0] * n_bins for v in values}
        for s in samples:
            bin_idx = min(
                int((s.timestamp % period_s) / period_s * n_bins),
                n_bins - 1,
            )
            bin_counts[s.curr_features[feat_name]][bin_idx] += 1

        day_labels = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"]
        print(f"  {'':12}" + "".join(f"  {d:>5}" for d in day_labels))
        for v in sorted(values):
            total = sum(bin_counts[v])
            if total == 0:
                continue
            row = f"  {v:<12}"
            for cnt in bin_counts[v]:
                row += f"  {cnt/total*100:4.0f}%"
            print(row)

    modell_enc = maps.maps.get("modell")
    if modell_enc:
        modell_values = [v for v in modell_enc if v != NONE_TOKEN]
        trans: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for s in samples:
            if s.prev_features is not None:
                prev_m = s.prev_features.get("modell", NONE_TOKEN)
                curr_m = s.curr_features.get("modell", NONE_TOKEN)
                if prev_m != NONE_TOKEN:
                    trans[prev_m][curr_m] += 1

        if trans:
            print("\n  Model transitions (prev → curr, %):")
            print(f"  {'→':<10}" + "".join(f"  {v:>8}" for v in modell_values))
            for prev_m in sorted(trans.keys()):
                row_total = sum(trans[prev_m].values())
                if row_total == 0:
                    continue
                row = f"  {prev_m:<10}"
                for curr_m in modell_values:
                    pct = trans[prev_m][curr_m] / row_total * 100
                    row += f"  {pct:6.1f}%  "
                print(row)


def prepare_product_data(
    orders_paths: Sequence[Path],
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    time_periods: Optional[List[str]] = None,
    ewma_alpha: float = 0.3,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, ProductEncodingMaps]:
    """Full pipeline: order-log CSVs → training-ready arrays.

    feature_names fixes the head order and must start with 'modell'; ewma_alpha
    must lie in (0, 1] and is persisted to metadata.json for inference.
    """
    print("=" * 60)
    print("PRODUCT DATA PREPARATION (sequences + time encoding)")
    print("=" * 60)

    if time_periods is None:
        time_periods = DEFAULT_PERIODS

    unknown = [p for p in time_periods if p not in NAMED_PERIODS_SECONDS]
    if unknown:
        raise ValueError(
            f"Unknown periods: {unknown}. "
            f"Allowed: {list(NAMED_PERIODS_SECONDS.keys())}"
        )

    print("\n1. Loading order-log CSVs...")
    print(f"   Files: {[p.name for p in orders_paths]}")
    orders = load_orders(orders_paths)
    print(f"   → {len(orders):,} completed orders loaded")

    if not orders:
        raise ValueError("No completed orders found.")

    if feature_names is None:
        all_feat_keys: set = set()
        for o in orders:
            all_feat_keys.update(o.features.keys())
        feature_names = sorted(all_feat_keys)
        if "modell" in feature_names:
            feature_names.remove("modell")
            feature_names.insert(0, "modell")

    print(f"   Feature names: {feature_names}")
    print(f"   Periods:      {time_periods}")

    print("\n2. Extracting sequence pairs...")
    samples = extract_sequence_samples(orders, feature_names)
    print(f"   → {len(samples):,} training pairs")

    if not samples:
        raise ValueError(
            "No sequence pairs found. "
            "Check that the order-log CSVs contain feature columns."
        )

    # In DeepProduct the encoder (EWMA block) and decoder (classification
    # heads) share one vocabulary — modell/feature_a/feature_b are context AND
    # target classes. A train-only fit would deny the heads val/test targets
    # (KeyError). Hence full-dataset vocabulary; OOV diagnostics against a
    # hypothetical train-only vocabulary go to metadata (oov_stats).
    print("\n3. Building encoding maps + time encoding + EWMA...")
    maps = build_encoding_maps(samples, feature_names, time_periods)
    X, y = encode_samples(samples, maps, ewma_alpha=ewma_alpha)
    print(f"   Input dim: {maps.feature_dim}  "
          f"({maps.n_time_features} time + {maps.prev_order_dim} prev_order EWMA, alpha={ewma_alpha})")

    n_train = chronological_train_end_idx(len(samples), train_ratio=train_ratio)
    train_vocabs: Dict[str, Set[str]] = {
        f: {s.curr_features[f] for s in samples[:n_train]}
        for f in feature_names
    }
    extractors = {
        f: (lambda s, fn=f: s.curr_features[fn]) for f in feature_names
    }
    oov_counts = count_oov(samples, n_train, extractors, train_vocabs)
    n_holdout = len(samples) - n_train
    oov_stats = summarize_oov(oov_counts, n_holdout)
    if n_holdout > 0:
        max_ratio = max((v["ratio"] for v in oov_stats.values()), default=0.0)
        print(f"   OOV diagnostics (against train-only vocab, n_holdout={n_holdout}): "
              f"max_ratio={max_ratio:.4f}")

    print_statistics(samples, maps)

    print("\n4. Saving...")
    save(
        X, y, maps, output_dir,
        ewma_alpha=ewma_alpha,
        train_ratio=train_ratio,
        n_train_vocab_fit=n_train,
        oov_stats=oov_stats,
        vocab_fit_scope="full_dataset",
    )

    return X, y, maps


def main():
    parser = argparse.ArgumentParser(
        description="Prepares training data for DeepProduct (with time encoding).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path)
    group.add_argument("--orders", type=Path, nargs="+")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/product_training"))
    parser.add_argument("--features", type=str, nargs="+")
    parser.add_argument(
        "--periods", type=str, nargs="+",
        default=DEFAULT_PERIODS,
        choices=list(NAMED_PERIODS_SECONDS.keys()),
        help=f"Time periods for sin/cos encoding (default: {DEFAULT_PERIODS})",
    )
    args = parser.parse_args()

    if args.data_dir:
        orders_paths = sorted(args.data_dir.rglob("*orders*.csv"))
        if not orders_paths:
            parser.error(f"No *orders*.csv in {args.data_dir}")
    else:
        orders_paths = args.orders

    prepare_product_data(
        orders_paths=orders_paths,
        output_dir=args.output,
        feature_names=args.features,
        time_periods=args.periods,
    )


if __name__ == "__main__":
    main()