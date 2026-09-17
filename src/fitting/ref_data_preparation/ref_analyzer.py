"""
RefSimAnalyzer — extracts statistical parameters from simulation data.

For parity with DeepSim, each run CSV keeps only the first ``train_ratio``
fraction of events chronologically and restricts orders to ``timestamp_creation``
within the training window; this prevents data leakage with the eval logs.
``train_ratio=1.0`` disables the cut (full data).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import CensoredData, weibull_min

from src.dynamics.foundation_dynamics import END_TOKEN
from src.config.routing_keys import full_variant_key, product_type_key


def _dequantized_normal(xs) -> tuple[float, float]:
    """Latent (mean, std) from ceil-quantized whole-second observations.

    Sheppard's corrections for grouped data: the quantizer adds +0.5 to the
    mean and +1/12 to the variance, so the latent moments are mean-0.5 and
    sqrt(max(var - 1/12, 0)). Derived from the declared log quantization
    (integer time contract), not from generator knowledge.
    """
    mean = float(np.mean(xs)) - 0.5
    var = float(np.var(xs)) - 1.0 / 12.0
    return mean, float(np.sqrt(max(var, 0.0)))


class RefSimAnalyzer:
    def __init__(self, data_dir: Path, train_ratio: float = 0.70) -> None:
        if not 0.0 < train_ratio <= 1.0:
            raise ValueError(
                f"train_ratio must be in (0, 1]; got {train_ratio}"
            )
        self._data_dir = Path(data_dir)
        self._train_ratio = float(train_ratio)
        self._events_paths = sorted(self._data_dir.rglob("*_events_*.csv"))
        self._orders_paths = sorted(self._data_dir.rglob("*_orders_*.csv"))
        if not self._events_paths:
            raise FileNotFoundError(f"No *_events_*.csv found in {self._data_dir}")
        if not self._orders_paths:
            raise FileNotFoundError(f"No *_orders_*.csv found in {self._data_dir}")
        self._cut_metadata: List[Dict[str, Any]] = []

    def extract_all(self) -> Dict[str, Any]:
        self._cut_metadata = []
        events_all: List[Dict[str, Any]] = []
        orders_features_all: Dict[str, Dict[str, str]] = {}
        orders_completions_all: Dict[str, Optional[float]] = {}

        for events_path, orders_path in self._pair_files():
            run_events, cut_ts = self._load_and_cut_events(events_path)
            events_all.extend(run_events)
            run_features, run_completions = self._load_and_cut_orders(
                orders_path, cut_ts
            )
            orders_features_all.update(run_features)
            orders_completions_all.update(run_completions)

        mttf_data, repair_times = self._extract_breakdowns_and_repairs(events_all)

        return {
            "process_times": self._extract_process_times(events_all, orders_features_all),
            "transition_probs": self._extract_transitions(
                events_all, orders_features_all, orders_completions_all
            ),
            "process_times_variant": self._extract_process_times_conditioned(
                events_all, orders_features_all
            ),
            "transition_probs_variant": self._extract_transitions_conditioned(
                events_all, orders_features_all, orders_completions_all
            ),
            "weibull_ttf": self._extract_weibull_ttf(events_all),
            "mttf_data": mttf_data,
            "repair_times": repair_times,
            "product_features": self._extract_product_features(orders_features_all),
            "train_cut_metadata": list(self._cut_metadata),
        }

    def _pair_files(self) -> List[Tuple[Path, Path]]:
        pairs: List[Tuple[Path, Path]] = []
        for ep in self._events_paths:
            op = ep.with_name(ep.name.replace("_events_", "_orders_", 1))
            if not op.exists():
                raise FileNotFoundError(
                    f"No matching orders file for {ep.name}: expected {op.name}"
                )
            pairs.append((ep, op))
        return pairs

    def _load_and_cut_events(self, path: Path) -> Tuple[List[Dict[str, Any]], float]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "order_id": r["order_id"],
                    "timestamp_event_start": float(r["timestamp_event_start"]),
                    "station": r["station"],
                    "station_type": r.get("station_type", "machine"),
                    "time_processing": float(r["time_processing"]),
                    "is_breakdown": r["is_breakdown"] == "True",
                    "net_process_time": float(r["net_process_time"]),
                    "repair_time": float(r["repair_time"]),
                })
        rows.sort(key=lambda e: e["timestamp_event_start"])
        n_total = len(rows)

        if self._train_ratio >= 1.0:
            kept = rows
            cut_ts = float("inf")
            n_train = n_total
        else:
            n_train = int(np.floor(self._train_ratio * n_total))
            kept = rows[:n_train]
            cut_ts = kept[-1]["timestamp_event_start"] if kept else 0.0

        self._cut_metadata.append({
            "run_key": path.stem,
            "events_file": path.name,
            "n_total_events": n_total,
            "n_train_events": n_train,
            "train_cut_timestamp": cut_ts,
            "train_ratio": self._train_ratio,
        })
        return kept, cut_ts

    def _load_and_cut_orders(
        self, path: Path, cut_ts: float
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Optional[float]]]:
        features: Dict[str, Dict[str, str]] = {}
        completions: Dict[str, Optional[float]] = {}
        meta_cols = {"order_id", "timestamp_creation", "timestamp_completion"}
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if cut_ts != float("inf"):
                    tc_create = r.get("timestamp_creation", "")
                    if not tc_create:
                        raise ValueError(
                            f"Order in {path.name} has an empty "
                            f"'timestamp_creation' field "
                            f"(order_id={r.get('order_id', '<missing>')}). "
                            f"Generator scripts should write valid floats."
                        )
                    try:
                        t_create = float(tc_create)
                    except ValueError as exc:
                        raise ValueError(
                            f"Corrupt 'timestamp_creation' value "
                            f"{tc_create!r} in {path.name} "
                            f"(order_id={r.get('order_id', '<missing>')})."
                        ) from exc
                    if t_create > cut_ts:
                        continue
                oid = r["order_id"]
                features[oid] = {k: v for k, v in r.items() if k not in meta_cols}
                tc = r.get("timestamp_completion", "")
                t_complete = float(tc) if tc and tc != "None" else None
                # A completion after the train cut is post-cut knowledge: within
                # the training window the order is still in flight, so no END
                # transition may be counted for it.
                if (t_complete is not None and cut_ts != float("inf")
                        and t_complete > cut_ts):
                    t_complete = None
                completions[oid] = t_complete
        return features, completions

    def _extract_process_times(
        self,
        events: List[Dict],
        orders_features: Dict[str, Dict[str, str]],
    ) -> Dict[str, Tuple[float, float]]:
        """Sample mean/std of net_process_time per station (station-marginal)."""
        times: Dict[str, List[float]] = defaultdict(list)
        n_orphan = 0
        for e in events:
            if e["is_breakdown"] or e["net_process_time"] <= 0:
                continue
            feats = orders_features.get(e["order_id"])
            if feats is None:
                # Warmup-boundary edge case: event for an order created during
                # warmup but not persisted to orders.csv. _extract_transitions
                # skips these events as well.
                n_orphan += 1
                continue
            times[e["station"]].append(e["net_process_time"])
        if n_orphan:
            print(f"  [stat] Skipped {n_orphan} orphan event(s) "
                  "(order_id not in orders.csv, likely warmup boundary).")
        return {s: _dequantized_normal(v) for s, v in times.items()}

    def _extract_transitions(self, events: List[Dict], orders_features: Dict, orders_completions: Dict) -> Dict[str, Dict[str, float]]:
        """P(target|station) per station, marginalized over model/feature."""
        productive = [
            e for e in events
            if not e["is_breakdown"]
            and e["order_id"] != "BREAKDOWN"
            and e["station_type"] != "overflow"
            and e["order_id"] in orders_features
        ]

        events_by_order = defaultdict(list)
        for e in productive:
            events_by_order[e["order_id"]].append(e)

        for oid in events_by_order:
            events_by_order[oid].sort(key=lambda e: e["timestamp_event_start"])

        counts = defaultdict(lambda: defaultdict(int))

        for oid, oe in events_by_order.items():
            for i in range(len(oe) - 1):
                current_station = oe[i]["station"]
                next_station = oe[i + 1]["station"]
                counts[current_station][next_station] += 1

            if orders_completions.get(oid) is not None and oe:
                last_station = oe[-1]["station"]
                counts[last_station][END_TOKEN] += 1

        out = {}
        for s, v in counts.items():
            total = sum(v.values())
            out[s] = {t: c / total for t, c in v.items()}
        return out

    _MIN_CELL_OBS = 2  # cells with <2 observations: variance undefined, fall back

    def _extract_process_times_conditioned(
        self,
        events: List[Dict],
        orders_features: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """Normal MLE per (station, variant) and per (station, product type).

        Cells below _MIN_CELL_OBS fall back to the coarser level at runtime.
        """
        vals_v: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        vals_p: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for e in events:
            if e["is_breakdown"] or e["net_process_time"] <= 0:
                continue
            feats = orders_features.get(e["order_id"])
            if feats is None:
                continue  # orphan tolerance, as in _extract_process_times
            vals_v[e["station"]][full_variant_key(feats)].append(e["net_process_time"])
            vals_p[e["station"]][product_type_key(feats)].append(e["net_process_time"])

        by_variant, cov_v = self._fit_normal_cells(vals_v)
        by_producttype, cov_p = self._fit_normal_cells(vals_p)
        return {
            "by_variant": by_variant,
            "by_producttype": by_producttype,
            "coverage": {"variant": cov_v, "producttype": cov_p},
        }

    def _fit_normal_cells(
        self, vals: Dict[str, Dict[str, List[float]]]
    ) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, int]]:
        out: Dict[str, Dict[str, Tuple[float, float]]] = {}
        n_cells = n_kept = n_dropped = 0
        for station, by_key in vals.items():
            kept: Dict[str, Tuple[float, float]] = {}
            for key, xs in by_key.items():
                n_cells += 1
                if len(xs) >= self._MIN_CELL_OBS:
                    kept[key] = _dequantized_normal(xs)
                    n_kept += 1
                else:
                    n_dropped += 1
            if kept:
                out[station] = kept
        return out, {"n_cells": n_cells, "n_kept": n_kept, "n_dropped_lt2obs": n_dropped}

    def _extract_transitions_conditioned(
        self,
        events: List[Dict],
        orders_features: Dict,
        orders_completions: Dict,
    ) -> Dict[str, Any]:
        """Categorical routing distribution per (station, variant) and (station, product type).

        Mirrors the filter/END logic of _extract_transitions exactly, only
        additionally grouped by variant/product type. Cells with <2 transitions
        fall back.
        """
        productive = [
            e for e in events
            if not e["is_breakdown"]
            and e["order_id"] != "BREAKDOWN"
            and e["station_type"] != "overflow"
            and e["order_id"] in orders_features
        ]
        events_by_order = defaultdict(list)
        for e in productive:
            events_by_order[e["order_id"]].append(e)
        for oid in events_by_order:
            events_by_order[oid].sort(key=lambda e: e["timestamp_event_start"])

        counts_v: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        counts_p: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))

        for oid, oe in events_by_order.items():
            feats = orders_features[oid]
            vk, pk = full_variant_key(feats), product_type_key(feats)
            for i in range(len(oe) - 1):
                cur, nxt = oe[i]["station"], oe[i + 1]["station"]
                counts_v[cur][vk][nxt] += 1
                counts_p[cur][pk][nxt] += 1
            if orders_completions.get(oid) is not None and oe:
                last = oe[-1]["station"]
                counts_v[last][vk][END_TOKEN] += 1
                counts_p[last][pk][END_TOKEN] += 1

        by_variant, cov_v = self._normalize_prob_cells(counts_v)
        by_producttype, cov_p = self._normalize_prob_cells(counts_p)
        return {
            "by_variant": by_variant,
            "by_producttype": by_producttype,
            "coverage": {"variant": cov_v, "producttype": cov_p},
        }

    def _normalize_prob_cells(
        self, counts: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, int]]:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        n_cells = n_kept = n_dropped = 0
        for station, by_key in counts.items():
            kept: Dict[str, Dict[str, float]] = {}
            for key, tgt_counts in by_key.items():
                n_cells += 1
                total = sum(tgt_counts.values())
                if total >= self._MIN_CELL_OBS:
                    kept[key] = {t: c / total for t, c in tgt_counts.items()}
                    n_kept += 1
                else:
                    n_dropped += 1
            if kept:
                out[station] = kept
        return out, {"n_cells": n_cells, "n_kept": n_kept, "n_dropped_lt2obs": n_dropped}

    def _extract_ttf_spells(
        self, events: List[Dict]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Reconstruct TTF spells (operating seconds) per machine station.

        Mirrors ``prepare_survival_data.extract_survival_samples``: per station,
        chronologically; Σ net_process_time of productive jobs = spell duration; a
        stoppage (is_breakdown or order_id=='BREAKDOWN') closes an uncensored
        spell; the cycle still open at log end is right-censored.
        """
        by_station: Dict[str, List[Dict]] = defaultdict(list)
        for e in events:
            if e["station_type"] == "machine":
                by_station[e["station"]].append(e)
        for s in by_station:
            by_station[s].sort(key=lambda e: e["timestamp_event_start"])

        out: Dict[str, Dict[str, List[float]]] = {}
        for station, evs in by_station.items():
            unc: List[float] = []
            cen: List[float] = []
            op = 0.0
            jobs = 0
            for e in evs:
                if e["is_breakdown"] or e["order_id"] == "BREAKDOWN":
                    if jobs > 0 and op > 0:
                        unc.append(op)
                    op = 0.0
                    jobs = 0
                    continue
                op += e["net_process_time"]
                jobs += 1
            if jobs > 0 and op > 0:
                cen.append(op)
            out[station] = {"uncensored": unc, "censored": cen}
        return out

    def _extract_weibull_ttf(self, events: List[Dict]) -> Dict[str, Any]:
        """Weibull MLE (shape, scale) per station with right-censoring."""
        spells = self._extract_ttf_spells(events)
        params: Dict[str, Tuple[float, float]] = {}
        coverage: Dict[str, Dict[str, int]] = {}
        for station, d in spells.items():
            unc, cen = d["uncensored"], d["censored"]
            coverage[station] = {
                "n_spells": len(unc) + len(cen),
                "n_uncensored": len(unc),
                "n_censored": len(cen),
            }
            fit = self._fit_weibull_censored(unc, cen)
            if fit is not None:
                params[station] = fit
        return {"params": params, "coverage": coverage}

    @staticmethod
    def _fit_weibull_censored(
        uncensored: List[float], censored: List[float]
    ) -> Optional[Tuple[float, float]]:
        """Weibull MLE (loc=0) on uncensored + right-censored spells.

        Returns None for <2 uncensored spells.
        """
        if len(uncensored) < 2:
            return None
        unc = np.asarray(uncensored, dtype=float)
        if censored:
            data: Any = CensoredData(uncensored=unc, right=np.asarray(censored, dtype=float))
        else:
            data = unc
        shape, _loc, scale = weibull_min.fit(data, floc=0)
        return (float(shape), float(scale))

    def _extract_breakdowns_and_repairs(self, events: List[Dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
        operating_time = defaultdict(float)
        bd_count = defaultdict(int)
        repairs = defaultdict(list)

        for e in events:
            s = e["station"]
            if e["is_breakdown"]:
                bd_count[s] += 1
                repairs[s].append(e["repair_time"])
            elif e["order_id"] != "BREAKDOWN":
                # Sum actual processing time for the MTTF.
                operating_time[s] += e["net_process_time"]

        mttf_dict = {}
        repair_scales: Dict[str, float] = {}
        for s in set(operating_time.keys()) | set(bd_count.keys()):
            count = bd_count.get(s, 0)
            mttf_dict[s] = operating_time[s] / count if count > 0 else float('inf')

            # Exponential MLE on ceil-quantized data: latent scale = mean - 0.5
            # (integer time contract; exact dequantization for the exponential
            # is scale = -1/log(1 - 1/mean_geom), but the -0.5 first-order form
            # is within 1e-4 relative at these scales and keeps it simple).
            reps = repairs.get(s, [])
            repair_scales[s] = max(float(np.mean(reps)) - 0.5, 0.5) if reps else 0.0

        return mttf_dict, repair_scales

    def _extract_product_features(self, orders_features: Dict) -> Dict[str, Dict[str, float]]:
        counts = defaultdict(lambda: defaultdict(int))
        for features in orders_features.values():
            for fn, fv in features.items():
                counts[fn][fv] += 1
        out = {}
        for fn, vc in counts.items():
            total = sum(vc.values())
            out[fn] = {v: c / total for v, c in vc.items()}
        return out
