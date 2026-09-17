"""Routing-probability comparison per decision point and product variant.

Predicted vectors are scored against the configuration truth by L1 and
KL(true‖sys).
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import topology_4stage as cfg
from src.simulation.order import Order
from src.config.routing_keys import full_variant_key, station_admissible_targets

_KL_EPS = 1e-12


def all_variants() -> List[Tuple[str, Dict[str, str]]]:
    """All full product variants as (variant_key, features)."""
    pf = cfg.PRODUCT_FEATURES
    models = sorted(pf["modell"])
    fas = sorted({v for m in pf["feature_a"].values() for v in m})
    fbs = sorted({v for m in pf["feature_b"].values() for v in m})
    out = []
    for m, fa, fb in product(models, fas, fbs):
        feats = {"modell": m, "feature_a": fa, "feature_b": fb}
        out.append((full_variant_key(feats), feats))
    return out


def admissible_targets(process_config, station_id: str) -> set:
    sc = {s.id: s for s in process_config.stations}[station_id]
    return station_admissible_targets(sc)


def decision_points(process_config) -> List[str]:
    """Stations with more than one admissible target."""
    return [s.id for s in process_config.stations
            if len(admissible_targets(process_config, s.id)) > 1]


def config_true_vector(
    process_config, station_id: str, features: Dict[str, str], *, visit: int = 1,
) -> Dict[str, float]:
    """True routing vector from the config, normalized.

    Resolved the same way the generator resolves it, visit index included, so
    a station with a repeat-visit row is compared against what it actually
    does rather than against its first-visit distribution.
    """
    sc = {s.id: s for s in process_config.stations}[station_id]
    order = Order(id="probe", features=features, timestamp_creation=0.0,
                  visits={station_id: visit})
    key = order.resolve_transition_key(sc.transitions, station_id=station_id)
    tm = sc.transitions[key]
    total = sum(tm.values())
    return {t: v / total for t, v in tm.items()}


def _l1(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def _kl(true: Dict[str, float], sys: Dict[str, float]) -> float:
    """KL(true‖sys) over the support of true; sys clipped at _KL_EPS."""
    s = 0.0
    for k, t in true.items():
        if t > 0.0:
            s += t * np.log(t / max(sys.get(k, 0.0), _KL_EPS))
    return float(s)


def deep_shadow_vectors(shadow_dir: Path, system: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Mean predicted vector per (station, variant) from the shadow calls."""
    import pandas as pd
    tr = shadow_dir / f"{system}__transition.csv"
    sc = shadow_dir / "_context_variant.csv"
    if not tr.exists() or not sc.exists():
        return {}
    dfv = pd.read_csv(sc, dtype={"context_id": str, "variant": str})
    var_by_ctx = dict(zip(dfv["context_id"], dfv["variant"], strict=True))
    df = pd.read_csv(tr, dtype={"context_id": str})
    acc: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt: Dict[Tuple[str, str], int] = defaultdict(int)
    for _, r in df.iterrows():
        p = r["params"]
        if not isinstance(p, str) or not p.strip():
            continue
        probs = json.loads(p).get("probs", {})
        key = (str(r["station"]), var_by_ctx.get(str(r["context_id"]), ""))
        for t, v in probs.items():
            acc[key][t] += float(v)
        cnt[key] += 1
    return {k: {t: v / cnt[k] for t, v in d.items()} for k, d in acc.items() if cnt[k] > 0}


def ref_fitted_vectors(ref_transition_strategy, process_config, points, variants
                       ) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Ref vectors per (station, variant) from ``distribution_params``."""
    strat = ref_transition_strategy
    strat.initialize({s.id: s for s in process_config.stations})
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for station in points:
        avail = admissible_targets(process_config, station)
        for vkey, feats in variants:
            order = Order(id="probe", features=feats, timestamp_creation=0.0)
            dp = strat.distribution_params(station, order, available_targets=avail)
            out[(station, vkey)] = dict(dp["probs"])
    return out


def compare_rows(process_config, *, deep_vectors: Dict, ref_vectors: Dict,
                 system_name: str, use_deep: bool) -> List[dict]:
    """L1 + KL per (station, variant) against config truth for one system."""
    points = decision_points(process_config)
    rows: List[dict] = []
    for station in points:
        for vkey, feats in all_variants():
            true_v = config_true_vector(process_config, station, feats)
            if len(true_v) <= 1:
                continue
            src = deep_vectors if use_deep else ref_vectors
            sysv = src.get((station, vkey))
            if not sysv:
                continue
            rows.append({"system": system_name, "station": station, "variant": vkey,
                         "l1": _l1(true_v, sysv), "kl": _kl(true_v, sysv),
                         "n_targets": len(true_v)})
    return rows
