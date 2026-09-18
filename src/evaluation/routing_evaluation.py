"""Routing-probability comparison per decision point and product variant."""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from src.config import topology_4stage as cfg
from src.simulation.order import Order
from src.config.routing_keys import full_variant_key, station_admissible_targets

_KL_EPS = 1e-12


def all_variants() -> List[Tuple[str, Dict[str, str]]]:
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


def config_true_vector(
    process_config, station_id: str, features: Dict[str, str], *, visit: int = 1,
) -> Dict[str, float]:
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
    s = 0.0
    for k, t in true.items():
        if t > 0.0:
            s += t * np.log(t / max(sys.get(k, 0.0), _KL_EPS))
    return float(s)


Key = Tuple[str, str, int]


def deep_shadow_vectors(shadow_dir: Path, system: str) -> Dict[Key, Dict[str, float]]:
    import pandas as pd
    tr = shadow_dir / f"{system}__transition.csv"
    sc = shadow_dir / "_context_variant.csv"
    if not tr.exists() or not sc.exists():
        return {}
    dfv = pd.read_csv(sc, dtype={"context_id": str, "variant": str, "visit": int})
    ctx_key = {c: (v, int(n)) for c, v, n in zip(dfv["context_id"], dfv["variant"], dfv["visit"], strict=True)}
    df = pd.read_csv(tr, dtype={"context_id": str})
    acc: Dict[Key, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cnt: Dict[Key, int] = defaultdict(int)
    for _, r in df.iterrows():
        p = r["params"]
        if not isinstance(p, str) or not p.strip():
            continue
        probs = json.loads(p).get("probs", {})
        variant, visit = ctx_key[str(r["context_id"])]
        key = (str(r["station"]), variant, visit)
        for t, v in probs.items():
            acc[key][t] += float(v)
        cnt[key] += 1
    return {k: {t: v / cnt[k] for t, v in d.items()} for k, d in acc.items() if cnt[k] > 0}


def ref_fitted_vectors(ref_transition_strategy, process_config, keys: Iterable[Key]
                       ) -> Dict[Key, Dict[str, float]]:
    strat = ref_transition_strategy
    strat.initialize({s.id: s for s in process_config.stations})
    feats_of = dict(all_variants())
    out: Dict[Key, Dict[str, float]] = {}
    for station, variant, visit in keys:
        avail = admissible_targets(process_config, station)
        order = Order(id="probe", features=feats_of[variant], timestamp_creation=0.0,
                      visits={station: visit})
        dp = strat.distribution_params(station, order, available_targets=avail)
        out[(station, variant, visit)] = dict(dp["probs"])
    return out


def compare_rows(process_config, vectors: Dict[Key, Dict[str, float]],
                 system_name: str) -> List[dict]:
    feats_of = dict(all_variants())
    rows: List[dict] = []
    for (station, variant, visit), sysv in sorted(vectors.items()):
        true_v = config_true_vector(process_config, station, feats_of[variant], visit=visit)
        if len(true_v) <= 1:
            continue
        rows.append({"system": system_name, "station": station, "variant": variant,
                     "visit": visit, "l1": _l1(true_v, sysv), "kl": _kl(true_v, sysv),
                     "n_targets": len(true_v)})
    return rows
