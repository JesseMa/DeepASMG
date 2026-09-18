"""What preparation serializes must be what inference reads."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.dynamics.foundation_dynamics import NONE_TOKEN, VISIT_CAP, compile_offsets, visit_token
from src.fitting.deep_data_preparation.data_io import build_feature_layout
from src.fitting.deep_data_preparation.prepare_transition_data import (
    TransitionSample,
    build_encoding_maps,
    encode_samples,
)


def _groups(maps):
    groups = [("modell", maps.modell), ("feature_a", maps.feature_a),
              ("feature_b", maps.feature_b), ("from_station", maps.from_station),
              ("visit", maps.visit)]
    for k in range(maps.n_hist_slots):
        groups += [(f"hist_modell_{k}", maps.modell),
                   (f"hist_target_{k}", maps.slot_target)]
    return groups

K = 2
STATIONS = ("M5", "B3.1.1", "M4.1")


def _samples(n: int = 60):
    return [
        TransitionSample(
            modell="A" if i % 2 else "B",
            feature_a=f"a.{i % 2 + 1}",
            feature_b=f"b.{i % 3 + 1}",
            from_station=STATIONS[i % len(STATIONS)],
            visit=visit_token(i % (VISIT_CAP + 2) + 1),
            hist_modells=(NONE_TOKEN,) * K,
            hist_targets=(NONE_TOKEN,) * K,
            to_station="End" if i % 4 else "B5",
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def maps():
    s = _samples()
    return build_encoding_maps(s, s, n_hist_slots=K)


def test_every_encoded_block_survives_serialization(maps):
    serialized = json.loads(json.dumps(maps.to_dict()))
    for block in ("modell", "feature_a", "feature_b", "from_station", "visit",
                  "slot_target", "to_station"):
        assert block in serialized, f"encoding_maps loses '{block}' on the way to disk"
        assert serialized[block], f"'{block}' serializes as an empty vocabulary"


def test_layout_and_vocabulary_agree_on_the_same_blocks(maps):
    serialized = json.loads(json.dumps(maps.to_dict()))
    layout, _ = build_feature_layout(_groups(maps))
    offsets = compile_offsets(layout)
    categorical = {k for k, v in serialized.items() if isinstance(v, dict)}

    def vocabulary_of(block: str) -> str:
        if block.startswith("hist_modell_"):
            return "modell"
        if block.startswith("hist_target_"):
            return "slot_target"
        return block

    for name in offsets:
        assert vocabulary_of(name) in categorical, (
            f"layout block '{name}' has no serialized vocabulary"
        )
    assert "visit" in categorical


def test_feature_dim_matches_the_encoded_width(maps):
    X, _ = encode_samples(_samples(), maps)
    assert maps.to_dict()["feature_dim"] == X.shape[1]


def test_visit_block_is_one_hot_and_capped(maps):
    layout, _ = build_feature_layout(_groups(maps))
    offsets = compile_offsets(layout)
    off = offsets["visit"]
    width = len(maps.visit)
    assert width == VISIT_CAP

    X, _ = encode_samples(_samples(), maps)
    block = X[:, off:off + width]
    assert np.all(block.sum(axis=1) == 1.0)

    assert visit_token(VISIT_CAP) == visit_token(VISIT_CAP + 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
