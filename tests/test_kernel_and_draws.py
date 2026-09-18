"""Departure discipline and the single-draw contract of the categorical draws."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.schema import StationConfig
from src.dynamics.foundation_dynamics import (
    masked_categorical_draw, masked_categorical_prepare, masked_categorical_probs,
    weighted_draw,
)
from src.simulation.order import Order
from src.simulation.station import Station


def test_departures_leave_in_arrival_order():
    st = Station(StationConfig(id="S", capacity=3))
    for i in range(3):
        st.add_to_departure(Order(id=f"O{i}", features={}, timestamp_creation=0.0), "T")
    assert [st.pop_departure()[0].id for _ in range(3)] == ["O0", "O1", "O2"]


def test_weighted_draw_matches_numpy_choice_and_its_generator_state():
    values = ["a", "b", "c", "d"]
    p = np.array([0.1, 0.4, 0.3, 0.2])
    for seed in range(300):
        r1, r2 = np.random.default_rng(seed), np.random.default_rng(seed)
        got = weighted_draw(values, p, r1)
        ref = r2.choice(values, p=p)
        assert got == ref, seed
        assert r1.bit_generator.state == r2.bit_generator.state, "RNG consumption differs"
    # integer form returns the index
    rng = np.random.default_rng(7)
    idx = weighted_draw(4, p, rng)
    assert idx == values.index(weighted_draw(values, p, np.random.default_rng(7)))


def test_masked_table_renormalizes_and_the_draw_consumes_one_number():
    targets, weights = ["A", "B", "C"], np.array([0.5, 0.3, 0.2])
    table = masked_categorical_prepare(targets, weights, {"B", "C"})
    assert table.removed_mass == pytest.approx(0.5) and not table.fallback
    assert masked_categorical_probs(table) == pytest.approx({"B": 0.6, "C": 0.4})
    rng = np.random.default_rng(0)
    draws = [masked_categorical_draw(rng, table) for _ in range(20000)]
    assert draws.count("A") == 0
    assert draws.count("B") / 20000 == pytest.approx(0.6, abs=0.02)
    state_before = np.random.default_rng(0).bit_generator.state
    r = np.random.default_rng(0)
    masked_categorical_draw(r, table)
    r_ref = np.random.default_rng(0)
    r_ref.random()
    assert r.bit_generator.state == r_ref.bit_generator.state
    assert state_before != r.bit_generator.state


def test_masked_table_falls_back_to_uniform_over_the_admissible_set():
    table = masked_categorical_prepare(["A", "B"], np.array([1.0, 0.0]), {"B", "X"})
    assert table.fallback and table.removed_mass == pytest.approx(1.0)
    assert masked_categorical_probs(table) == {"B": 0.5, "X": 0.5}
    with pytest.raises(ValueError, match="mask empty"):
        masked_categorical_prepare(["A"], np.array([1.0]), set())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
