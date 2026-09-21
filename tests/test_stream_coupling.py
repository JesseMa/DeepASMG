"""Only a true module inside the true core may share the target realization's random streams."""

import numpy as np
import pytest

from src.experiments.ablation_runner import CONFIGS, G
from src.experiments.sim_runner import MODULES, SimFactorySet
from src.config.simulation_config import SEED_OFFSET_FLOOR


def _first_draws(seed: int, coupled: frozenset) -> dict:
    rngs = SimFactorySet._module_rngs(seed, coupled)
    return {m: rngs[m].random() for m in MODULES}


def test_target_streams_belong_to_the_target_only():
    seed = 1234
    target = {m: np.random.default_rng(s).random()
              for m, s in zip(MODULES, np.random.SeedSequence(seed).spawn(len(MODULES)), strict=True)}
    offset = {m: np.random.default_rng(s).random()
              for m, s in zip(MODULES, np.random.SeedSequence(seed + SEED_OFFSET_FLOOR).spawn(len(MODULES)), strict=True)}
    assert _first_draws(seed, frozenset(MODULES)) == target
    assert _first_draws(seed, frozenset()) == offset
    mixed = _first_draws(seed, frozenset({"pt", "tr"}))
    assert mixed["pt"] == target["pt"] and mixed["tr"] == target["tr"]
    assert all(mixed[m] == offset[m] for m in ("sv", "rt", "pr"))


def test_grid_couples_only_true_modules_inside_the_true_core():
    for name, (kinds, coupled) in CONFIGS.items():
        for kind, slot in zip(kinds, MODULES, strict=True):
            if kind != G:
                assert slot not in coupled, f"{name}: candidate module {slot} shares the target stream"
        if all(k != G for k in kinds) or sum(k == G for k in kinds) == 1:
            assert coupled == frozenset(), f"{name}: a candidate core must draw independently"
    assert CONFIGS["Target (GroundSim)"][1] == frozenset(MODULES)
    assert CONFIGS["Floor (Full)"][1] == frozenset()
    for slot in MODULES:
        only = [n for n, (kinds, _) in CONFIGS.items() if n.endswith("only)") and kinds[MODULES.index(slot)] != G]
        for n in only:
            assert CONFIGS[n][1] == frozenset(MODULES) - {slot}, n


def test_compose_rejects_a_coupled_candidate_module():
    fs = SimFactorySet.__new__(SimFactorySet)
    with pytest.raises(ValueError):
        fs.compose(1000, 1, ("deep", "ground", "ground", "ground", "ground"), coupled=frozenset(MODULES))
    with pytest.raises(ValueError):
        fs.compose(1000, 1, ("ground",) * 5, coupled=frozenset({"xx"}))


def test_replicate_stream_sets_use_seeds_no_other_run_uses():
    from scripts.run_stream_replicates import compose_seed, N_SETS
    seeds = list(range(1000, 1010))
    targets = set(seeds)
    used = set()
    for k in range(1, N_SETS + 1):
        for seed in seeds:
            stream_seed = compose_seed(seed, k) + SEED_OFFSET_FLOOR
            assert stream_seed not in targets
            assert stream_seed not in used
            used.add(stream_seed)
    assert compose_seed(1000, 1) == 1000
