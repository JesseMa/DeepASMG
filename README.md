# DeepASMG

Code and compact research artifacts for **DeepASMG: A Modular,
Kernel-Integrated Surrogate Framework for Automatic Simulation Model Generation
with Sim-to-Sim Verification**.

DeepASMG couples a discrete-event manufacturing simulation (`GroundSim`) with
learned conditional mechanisms (`DeepSim`) and distributional reference
mechanisms (`RefSim`). The release contains the synthetic testbed, training and
verification code, exact trained models, derived result tables, and
validation evidence used by the associated article.

All study inputs are synthetic. This repository contains no participant,
customer, or operational company data.

## Contents

| Path | Contents |
|---|---|
| `src/config/` | Run constants (`simulation_config.py`), the process schema (`schema.py`), and the instantiated four-stage topology (`topology_4stage.py`). |
| `src/simulation/` | The discrete-event kernel: engine, stations, events, orders, routing, deadlock and failure handling. It executes a run without knowing which mechanism family it is running. |
| `src/dynamics/` | The five mechanism contracts and their three interchangeable implementations: GroundSim, DeepSim, RefSim. |
| `src/fitting/` | Offline derivation: turns GroundSim logs into the DeepSim models and the RefSim parameters. |
| `src/recording/` | Station-pass and order-lifecycle recording and run persistence. |
| `src/experiments/` | Wires the mechanism families into systems, assigns the random streams and drives the replication series. |
| `src/evaluation/` | Scores finished runs: proper scores, aggregation, routing, hazard, KPI and system-level comparison. |
| `scripts/` | Data generation, model training, verification, and result-production commands. |
| `models/` | Exact trained release models and inference metadata used for the article. |
| `results/verification/` | Derived CSV tables supporting the reported numerical results. |
| `results/execution_audit/` | Derived runtime safeguard-audit evidence. |
| `ARTIFACTS.md` | Inclusion, exclusion, licensing, and archival scope. |

## Reading the code

A reviewer checking the reported results can follow one path:

1. `scripts/run_closed_loop.py` names the six evaluated systems.
2. `src/experiments/sim_runner.py` builds each of them: `SimFactorySet.compose`
   wires one module of a given kind (`ground`, `deep`, `stat`) per mechanism
   plus five independent RNG streams into one `SimulationConfig`; the named
   systems and the substitution grid are all compositions.
3. `src/config/simulation_config.py` holds that config and the run constants.
4. `src/simulation/engine.py` executes it. Process time, routing and released-order
   attributes are sampled there; time-to-failure and repair duration are driven by
   `src/simulation/failure_manager.py`, which owns the per-cycle TTF state.
5. `src/dynamics/foundation_dynamics.py` defines the five mechanism contracts, and
   `src/dynamics/{GroundSim,DeepSim,RefSim}/` implement each of them once per family:

| Mechanism | GroundSim (truth) | DeepSim (learned) | RefSim (statistical) |
|---|---|---|---|
| process time | `ground_process_time.py` | `deep_process_time.py` | `ref_process_time.py` |
| transition | `ground_transition.py` | `deep_transition.py` | `ref_transition.py` |
| survival | `ground_survival.py` | `deep_survival.py` | `ref_survival.py` |
| repair | `ground_repair.py` | `deep_repair.py` | `ref_repair.py` |
| released order | `ground_product.py` | `deep_product.py` | `ref_product.py` |

Each mechanism appears under three names: the module prefix above, the two-letter
slot key (`pt`, `tr`, `sv`, `rt`, `pr`) used by the runners and the model
files, and the `component` value in the result CSVs (`processing`, `transition`,
`survival`, `repair`, `arrival`).

### Tracing a reported number back to the data

The reported system-level distances are **means over the ten evaluation seeds**,
not single values. For the throughput-time Wasserstein-1 distance of *DeepSim*:

1. `results/verification/system_distances.csv` holds one row per system and seed;
   `w1_summary.csv` averages the ten `w1` values per system and reports their
   standard deviation, `w1_wilcoxon.csv` the paired difference of DeepSim
   against the decorrelated generator with its t-based confidence interval.
2. `scripts/produce_results.py` writes that table.
3. `src/evaluation/system_evaluation.py` computes the per-seed distances, pairing
   each candidate run with the GroundSim run of the same seed.
4. `src/evaluation/kpi_report.py` extracts the throughput times those distances
   are computed on.

`results/verification/manifest.json` maps every reported table to the CSV that
backs it.

## Environment

The release was produced with Python 3.14.5 on Darwin arm64. The direct runtime
dependencies are pinned in `requirements.txt`.

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch availability can vary by operating system and hardware. If the pinned
PyTorch wheel is unavailable for a platform, follow the official PyTorch
installation instructions for that platform while retaining the remaining
dependency versions.

## Use the archived models

The release models are ready for inference in `models/`; retraining is not
required to inspect or rerun the closed-loop verification.

```bash
python -m scripts.run_closed_loop        # 6 systems x 10 seeds -> closed_loop_runs.pkl
python -m scripts.run_shadow_evaluation  # per-decision scoring bundle
python -m scripts.run_substitutions      # two-way component substitutions
python -m scripts.run_safeguard_audit    # instrumented rerun + safeguard counters
python -m scripts.produce_results        # all result tables, after every producer
python -m scripts.write_checksums        # refresh checksums.sha256
```

`produce_results` runs last on purpose: it reads what the four producers above
write. `write_checksums` regenerates the manifest from declared patterns rather
than by hand.

`run_substitutions` runs the two-way component substitutions reported in the article,
together with the module-selection configuration (the learned core with the
statistical repair module), reported under the `module_selection` direction in
`component_substitutions.csv`.
`run_safeguard_audit` reruns the closed loop with the read-only safeguard
counters enabled and checks every run against the authoritative bundle. The
result is reported as PASS or FAIL on the console and as the field
`no_behavior_change_pass` in `results/execution_audit/m6_safeguard_audit.json`. The script
needs `run_closed_loop` to have run first.

These are full research runs: the default closed-loop and shadow evaluations
each use ten fixed seeds and a 31-day evaluation horizon after a 24 h warm-up;
the shadow run scores only the decisions made inside that window, as the
closed loop records only the orders released inside it. Raw bundles are
ignored; the compact derived CSVs are written to `results/verification/`.

Processing is scored on total duration, station setup time included, in
`results/verification/scores_continuous.csv` like every other component. All
five systems report on that one scale, so no reconciliation step is involved.
The result manifest records this distinction and reports run metadata only when
it is present in, or can be derived from, the corresponding raw bundle.

## Regenerate everything from scratch

Training data are generated entirely by GroundSim. The full chain, in the one
order that satisfies every dependency:

```bash
# 1. derivation logs
python -m scripts.generate_training_data
TRAINING_DATA_DIR="$(find data/training -maxdepth 1 -type d -name 'data_*' | sort | tail -n 1)"

# 2. hyperparameters (writes models/hpo/*_best_params.json)
python -m scripts.optimize_hyperparameters --data-dir "$TRAINING_DATA_DIR"

# 3. models + RefSim parameters + training manifest
python -m scripts.train_models --data-dir "$TRAINING_DATA_DIR"

# 4. experiments (any order among these four)
python -m scripts.run_closed_loop
python -m scripts.run_shadow_evaluation
python -m scripts.run_substitutions
python -m scripts.run_safeguard_audit

# 5. tables, then the checksum manifest
python -m scripts.produce_results
python -m scripts.write_checksums

# 6. optional: data-regime sweeps (hours; separate tree)
python -m scripts.build_sensitivity_tree --root /path/to/sensitivity_tree
python -m scripts.run_sensitivity_sweeps --sensitivity-root /path/to/sensitivity_tree
```

Step 2 is not optional: `train_models` reads the search result and fails if it
is absent, so that the production models and the sensitivity tree cannot end up
on different hyperparameters.

The settings that produced a model set and their provenance are recorded in
`models/production_training_manifest.json`, which `train_models` emits from the
run itself.
The five data-preparation modules under
`src/fitting/deep_data_preparation/` also expose standalone CLIs for
inspecting individual preparation steps; `train_models` runs them implicitly.

Hyperparameters come from one place: `train_models` reads the best parameters
written by `scripts.optimize_hyperparameters` into `--hpo-dir` (default
`models/hpo/`) and fails if they are absent. The `*_best_params.json` files
are release artifacts: the ignore rules admit them and `write_checksums`
lists them, so a model set ships together with the search result it was
trained from and can be retrained without repeating the search. A study is
named after the model and a fingerprint of the training log, so a search
resumes on the same log and never continues on a different one. To explore a
different search, point both scripts at the same separate directory:

```bash
python -m scripts.optimize_hyperparameters --data-dir "$TRAINING_DATA_DIR" --hpo-dir models/hpo_explore
python -m scripts.train_models --data-dir "$TRAINING_DATA_DIR" --hpo-dir models/hpo_explore
```

The sensitivity tree uses the same production hyperparameters for every
configuration, so the data-regime comparison varies data volume alone.

Retraining replaces the release models in `models/`; use a separate
`--model-dir` to retain the archived model files unchanged.

## Additional commands

`scripts.build_sensitivity_tree` rebuilds the sensitivity-configuration tree —
one training-data draw and one trained model set per derivation horizon and per
derivation seed — that the data-regime analysis needs. This is a long-running
command (several hours of data generation and training).
`scripts.run_sensitivity_sweeps` then computes the sweeps behind
`results/verification/sweep_*.csv` fresh from that tree:

```bash
python -m scripts.build_sensitivity_tree --root /path/to/sensitivity_tree
python -m scripts.run_sensitivity_sweeps --sensitivity-root /path/to/sensitivity_tree \
    --production-data-dir "$TRAINING_DATA_DIR"
```

`run_sensitivity_sweeps` writes the raw bundle
`results/verification/sensitivity_sweeps.pkl`; the sweep tables themselves are
written by the subsequent `python -m scripts.produce_results` run, which reads
this bundle.

## Data scope

The repository includes the exact trained models and compact derived evidence
underlying the article. It excludes approximately 48 GB of event-level logs,
training checkpoints, raw shadow bundles, and raw closed-loop pickles. These are
synthetic, deterministic or regenerable intermediate artifacts rather than
independent observational data. See `ARTIFACTS.md` and `checksums.sha256` for the
release inventory and integrity information.

Verify the released artifacts from the repository root with:

```bash
shasum -a 256 -c checksums.sha256      # Linux: sha256sum -c checksums.sha256
```

Verify before rerunning the pipeline. `produce_results` always rewrites
`results/verification/manifest.json`, which is one of the checksummed entries,
so the check reports a mismatch on a working tree where the verification
commands have already been rerun.

## Citation and archive

The tagged `v1.0.0` release is archived on Zenodo via the GitHub-release
integration. Cite the version-specific DOI of that archived release rather than
a development branch.

## Licenses

- Source code: MIT (`LICENSE`).
- Trained models and derived research data: CC BY 4.0 (`DATA_LICENSE.md`).
