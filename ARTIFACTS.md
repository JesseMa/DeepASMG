# Publication artifacts

This repository contains the compact evidence package associated with the
DeepASMG article. All study inputs are synthetic; it contains no participant,
customer, or operational company data.

## Included

- `models/`: the exact trained TorchScript models, their inference metadata,
  the hyperparameter search results and training manifest behind them, and
  the fitted reference-distribution parameters used for the paper.
- `results/verification/`: compact derived CSV tables supporting the reported
  component, routing, system, and sensitivity results, plus two hazard
  diagnostics that the article does not report.
- `results/execution_audit/`: derived evidence for the runtime safeguard audit,
  regenerable with `scripts/run_safeguard_audit.py`, together with the
  hand-authored `SAFEGUARD_INVENTORY.csv`.
- `checksums.sha256`: SHA-256 checksums for all released models and research-data
  artifacts.

## Excluded

Approximately 13 GB of intermediate artifacts remain in the development
workspace and are not part of this release:

- Event-level simulation logs and run directories (`data/`, `runs/`, `logs/`).
- Raw verification bundles: `results/verification/closed_loop_runs*.pkl`,
  `results/verification/shadow/`, `results/verification/sensitivity_sweeps.pkl`.
- Model-training intermediates: `models/checkpoints/`, `models/hpo*/` except
  the `*_best_params.json` files, and the per-component `models/*_data/data.npz`
  training caches.
- Paper figure sources, input tables, and rendered artwork; the manuscript
  distributes the final figures.
- The sensitivity-configuration tree (approximately 5 GB): per-horizon and
  per-seed training data, trained models, and fitted reference parameters for
  the data-regime analysis.

All of these are deterministic or regenerable from the included synthetic
generator, testbed configuration, fixed seed protocol, and trained release
models.

## Regeneration scope

Every table in `results/verification/` is regenerable from this release. The data-regime tables
(`sweep_horizon.csv`, `sweep_draw31.csv`, `sweep_draw365.csv`, `observation_counts.csv`) additionally require the
sensitivity-configuration tree — one training-data draw and one trained model
set per derivation horizon and per derivation seed. That tree is excluded from
the release for size, and `scripts/build_sensitivity_tree.py` rebuilds it from
scratch; expect several hours of data generation and training. Afterwards,
`scripts/run_sensitivity_sweeps.py` computes every configuration fresh
(DeepSim with the configuration's own model set, refit RefSim references, the
shared paired GroundSim and GroundSim-DEC runs, and — for the 31-day draws —
the two module-selection variants released as the hybrid columns of
`sweep_draw31.csv`).

`results/verification/observation_counts.csv` records the per-component
observation counts of every sensitivity configuration: the six derivation
horizons, the five 365-day draws and the five 31-day draws. The underlying training logs are
excluded, so this table keeps the reported observation counts verifiable without
shipping the logs. `results/verification/kpi_absolute.csv` holds the unscaled
per-system, per-seed macro-KPI values behind the relative deviations in
`kpi_panel.csv`.

## Reading notes

- `component_substitutions.csv` carries the column `coupled_slots`: the module
  slots (`pt`, `tr`, `sv`, `rt`, `pr`) that share the target realization's
  random streams in that configuration. Candidate systems and every "Perfect X"
  hybrid show `none`; the one-module substitutions and their floors show the
  four true-core slots. All other evaluated systems draw from the second stream
  set of the seed, which they share among themselves.
- In `component_substitutions.csv` the three `component=all` configurations
  appear under both substitution directions with identical values by
  construction; they are one measurement each, not two.
- `RefSim Hybrid (Perfect Orders)` is bit-identical to `RefSim (All
  Statistical)` in every seed: no RefSim-M module conditions on released-order
  attributes, so restoring the true released-order mechanism changes nothing
  downstream.
- The release model set (365-day log, derivation seed 101) appears as
  `production` in `sweep_draw365.csv` and as `days_0365` in `sweep_horizon.csv`;
  both coincide with the DeepSim and RefSim-M rows of `system_distances.csv`.
  `seed_0101` in `sweep_draw31.csv` is the separately generated 31-day log of
  the same derivation seed, with its own model set; it coincides with
  `days_0031` in `sweep_horizon.csv`.
- In the `results/execution_audit/` tables every system reports the same
  `routing.mask_*` counters, but their renormalization rates are not comparable
  across families: DeepTransition's softmax puts positive mass on every target,
  so each masked draw counts as a renormalization (rate 1.0), whereas the fitted
  RefSim tables carry no mass on inadmissible targets (rate 0.0).
- `hazard_true_grid.csv` and `hazard_params.csv` are diagnostics of the
  GroundSim bathtub hazard against the fitted RefSim-M and RefSim-W parameters,
  written by `produce_results`; no figure or table of the manuscript and no
  script in this release consumes them.
- RefSim-M and RefSim-W rows are bit-identical in every table outside the
  survival component (routing, processing, repair, arrival), because the two
  variants differ only in the survival module.
- The `groundsim_dec_w1` column in `sweep_draw31.csv` repeats the
  GroundSim-DEC reference from `system_distances.csv` as a reference line; it
  is identical across the five configurations by construction.
- In `m6_routine_constraint_enforcement.csv` the `routing_renorm_rate` column
  contains the string `NOT_APPLICABLE` for the two GroundSim rows; parse the
  column as text or filter those rows before casting.
- In `scores_continuous.csv` and `scores_categorical.csv` the `station` and
  `head` cells are empty where the column does not apply: released-order rows
  are scored system-wide rather than per station, and only the categorical
  components have prediction heads.
- The SHA-256 hashes of the five model files are recorded both in
  `checksums.sha256` and in `production_training_manifest.json`
  (`artifacts.<component>.sha256`); keep the two in sync on any re-release.

## Repository deposit

Archive the tagged `v1.0.0` release on Zenodo via the GitHub-release
integration and add the minted version DOI to the manuscript data availability
statement before journal submission. Cite the version-specific DOI rather than
an unversioned development branch.

## Licenses

Software is MIT licensed (`LICENSE`). Trained
models and derived data are CC BY 4.0 licensed (`DATA_LICENSE.md`).
