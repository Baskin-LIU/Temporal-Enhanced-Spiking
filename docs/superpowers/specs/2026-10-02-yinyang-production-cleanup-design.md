# Yin-Yang Production Cleanup Design

## Purpose

Turn the research/debugging implementation into a human-readable and efficient
training codebase without changing the validated TeS event dynamics, timing
losses, or gradients. The cleaned repository will expose every experimental
hyperparameter, make the best held-out-test configuration the default, retain
only scientifically useful artifacts, and remove obsolete debugging machinery.

## Canonical experiment

The canonical default is the 40-epoch squared-hinge experiment with hidden-row
restart offset `0.1`, selected because it achieved the highest held-out test
accuracy (`0.935`) among the completed runs. The `0.25` restart run reached a
higher single validation result (`0.955`) but a lower held-out test accuracy
(`0.920`), so it is not the default.

The canonical values include:

- squared-hinge timing loss with margin `2.0 ms`;
- Adam learning rate `0.01`, batch size `50`, and the existing beta/epsilon and
  gradient-norm-cap values;
- 40-epoch cosine annealing with `T_max=40` and minimum rate `0.0`;
- 50 ms full-horizon simulation with training early stopping disabled;
- threshold-sensitivity reciprocal clamp `2.0`;
- hidden restart offset `0.1`, silence threshold fraction `0.5`, and restart
  seed `3001`;
- `tau_v=20 ms`, `alpha=1/40 ms^-1`, fast angular frequency `1/4 rad/ms`, and
  slow angular frequency `1/6 rad/ms`;
- `d_max=5 ms`, `c0=20`, and `c1=2`;
- the dataset sizes, model dimensions, initialization distributions,
  calibration controls, and random seeds recorded by the retained run.

## Configuration interface

`configs/yinyang_best.json` is the canonical, human-readable source of default
experiment settings. It uses nested sections for data, model dynamics,
initialization, calibration, loss, optimization, restart behavior, and artifact
policy.

`Code/yinyang_config.py` owns typed configuration records, JSON loading,
startup validation, serialization, and command-line override mapping. Every
experimentally meaningful field is configurable, including:

- data sizes and split seeds;
- model width, horizon, numerical step, threshold/reset values, membrane and
  oscillator constants, delay constants, and reciprocal-slope clamp;
- hidden and readout initialization means, standard deviations, gains, row
  scales or target row sums, heterogeneity, and seeds;
- calibration target, scale, and maximum rounds;
- loss name and margin;
- epoch count, batch size, shuffle seed, all Adam values, gradient cap, and all
  cosine schedule values;
- hidden silence threshold, restart distribution scale/offset, and seed; and
- result, checkpoint, plot, and raster controls.

`Code/train_yinyang.py` becomes a small command-line entry point. It loads the
canonical JSON by default, applies only explicitly supplied CLI overrides,
runs the experiment, and prints the result path. Every resolved value is saved
verbatim under the result JSON's `configuration` field. Training always calls
the numerical solver with `early_stop=False`; this is an experiment invariant,
not a switch in the production configuration.

## Production code structure

- `Code/neurons.py`, `Code/tes_events.py`, `Code/tes_roots.py`,
  `Code/tes_loss.py`, and `Code/tes_psp.py` retain the numerical model.
- `Code/tes_training.py` contains optimizer, batching, evaluation, and silent
  hidden-row restart primitives.
- `Code/yinyang_config.py` contains configuration and CLI mapping.
- `Code/yinyang_experiment.py` contains initialization, readout calibration,
  epoch orchestration, and resume behavior.
- `Code/yinyang_reporting.py` contains checkpoint, compact JSON, and plotting
  responsibilities.
- `Code/train_yinyang.py` is only the executable adapter.
- `Code/debug.py` retains a small selectable finite-difference gradient check
  and compact result/checkpoint inspection; it is not imported by production
  training.

The numerical solver's event ordering, threshold semantics, exact flows, root
certification, silence semantics, and forward/backward equations do not change.
Large-file extraction is by responsibility, not by rewriting numerical
algorithms.

## Runtime checks and diagnostics

Normal execution removes redundant hot-path validation, legacy configuration
normalization, obsolete checkpoint-schema support, and detailed per-sample
failure-tape construction. Algorithmic branches required by event ordering,
silence handling, and gradient transport remain.

Production checks are limited to:

- configuration validation once at startup;
- weight-shape validation at model construction;
- checkpoint compatibility validation when loading;
- scientifically meaningful root ambiguity/numerical exceptions; and
- one finite-weight check after each completed epoch, before validation and
  artifact writing.

There is no per-update finite-gradient or finite-weight check. A non-finite
epoch ends with an error identifying the epoch. Detailed result, checkpoint,
and finite-difference inspection is available through the separate opt-in
`Code.debug` commands, without adding a dormant production-training flag.

Only cosine annealing is retained. The obsolete one-time step-decay fields and
their checkpoint compatibility branches are removed.

## Artifact policy

Ordinary training does not save a checkpoint and plot every epoch. It keeps the
best-validation checkpoint, the final tested checkpoint when a test evaluation
is requested, compact epoch summaries, final plots, and optional raster plots.
Intermediate checkpoint and plot intervals remain explicit operational
arguments and are disabled by default.

The cleaned retained artifacts are:

```text
results/
  yinyang_best/
    config.json
    metrics.json
    checkpoints/
      best_validation.pt
      final_tested.pt
    figures/
      training.png
      raster_pre_training.png
      raster_correct.png
      raster_incorrect.png
  yinyang_experiment_summary.json
```

`metrics.json` contains epoch-level learning-rate, loss, accuracy, silence,
restart, and cap-activation summaries plus final metrics, per-class accuracy,
confusion matrices, runtime, and artifact names. It omits batch histories and
sample-index arrays. `yinyang_experiment_summary.json` preserves headline
results and key configuration differences for the two squared-hinge runs and
the three modular DelGrad runs.

All other generated checkpoints, figures, raw tuning sweeps, temporary
diagnostics, caches, and platform metadata are deleted. Because this workspace
has no Git metadata, these deletions are not recoverable from the repository.

## Obsolete code and documentation

The following research/debugging modules and their dedicated tests are removed:

- `Code/ttfs_landscape.py`;
- `Code/refined_landscape.py`;
- `Code/landscape_diagnosis.py`;
- `Code/continuous_landscape.py`;
- `Code/continuous_gradient_check.py`;
- `Code/continuous_yinyang.py`;
- `Code/gradient_check.py`;
- `Code/benchmark_tes_core.py`;
- `Code/tune_yinyang.py`; and
- `Code/test.ipynb`.

Tests that use a removed helper merely as a fixture are migrated to direct
production fixtures rather than deleted. Tests whose only purpose is a removed
landscape, sweep, or benchmark utility are deleted.

Valid model contracts and empirical guidance are consolidated into
`docs/tes_code_contracts.md` and `docs/yinyang_empirical_tuning_guide.md`.
Stale implementation plans/specifications and the completed oscillator
migration-debt note are then removed. The `Latex/` theory tree is untouched.

## Verification

Before changing production numerical code, record a deterministic golden
snapshot containing forward events, physical/effective readout times, selected
losses, and input/readout weight gradients. Completion requires:

1. The canonical JSON exactly matches the retained `0.1` run configuration.
2. Every configuration field has a CLI override and appears in resolved output.
3. Golden forward/backward values remain numerically unchanged.
4. All retained unit and integration tests pass.
5. A small one-epoch smoke run completes with early stopping disabled.
6. Default output contains only compact results, best/final checkpoints, and
   final figures.
7. Runtime and generated-byte counts are compared with the original path.
8. Every retained artifact referenced by the manifests exists.
9. An independent agent reviews the implementation and cleanup.

The completed 40-epoch experiment is not rerun. Its measured report and tested
checkpoint remain the evidence for the canonical result.
