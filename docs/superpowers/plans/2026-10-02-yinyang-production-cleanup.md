# Yin-Yang Production Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the Yin-Yang research training stack into a compact, configurable production path while preserving validated numerical behavior and the best measured experiment.

**Architecture:** A canonical nested JSON config feeds typed configuration records and explicit CLI overrides. The existing monolithic driver is split into configuration, experiment orchestration, and reporting modules; the TeS numerical core remains mathematically unchanged. Generated artifacts are reduced to compact reports plus selected checkpoints/figures, and obsolete research utilities are replaced by one small opt-in debug module.

**Tech Stack:** Python 3, PyTorch float64, NumPy, Matplotlib, pytest, JSON.

**Spec:** `docs/superpowers/specs/2026-10-02-yinyang-production-cleanup-design.md`

## Global Constraints

- Preserve continuous TeS event ordering, root certification, full-horizon behavior, loss values, and gradients.
- Training always calls the solver with `early_stop=False`.
- The canonical default is the squared-hinge `margin=2.0`, restart-offset `0.1`, reciprocal-slope-clamp `2.0`, 40-epoch run that measured `0.935` held-out test accuracy.
- Perform no new 40-epoch run and report no unmeasured accuracy.
- Check finite weights once after each epoch; perform no per-update finite-gradient or finite-weight checks.
- Keep necessary startup, shape, checkpoint, and scientific root-error validation.
- Add no dependency beyond the libraries already used by the project.
- Do not modify the `Latex/` tree.
- The workspace has no Git metadata; per-task Git commit steps are replaced by test results and explicit file inventories.

## Review Focus

- CLI values omitted by the user must come from the JSON config, while explicit falsey values such as `0.0` must override it.
- A resumed run must reject a checkpoint whose resolved scientific configuration differs, before mutating model or optimizer state.
- Disabling intermediate artifacts must not prevent creation of the best-validation and final-tested checkpoints.
- A non-finite weight introduced during an epoch must be detected at the epoch boundary before validation or checkpointing.
- Compacting the report must retain restart identities, silence counts, per-class metrics, confusion matrices, and the exact resolved configuration.

---

### Task 1: Freeze current numerical and artifact baselines

**Files:**
- Create: `Code/tests/data/yinyang_numerical_golden.json`
- Create: `Code/tests/test_yinyang_numerical_regression.py`
- Create: `results/yinyang_cleanup_baseline.json`

**Interfaces:**
- Consumes: current `initialize_yinyang_model`, continuous forward/loss/backward APIs, and the retained `0.1` report.
- Produces: immutable numerical expectations and pre-cleanup runtime/disk counts used by all later verification.

- [ ] **Step 1: Record one deterministic numerical fixture**

Use the canonical seeds and a fixed training-split sample index. Record input events, continuous hidden/output threshold times and identities, effective readouts, squared-hinge loss/gradient, and `grad_W_in`/`grad_W_out` in `yinyang_numerical_golden.json` with full float precision.

- [ ] **Step 2: Write the regression test**

Add `test_canonical_forward_loss_and_backward_match_golden()` and assert exact integer/event fields plus `torch.testing.assert_close` for float values at tight float64 tolerances.

- [ ] **Step 3: Verify the golden test passes before refactoring**

Run: `pytest -q Code/tests/test_yinyang_numerical_regression.py`

Expected: PASS against the current implementation.

- [ ] **Step 4: Record cleanup baselines**

Write the current number and total bytes of checkpoints, JSON records, and figures, the current code/test line counts, and a timed deterministic small training run to `results/yinyang_cleanup_baseline.json`.

- [ ] **Step 5: Inventory Task 1 output**

Run: `find Code/tests/data results -maxdepth 2 -type f -print | sort`

Expected: the golden fixture, baseline record, and no deletion of existing artifacts.

---

### Task 2: Add the canonical typed JSON configuration

**Files:**
- Create: `configs/yinyang_best.json`
- Create: `Code/yinyang_config.py`
- Create: `Code/tests/test_yinyang_config.py`
- Modify: `Code/neurons.py`
- Modify: `Code/tes_training.py`

**Interfaces:**
- Consumes: the exact resolved configuration stored in `Code/output/yinyang_quarter_sixth_hinge_restart010_epoch40_2026-10-01/yinyang_training.json` plus current hard-coded dynamics/initializer constants.
- Produces: `YinYangConfig.from_dict(values)`, `YinYangConfig.to_dict()`, `load_yinyang_config(path)`, `parse_yinyang_args(argv)`, `apply_cli_overrides(config, overrides)`, `YinYangConfig.to_tes_config()`, and `YinYangConfig.to_training_config()`.

- [ ] **Step 1: Write failing canonical-config tests**

Tests must assert that the JSON resolves to the approved scientific values, round-trips without data loss, rejects unknown/malformed fields, and reproduces the retained run's configuration after adding the newly explicit hard-coded constants.

- [ ] **Step 2: Write failing CLI coverage tests**

Parameterize over every leaf key in the JSON and assert there is a corresponding explicit CLI override. Separately prove that omitted flags preserve JSON values and falsey overrides (`0`, `0.0`, and disabled booleans) are not mistaken for omission.

- [ ] **Step 3: Implement nested immutable config records**

Define focused dataclasses for data, architecture/dynamics, initialization, calibration, loss, optimization, restart, and artifacts, composed by `YinYangConfig`. Centralize startup validation in `YinYangConfig.from_dict` and keep serialization JSON-native.

- [ ] **Step 4: Make formerly hard-coded scientific values configurable**

Extend `TeSConfig` construction to receive `tau_v`, `alpha`, `threshold`, `omega_fast`, `omega_slow`, `reset_drop`, the silent-target gap, delay constants, and the reciprocal-slope clamp. Extend initialization configuration to receive dimensions, distribution parameters, gains, row settings, heterogeneity, and all seeds.

- [ ] **Step 5: Simplify `TrainingConfig` to the active schedule**

Remove `learning_rate_decay_start_epoch` and `learning_rate_decay_factor`. Keep Adam fields, batch size, horizon, loss, margin, shuffle seed, cosine `T_max`, and cosine minimum rate. Permit configured batch sizes while preserving the canonical default of `50`.

- [ ] **Step 6: Run focused tests**

Run: `pytest -q Code/tests/test_yinyang_config.py Code/tests/test_tes_loss.py Code/tests/test_tes_psp.py`

Expected: PASS.

- [ ] **Step 7: Re-run numerical golden regression**

Run: `pytest -q Code/tests/test_yinyang_numerical_regression.py`

Expected: PASS with no changed event, loss, or gradient value.

---

### Task 3: Extract the experiment orchestrator and simplify the training hot path

**Files:**
- Create: `Code/yinyang_experiment.py`
- Modify: `Code/tes_training.py`
- Modify: `Code/train_yinyang.py`
- Modify: `Code/tests/test_tes_training.py`
- Create: `Code/tests/test_yinyang_experiment.py`

**Interfaces:**
- Consumes: `YinYangConfig`, TeS numerical APIs, `save_checkpoint`, `load_checkpoint`, batching/evaluation primitives.
- Produces: `TrainingArtifacts` in `Code/yinyang_experiment.py`, `initialize_yinyang_model(config: YinYangConfig) -> TeSNeurons`, `calibrate_readouts(model, split, config) -> CalibrationHistory`, and `run_yinyang_training(config, run_dir, *, resume_checkpoint=None, evaluate_test=False) -> TrainingArtifacts`.

- [ ] **Step 1: Write failing orchestration tests**

Cover deterministic initialization, forced `early_stop=False`, calibrated readout coverage, causal hidden-silence restart selection, one final test evaluation at most, and compatible resume behavior.

- [ ] **Step 2: Write the epoch-boundary finite-weight test**

Inject a non-finite weight during a mocked epoch, assert no per-update checker is called, and assert the orchestrator raises before validation/checkpointing with the epoch number.

- [ ] **Step 3: Move experiment responsibilities without changing equations**

Move initialization, calibration, epoch summaries, resume reconstruction, and the training loop from `train_yinyang.py` into `yinyang_experiment.py`. Use the typed config conversions instead of independent defaults.

- [ ] **Step 4: Remove normal-path diagnostic tape construction**

Remove `TrainingFailure`, `TrainingSampleError`, `build_training_failure`, and serialized full event/sample failure records from production training. Normal errors retain phase, epoch, batch, and sample context; detailed diagnosis is delegated to the opt-in debug path.

- [ ] **Step 5: Remove per-update finite checks**

Delete `_all_finite` checks from `adam_update`; retain shape assumptions established by model/config construction and compute the update directly. Add a single epoch-boundary finite-weight predicate in the orchestrator.

- [ ] **Step 6: Remove obsolete compatibility branches**

Delete legacy schedule normalization and old checkpoint-schema support. A checkpoint must match the current schema and complete resolved scientific configuration, except that an increased total epoch budget is permitted.

- [ ] **Step 7: Run focused training tests**

Run: `pytest -q Code/tests/test_yinyang_experiment.py Code/tests/test_tes_training.py`

Expected: PASS after deleting/replacing assertions that exist only for removed detailed failure tapes or step decay.

- [ ] **Step 8: Re-run numerical golden regression**

Run: `pytest -q Code/tests/test_yinyang_numerical_regression.py`

Expected: PASS.

---

### Task 4: Extract reporting and enforce the compact artifact policy

**Files:**
- Create: `Code/yinyang_reporting.py`
- Create: `Code/tests/test_yinyang_reporting.py`
- Modify: `Code/yinyang_experiment.py`
- Replace: `Code/train_yinyang.py`

**Interfaces:**
- Consumes: `TrainingArtifacts`, epoch/evaluation metrics, resolved `YinYangConfig`, and checkpoint payloads.
- Produces: `compact_epoch(record) -> dict`, `write_training_report(path, artifacts, config) -> Path`, `render_training_figure(path, artifacts) -> Path`, `render_spike_rasters(directory, examples) -> tuple[Path, ...]`, and the selected-checkpoint artifact policy.

- [ ] **Step 1: Write failing compact-report tests**

Assert the report contains the complete resolved configuration, epoch-level learning rate/loss/accuracy/silence/restart/cap data, final and test metrics, per-class accuracy, confusion matrices, runtime, and relative artifact names. Assert it excludes batches, sample-index arrays, absolute project paths, and diagnostic tapes.

- [ ] **Step 2: Write failing artifact-policy tests**

For default policy, assert only `best_validation.pt`, `final_tested.pt` when test evaluation is enabled, `metrics.json`, the final training plot, and requested raster plots are created. Assert intermediate checkpoint/plot intervals are opt-in and `0` disables them.

- [ ] **Step 3: Move plotting and serialization functions**

Extract plot, raster, JSON conversion, and atomic-write helpers from the old driver. Make every manifest path relative to the run directory and render only after training unless an explicit interval requests otherwise.

- [ ] **Step 4: Implement selected checkpoint ownership**

Track best validation accuracy during the run, overwrite only `best_validation.pt` on improvement, and write `final_tested.pt` only for the exact state evaluated on the held-out test.

- [ ] **Step 5: Replace the CLI entry point**

Make `train_yinyang.py` load `configs/yinyang_best.json`, apply explicit overrides, call `run_yinyang_training`, and print the compact report path. Keep no model, plotting, or optimizer implementation in the entry point.

- [ ] **Step 6: Run reporting and CLI tests**

Run: `pytest -q Code/tests/test_yinyang_reporting.py Code/tests/test_yinyang_config.py Code/tests/test_yinyang_experiment.py`

Expected: PASS.

---

### Task 5: Replace obsolete research utilities with one opt-in debug module

**Files:**
- Create: `Code/debug.py`
- Create: `Code/tests/test_debug.py`
- Create or modify: `Code/tests/conftest.py`
- Modify: `Code/tests/test_tes_backward.py`
- Delete: `Code/ttfs_landscape.py`
- Delete: `Code/refined_landscape.py`
- Delete: `Code/landscape_diagnosis.py`
- Delete: `Code/continuous_landscape.py`
- Delete: `Code/continuous_gradient_check.py`
- Delete: `Code/continuous_yinyang.py`
- Delete: `Code/gradient_check.py`
- Delete: `Code/benchmark_tes_core.py`
- Delete: `Code/tune_yinyang.py`
- Delete: `Code/test.ipynb`
- Delete: dedicated tests for the removed modules.

**Interfaces:**
- Consumes: public model/config/checkpoint/report APIs only.
- Produces: `check_weight_gradient(config, *, split, sample_index, parameter, index, epsilon) -> GradientCheckResult`, `summarize_result(path) -> dict`, and `inspect_checkpoint(path) -> dict`.

- [ ] **Step 1: Write failing debug utility tests**

Assert the finite-difference checker is deterministic, does not mutate the source model, reports analytic/numeric gradients and relative error, and rejects an index outside the selected weight matrix. Assert result/checkpoint inspectors return concise metadata without dumping tensors.

- [ ] **Step 2: Implement the minimal debug module**

Use one centered finite-difference comparison and one analytic backward pass. Do not retain sweep generation, landscapes, plotting, tuning orchestration, topology taxonomies, or benchmarking frameworks.

- [ ] **Step 3: Move reusable test fixtures out of removed modules**

Put small deterministic model builders in `Code/tests/conftest.py` or the owning test file. In particular, remove `test_tes_backward.py`'s dependency on `refined_landscape.py` without weakening its gradient assertions.

- [ ] **Step 4: Delete obsolete modules and dedicated tests**

Use the explicit file list above. Confirm no production or retained test import references a deleted module.

- [ ] **Step 5: Run debug and numerical-core tests**

Run: `pytest -q Code/tests/test_debug.py Code/tests/test_tes_forward.py Code/tests/test_tes_backward.py Code/tests/test_tes_continuous_backward.py Code/tests/test_tes_continuous_multispike.py Code/tests/test_tes_roots.py Code/tests/test_tes_threshold_semantics.py`

Expected: PASS.

- [ ] **Step 6: Check module boundaries**

Run: `rg 'ttfs_landscape|refined_landscape|landscape_diagnosis|continuous_landscape|continuous_gradient_check|continuous_yinyang|gradient_check|benchmark_tes_core|tune_yinyang' Code --glob '*.py'`

Expected: no imports; only intentional migration notes, if any.

---

### Task 6: Curate retained experiment artifacts and remove generated history

**Files:**
- Create: `results/yinyang_best/config.json`
- Create: `results/yinyang_best/metrics.json`
- Create: `results/yinyang_best/checkpoints/best_validation.pt`
- Create: `results/yinyang_best/checkpoints/final_tested.pt`
- Create: `results/yinyang_best/figures/training.png`
- Create: the three retained raster figures.
- Create: `results/yinyang_experiment_summary.json`
- Delete: all previous content under `Code/output/` and `Figure/` after retained copies and manifests validate.
- Delete: all `__pycache__/`, `.pytest_cache/`, and `.DS_Store` entries in the workspace.

**Interfaces:**
- Consumes: the five 40-epoch modular-loss reports and the canonical `0.1` run's epoch 14/40 checkpoints and four selected figures.
- Produces: the exact `results/` layout and compact manifests described by the design.

- [ ] **Step 1: Build the compact canonical metrics record**

Transform the retained `0.1` report into compact epoch summaries and final metrics without changing values. Record provenance names and measured `0.935` test accuracy; do not recompute or reinterpret it.

- [ ] **Step 2: Build the experiment comparison summary**

Record loss, margin, restart offset, best validation epoch/accuracy, final validation, test accuracy, readout silence, restart epochs, and runtime for hinge offsets `0.1`/`0.25` and DelGrad margins `0.16`/`1.0`/`2.0`.

- [ ] **Step 3: Copy and rename the retained checkpoints and figures**

Copy the canonical run's epoch 14 checkpoint to `best_validation.pt`, epoch 40 tested checkpoint to `final_tested.pt`, and the final/raster plots to the specified stable names. Validate file sizes and checkpoint readability before deleting sources.

- [ ] **Step 4: Validate the retained manifests**

Add tests or a verification command that resolves every relative artifact path and loads both checkpoints with the current schema without mutating an active model.

- [ ] **Step 5: Delete obsolete generated artifacts**

After Step 4 passes, remove only the explicitly resolved `Code/output/` and `Figure/` contents, then remove caches and platform metadata. Recreate no empty legacy output directories unless a compatibility test requires them.

- [ ] **Step 6: Verify disk reduction**

Run: `du -sh results Code Figure 2>/dev/null` and count retained `.pt`, `.json`, and `.png` files.

Expected: two retained checkpoints, compact JSONs, four canonical figures, and a reduction close to the estimated 480–500 MB.

---

### Task 7: Consolidate active documentation and remove stale plans

**Files:**
- Modify: `docs/tes_code_contracts.md`
- Modify: `docs/yinyang_empirical_tuning_guide.md`
- Create: `docs/yinyang_training.md`
- Delete: `docs/oscillator_parameter_migration_debt.md`
- Delete: obsolete files under `docs/superpowers/specs/` and `docs/superpowers/plans/`, preserving this cleanup spec and plan.

**Interfaces:**
- Consumes: final module/config/CLI behavior and the compact experiment summary.
- Produces: one current model contract, one empirical guide, and one short operator guide.

- [ ] **Step 1: Update the code contract**

Document the new module boundaries, configurable dynamics, absence of training early stop, one end-of-epoch finite-weight check, current checkpoint schema, and retained scientific exceptions.

- [ ] **Step 2: Update the tuning guide**

Add the completed hinge and DelGrad campaign results, name the canonical config, replace outdated claims about untested cosine scheduling/nonnegative projections, and retain uncertainty where the experiments do not establish a range.

- [ ] **Step 3: Write the operator guide**

Document the default command, config/CLI precedence, common overrides, resume rules, artifact policy, debug command, and the fact that test evaluation should be requested only for a final run.

- [ ] **Step 4: Remove stale workflow documents**

Delete completed migration debt and old Superpowers plans/specifications only after their still-valid contracts have been incorporated into the three active documents.

- [ ] **Step 5: Validate documentation references**

Run: `rg 'Code/output|Figure/|tune_yinyang|continuous_gradient_check|learning_rate_decay_start_epoch' docs configs README* 2>/dev/null`

Expected: no stale operational reference; historical experiment names may appear only in provenance tables.

---

### Task 8: Full verification, efficiency report, and independent review

**Files:**
- Create: `results/yinyang_cleanup_report.json`

**Interfaces:**
- Consumes: all prior tasks and `results/yinyang_cleanup_baseline.json`.
- Produces: final test evidence, performance/storage comparison, and reviewed production cleanup.

- [ ] **Step 1: Run the retained full test suite**

Run: `pytest -q`

Expected: all retained tests PASS; no known-failure allowance for deleted legacy utilities.

- [ ] **Step 2: Run the golden numerical regression separately**

Run: `pytest -q Code/tests/test_yinyang_numerical_regression.py`

Expected: PASS.

- [ ] **Step 3: Run a small one-epoch smoke experiment**

Use a temporary result directory, reduced data sizes divisible by the configured batch size, one epoch, no intermediate artifacts, and `evaluate_test=False`. Assert all forward calls use `early_stop=False`, the report is compact, and only the selected default artifacts exist.

- [ ] **Step 4: Measure post-cleanup efficiency**

Repeat Task 1's timed small run with identical scientific settings. Record wall time, artifact bytes, file counts, and code/test line counts in `results/yinyang_cleanup_report.json`; state the measurement conditions and avoid extrapolating to a new 40-epoch accuracy.

- [ ] **Step 5: Inspect CLI and retained artifacts**

Run the help command, load `configs/yinyang_best.json`, load both retained checkpoints, resolve every manifest path, and confirm the canonical resolved config equals the preserved experiment config.

- [ ] **Step 6: Request independent code review**

Use the `superpowers:requesting-code-review` workflow. Ask the reviewer to focus on numerical preservation, config completeness, hidden runtime checks, artifact provenance, accidental deletion, and whether tests exercise the five Review Focus items.

- [ ] **Step 7: Resolve review findings and re-run affected verification**

Apply only evidence-backed fixes, run the focused tests for every changed area, then rerun the full retained suite.

- [ ] **Step 8: Deliver the cleanup report**

Report the retained best result without retraining, exact deleted/retained artifact counts, reclaimed disk space, before/after smoke-runtime and output-byte measurements, test results, independent-review conclusion, and any further cleanup opportunities that were deliberately left out of scope.
