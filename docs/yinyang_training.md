# Yin-Yang training operator guide

## Default run

Run the canonical 40-epoch configuration with:

```bash
python -m Code.train_yinyang
```

Defaults come from `configs/yinyang_best.json`. The default result directory is
`results/yinyang_run`. This command does **not** evaluate the held-out test.
Use a distinct `--run-dir` for every experiment so an earlier result is not
overwritten.

Configuration precedence is simple:

1. load the JSON named by `--config`;
2. apply only CLI values explicitly supplied by the caller; and
3. validate and save the complete resolved configuration.

Falsey overrides are real overrides. For example,
`--checkpoint-interval 0`, `--cosine-eta-min 0`, and
`--no-save-rasters` do not fall back to the JSON value.

## Common experiments

A short debugging run that preserves the canonical dynamics is:

```bash
python -m Code.train_yinyang \
  --run-dir results/debug_run \
  --epochs 10 \
  --cosine-t-max 10 \
  --no-save-rasters
```

Examples of focused scientific overrides are:

```bash
python -m Code.train_yinyang --learning-rate 0.008
python -m Code.train_yinyang --hidden-bump-amount 0.15
python -m Code.train_yinyang --loss delgrad --margin 1.0
python -m Code.train_yinyang --d-max 4.5 --c0 18 --c1 1.8
```

Use `python -m Code.train_yinyang --help` for the complete set. Every leaf in
the JSON has an explicit CLI override. Training, validation, and raster
collection always simulate the full requested horizon with
`early_stop=False`.

## Learning per-neuron frequencies

Frequency learning is opt-in, so existing commands retain the original fixed
fast/slow groups. Enable one independently adapted angular frequency per hidden
and output neuron with:

```bash
python -m Code.train_yinyang \
  --run-dir results/learn_omega \
  --learn_omega
```

`--learn-omega` is an equivalent spelling. The resolved `config.json` remembers
this option when reused with `--config`; pass `--no-learn-omega` to override a
saved enabled setting. This mode uses the exact continuous event gradient, a
separate Adam moment stream and step counter, and independent gradient clipping
for omega. The omega learning rate follows the weight cosine schedule with a
`0.01` multiplier, and every update is projected into `[0.10, 0.35] rad/ms`.

The multiplier is deliberately conservative. On the first ten canonical
batches, a `0.1` multiplier moved the most mobile frequency by about `5.9%`;
the selected `0.01` multiplier reduced that to about `0.6%`. This keeps omega
adaptation slow relative to weight learning while still producing a measurable
update in every batch.

The multiplier can be overridden independently with
`--omega-learning-rate-scale`. For example, halving the weight rate while
preserving the original absolute omega rate uses:

```bash
python -m Code.train_yinyang \
  --learn_omega \
  --learning-rate 0.005 \
  --omega-learning-rate-scale 0.02
```

Both values follow the same cosine multiplier, so the absolute omega schedule
remains identical to the default `0.01 * 0.01 = 0.0001` schedule.

## Lightweight in-epoch hidden rescue

The optional deterministic rescue reuses causal hidden activity already
observed during training, so it performs no additional simulations. The
following configuration examines each 10-batch window and adds exactly `0.05`
to every input weight of a hidden neuron only when its silence fraction is
strictly greater than `0.5`:

```bash
python -m Code.train_yinyang \
  --learn_omega \
  --in-epoch-bump-interval 10 \
  --in-epoch-bump-silence-fraction 0.5 \
  --in-epoch-bump-amount 0.05 \
  --hidden-silence-fraction 0.6
```

The original stochastic validation restart remains enabled, but the final flag
raises its strict silence threshold to `0.6`. In-epoch bumps and validation
restarts are both written to `restart_history` with distinct `kind` values.

## Held-out test discipline

Do not add `--evaluate-test` to exploratory runs. Select and freeze a
configuration from validation evidence first, then run the held-out test once:

```bash
python -m Code.train_yinyang \
  --run-dir results/final_run \
  --evaluate-test
```

Only a run that evaluates the test writes
`checkpoints/final_tested.pt`. The measured canonical result already retained
under `results/yinyang_best/` is `0.935`; cleanup did not retrain or recompute
that value.

## Resume rules

Resume with:

```bash
python -m Code.train_yinyang \
  --run-dir results/resumed_run \
  --resume-checkpoint results/partial_run/checkpoints/epoch_020.pt \
  --epochs 40
```

The resolved configuration must match the checkpoint before weights or Adam
state are restored. The usual allowed difference is a larger total epoch
budget. This is useful when the original run was created with, for example,
`epochs=20` and `cosine_t_max=40`, then extended to 40 epochs. An explicit
`--learn_omega` may also upgrade a weight-only or schema-v4 checkpoint: the
saved frequencies are preserved and their Adam moments start at zero. The
loss, dynamics, data, seeds, initialization, weight optimizer, cosine horizon,
restart, and artifact settings must otherwise be identical.

## Artifact policy

An ordinary run writes:

```text
run_dir/
  config.json
  metrics.json
  checkpoints/best_validation.pt
  figures/training.png
```

Rasters are added when `save_rasters` is enabled. `final_tested.pt` is added
only after an explicit test evaluation. Intermediate checkpoints and plots are
opt-in through positive `--checkpoint-interval` and `--plot-interval` values;
zero disables them.

`metrics.json` contains resolved configuration, calibration, compact epoch
summaries, causal and full-horizon silence, restart identities, per-class
accuracy, confusion matrices, final metrics, runtime, relative artifact paths,
and the final omega range and values. It deliberately omits batch histories and
sample event tapes. Current checkpoints retain omega plus its Adam moments and
step counter.

## Debugging and inspection

Inspect a compact report or checkpoint without dumping tensors:

```bash
python -m Code.debug result results/yinyang_best/metrics.json
python -m Code.debug checkpoint results/yinyang_best/checkpoints/final_tested.pt
```

Check one analytic gradient against a centered finite difference:

```bash
python -m Code.debug gradient \
  --parameter W_out \
  --row 0 \
  --column 0 \
  --sample-index 0 \
  --epsilon 1e-5
```

Run the topology-aware omega loss-landscape test on the retained best
checkpoint:

```bash
python -m Code.omega_landscape
```

The default diagnostic sweeps hidden neuron 11 and output neuron 0 (global
neuron 12) around their operating frequency on training sample 4. It treats
the continuous event solution as the tangent test and overlays a discrete
simulation at `dt=0.005 ms` using the same weights and input sample, with input
delivery requantized onto that grid. The JSON records and PNG plots are written
to `results/omega_landscape/`; each JSON record includes the validated
checkpoint epoch and SHA-256 digest plus the config, split, and sample identity.

The production hot path checks finite weights once at the end of each epoch.
Root ambiguity and unresolved-crossing exceptions remain scientific errors and
must be diagnosed rather than suppressed.
