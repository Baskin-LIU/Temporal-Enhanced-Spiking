# TeS core code contracts

This document is the input contract for the consolidated feed-forward TeS core in
`Code/neurons.py`, `Code/tes_events.py`, and `Code/tes_roots.py`. The hot loops
assume these contracts are obeyed. Except for weight-shape consistency, causal
early-stop safety, and scientifically ambiguous root cases, malformed inputs are
caller errors and are deliberately not checked at runtime.

## Numerical model

Each neuron has the float64 state `(V, I, Q)` and, between events, follows

```text
tau_v dV/dt = -V + I
dI/dt       = -alpha I - omega Q
dQ/dt       =  omega I - alpha Q
```

An arrival with synaptic weight `w` applies `Q <- Q - w`. A threshold event at
`V = threshold` applies the subtractive reset `V <- V - reset_drop`; `I` and `Q`
are continuous. A hidden threshold emits one delayed event with

```text
delay = d_plus * sigmoid(c1 - c0 * slope)
slope = (-V + I) / tau_v
```

There is no bias and no fifth input channel. Frequencies are initialized in
contiguous groups: the first `ceil(n_hidden/2)` hidden neurons are fast and all
remaining hidden neurons are slow; every readout is slow. The original
forward/backward APIs keep this fixed grouped-frequency contract. The separate
`*_omega` APIs use and differentiate every entry of `omega` independently, so
callers may adapt neurons at different rates. All forward solvers allow any
neuron to spike repeatedly. The network is strictly feed-forward: input to
hidden to readout.

## Configuration and model attributes

### `TeSConfig`

The numerical record is immutable but its scientific constants are
configurable. Production experiments construct it through
`YinYangConfig.to_tes_config()`, after the complete nested configuration has
been validated once at startup. Direct low-level callers are responsible for
supplying meaningful finite values; the record itself retains only the
frequency and reciprocal-slope-cap guards needed by the solver.

| Attribute | Type | Meaning / value |
|---|---:|---|
| `dt` | `float` | Discrete grid interval in ms; arbitrary positive value selected at initialization. |
| `tau_v` | `float` | Voltage time constant; canonical value `20.0` ms. |
| `alpha` | `float` | Oscillator decay; canonical value `1/40 = 0.025 ms^-1`. |
| `threshold` | `float` | Voltage threshold; canonical value `1.0`. |
| `reset_drop` | `float` | Subtractive voltage reset; canonical value `5.0`. |
| `fast_period_ms` | derived property | `2*pi/omega_fast`; canonical value `8*pi ~= 25.1327` ms. |
| `slow_period_ms` | derived property | `2*pi/omega_slow`; canonical value `12*pi ~= 37.6991` ms. |
| `omega_fast` | `float` | Fast angular frequency; canonical value `1/4` radians/ms. |
| `omega_slow` | `float` | Slow angular frequency; canonical value `1/6` radians/ms. |
| `d_plus` | `float` | Maximum hidden-to-readout delay in ms. |
| `c0`, `c1` | `float` | Slope-to-delay logistic parameters. |
| `slope_epsilon` | `float` | Scientific positive-slope tolerance, `1e-12`. |
| `threshold_sensitivity_cap` | `float` | Positive finite clamp on `1/(tau_V * dV/dt)`; low-level default `10.0`, canonical training value `2.0`. |
| `silent_target_gap_ms` | `float` | Effective-time gap after the latest physical readout for a silent target; canonical value `2.0` ms. |

### `TeSNeurons`

`TeSNeurons(n_in, n_hidden, n_out, w_in, w_out, config)` requires weights with shapes
`W_in=(n_hidden,n_in)` and `W_out=(n_out,n_hidden)`. The combined shape check is
the retained construction-time guard. Production configuration validation owns
the layer-size and initialization-value checks.

Public attributes are:

| Attribute | Type / shape | Contract |
|---|---|---|
| `n_in`, `n_hidden`, `n_out` | `int` | Layer sizes. |
| `n_neurons` | `int` | `n_hidden + n_out`; global neuron indices put hidden neurons first. |
| `config` | `TeSConfig` | Immutable numerical configuration. |
| `omega` | float64 tensor `(n_neurons,)` | Per-neuron angular frequencies, initialized from the two configured groups. The `*_omega` APIs use the current entries independently. |
| `W_in` | float64 tensor `(n_hidden,n_in)` | Detached cloned input weights; not a trainable `Parameter`. |
| `W_out` | float64 tensor `(n_out,n_hidden)` | Detached cloned readout weights. |
| `dCdW_in`, `dCdW_out` | float64 tensors matching weights | Most recently computed gradients; zero after `reset`. |
| `state` | float64 tensor `(n_neurons,3)` | Current discrete state ordered as `V,I,Q`. |
| `V`, `I`, `Q` | float64 tensor views `(n_neurons,)` | Views of the three state columns. |
| `arrival_records` | list of `DiscreteArrival` | Current discrete run's delivered arrivals. |
| `threshold_events` | list of `DiscreteThreshold` | Current discrete run's thresholds. |

Internal cached attributes (`_inv_tau_v`, `_inv_dt`, frequency index tuples,
analytical-flow coefficients, curvature constants, root-panel widths, adjoint
mass matrices, and one-step forward/adjoint transitions) are derived once at
construction. `_queue` and `_sequence` are reset for each discrete run.

All numerical tensors are float64. Construct on CPU, or move the complete module
to one device before use; weights, state, inputs used in low-level methods, and
adjoints must not be split across devices.

## Public methods

### Forward, loss, and backward

`forward_discrete(inputs, t_end, *, early_stop=False, record_state=True)` takes
an ordered sequence of `InputSpike` and a finite `t_end >= 0` in ms. `grid_step`,
not `exact_time`, controls delivery. Due arrivals are accumulated into one jump
before exact analytical flow over each grid interval. A crossing is detected
from below across the interval and recorded at its right endpoint; no root
interpolation is done. A hidden delay is quantized with `ceil(delay/dt)`. With
`record_state=False`, `times`, `V`, `I`, and `Q` are `None`; event and backward
tapes remain complete. Returns `DiscreteForwardResult`.

`forward_continuous(inputs, t_end, *, early_stop=False)` uses `exact_time`, exact
analytical flow, and certified first-upward-root isolation. Same-time arrivals
are summed. A threshold exactly at an arrival boundary is recorded and reset
before the arrival jump. Hidden streams are independent, their emitted arrivals
are merged deterministically, and readout streams are then solved independently.
Returns `ContinuousForwardResult`.

`forward_discrete_omega(inputs, t_end, *, early_stop=False)` uses the current
per-neuron `omega` entries and always retains dense float64 `I,Q` histories;
`V` is omitted. `forward_continuous_omega(inputs, t_end, *, early_stop=False)`
uses the current per-neuron entries and retains only post-event oscillator
states as `ContinuousOmegaCheckpoint` records. The continuous omega path
currently requires `early_stop=False`, because its sparse gradient tape is
defined over the requested full horizon. Both methods reject nonpositive or
nonfinite learned frequencies.

Timing objectives live in `Code/tes_loss.py` and consume only the immutable
`EffectiveReadout` records, label, objective name, and optional margin. The
`delgrad` objective defaults to `margin=1.0 ms` and adds `0.5*e_j^2` for every
physical wrong output. The separately selectable `squared_hinge` objective
defaults to `margin=2.0 ms` and adds that term only when `e_j < 0`, where
`e_j = t_j - t_label - margin`. `TeSNeurons.delgrad_loss` and
`TeSNeurons.squared_hinge_loss` are compatibility wrappers around the module;
training and evaluation select the module through `TrainingConfig.loss_name`.

For both objectives, silent wrong outputs are excluded. A silent target is assigned
`latest_physical + 2 ms`; physical wrong
outputs still receive loss against that clipped target, and the target-time
derivative is routed through the unique latest physical output. If the latest
physical time is tied and loss-bearing, `output_time_grad` is `None` and
`nondifferentiable_reason` explains why. If every readout is silent, loss and
gradient are exactly zero. Fake readouts never receive a gradient. Returns
`DelGradLoss` as the common timing-loss result record.

`backward_discrete(forward, output_time_grad)` and
`backward_continuous(forward, output_time_grad)` require a float-like tensor of
shape `(n_out,)` associated with physical first-readout times. They return
`BackwardResult`, update `dCdW_in/dCdW_out`, sum all repeated uses of a shared
weight, and reverse only to the latest physical first readout. All-silent trials
return exact zero weight gradients. Continuous reverse uses exact parent IDs and
the full subtractive-reset adjoint, including a nonzero post-reset adjoint.
Later readout spikes are retained by full forward mode but do not enter the
first-spike objective.

`backward_discrete_omega(forward, output_time_grad)` returns one frequency
gradient per neuron by midpoint integration of `lambda_I*Q-lambda_Q*I` over
the stored grid trajectory. `backward_continuous_omega(forward,
output_time_grad)` evaluates the same integral exactly on each event-free
interval using the stored left-boundary `I,Q` pair and the right-boundary
adjoint. Both return `OmegaBackwardResult`, which preserves the ordinary
weight/timing gradients and adds `grad_omega (n_neurons,)`. They do not mutate
`omega`; the caller owns any separately scaled frequency update. The canonical
Yin-Yang trainer provides that update behind `--learn_omega`: it uses a separate
Adam stream at `0.01` times the weight learning rate, clips its gradient
independently, and projects every neuron into `[0.10, 0.35] rad/ms`.

Both backward methods evaluate every reciprocal threshold factor through
`min(1 / (tau_v * slope), threshold_sensitivity_cap)`, including the physical-output seed, reset
transport, timing source, and delay-acceleration correction. This is a deliberate
cap-active approximation: the analytical adjoint remains exact only while the
cap is inactive. `BackwardResult.threshold_sensitivity_cap_activations` counts
traversed thresholds whose uncapped sensitivity exceeded the configured cap, and
`BackwardResult.minimum_causal_slope` reports the smallest slope traversed by
that backward call. The configured `slope_epsilon` is separately passed to root
isolation and remains a crossing-classification tolerance, not a gradient cap.

At coincident times, the forward ordering is threshold/reset then arrival jump;
reverse ordering is therefore arrival jump then threshold. An arrival exactly at
an output's first threshold is noncausal for that threshold. Events strictly
after the latest physical first readout do not contribute to gradients.

### Early stopping

Both forwards accept `early_stop=True`. Stopping is allowed after all readouts
have physical first spikes, or after every unresolved readout is certified
silent. The silence certificate is strict:

```text
max(V, hypot(I,Q)) < threshold
```

It may be used for hidden neurons only after all external inputs are delivered,
and for readouts only after every hidden stream is closed and all scheduled
hidden arrivals are drained. Violating this causal prerequisite raises the
retained internal `RuntimeError`. Early and full modes must have identical
effective readouts, loss, and gradients. `StopInfo.actual_t_end` reports the
physical/certification terminal; backward still terminates at the latest
physical first readout, never at a fake or certification time.

### Low-level/diagnostic methods

These methods expose the same core without extra validation:

| Method | Input / return contract |
|---|---|
| `system_matrix(omega)` | Returns the float64 `3x3` generator on the state device. |
| `flow_state(state, delta)` | Exact batched flow for `(n_neurons,3)` over finite `delta >= 0`; returns a new tensor. |
| `reset()` | Zeros state/gradient buffers and creates empty discrete tapes/queue. |
| `delay_from_slope(slope)` | Returns the logistic delay in `(0,d_plus)` for finite slope. |
| `quantize_delay(delay)` | Returns integer `ceil(delay/dt)`. |
| `enqueue_input(spike)` | Adds a contract-valid input to the discrete queue. |
| `apply_due_arrivals(step)` | Applies all queued arrivals whose step equals `step`, batching each layer's jump. |
| `step(step, delta=None)` | Applies due arrivals, flows once, handles every endpoint crossing/reset/emission, and returns the post-arrival/pre-flow state. |

Callers using `step` directly must choose `dt`/`reset_drop` so a reset leaves
`V < threshold`; this is no longer checked in the hot loop.

## Record contracts

All records are frozen dataclasses. Tuples in result records are immutable tapes.
Indices are zero-based.

### Inputs and arrivals

| Record | Fields and invariants |
|---|---|
| `InputSpike` | `exact_time: float` in ms; `grid_step: int >= 0`; `channel: int` in `[0,n_in)`. Continuous mode uses `exact_time`; discrete mode uses `grid_step`. |
| `DiscreteArrival` | `event_id` is its index in `arrivals`; `time=step*dt`; `step`; `kind` is `input` or `hidden`; `source` indexes input channel or hidden neuron; `exact_time` preserves input time or unquantized hidden arrival; hidden records have `parent_threshold_id`, inputs have `None`. |
| `ContinuousArrival` | `event_id` is list-local; `sequence` is the deterministic tie-break; exact `time`; `kind/source`; every hidden arrival has the global parent threshold ID. |

### Thresholds and readouts

| Record | Fields and invariants |
|---|---|
| `DiscreteThreshold` | Global `event_id`; endpoint `time` and `step`; global `neuron`; positive `slope`; pre-event `v_pre/i_pre/q_pre`; `v_post=v_pre-reset_drop`. Hidden events have `continuous_delay`, integer `delay_steps`, and quantized `arrival_time`; readout fields are `None`. |
| `ContinuousThreshold` | Globally ordered `event_id`; `layer`; layer-local `neuron`; exact `time`; positive `slope`; `acceleration_pre`; pre/post state; hidden-only exact delay/arrival. Global ordering is `(time, hidden-before-output, neuron, local sequence)`. |
| `EffectiveReadout` | Output `neuron`; optional first physical threshold ID/time; always-finite `effective_time`; `censored` means fake; `anchor_output` identifies the latest physical source for a fake time, or is `None`. |
| `StopInfo` | `reason` is `all_readouts_physical`, `all_readouts_resolved`, `all_readouts_silent`, or `requested_horizon`; requested/actual end times; hidden/output flow interval and root-panel work counters. |

### Results

| Record | Fields and invariants |
|---|---|
| `DiscreteForwardResult` | Optional `times` `(T,)`; optional `V/I/Q` `(T,n_neurons)`; arrival and threshold tapes; exactly `n_out` readouts; warning tuple; `StopInfo`. State histories are all present or all `None`. |
| `ContinuousForwardResult` | Exact input-arrival, hidden-threshold, hidden-arrival, and output-threshold tapes; exactly `n_out` readouts; `StopInfo`. Full mode retains repeated spikes through `t_end`; early mode retains the causal prefix only. |
| `ContinuousOmegaForwardResult` | Extends `ContinuousForwardResult` with sparse post-event `omega_checkpoints`; each checkpoint stores global neuron, time, `I`, and `Q`. |
| `DelGradLoss` | Scalar `loss`; `effective_times` and `physical_mask` of length `n_out`; physical `active_wrong_outputs`; optional fake anchor; optional float64 `output_time_grad (n_out,)`; optional nondifferentiability reason. |
| `BackwardResult` | Copied `output_time_grad (n_out,)`; `hidden_spike_time_grad (number of hidden thresholds,)` in hidden-tape order; `grad_W_in` and `grad_W_out` matching weight shapes; `threshold_sensitivity_cap_activations`; and optional `minimum_causal_slope`. |
| `OmegaBackwardResult` | Extends `BackwardResult` with independent `grad_omega (n_neurons,)`. |

### Root isolation and internal scheduling

`UpwardCrossing` contains root `offset`, float64 three-state `state`, positive
`slope`, signed `residual`, `time_uncertainty`, certified
`residual_tolerance`, and `panels_visited`. `first_upward_crossing` returns the
first certified upward root in `[0,horizon]` or `None`. It deliberately raises
`GrazingCrossingError` for an unresolved stationary contact and
`UnresolvedCrossingError` when subdivision or residual certification cannot
resolve the event; these failures must not be hidden. Its optional
`panel_counter` is a one-element mutable integer list initialized by the
caller. Every visited isolation panel increments that element, including when
the search returns `None`; `TeSNeurons` uses it to make `StopInfo.root_panels`
a complete work counter rather than a successful-root-only count.

`ScheduledArrival` is an internal frozen priority-queue record ordered only by
`step` then `sequence`; `kind`, `source`, `exact_time`, and parent ID do not
affect queue order.

## Yin-Yang proof-of-learning training contract

The active module boundaries are:

- `Code/yinyang_config.py`: nested typed configuration, JSON loading, startup
  validation, and explicit CLI overrides;
- `Code/yinyang_experiment.py`: deterministic initialization, readout
  calibration, epoch orchestration, restart decisions, and resume behavior;
- `Code/tes_training.py`: data splits, evaluation, batching, Adam, and the
  hidden-row restart primitive;
- `Code/yinyang_reporting.py`: selected checkpoints, compact JSON, training
  plots, and spike rasters; and
- `Code/train_yinyang.py`: a small executable adapter only.

Initialization, calibration, training, validation, rasters, and the optional
single final test evaluation all use continuous full-horizon calls with
`early_stop=False`. This is an invariant rather than a configuration switch.
The canonical deterministic split is 5,000/1,000/1,000; Adam uses batch size
50, float64 moments outside autograd, and one combined gradient-norm cap. Both
weight matrices start positive but are signed and unconstrained thereafter.

Initialization compensates for the changed first positive lobe using the
closed-form unit-weight PSP

```text
V(t) = exp(-alpha*t) *
       ((beta-alpha)*sin(omega*t) - omega*cos(omega*t)
        + omega*exp(-(beta-alpha)*t))
       / (tau_v * ((beta-alpha)^2 + omega^2)),  beta = 1/tau_v.
```

The historical vanilla exponential-current reference has `tau_v=20 ms`,
`alpha_ref=0.1 /ms`, and unit peak `0.25`. Deterministic first-lobe calibration
gives fast `(t_peak=11.1676331128 ms, peak=0.2591420990, gain=0.9647216756)`
and slow `(t_peak=15.8712273368 ms, peak=0.3202711116, gain=0.7805886666)`.
The analytical gain is applied by postsynaptic group before the optional
fast-hidden multiplier. Default readout target row sums receive the slow gain
and are normalized to the configured `readout_reference_hidden` width, so the
total initial excitatory row sum remains comparable when hidden width changes.
Explicit target row sums override that automatic scaling. Coverage calibration
then multiplies each failing readout row in place until the configured training
coverage is reached or the round budget is exhausted.

The 50 ms trial horizon contains only a finite portion of each response: it
includes the complete first positive lobe for both groups but cannot establish
benefits from arbitrarily late lobes. Claims about extended temporal
expressiveness must therefore be supported by event timing within this window.

Prediction uses physical first-spike times only. A uniquely earliest physical
correct output wins even when incorrect outputs are silent. All-output silence,
a silent target defeated by a physical output, and an exact earliest-time tie
are incorrect; ties are not assigned to class zero.

After each validation, a hidden input row is stochastically restarted only if
it is silent on more than 500 of the 1,000 validation samples. Exactly 500 is
not an intervention. The restart adds an independent standard-normal draw plus
the configured positive offset to each of the row's four weights. The draw uses
a recorded deterministic epoch seed, is not sign-clamped, and does not change
any Adam moment or step. Readout coverage calibration uses training inputs only;
configuration selection never uses the test split. Causal hidden activity is
measured through the latest first threshold across all readouts; when not every
readout crosses, the full 50 ms horizon is the cutoff. A later hidden spike is
retained in full-horizon diagnostics but counts as silent for restart purposes.

An opt-in in-epoch rescue uses the same causal definition without another
forward pass. It accumulates per-neuron silence over a configured number of
complete training batches and adds a fixed positive offset to every incoming
weight only when silence is strictly above the configured fraction. The
canonical lightweight experiment uses 10 batches, fraction `0.5`, and offset
`0.05`; its separate validation restart uses fraction `0.6`.

The only production finite check is performed once after every completed
training epoch and before validation or artifact writing. Adam deliberately
contains no per-update finite-gradient or finite-weight scan. Configuration is
validated once at startup, weight shapes at construction, checkpoint
compatibility before restoring state, and scientifically ambiguous roots at
the point of isolation.

Ordinary output is compact: `config.json`, `metrics.json`, the best-validation
checkpoint, a final-tested checkpoint only when the held-out test was actually
evaluated, one final training plot, and optional rasters. Intermediate
checkpoint and plot intervals are explicit and default to zero. Reports retain
epoch loss/accuracy, silence, class metrics, confusion matrices, restart
identities, cap activations, runtime, and relative artifact paths, but omit
batches, sample-index arrays, and full diagnostic event tapes.

Current-schema checkpoints store the resolved scientific configuration,
weights, Adam state, calibration and restart history, compact completed epochs,
and the best-validation snapshot. Resume permits only a larger total epoch
budget; every other resolved field must match. This comparison occurs before
the target model or optimizer is mutated. Detailed finite-difference and file
inspection work is opt-in through `Code/debug.py`, outside the training hot
path.

## Reviewer checklist

The following rows were compared with the live signatures and dataclass fields
after consolidation.

| API / record | Parameters or fields checked | Semantics checked | Status |
|---|:---:|:---:|---|
| `TeSConfig` | yes | yes | checked 2026-10-03 |
| `TeSNeurons.__init__` and attributes | yes | yes | checked 2026-09-21 |
| `forward_discrete` | yes | yes | checked 2026-09-21 |
| `forward_continuous` | yes | yes | checked 2026-09-21 |
| modular timing losses and compatibility wrappers | yes | yes | checked 2026-10-02 |
| `backward_discrete` | yes | yes | checked 2026-09-22 |
| `backward_continuous` | yes | yes | checked 2026-09-22 |
| Low-level/diagnostic methods | yes | yes | checked 2026-09-21 |
| `InputSpike` | yes | yes | checked 2026-09-21 |
| `DiscreteArrival` | yes | yes | checked 2026-09-21 |
| `DiscreteThreshold` | yes | yes | checked 2026-09-21 |
| `ContinuousArrival` | yes | yes | checked 2026-09-21 |
| `ContinuousThreshold` | yes | yes | checked 2026-09-21 |
| `EffectiveReadout` | yes | yes | checked 2026-09-21 |
| `StopInfo` | yes | yes | checked 2026-09-21 |
| `DiscreteForwardResult` | yes | yes | checked 2026-09-21 |
| `ContinuousForwardResult` | yes | yes | checked 2026-09-21 |
| `DelGradLoss` | yes | yes | checked 2026-09-21 |
| `BackwardResult` | yes | yes | checked 2026-09-22 |
| `ScheduledArrival` | yes | yes | checked 2026-09-21 |
| `UpwardCrossing` and root errors | yes | yes | checked 2026-09-21 |
| Yin-Yang config/orchestration/reporting/checkpoints | yes | yes | checked 2026-10-03 |
