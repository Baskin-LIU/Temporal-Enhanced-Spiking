# Empirical Hyperparameter Guide for TeS Yin–Yang Training

## Purpose and limits

This note records the practical lessons from the first continuous TeS Yin–Yang
training campaign. Its purpose is to narrow future searches and to make a
failed run diagnostically useful, rather than to claim universal optimal
hyperparameters.

The empirical base has two regimes. The historical broad-search regime contains:

- a 4×3 gain-by-heterogeneity factorial readout screen with 12 initializations;
- a 36-candidate, 4-epoch Latin-hypercube search;
- 10 candidates continued to 10 epochs;
- a 12-candidate, 4-epoch local refinement;
- 5 local candidates continued to 10 epochs; and
- 3 finalists run on 3 seed bundles.

This produced 72 reports. Those searches used validation only and kept the test
split sealed. They apply most directly to the 4-input, 12-hidden, 3-readout
network with projected nonnegative weights and 10 constant-rate epochs.
The recorded campaign used the superseded `alpha=0.1 /ms`, `15/25 ms`
oscillator periods, and equality-to-margin timing loss. The current training
code instead uses `alpha=0.025 /ms`, `8pi/12pi ms` periods, signed weights after
initialization, stochastic hidden-row restarts, and a modular timing-loss layer.
It can select original DelGrad or the strict squared hinge without changing
neuron dynamics.

The active regime now also has five completed 40-epoch cosine runs with signed
weights and restart offset `0.1` or `0.25`: two squared-hinge runs and DelGrad
with margins `0.16`, `1.0`, and `2.0`. Their measured headline results are
preserved in `results/yinyang_experiment_summary.json`. These five runs are
enough to choose a practical default, but not enough to establish independent
effects for every parameter.

Ranges below are **starting envelopes**, not confidence intervals. A range is
called "acceptable" when it contained several healthy runs or is tightly
centered on the most useful local configurations. Mixed-search correlations
are suggestive because several parameters changed simultaneously.

## Start here

For the present 4–12–3 network, a sensible next configuration is:

| Quantity | Recommended starting value or range |
|---|---:|
| Adam learning rate | `0.01`, cosine-annealed over 40 epochs to `0` |
| Batch size | `50` (fixed by the proof-of-learning protocol) |
| Hidden restart offset | `+0.10` with unit-normal noise |
| Readout row-sum gain | `0.93–1.00` times the reference row sums |
| Readout row sums at gain 1 | `(24.72, 27.68, 24.89)` |
| Readout lognormal heterogeneity, `sigma` | `0.65–0.80` |
| Maximum delay, `d_plus` | `4.3–5.0 ms` |
| Delay sharpness, `c0` | `18–22 ms/V` |
| Delay midpoint slope, `c1/c0` | `0.09–0.105 V/ms` |
| Corresponding `c1` | normally `1.6–2.3`, computed as `c0 × midpoint` |
| Fast-hidden input gain | `1.0`; historical screen suggests `1.0–1.2` if activity is imbalanced |
| Readout coverage target | at least `0.99` for every output |

Do not sweep all of these again. First establish activity, then tune the
optimizer, and only then tune timing:

1. Screen the untrained model for hidden and readout activity.
2. Reject or repair initializations with readout coverage below 99%.
3. Run 4 epochs to reject under-learning, class collapse, or instability.
4. Continue only a small frontier to 10 epochs.
5. Compare finalists on complete seed bundles, including fresh seeds not used
   for selection.
6. Freeze the configuration before evaluating the held-out test exactly once.

The current canonical run reached `0.947` best validation accuracy, `0.929`
final validation accuracy, and `0.935` on the held-out test. It used squared
hinge with margin `2.0`, restart offset `0.1`, and the 40-epoch cosine schedule.
This is a single fixed seed bundle, so it establishes a working reference, not
seed robustness. The `0.25` restart run reached a higher validation peak
(`0.955`) but lower held-out test accuracy (`0.920`).

## What to monitor in every run

Accuracy alone is too late and too ambiguous. Retain these diagnostics:

- per-output readout coverage before training;
- all-silent and target-silent sample counts;
- causal hidden silence count for every hidden neuron, where a neuron is
  silent unless it spikes by the last output-threshold time (or by `50 ms`
  when not every output spikes), plus the separate full-horizon silence count;
- restarted-row count and identity, random seed, and exact perturbation per epoch;
- validation accuracy immediately before and after every hidden restart;
- per-class accuracy and the confusion matrix;
- median and spread of pairwise first-readout time gaps;
- epoch of peak validation accuracy and mean accuracy over the final 3 epochs;
- gradient-clipped batch fraction and signed `W_in`/`W_out` ranges;
- threshold-sensitivity cap activations and minimum causal crossing slope;
- delay values or sigmoid fractions, so delay saturation is visible; and
- calibration rounds and final row sums.

A healthy run usually has one-round readout calibration, nearly complete
readout coverage, no persistent all-silent population, no class with collapsed
accuracy, and a validation curve that does not fall sharply after its peak.
For model selection, use the final-three-epoch mean first, then worst-seed and
worst-class accuracy. A single favorable final epoch is not sufficient.

Across the nine 10-epoch finalist/seed runs, reference operating ranges were
`0–0.8%` clipped batches, `1.68–5.30` threshold-sensitivity cap activations per
1,000 training samples, and `1.0–6.2` projected weights per batch. Both healthy
and collapsed runs occurred inside these ranges. Nonzero counts are therefore
not failures by themselves in the historical projected regime. Projection
counts no longer exist in signed training; inspect weight ranges instead.

## Parameter guide

### 1. Adam learning rate

**Acceptable starting range.** For batch size 50, start at `0.009–0.012`. The
canonical active configuration uses `0.01` with cosine annealing over 40 epochs
and a zero minimum. For a 10-epoch constant-rate diagnostic, the same starting
range remains the best supported historical envelope.

**Observed behavior.** Learning rate was the clearest driver in the broad
4-epoch search. Spearman correlation with final-three accuracy was `+0.889`:
the lowest third (`0.0021–0.0036`) averaged about `0.481`, while the highest
third (`0.0066–0.0119`) averaged about `0.690`. Rates below about `0.006` were
mostly too slow for a 4-epoch screen.

The 10-epoch runs show the other side of the trade-off. In the
single-default-seed local search, rates near `0.014–0.018` learned quickly, but
several runs peaked around epochs 3–6 and then lost validation accuracy. Only
one high-rate candidate (`local_000`) was tested across multiple seeds, so a
causal link between high LR and seed sensitivity remains untested. Across the
three finalists, mean validation accuracy was generally best around epoch 4–7,
which motivates testing decay rather than adding more constant-rate epochs.

**What the active runs establish.** All five completed active-regime runs used
the 40-epoch cosine schedule and remained numerically trainable. The schedule
therefore has direct feasibility evidence, and the best run used it. There is
no matched constant-rate control under the same dynamics, so these results do
not isolate how much of the improvement came from decay. The obsolete one-time
step schedule has been removed rather than retained as a second tuning axis.

**If increased:** faster early learning, but more overshoot, large signed weights,
late-epoch degradation, readout silence, and seed sensitivity.

**If decreased:** smoother changes and potentially better late stability, but
clear under-training within 10 epochs.

**Tune it when:**

- increase it if epochs 1–4 improve monotonically, gradients are finite, and
  the run remains far below the better candidates;
- decrease the constant rate if accuracy peaks early and then falls, target
  silence grows, a class collapses, or signed weight magnitudes grow rapidly;
- treat persistent gradient clipping as an instability symptom, but remember
  that Adam's learning rate does not directly change the raw gradient norm.

There is no validated width-scaling law yet. For a larger network, begin with
the same or a slightly smaller rate and use clipping, weight ranges, silence, and
the early validation curve as the decision signals.

### 2. Hidden silence restart

The current rule restarts a hidden row only when it is silent on **more than**
half of the validation samples. Each selected weight receives
`Normal(0, 1) + b`, where `b` is the configured restart offset. The signed result
is retained and Adam state is left untouched.

**Current starting point.** Bias `b=0.10` with unit-normal noise. The two
matched squared-hinge runs support `0.10` over `0.25` as the practical default:
test accuracy was `0.935` versus `0.920`, despite the larger offset producing a
higher one-epoch validation peak. This comparison is too small to define a
smooth acceptable range.

**Observed behavior.** Larger bumps reliably rescued more silent trials, but
did not improve short-run accuracy by themselves. Across the broad search,
bump amount had essentially zero correlation with 4-epoch accuracy. A bump
usually caused an immediate validation penalty: about `-2.1` percentage points
on average across broad runs with bump events. In the local search, amounts
around `0.21–0.25` had the smallest mean immediate penalty (about `-0.3` point),
while the `0.27–0.31` group lost about `-1.8` points. Values near `0.4–0.5`
rescued more silence per intervention but sometimes caused 7–10 point drops.

**If increased:** fewer silent trials immediately, fewer repeated small rescue
steps, but a larger discontinuity in the representation and stronger chance of
destroying a useful decision boundary.

**If decreased:** gentler post-bump behavior, but the same row may be bumped in
many consecutive epochs without becoming reliably active.

**Tune it when:** compare validation immediately before and after the bump. A
good bump materially lowers that row's silence while losing no more than about
1–2 validation points, followed by recovery during the next epoch. Repeated
large accuracy losses mean the amount is too high. Repeated bumps with little
silence reduction mean it is too low or the row's initialization/dynamics are
the real problem. Track simultaneous multi-row bumps and cumulative addition
per row: repeated multi-row interventions indicate a network-wide activity
imbalance, not merely one underpowered neuron.

**Input-size scaling.** Preserve the row-sum intervention rather than the
per-weight increment. The present intervention is

`expected_offset_row_sum = 4 × 0.10 = 0.40`.

For `n_in` input channels, begin with `offset ≈ 0.40 / n_in`. Revalidate this if
the input encoding changes its amplitude or sparsity.

### 3. Readout weight initialization and row-sum gain

The explicit initializer normalizes each readout row to a target sum. The
reference sums are `(24.72, 27.68, 24.89)` for 12 hidden neurons. A scalar
gain multiplies all three sums.

**Acceptable starting range.** Gain `0.93–1.00`, giving approximate first-row
sums of `23.0–24.7`. `local_000` used gain `0.9565`, or row sums
`(23.64, 26.47, 23.81)`, and required only one calibration round.

**Observed behavior.** Explicit row sums reduced the legacy initialization's
19–23 calibration rounds to one. In the factorial initialization screen, every gain from
`0.90` to `1.15` produced high activity when heterogeneity was mild. At
`sigma=0.5`, even gain `0.90` gave per-output coverage above `99.2%`. Higher
gain further increased coverage but compressed timing differences: at
`sigma=0.5`, the median pairwise gap fell from about `0.596 ms` at gain `0.90`
to `0.538 ms` at gain `1.15`. Gains above `1.0` did not show a consistent
10-epoch advantage.

**If increased:** readouts spike earlier and more reliably, calibration becomes
easier, but readout times crowd together and the timing classifier can lose
contrast.

**If decreased:** larger timing separation and less excessive activity, but
more output silence, fake target times, and calibration work.

**Tune it when:**

- increase only the failing row if its coverage is below 99%;
- decrease a row if it spikes on essentially every sample much earlier than
  the others and pairwise gaps are compressed;
- do not use training accuracy to compensate for a silent initializer—repair
  activity before optimization.

When explicit target row sums are supplied, the older per-row scale is
normalized away and is not a useful independent tuning knob.

**Hidden-width scaling.** Let `p_active` be the fraction of hidden neurons that
deliver a useful spike. Expected delivered drive is roughly proportional to
`p_active × row_sum`. Therefore use

`new_row_sum ≈ reference_row_sum × reference_p_active / new_p_active`.

If the active fraction is unchanged, keep the total row sum approximately
constant as hidden width changes; the mean individual weight then scales as
`1 / n_hidden`. Do not multiply the row sum by width automatically.

### 4. Readout heterogeneity `sigma`

Readout weights receive lognormal factors and are then renormalized to the
requested row sum. Thus `sigma` changes concentration and pathway diversity,
not total drive.

**Acceptable starting range.** `0.65–0.80`. Use the lower end when silence or
seed fragility is the main problem; use the upper end only when timings are too
similar and coverage remains healthy.

**Observed behavior.** This was the clearest factorial initialization
trade-off. At fixed gain:

- `sigma=0.5` gave the best coverage and median readout gaps around
  `0.54–0.60 ms`;
- `sigma=0.8` enlarged median gaps to about `0.98–1.06 ms`, with modestly more
  silence; and
- `sigma=1.1` enlarged gaps to about `1.43–1.73 ms`, but minimum per-output
  coverage could fall to `94.6–96.9%`.

The broad and local searches mildly favored lower or moderate heterogeneity.
The multi-seed finalists all used about `0.76–0.80`; robustness above `0.80`
has not been tested across seeds. The single-default-seed searches only weakly
favored low or moderate values over values above `0.9`.

**If increased:** a few synapses dominate, readout time gaps grow, and pathway
specialization increases; coverage and redundancy decrease. Seed robustness
may also decrease, but that effect remains unconfirmed above `sigma=0.80`.

**If decreased:** more even aggregation and better coverage, but less temporal
diversity and potentially less class separation.

**Tune it when:** use coverage and gap statistics together. Low coverage plus
large gaps calls for lower `sigma`, not merely higher row sums. Complete
coverage with nearly coincident readouts suggests slightly higher `sigma`.

An approximate concentration diagnostic is

`N_eff = (sum w)^2 / sum(w^2)`.

For normalized lognormal weights, `N_eff / n_hidden` is roughly
`exp(-sigma^2)`. The recommended range corresponds to an effective fraction of
about `0.53–0.66`, or roughly 6–8 effective inputs out of 12. This is more
portable across hidden widths than `sigma` alone.

### 5. Maximum delay `d_plus`

The delay is

`delay = d_plus × sigmoid(c1 - c0 × slope)`.

**Acceptable starting range.** `4.3–5.0 ms` for the current 50 ms horizon and
20 ms membrane constant. The broad range `3–7 ms` produced valid runs, but the
local search favored the lower half of `4.0–5.7 ms`; the top three 4-epoch
local candidates (`local_005`, `local_004`, and `local_000`) were between
`4.30` and `4.85 ms`.

**If increased:** increases both temporal modulation and delay-gradient scale,
For hidden spikes, it pushes their arrivals later and increases the chance that
they occur after the relevant output threshold terminal, making them
non-causal for that output's first-threshold gradient. For output spikes, it
also lengthens the separate threshold-to-physical-readout delay used for
classification. It therefore amplifies sensitivity to slope noise at both
layers.

**If decreased:** makes arrival timing safer and gradients smaller, but can
remove useful phase separation until the network behaves like an almost
zero-delay feed-forward model.

**Tune it when:** inspect hidden and output delay distributions separately.
For gradient causality, compare each hidden arrival to the relevant output
**threshold time**, not the later physical-readout time. A hidden arrival can
precede the reported readout and still be non-causal if the output had already
crossed threshold. Increase `d_plus` only if delays occupy a narrow early band
and physical-readout gaps remain too small. Decrease it if many hidden arrivals
occur after output thresholds or outside the 50 ms dynamical window, or if
delay-gradient-related instability rises. A delayed physical readout beyond
50 ms is not by itself silent: the underlying output threshold occurred inside
the simulated window and its timing remains recorded.

**Time-scale scaling.** If every dynamical time constant and the horizon are
multiplied by `s`, multiply `d_plus` by `s`. The present starting point is about
`0.22–0.25 × tau_V` and `0.09–0.10 × t_end`.

### 6. Delay sharpness `c0`, midpoint `c1/c0`, and `c1`

Tune these as **sharpness plus midpoint**, not as two unrelated numbers. Define

`midpoint_slope = c1 / c0`.

Then the sigmoid fraction is `sigmoid(c0 × (midpoint_slope - slope))`.

This mapping is applied twice in the generalized network: a hidden threshold
uses it to schedule the hidden-to-output arrival, and an output threshold uses
it to define the later physical readout used for classification. Output state
and its first-threshold adjoint terminate at the threshold time, not at the
delayed physical-readout time.

**Acceptable starting range.** Use `c0=18–22 ms/V` and midpoint
`0.09–0.105 V/ms`, which gives `c1` around `1.6–2.3`. The current reference is
`c0=20`, midpoint `0.10`, and `c1=2`.

The local 4-epoch search showed an exploratory negative rank relationship for
`c0` over `18–28` (`rho=-0.620`, unadjusted `p≈0.032`). This came from only 12
mixed Latin-hypercube points across eight inspected axes, with no
multiple-comparison correction, so it is a follow-up signal rather than a
causal or statistically confirmed optimum. The strongest group was
concentrated around `18–22`, while values above about `25` were generally
weaker. The midpoint was less identifiable because it interacted with both
`c0` and the hidden-slope distribution.

The maximum delay sensitivity is

`max |d(delay)/d(slope)| = d_plus × c0 / 4`.

Start with `d_plus × c0` around `80–110`; the broad search enforced a ceiling
of `180` to avoid extreme sensitivity.

**Increasing `c0` at fixed midpoint:** sharpens the transition and increases
the maximum delay derivative, but saturates more spikes near zero or maximum
delay. Decreasing it spreads useful gradients across a wider slope range but
weakens slope selectivity.

**Increasing `c1` at fixed `c0`:** raises the midpoint slope and lengthens the
delay for every fixed slope. Decreasing it shifts delays toward zero.

**Tune them when:** plot the sigmoid fraction or delay divided by `d_plus`.
If most values are below `0.05` or above `0.95`, the mapping is saturated:
move the midpoint toward the observed slopes and/or lower `c0`. If all values
cluster near `0.5`, increase `c0` only after confirming that slope-dependent
timing would help. A good midpoint lies inside the central mass of causal
hidden threshold slopes; a useful sharpness spans a substantial part of that
distribution rather than separating only a few outliers.

**Time-scale scaling.** When all times are multiplied by `s`, slopes scale as
`1/s`. Multiply `c0` by `s`, keep dimensionless `c1` approximately constant,
and let the midpoint slope scale as `1/s`.

### 7. Fast-hidden input gain

The current initializer multiplies the first 6 of 12 hidden input rows by this
gain. It changes hidden activity before any optimizer update.

**Acceptable starting range.** `1.0–1.2`; start near `1.1–1.17`. The local
leaders used `1.035`, `1.142`, and `1.172`. Large broad-search values up to
`1.6` occasionally learned quickly, but did not remain consistently strong at
10 epochs.

**If increased:** fast hidden neurons spike more readily and usually earlier,
reducing their silence but increasing fast/slow imbalance. Their sharper
crossings can also drive the delay sigmoid toward shorter delays.

**If decreased:** preserves more balanced phase pathways but can leave weak
fast rows silent.

**Tune it when:** compare silence and spike times separately for fast and slow
hidden groups. Increase only if the fast group is systematically silent or too
late. Decrease if fast neurons dominate causal arrivals, slow neurons stop
affecting readouts, or increased gain produces rapidly growing signed weights.

For a different hidden width, keep the fast fraction near the intended model
design (currently one half) and do not scale the gain with width.

### 8. Readout calibration controls

The current calibration target is 99% per-output train coverage, with failing
rows multiplied by `1.05` per round for at most 40 rounds.

These values were not independently swept. They behaved correctly, but the
legacy initializer required 19–23 rounds. That is a symptom of poor starting
row sums, not a reason to spend more calibration rounds. Explicit target sums
reduced calibration to one round.

Use calibration as a guardrail:

- keep the target at `0.99` during debugging;
- prefer direct row-sum initialization over a more aggressive multiplier;
- if more than about 3–5 rounds are regularly needed, retune row sums or
  heterogeneity;
- avoid overshooting coverage with a large calibration factor because excess
  drive compresses readout timing gaps.

### 9. Unswept fixed controls

Except for the staged epoch budget, the campaign held the following controls
fixed. Their current values are part of the empirical regime, but the runs do
**not** establish acceptable ranges for them.

| Control | Current value | Role and warning |
|---|---:|---|
| Training epochs | `4–10` for screens; `40` for completed final runs | No solver early stopping; intermediate model selection remains validation-only |
| Batch size | `50` | Required proof-of-learning batch size |
| Full dynamical horizon | `50 ms` | All initialization, training, and validation calls use `early_stop=False` |
| Input encoding | `(x, y, 1-x, 1-y)` over `5–25 ms` | Four complementary channels; changing this window alters hidden slopes and delay transferability |
| Timing objective and margin | original DelGrad: `1 ms`; squared hinge: `2 ms` | DelGrad pulls every physical wrong output to the margin; hinge penalizes only violations. Treat their LR/accuracy behavior as separate regimes. |
| Hidden restart trigger | causal silence on `>0.5` of validation samples | Strict majority; spikes after the last output threshold do not prevent a restart; full-horizon silence is retained only as a diagnostic |
| Gradient-norm cap | `10` | Global batch-gradient clipping, separate from threshold sensitivity |
| Threshold-sensitivity cap | `2` on `1/(tau_V × slope)` | Canonical numerical safeguard; nonzero activation is expected occasionally |
| Adam `beta1`, `beta2`, `epsilon` | `0.9`, `0.999`, `1e-8` | Not swept; all LR conclusions assume these values |
| Configured grid step `dt` | `0.05 ms` | Used by grid-resolved paths; continuous training is event/root based and does not quantize events to `dt` |
| Slope tolerance | `1e-12` | Crossing-validity tolerance, not an optimization knob in this campaign |
| Membrane constant | `20 ms` | Defines the time-scale regime used by the delay recommendations |
| Synaptic decay `alpha` | `1/40 = 0.025 /ms` | Canonical configurable oscillator-envelope decay; historical broad-search evidence used `0.1 /ms` |
| Fast/slow periods | `8pi/12pi ms` (`~25.13/~37.70 ms`) | Canonical configurable populations with `omega=1/4, 1/6 rad/ms`; historical evidence used `15/25 ms` |
| Threshold/reset drop | `1/5 V` | Fixed event and subtractive-reset semantics |
| Weight constraint | none after positive initialization | Both matrices may become signed; monitor minimum and maximum values |
| Calibration target/step/max | `0.99`, `1.05`, `40` | Guardrail only; direct target row sums are preferred |

Change one of these only in a separate, named experiment. In particular, do
not transfer the present LR range unchanged after changing the loss margin,
batch size, or gradient cap.

## Unresolved questions

The completed campaign does not resolve the following points:

- the causal benefit of cosine decay relative to a matched constant-rate run,
  and whether its 40-epoch horizon or zero minimum should change;
- seed robustness for readout heterogeneity above `sigma=0.80`;
- a causal optimum for `c0`, `d_plus`, or their product—the observed
  associations came from mixed searches;
- how the suggested row-sum, bump, and delay scaling rules perform at a
  different network width or time scale; and
- preferred ranges for the fixed controls in the preceding table.

The scaling formulas remain useful hypotheses for choosing a first point, but
they are not substitutes for a small activity screen in a changed regime.

## Interactions that matter most

### Learning rate × training horizon

The high rates that win a 4-epoch screen are not automatically the best final
rates. The active end-to-end runs show that cosine annealing over 40 epochs is
viable, while the older constant-rate screen explains early-learning behavior.
Because there is no matched schedule ablation, do not attribute the active
accuracy improvement to cosine decay alone.

### Restart offset/noise × learning rate

Both change hidden weights, but the stochastic restart is discontinuous. Adam
state is deliberately retained. If a high-rate run has a large negative
post-restart accuracy delta, reduce the restart noise or offset before concluding
that the learning rate itself is wrong.

### Readout gain × heterogeneity

Higher row sums can hide the coverage failure caused by high heterogeneity,
but they also compress timing. When coverage is poor and gaps are already
large, lower `sigma` first. When weights are even and all gaps are tiny, modest
heterogeneity is preferable to simply lowering all row sums.

### `d_plus` × `c0`

Their product sets the maximum slope-to-delay derivative. Do not raise both
independently. Search in terms of maximum delay, midpoint slope, and the product
`d_plus × c0`.

### Hidden gain × delay midpoint

Increasing hidden input gain generally makes threshold crossings sharper.
That shifts slopes relative to the fixed delay midpoint and can collapse delays
toward zero. After a material hidden-gain change, recheck the delay-fraction
distribution before tuning `d_plus`.

## Symptom-to-action guide

| Observed phenomenon | Most likely first action | Secondary checks |
|---|---|---|
| Readout coverage below 99% before training | Increase only that row sum or lower `sigma` | Hidden activity; calibration rounds |
| All readouts spike but times are nearly identical | Slightly increase `sigma` or timing sensitivity | Avoid raising row sums; inspect delay saturation |
| Large gaps but many silent outputs | Lower `sigma`; then modestly raise affected row sums | Check hidden active fraction |
| Accuracy rises quickly then falls after epoch 4–6 | Use the canonical cosine schedule before lowering the base rate | Schedule causality remains unresolved; compare matched configurations |
| Accuracy barely changes during first 4 epochs | Raise constant LR within `0.009–0.012`; test higher rates separately | Confirm gradients are nonzero and activity is healthy |
| Same hidden row is restarted every epoch | Inspect whether task gradients suppress it before increasing restart strength | Initialization, row slope, optimizer gradients |
| Several hidden rows are repeatedly restarted together | Repair global hidden initialization/activity balance before enlarging restarts | Exact perturbations; fast/slow imbalance |
| Restart rescues silence but validation drops >2 points | Reduce restart noise or offset | Delay/phase change caused by the row |
| One class collapses while overall accuracy looks fair | Reject the run; inspect its readout silence and confusion matrix | Row-specific initialization, seed robustness |
| Target/all-silent counts grow late in training | Lower the constant LR | Row sums, signed weight ranges, high `sigma` |
| Delay fractions mostly near 0 or 1 | Move midpoint into slope distribution and/or lower `c0` | Hidden gain and slope distribution |
| Many hidden arrivals occur after the relevant output threshold | Lower `d_plus` or shift delay mapping shorter | Compare to threshold time, not delayed physical-readout time |
| Cap-activation rate rises above the run's earlier epochs with falling accuracy | Inspect near-tangent spikes; strengthen weak hidden rows carefully | Reference campaign range was `1.68–5.30` per 1,000 samples |
| Clipped-batch rate rises with high `d_plus*c0` and falling accuracy | Lower timing sensitivity product | Healthy and collapsed finalists both lay within `0–0.8%`; use trends, not nonzero counts |
| Signed weight magnitude grows with silence or class collapse | Lower constant LR and inspect gradient signs | Compare `W_in` and `W_out` minima/maxima by epoch |
| Strong default seed, weak new seeds | Prefer the robust candidate; do not open test | Add fresh complete seed bundles |

## Efficient tuning protocol

### Stage A: deterministic activity screen

Use 500–5,000 training samples without optimization. Measure hidden activity,
per-output coverage, all-physical fraction, readout gaps, slope distribution,
and delay fractions. This stage should take seconds to minutes, not epochs.

Reject configurations with:

- any output below 99% coverage after reasonable row-sum calibration;
- widespread all-silent trials;
- delay values almost entirely within 5% of one bound; or
- a gross fast/slow hidden activity imbalance.

### Stage B: short optimization screen

Run 4 epochs on two contrasting seed bundles. Search only the few remaining
high-impact axes:

- base learning rate;
- bump amount if bumps are actually triggered; and
- one timing composite at a time (`d_plus`, midpoint, or `d_plus*c0`).

Do not rank solely by epoch 4. Use mean epochs 2–4, worst-class accuracy,
silence, and whether the curve is still improving.

### Stage C: 10-epoch continuation

Continue only the Pareto frontier: strong stable accuracy, acceptable worst
class, healthy activity, and low calibration cost. Preserve all 10 epochs and
rank by stable validation behavior. A future schedule study should be recorded
as a separate campaign rather than mixed into the evidence summarized here.

### Stage D: independent seed confirmation

Run complete configurations on at least three seed bundles. Do not mix partial
seed results into rankings. At least one final confirmation seed should be new,
not used in broad or local selection. Rank by mean stable accuracy, then worst
seed and worst class.

### Stage E: freeze and test

Write the chosen configuration and selection rationale before evaluating the
held-out test. Run the test once. Do not return to validation tuning based on
that result.

## Current empirical frontier

The active 40-epoch comparisons are the primary reference:

| Loss | Margin | Restart offset | Best validation | Final validation | Test |
|---|---:|---:|---:|---:|---:|
| Squared hinge | `2.0` | `0.10` | `0.947` | `0.929` | **`0.935`** |
| Squared hinge | `2.0` | `0.25` | `0.955` | `0.924` | `0.920` |
| DelGrad | `0.16` | `0.10` | `0.880` | `0.870` | `0.886` |
| DelGrad | `1.0` | `0.10` | `0.911` | `0.905` | `0.908` |
| DelGrad | `2.0` | `0.10` | `0.946` | `0.613` | `0.630` |

The DelGrad margin-2 run is the clearest warning against selecting from one
validation peak: it reached `0.946` and then collapsed. DelGrad margin `1.0`
was more stable than either `0.16` or `2.0`, while squared hinge margin `2.0`
produced the strongest held-out result. The test values are completed-run
records, not a basis for further tuning.

The older multi-seed projected-weight references remain useful for qualitative
robustness context:

| Candidate | LR | Bump | Gain | Sigma | `d_plus` | `c0` | Midpoint | Fast gain | Three-seed stable mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 0.0100 | 0.200 | calibrated | 0.800 | 5.000 | 20.00 | 0.100 | 1.000 | 0.8057 |
| `local_000` | 0.0166 | 0.152 | 0.957 | 0.758 | 4.854 | 19.42 | 0.095 | 1.172 | 0.8009 |
| `candidate_014` | 0.0111 | 0.261 | 0.932 | 0.757 | 4.932 | 24.99 | 0.104 | 1.112 | 0.7908 |

The historical legacy control had the strongest three-seed stable mean in that
regime, but its absolute accuracy and projected-weight dynamics are not directly
comparable with the active signed-weight model.

## Evidence locations

- Canonical measured run: `results/yinyang_best/metrics.json`
- Five-run active comparison: `results/yinyang_experiment_summary.json`
- Canonical resolved defaults: `configs/yinyang_best.json`

The raw historical sweeps were removed during production cleanup after their
empirical conclusions were consolidated here. The guide therefore distinguishes
historical aggregate observations from the retained active-run records.

The guide should be updated after each materially different tuning campaign.
Any future schedule study, or any change in network width or time constants,
should be recorded as a separate empirical regime rather than silently merged
into these ranges.
