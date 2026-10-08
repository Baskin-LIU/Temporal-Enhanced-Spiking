"""Deterministic data, metrics, and optimization for TeS proof-of-learning runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import time
from typing import Literal, Sequence

import numpy as np
import torch

from Code.neurons import InputSpike, TeSNeurons
from Code.tes_events import EffectiveReadout
from Code.tes_loss import TimingLossName, evaluate_timing_loss, get_timing_loss
from Code.yinyang import YinYang, encode_events


CHECKPOINT_SCHEMA_VERSION = 5
LEGACY_CHECKPOINT_SCHEMA_VERSION = 4
MODEL_SEMANTICS_VERSION = "quarter-sixth-oscillator-psp-calibrated-v1"
TRAINING_SEMANTICS_VERSION = "modular-timing-loss-signed-restart-v1"
HIDDEN_SILENCE_METRIC_VERSION = "last-output-threshold-v1"
OMEGA_LEARNING_RATE_SCALE = 0.01
OMEGA_MIN = 0.1
OMEGA_MAX = 0.35


@dataclass(frozen=True)
class TrainingSample:
    index: int
    point: tuple[float, float]
    label: int
    events: tuple[InputSpike, ...]


@dataclass(frozen=True)
class YinYangSplit:
    name: Literal["train", "validation", "test"]
    seed: int
    class_counts: tuple[int, int, int]
    samples: tuple[TrainingSample, ...]


@dataclass(frozen=True)
class Prediction:
    predicted: int | None
    correct: bool
    all_silent: bool
    target_silent: bool
    ambiguous_tie: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    split: str
    samples: int
    mean_loss: float
    accuracy: float
    correct: int
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class_accuracy: tuple[float | None, ...]
    all_silent: int
    target_silent: int
    ambiguous_ties: int
    output_silence_counts: tuple[int, ...]
    hidden_silence_counts: tuple[int, ...]
    hidden_spikes_per_sample: float
    output_spikes_per_sample: float
    threshold_sensitivity_cap_activations: int
    minimum_causal_slope: float | None
    minimum_recorded_slope: float | None
    runtime_seconds: float
    hidden_full_horizon_silence_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class AdamConfig:
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_norm_cap: float = 10.0


@dataclass
class AdamState:
    m_in: torch.Tensor
    v_in: torch.Tensor
    m_out: torch.Tensor
    v_out: torch.Tensor
    m_omega: torch.Tensor
    v_omega: torch.Tensor
    step: int = 0
    omega_step: int = 0

    @classmethod
    def zeros_like(cls, model: TeSNeurons) -> AdamState:
        return cls(
            torch.zeros_like(model.W_in),
            torch.zeros_like(model.W_in),
            torch.zeros_like(model.W_out),
            torch.zeros_like(model.W_out),
            torch.zeros_like(model.omega),
            torch.zeros_like(model.omega),
        )

    def to_payload(self) -> dict[str, torch.Tensor | int]:
        return {
            "m_in": self.m_in.clone(),
            "v_in": self.v_in.clone(),
            "m_out": self.m_out.clone(),
            "v_out": self.v_out.clone(),
            "m_omega": self.m_omega.clone(),
            "v_omega": self.v_omega.clone(),
            "step": self.step,
            "omega_step": self.omega_step,
        }


@dataclass(frozen=True)
class UpdateMetrics:
    raw_gradient_norm: float
    clipped_gradient_norm: float
    clipping_factor: float
    negative_w_in: int
    negative_w_out: int
    w_in_min: float
    w_in_max: float
    w_out_min: float
    w_out_max: float
    raw_omega_gradient_norm: float | None = None
    clipped_omega_gradient_norm: float | None = None
    omega_clipping_factor: float | None = None
    omega_max_absolute_update: float | None = None
    omega_clamped_entries: int = 0
    omega_min: float | None = None
    omega_max: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 50
    t_end: float = 50.0
    loss_name: TimingLossName = "squared_hinge"
    margin: float | None = None
    shuffle_seed: int = 2001
    adam: AdamConfig = field(default_factory=AdamConfig)
    cosine_annealing_t_max: int | None = None
    cosine_annealing_eta_min: float = 0.0
    learn_omega: bool = False
    omega_learning_rate_scale: float = OMEGA_LEARNING_RATE_SCALE
    omega_min: float = OMEGA_MIN
    omega_max: float = OMEGA_MAX
    in_epoch_bump_interval_batches: int = 0
    in_epoch_bump_silence_fraction: float = 0.5
    in_epoch_bump_amount: float = 0.05

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")
        if not isinstance(self.learn_omega, bool):
            raise TypeError("learn_omega must be boolean")
        if (
            isinstance(self.in_epoch_bump_interval_batches, bool)
            or not isinstance(self.in_epoch_bump_interval_batches, int)
            or self.in_epoch_bump_interval_batches < 0
            or not math.isfinite(self.in_epoch_bump_silence_fraction)
            or not 0.0 <= self.in_epoch_bump_silence_fraction <= 1.0
            or not math.isfinite(self.in_epoch_bump_amount)
            or self.in_epoch_bump_amount <= 0.0
        ):
            raise ValueError("invalid in-epoch hidden bump configuration")
        if (
            not math.isfinite(self.omega_learning_rate_scale)
            or self.omega_learning_rate_scale <= 0.0
            or not math.isfinite(self.omega_min)
            or not math.isfinite(self.omega_max)
            or self.omega_min <= 0.0
            or self.omega_min >= self.omega_max
        ):
            raise ValueError("invalid omega learning-rate scale or clamp range")
        get_timing_loss(self.loss_name)
        if self.t_end <= 0.0 or (
            self.margin is not None
            and (not math.isfinite(self.margin) or self.margin < 0.0)
        ):
            raise ValueError("invalid horizon or margin")
        cosine_t_max = self.cosine_annealing_t_max
        eta_min = self.cosine_annealing_eta_min
        if cosine_t_max is None:
            if eta_min != 0.0:
                raise ValueError(
                    "cosine annealing eta_min requires a T_max"
                )
        elif (
            isinstance(cosine_t_max, bool)
            or not isinstance(cosine_t_max, int)
            or cosine_t_max <= 0
            or not math.isfinite(eta_min)
            or eta_min < 0.0
            or eta_min > self.adam.learning_rate
            or (eta_min == 0.0 and self.epochs > cosine_t_max)
        ):
            raise ValueError("invalid cosine annealing configuration")

    @property
    def resolved_margin(self) -> float:
        """Configured margin, or the selected loss module's own default."""
        if self.margin is not None:
            return float(self.margin)
        return get_timing_loss(self.loss_name).default_margin_ms

    def adam_for_epoch(self, epoch: int) -> AdamConfig:
        """Return Adam settings at a one-based training epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        if self.cosine_annealing_t_max is not None:
            completed_epochs = min(epoch - 1, self.cosine_annealing_t_max)
            cosine = math.cos(
                math.pi * completed_epochs / self.cosine_annealing_t_max
            )
            learning_rate = self.cosine_annealing_eta_min + 0.5 * (
                self.adam.learning_rate - self.cosine_annealing_eta_min
            ) * (1.0 + cosine)
            return replace(self.adam, learning_rate=learning_rate)
        return self.adam


@dataclass(frozen=True)
class BatchMetrics:
    epoch: int
    batch: int
    samples: int
    mean_loss: float
    accuracy: float
    correct: int
    all_silent: int
    target_silent: int
    ambiguous_ties: int
    hidden_thresholds: int
    output_thresholds: int
    threshold_sensitivity_cap_activations: int
    minimum_causal_slope: float | None
    update: UpdateMetrics
    runtime_seconds: float
    hidden_silence_counts: tuple[int, ...] = ()
    in_epoch_bumped_hidden: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrainingState:
    epoch: int
    config: TrainingConfig
    adam: AdamState
    calibration_history: tuple[object, ...]
    bump_history: tuple[object, ...]
    epoch_metrics: tuple[object, ...]
    run_config: dict[str, object] | None
    data_identity: dict[str, object] | None
    model_config: dict[str, object] | None
    baseline_train: object | None
    baseline_validation: object | None
    best_validation_snapshot: dict[str, object] | None


class UnavailableTrainingGradient(RuntimeError):
    """A sample has no unique differentiable training source."""


class CheckpointMismatchError(RuntimeError):
    """A saved state is incompatible with the requested model or run."""


def adam_update(
    model: TeSNeurons,
    grad_w_in: torch.Tensor,
    grad_w_out: torch.Tensor,
    state: AdamState,
    config: AdamConfig,
    *,
    grad_omega: torch.Tensor | None = None,
    omega_learning_rate_scale: float = OMEGA_LEARNING_RATE_SCALE,
    omega_min: float = OMEGA_MIN,
    omega_max: float = OMEGA_MAX,
) -> UpdateMetrics:
    """Apply one combined-gradient-clipped float64 Adam update in place."""
    grad_w_in = torch.as_tensor(
        grad_w_in, dtype=torch.float64, device=model.W_in.device
    )
    grad_w_out = torch.as_tensor(
        grad_w_out, dtype=torch.float64, device=model.W_out.device
    )
    if grad_w_in.shape != model.W_in.shape or grad_w_out.shape != model.W_out.shape:
        raise ValueError("gradient shapes do not match model weights")
    if config.gradient_norm_cap <= 0.0 or config.learning_rate <= 0.0:
        raise ValueError("Adam learning rate and gradient cap must be positive")
    if (
        not math.isfinite(omega_learning_rate_scale)
        or omega_learning_rate_scale <= 0.0
        or not math.isfinite(omega_min)
        or not math.isfinite(omega_max)
        or omega_min <= 0.0
        or omega_min >= omega_max
    ):
        raise ValueError("invalid omega learning-rate scale or clamp range")
    omega_gradient: torch.Tensor | None = None
    if grad_omega is not None:
        omega_gradient = torch.as_tensor(
            grad_omega, dtype=torch.float64, device=model.omega.device
        )
        if omega_gradient.shape != model.omega.shape:
            raise ValueError("omega gradient shape does not match model omega")
    raw_norm_tensor = torch.sqrt(
        torch.sum(grad_w_in * grad_w_in)
        + torch.sum(grad_w_out * grad_w_out)
    )
    raw_norm = float(raw_norm_tensor)
    factor = (
        1.0
        if raw_norm == 0.0
        else min(1.0, config.gradient_norm_cap / raw_norm)
    )
    clipped_in = grad_w_in * factor
    clipped_out = grad_w_out * factor
    next_step = state.step + 1

    next_m_in = config.beta1 * state.m_in + (1.0 - config.beta1) * clipped_in
    next_v_in = config.beta2 * state.v_in + (1.0 - config.beta2) * clipped_in.square()
    next_m_out = config.beta1 * state.m_out + (1.0 - config.beta1) * clipped_out
    next_v_out = config.beta2 * state.v_out + (1.0 - config.beta2) * clipped_out.square()
    first_moment_correction = 1.0 - config.beta1**next_step
    second_moment_correction = 1.0 - config.beta2**next_step
    update_in = config.learning_rate * (next_m_in / first_moment_correction) / (
        torch.sqrt(next_v_in / second_moment_correction) + config.epsilon
    )
    update_out = config.learning_rate * (next_m_out / first_moment_correction) / (
        torch.sqrt(next_v_out / second_moment_correction) + config.epsilon
    )
    candidate_in = model.W_in - update_in
    candidate_out = model.W_out - update_out
    negative_w_in = int(torch.count_nonzero(candidate_in < 0.0))
    negative_w_out = int(torch.count_nonzero(candidate_out < 0.0))
    state.m_in.copy_(next_m_in)
    state.v_in.copy_(next_v_in)
    state.m_out.copy_(next_m_out)
    state.v_out.copy_(next_v_out)
    state.step = next_step
    model.W_in.copy_(candidate_in)
    model.W_out.copy_(candidate_out)
    raw_omega_norm: float | None = None
    clipped_omega_norm: float | None = None
    omega_factor: float | None = None
    omega_max_update: float | None = None
    omega_clamped_entries = 0
    if omega_gradient is not None:
        raw_omega_norm = float(torch.linalg.vector_norm(omega_gradient))
        omega_factor = (
            1.0
            if raw_omega_norm == 0.0
            else min(1.0, config.gradient_norm_cap / raw_omega_norm)
        )
        clipped_omega = omega_gradient * omega_factor
        clipped_omega_norm = raw_omega_norm * omega_factor
        next_omega_step = state.omega_step + 1
        next_m_omega = (
            config.beta1 * state.m_omega
            + (1.0 - config.beta1) * clipped_omega
        )
        next_v_omega = (
            config.beta2 * state.v_omega
            + (1.0 - config.beta2) * clipped_omega.square()
        )
        omega_m_correction = 1.0 - config.beta1**next_omega_step
        omega_v_correction = 1.0 - config.beta2**next_omega_step
        omega_update = (
            config.learning_rate
            * omega_learning_rate_scale
            * (next_m_omega / omega_m_correction)
            / (
                torch.sqrt(next_v_omega / omega_v_correction)
                + config.epsilon
            )
        )
        candidate_omega = model.omega - omega_update
        clamped_omega = candidate_omega.clamp(omega_min, omega_max)
        omega_clamped_entries = int(
            torch.count_nonzero(candidate_omega != clamped_omega)
        )
        omega_max_update = float(
            torch.max(torch.abs(clamped_omega - model.omega))
        )
        state.m_omega.copy_(next_m_omega)
        state.v_omega.copy_(next_v_omega)
        state.omega_step = next_omega_step
        model.omega.copy_(clamped_omega)
    return UpdateMetrics(
        raw_gradient_norm=raw_norm,
        clipped_gradient_norm=raw_norm * factor,
        clipping_factor=factor,
        negative_w_in=negative_w_in,
        negative_w_out=negative_w_out,
        w_in_min=float(model.W_in.min()),
        w_in_max=float(model.W_in.max()),
        w_out_min=float(model.W_out.min()),
        w_out_max=float(model.W_out.max()),
        raw_omega_gradient_norm=raw_omega_norm,
        clipped_omega_gradient_norm=clipped_omega_norm,
        omega_clipping_factor=omega_factor,
        omega_max_absolute_update=omega_max_update,
        omega_clamped_entries=omega_clamped_entries,
        omega_min=float(model.omega.min()),
        omega_max=float(model.omega.max()),
    )


def bump_silent_hidden(
    model: TeSNeurons,
    silent_counts: Sequence[int],
    state: AdamState,
    sample_count: int,
    threshold_fraction: float = 0.5,
    amount: float = 0.2,
    *,
    seed: int,
    noise_std: float = 1.0,
) -> tuple[int, ...]:
    """Stochastically restart silent hidden rows without changing Adam state."""
    if len(silent_counts) != model.n_hidden:
        raise ValueError("one silence count is required per hidden neuron")
    if (
        sample_count <= 0
        or not 0.0 <= threshold_fraction <= 1.0
        or not math.isfinite(amount)
        or amount <= 0.0
        or not math.isfinite(noise_std)
        or noise_std < 0.0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("invalid hidden bump configuration")
    bumped = tuple(
        hidden
        for hidden, count in enumerate(silent_counts)
        if count > threshold_fraction * sample_count
    )
    if bumped:
        generator = torch.Generator(device=model.W_in.device).manual_seed(seed)
        perturbation = torch.randn(
            model.W_in.shape,
            dtype=model.W_in.dtype,
            device=model.W_in.device,
            generator=generator,
        ).mul_(noise_std).add_(amount)
        for hidden in bumped:
            model.W_in[hidden].add_(perturbation[hidden])
    return bumped


def bump_silent_hidden_deterministic(
    model: TeSNeurons,
    silent_counts: Sequence[int],
    sample_count: int,
    threshold_fraction: float = 0.5,
    amount: float = 0.05,
) -> tuple[int, ...]:
    """Add the same small positive offset to sufficiently silent input rows."""
    if len(silent_counts) != model.n_hidden:
        raise ValueError("one silence count is required per hidden neuron")
    if (
        sample_count <= 0
        or not math.isfinite(threshold_fraction)
        or not 0.0 <= threshold_fraction <= 1.0
        or not math.isfinite(amount)
        or amount <= 0.0
    ):
        raise ValueError("invalid deterministic hidden bump configuration")
    bumped = tuple(
        hidden
        for hidden, count in enumerate(silent_counts)
        if count > threshold_fraction * sample_count
    )
    if bumped:
        for hidden in bumped:
            model.W_in[hidden].add_(amount)
    return bumped


def _rotating_class_counts(size: int, extra_start: int) -> tuple[int, int, int]:
    base, remainder = divmod(size, 3)
    counts = [base, base, base]
    for offset in range(remainder):
        counts[(extra_start + offset) % 3] += 1
    return tuple(counts)


def build_yinyang_splits(
    train_size: int = 5000,
    validation_size: int = 1000,
    test_size: int = 1000,
    seeds: tuple[int, int, int] = (1001, 1002, 1003),
    *,
    dt: float = 0.05,
) -> tuple[YinYangSplit, YinYangSplit, YinYangSplit]:
    """Build encoded, deterministic, near-balanced, disjoint data splits."""
    sizes = (train_size, validation_size, test_size)
    names = ("train", "validation", "test")
    if any(size <= 0 for size in sizes):
        raise ValueError("split sizes must be positive")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct split seeds are required")

    dataset = YinYang()
    splits: list[YinYangSplit] = []
    extra_start = 0
    seen: set[tuple[float, float]] = set()
    for name, size, seed in zip(names, sizes, seeds, strict=True):
        counts = _rotating_class_counts(size, extra_start)
        extra_start = (extra_start + size % 3) % 3
        coordinates, labels = dataset.sample_balanced(max(counts), seed=seed)
        selected: list[tuple[tuple[float, float], int]] = []
        for label, count in enumerate(counts):
            class_coordinates = coordinates[labels == label][:count]
            selected.extend(
                ((float(point[0]), float(point[1])), label)
                for point in class_coordinates
            )
        order = np.random.default_rng(seed + 10_000).permutation(len(selected))
        samples = []
        for index, source_index in enumerate(order):
            point, label = selected[int(source_index)]
            if point in seen:
                raise ValueError(f"overlapping coordinate {point!r} across data splits")
            if dataset.which_class(*point) != label:
                raise ValueError(f"stored label disagrees with Yin-Yang geometry at {point!r}")
            if math.hypot(point[0] - 0.5, point[1] - 0.5) > 0.5:
                raise ValueError(f"point lies outside the Yin-Yang disk: {point!r}")
            seen.add(point)
            samples.append(
                TrainingSample(
                    index=index,
                    point=point,
                    label=label,
                    events=encode_events(point, dt).events,
                )
            )
        splits.append(YinYangSplit(name, seed, counts, tuple(samples)))
    return tuple(splits)


def physical_prediction(
    readouts: Sequence[EffectiveReadout], label: int
) -> Prediction:
    """Classify from physical first spikes; silence and exact ties are errors."""
    physical = tuple(
        (output, float(readout.physical_time))
        for output, readout in enumerate(readouts)
        if readout.physical_time is not None
    )
    target_silent = readouts[label].physical_time is None
    if not physical:
        return Prediction(None, False, True, target_silent, False)
    earliest = min(time_value for _, time_value in physical)
    winners = tuple(output for output, time_value in physical if time_value == earliest)
    if len(winners) != 1:
        return Prediction(None, False, False, target_silent, True)
    predicted = winners[0]
    return Prediction(predicted, predicted == label, False, target_silent, False)


def evaluate_split(
    model: TeSNeurons,
    split: YinYangSplit,
    t_end: float,
    margin: float | None,
    *,
    loss_name: TimingLossName = "squared_hinge",
    phase: str | None = None,
    epoch: int | None = None,
    learn_omega: bool = False,
) -> EvaluationMetrics:
    """Evaluate a split with full-horizon forwards and no backward passes."""
    del phase, epoch
    started = time.perf_counter()
    losses: list[float] = []
    correct = 0
    all_silent = 0
    target_silent = 0
    ambiguous_ties = 0
    confusion = [[0 for _ in range(model.n_out)] for _ in range(model.n_out)]
    class_totals = [0 for _ in range(model.n_out)]
    class_correct = [0 for _ in range(model.n_out)]
    output_silence = [0 for _ in range(model.n_out)]
    hidden_silence = [0 for _ in range(model.n_hidden)]
    hidden_full_horizon_silence = [0 for _ in range(model.n_hidden)]
    hidden_spikes = 0
    output_spikes = 0
    slopes: list[float] = []

    for sample in split.samples:
        forward_method = (
            model.forward_continuous_omega
            if learn_omega
            else model.forward_continuous
        )
        forward = forward_method(sample.events, t_end, early_stop=False)
        loss = evaluate_timing_loss(
            forward.readouts,
            label=sample.label,
            loss_name=loss_name,
            margin=margin,
        )
        losses.append(float(loss.loss))
        prediction = physical_prediction(forward.readouts, sample.label)
        class_totals[sample.label] += 1
        correct += int(prediction.correct)
        class_correct[sample.label] += int(prediction.correct)
        all_silent += int(prediction.all_silent)
        target_silent += int(prediction.target_silent)
        ambiguous_ties += int(prediction.ambiguous_tie)
        if prediction.predicted is not None:
            confusion[sample.label][prediction.predicted] += 1
        for output, readout in enumerate(forward.readouts):
            output_silence[output] += int(readout.physical_time is None)
        full_horizon_spiking_hidden = {
            event.neuron for event in forward.hidden_spikes
        }
        first_output_thresholds: dict[int, float] = {}
        for event in forward.output_spikes:
            first_output_thresholds.setdefault(event.neuron, event.time)
        causal_cutoff = (
            max(first_output_thresholds.values())
            if len(first_output_thresholds) == model.n_out
            else t_end
        )
        causal_spiking_hidden = {
            event.neuron
            for event in forward.hidden_spikes
            if event.time <= causal_cutoff
        }
        for hidden in range(model.n_hidden):
            hidden_silence[hidden] += int(hidden not in causal_spiking_hidden)
            hidden_full_horizon_silence[hidden] += int(
                hidden not in full_horizon_spiking_hidden
            )
        hidden_spikes += len(forward.hidden_spikes)
        output_spikes += len(forward.output_spikes)
        slopes.extend(event.slope for event in forward.hidden_spikes)
        slopes.extend(event.slope for event in forward.output_spikes)

    sample_count = len(split.samples)
    return EvaluationMetrics(
        split=split.name,
        samples=sample_count,
        mean_loss=sum(losses) / sample_count,
        accuracy=correct / sample_count,
        correct=correct,
        confusion_matrix=tuple(tuple(row) for row in confusion),
        per_class_accuracy=tuple(
            None if total == 0 else hits / total
            for hits, total in zip(class_correct, class_totals, strict=True)
        ),
        all_silent=all_silent,
        target_silent=target_silent,
        ambiguous_ties=ambiguous_ties,
        output_silence_counts=tuple(output_silence),
        hidden_silence_counts=tuple(hidden_silence),
        hidden_spikes_per_sample=hidden_spikes / sample_count,
        output_spikes_per_sample=output_spikes / sample_count,
        threshold_sensitivity_cap_activations=0,
        minimum_causal_slope=None,
        minimum_recorded_slope=min(slopes, default=None),
        runtime_seconds=time.perf_counter() - started,
        hidden_full_horizon_silence_counts=tuple(
            hidden_full_horizon_silence
        ),
    )


def train_batch(
    model: TeSNeurons,
    samples: Sequence[TrainingSample],
    state: AdamState,
    config: TrainingConfig,
    *,
    epoch: int,
    batch: int,
) -> BatchMetrics:
    """Accumulate exactly 50 sample gradients and apply one Adam update."""
    if len(samples) != config.batch_size:
        raise ValueError(
            f"batch has {len(samples)} samples; expected {config.batch_size}"
        )
    started = time.perf_counter()
    grad_w_in = torch.zeros_like(model.W_in)
    grad_w_out = torch.zeros_like(model.W_out)
    grad_omega = torch.zeros_like(model.omega) if config.learn_omega else None
    losses: list[float] = []
    correct = 0
    all_silent = 0
    target_silent = 0
    ambiguous_ties = 0
    hidden_thresholds = 0
    output_thresholds = 0
    cap_activations = 0
    causal_slopes: list[float] = []
    track_hidden_silence = config.in_epoch_bump_interval_batches > 0
    hidden_silence = (
        [0 for _ in range(model.n_hidden)] if track_hidden_silence else None
    )

    for sample in samples:
        forward_method = (
            model.forward_continuous_omega
            if config.learn_omega
            else model.forward_continuous
        )
        forward = forward_method(sample.events, config.t_end, early_stop=False)
        loss = evaluate_timing_loss(
            forward.readouts,
            label=sample.label,
            loss_name=config.loss_name,
            margin=config.margin,
            device=model.W_in.device,
        )
        if loss.output_time_grad is None:
            raise UnavailableTrainingGradient(
                loss.nondifferentiable_reason
                or f"training gradient unavailable at epoch {epoch}, batch {batch}, sample {sample.index}"
            )
        backward_method = (
            model.backward_continuous_omega
            if config.learn_omega
            else model.backward_continuous
        )
        backward = backward_method(forward, loss.output_time_grad)
        losses.append(float(loss.loss))
        grad_w_in.add_(backward.grad_W_in)
        grad_w_out.add_(backward.grad_W_out)
        if grad_omega is not None:
            grad_omega.add_(backward.grad_omega)
        prediction = physical_prediction(forward.readouts, sample.label)
        correct += int(prediction.correct)
        all_silent += int(prediction.all_silent)
        target_silent += int(prediction.target_silent)
        ambiguous_ties += int(prediction.ambiguous_tie)
        hidden_thresholds += len(forward.hidden_spikes)
        output_thresholds += len(forward.output_spikes)
        if hidden_silence is not None:
            first_output_thresholds: dict[int, float] = {}
            for event in forward.output_spikes:
                first_output_thresholds.setdefault(event.neuron, event.time)
            causal_cutoff = (
                max(first_output_thresholds.values())
                if len(first_output_thresholds) == model.n_out
                else config.t_end
            )
            causal_spiking_hidden = {
                event.neuron
                for event in forward.hidden_spikes
                if event.time <= causal_cutoff
            }
            for hidden in range(model.n_hidden):
                hidden_silence[hidden] += int(
                    hidden not in causal_spiking_hidden
                )
        cap_activations += backward.threshold_sensitivity_cap_activations
        if backward.minimum_causal_slope is not None:
            causal_slopes.append(backward.minimum_causal_slope)

    denominator = float(config.batch_size)
    grad_w_in.div_(denominator)
    grad_w_out.div_(denominator)
    if grad_omega is not None:
        grad_omega.div_(denominator)
    update = adam_update(
        model,
        grad_w_in,
        grad_w_out,
        state,
        config.adam,
        grad_omega=grad_omega,
        omega_learning_rate_scale=config.omega_learning_rate_scale,
        omega_min=config.omega_min,
        omega_max=config.omega_max,
    )
    return BatchMetrics(
        epoch=epoch,
        batch=batch,
        samples=len(samples),
        mean_loss=sum(losses) / denominator,
        accuracy=correct / denominator,
        correct=correct,
        all_silent=all_silent,
        target_silent=target_silent,
        ambiguous_ties=ambiguous_ties,
        hidden_thresholds=hidden_thresholds,
        output_thresholds=output_thresholds,
        threshold_sensitivity_cap_activations=cap_activations,
        minimum_causal_slope=min(causal_slopes, default=None),
        update=update,
        runtime_seconds=time.perf_counter() - started,
        hidden_silence_counts=(
            tuple(hidden_silence) if hidden_silence is not None else ()
        ),
    )


def train_epoch(
    model: TeSNeurons,
    split: YinYangSplit,
    state: AdamState,
    config: TrainingConfig,
    *,
    epoch: int,
) -> tuple[BatchMetrics, ...]:
    """Train one deterministic, complete epoch."""
    if split.name != "train":
        raise ValueError("train_epoch requires the training split")
    if len(split.samples) % config.batch_size != 0:
        raise ValueError("training split must divide into complete 50-sample batches")
    epoch_config = replace(config, adam=config.adam_for_epoch(epoch))
    order = np.random.default_rng(config.shuffle_seed + epoch).permutation(
        len(split.samples)
    )
    batches = []
    for batch_number, start in enumerate(
        range(0, len(order), config.batch_size), start=1
    ):
        batch_samples = tuple(
            split.samples[int(index)]
            for index in order[start : start + config.batch_size]
        )
        batch_metrics = train_batch(
            model,
            batch_samples,
            state,
            epoch_config,
            epoch=epoch,
            batch=batch_number,
        )
        batches.append(batch_metrics)
        interval = config.in_epoch_bump_interval_batches
        if interval and batch_number % interval == 0:
            window = batches[-interval:]
            silent_counts = tuple(
                sum(item.hidden_silence_counts[hidden] for item in window)
                for hidden in range(model.n_hidden)
            )
            sample_count = sum(item.samples for item in window)
            bumped = bump_silent_hidden_deterministic(
                model,
                silent_counts,
                sample_count,
                config.in_epoch_bump_silence_fraction,
                config.in_epoch_bump_amount,
            )
            batches[-1] = replace(
                batches[-1], in_epoch_bumped_hidden=bumped
            )
    return tuple(batches)


def _training_config_from_payload(payload: dict[str, object]) -> TrainingConfig:
    values = dict(payload)
    adam_values = values.pop("adam")
    if not isinstance(adam_values, dict):
        raise CheckpointMismatchError("checkpoint Adam configuration is malformed")
    return TrainingConfig(adam=AdamConfig(**adam_values), **values)


def _resolved_config_matches(
    saved: object,
    requested: dict[str, object],
    *,
    allow_epoch_extension: bool,
) -> bool:
    if not isinstance(saved, dict):
        return False
    if saved == requested:
        return True
    saved_optimization = saved.get("optimization")
    requested_optimization = requested.get("optimization")
    if (
        not allow_epoch_extension
        or not isinstance(saved_optimization, dict)
        or not isinstance(requested_optimization, dict)
    ):
        return False
    saved_epochs = saved_optimization.get("epochs")
    requested_epochs = requested_optimization.get("epochs")
    if (
        isinstance(saved_epochs, bool)
        or not isinstance(saved_epochs, int)
        or isinstance(requested_epochs, bool)
        or not isinstance(requested_epochs, int)
        or saved_epochs > requested_epochs
    ):
        return False
    candidate = dict(saved)
    candidate["optimization"] = dict(saved_optimization)
    candidate["optimization"]["epochs"] = requested_epochs
    return candidate == requested


def save_checkpoint(
    path: Path | str,
    model: TeSNeurons,
    adam_state: AdamState,
    *,
    epoch: int,
    config: TrainingConfig,
    calibration_history: Sequence[object] = (),
    bump_history: Sequence[object] = (),
    epoch_metrics: Sequence[object] = (),
    run_config: dict[str, object] | None = None,
    data_identity: dict[str, object] | None = None,
    model_config: dict[str, object] | None = None,
    baseline_train: object | None = None,
    baseline_validation: object | None = None,
    best_validation_snapshot: dict[str, object] | None = None,
) -> Path:
    """Atomically save a versioned, restartable training state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "hidden_silence_metric_version": HIDDEN_SILENCE_METRIC_VERSION,
        "epoch": int(epoch),
        "config": asdict(config),
        "w_in": model.W_in.detach().clone(),
        "w_out": model.W_out.detach().clone(),
        "omega": model.omega.detach().clone(),
        "adam": adam_state.to_payload(),
        "calibration_history": tuple(calibration_history),
        "bump_history": tuple(bump_history),
        "epoch_metrics": tuple(epoch_metrics),
        "run_config": run_config,
        "data_identity": data_identity,
        "model_config": model_config,
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "best_validation_snapshot": best_validation_snapshot,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: Path | str,
    model: TeSNeurons,
    *,
    expected_config: TrainingConfig | None = None,
    expected_run_config: dict[str, object] | None = None,
    allow_epoch_extension: bool = False,
    allow_omega_learning_upgrade: bool = False,
) -> TrainingState:
    """Validate and restore a saved training state into ``model``."""
    payload = torch.load(Path(path), map_location=model.W_in.device, weights_only=True)
    schema_version = payload.get("schema_version")
    if schema_version not in {
        LEGACY_CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise CheckpointMismatchError("unsupported checkpoint schema")
    if payload.get("model_semantics_version") != MODEL_SEMANTICS_VERSION:
        raise CheckpointMismatchError("checkpoint model semantics mismatch")
    if payload.get("training_semantics_version") != TRAINING_SEMANTICS_VERSION:
        raise CheckpointMismatchError("checkpoint training semantics mismatch")
    if (
        payload.get("hidden_silence_metric_version")
        != HIDDEN_SILENCE_METRIC_VERSION
    ):
        raise CheckpointMismatchError(
            "checkpoint hidden silence metric mismatch"
        )
    config = _training_config_from_payload(payload["config"])
    if expected_config is not None and config != expected_config:
        saved_config = asdict(config)
        requested_config = asdict(expected_config)
        compatible = dict(saved_config)
        if allow_epoch_extension and config.epochs <= expected_config.epochs:
            compatible["epochs"] = expected_config.epochs
        if (
            allow_omega_learning_upgrade
            and not config.learn_omega
            and expected_config.learn_omega
        ):
            compatible["learn_omega"] = True
        if compatible != requested_config:
            raise CheckpointMismatchError(
                "checkpoint training configuration mismatch"
            )
    if expected_run_config is not None and not _resolved_config_matches(
        payload.get("run_config"),
        expected_run_config,
        allow_epoch_extension=allow_epoch_extension,
    ):
        raise CheckpointMismatchError("checkpoint resolved configuration mismatch")
    w_in = payload["w_in"]
    w_out = payload["w_out"]
    omega = payload.get("omega", model.omega.detach().clone())
    if (
        w_in.shape != model.W_in.shape
        or w_out.shape != model.W_out.shape
        or omega.shape != model.omega.shape
        or w_in.dtype != torch.float64
        or w_out.dtype != torch.float64
        or omega.dtype != torch.float64
        or not torch.isfinite(omega).all()
        or torch.any(omega <= 0.0)
    ):
        raise CheckpointMismatchError(
            "checkpoint weight or omega shape, dtype, or value mismatch"
        )
    adam_payload = payload["adam"]
    m_omega = adam_payload.get("m_omega", torch.zeros_like(model.omega))
    v_omega = adam_payload.get("v_omega", torch.zeros_like(model.omega))
    tensors = (
        adam_payload["m_in"],
        adam_payload["v_in"],
        adam_payload["m_out"],
        adam_payload["v_out"],
        m_omega,
        v_omega,
    )
    expected_shapes = (
        model.W_in.shape,
        model.W_in.shape,
        model.W_out.shape,
        model.W_out.shape,
        model.omega.shape,
        model.omega.shape,
    )
    if any(
        tensor.shape != shape or tensor.dtype != torch.float64
        for tensor, shape in zip(tensors, expected_shapes, strict=True)
    ):
        raise CheckpointMismatchError("checkpoint Adam shape or dtype mismatch")
    model.W_in.copy_(w_in)
    model.W_out.copy_(w_out)
    model.omega.copy_(omega)
    adam = AdamState(
        m_in=tensors[0].clone(),
        v_in=tensors[1].clone(),
        m_out=tensors[2].clone(),
        v_out=tensors[3].clone(),
        m_omega=tensors[4].clone(),
        v_omega=tensors[5].clone(),
        step=int(adam_payload["step"]),
        omega_step=int(adam_payload.get("omega_step", 0)),
    )
    return TrainingState(
        epoch=int(payload["epoch"]),
        config=config,
        adam=adam,
        calibration_history=tuple(payload["calibration_history"]),
        bump_history=tuple(payload["bump_history"]),
        epoch_metrics=tuple(payload["epoch_metrics"]),
        run_config=payload.get("run_config"),
        data_identity=payload.get("data_identity"),
        model_config=payload.get("model_config"),
        baseline_train=payload.get("baseline_train"),
        baseline_validation=payload.get("baseline_validation"),
        best_validation_snapshot=payload.get("best_validation_snapshot"),
    )
