"""Immutable event tapes and compact results for TeS simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class InputSpike:
    exact_time: float
    grid_step: int
    channel: int


@dataclass(frozen=True)
class DiscreteArrival:
    event_id: int
    time: float
    step: int
    kind: Literal["input", "hidden"]
    source: int
    exact_time: float
    parent_threshold_id: int | None


@dataclass(frozen=True)
class DiscreteThreshold:
    event_id: int
    time: float
    step: int
    neuron: int
    slope: float
    v_pre: float
    i_pre: float
    q_pre: float
    v_post: float
    continuous_delay: float | None
    delay_steps: int | None
    arrival_time: float | None  # Exact delayed event time, not its grid delivery time.


@dataclass(frozen=True)
class ContinuousArrival:
    event_id: int
    sequence: int
    time: float
    kind: Literal["input", "hidden"]
    source: int
    parent_threshold_id: int | None


@dataclass(frozen=True)
class ContinuousThreshold:
    event_id: int
    layer: Literal["hidden", "output"]
    neuron: int
    time: float
    slope: float
    acceleration_pre: float
    v_pre: float
    i_pre: float
    q_pre: float
    v_post: float
    continuous_delay: float | None
    arrival_time: float | None  # Exact delayed threshold time for either layer.


@dataclass(frozen=True)
class EffectiveReadout:
    neuron: int
    physical_threshold_id: int | None
    physical_time: float | None
    effective_time: float
    censored: bool
    anchor_output: int | None


@dataclass(frozen=True)
class StopInfo:
    reason: Literal[
        "all_readouts_physical",
        "all_readouts_resolved",
        "all_readouts_silent",
        "requested_horizon",
    ]
    requested_t_end: float
    actual_t_end: float
    hidden_flow_intervals: int
    output_flow_intervals: int
    root_panels: int


@dataclass(frozen=True)
class DiscreteForwardResult:
    times: torch.Tensor | None
    V: torch.Tensor | None
    I: torch.Tensor | None
    Q: torch.Tensor | None
    arrivals: tuple[DiscreteArrival, ...]
    thresholds: tuple[DiscreteThreshold, ...]
    readouts: tuple[EffectiveReadout, ...]
    warnings: tuple[str, ...]
    stop: StopInfo


@dataclass(frozen=True)
class ContinuousForwardResult:
    input_arrivals: tuple[ContinuousArrival, ...]
    hidden_spikes: tuple[ContinuousThreshold, ...]
    hidden_arrivals: tuple[ContinuousArrival, ...]
    output_spikes: tuple[ContinuousThreshold, ...]
    readouts: tuple[EffectiveReadout, ...]
    stop: StopInfo


@dataclass(frozen=True)
class ContinuousOmegaCheckpoint:
    """Post-event oscillator state at one neuron's local event boundary."""

    neuron: int
    time: float
    i: float
    q: float


@dataclass(frozen=True)
class ContinuousOmegaForwardResult(ContinuousForwardResult):
    omega_checkpoints: tuple[ContinuousOmegaCheckpoint, ...]


@dataclass(frozen=True)
class DelGradLoss:
    loss: float
    effective_times: tuple[float, ...]
    physical_mask: tuple[bool, ...]
    active_wrong_outputs: tuple[int, ...]
    fake_anchor_output: int | None
    output_time_grad: torch.Tensor | None  # dL / d(effective delayed readout time)
    nondifferentiable_reason: str | None


@dataclass(frozen=True)
class BackwardResult:
    output_time_grad: torch.Tensor
    hidden_spike_time_grad: torch.Tensor
    grad_W_in: torch.Tensor
    grad_W_out: torch.Tensor
    threshold_sensitivity_cap_activations: int
    minimum_causal_slope: float | None


@dataclass(frozen=True)
class OmegaBackwardResult(BackwardResult):
    """Training gradients with one independent frequency gradient per neuron."""

    grad_omega: torch.Tensor
