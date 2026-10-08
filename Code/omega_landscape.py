"""Topology-aware loss-landscape diagnostics for per-neuron omega gradients.

The continuous event simulation is the primary tangent test.  A discrete
simulation at a fixed fine time step is evaluated from the same weights,
input times, and frequency vector and is shown as a numerical cross-check.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Literal, Sequence

import torch

from Code.neurons import TeSNeurons
from Code.tes_events import (
    ContinuousOmegaForwardResult,
    DelGradLoss,
    DiscreteForwardResult,
    InputSpike,
)
from Code.tes_loss import TimingLossName, evaluate_timing_loss
from Code.tes_training import build_yinyang_splits, load_checkpoint
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config
from Code.yinyang_experiment import initialize_yinyang_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "results" / "yinyang_best" / "checkpoints" / "best_validation.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "omega_landscape"

SCIENTIFIC_STATUS = (
    "continuous event-forward tangent test within fixed causal topology and "
    "with the threshold-sensitivity cap inactive; "
    "the returned fixed-step discrete adjoint is an overlaid numerical "
    "cross-check, not a secant derivative of the quantized grid loss"
)

TopologySignature = tuple[object, ...]


@dataclass(frozen=True)
class OmegaLandscapePoint:
    omega: float
    continuous_valid: bool
    continuous_missing_reason: str | None
    continuous_loss: float | None
    continuous_gradient: float | None
    continuous_topology_signature: TopologySignature | None
    continuous_topology_segment: int | None
    continuous_cap_activations: int | None
    continuous_minimum_causal_slope: float | None
    continuous_secant: float | None = None
    continuous_absolute_error: float | None = None
    continuous_relative_error: float | None = None
    discrete_valid: bool = False
    discrete_missing_reason: str | None = None
    discrete_loss: float | None = None
    discrete_gradient: float | None = None
    discrete_topology_signature: TopologySignature | None = None
    discrete_topology_segment: int | None = None
    discrete_cap_activations: int | None = None
    discrete_minimum_causal_slope: float | None = None
    continuous_comparison_signature: TopologySignature | None = None
    discrete_comparison_signature: TopologySignature | None = None


@dataclass(frozen=True)
class OmegaLandscape:
    settings: dict[str, object]
    points: tuple[OmegaLandscapePoint, ...]


@dataclass(frozen=True)
class _ModeEvaluation:
    valid: bool
    missing_reason: str | None
    loss: float | None
    gradient: float | None
    topology_signature: TopologySignature | None
    comparison_signature: TopologySignature | None
    cap_activations: int | None
    minimum_causal_slope: float | None


def _clone_model(model: TeSNeurons, *, dt: float) -> TeSNeurons:
    device = model.W_in.device
    clone = TeSNeurons(
        model.n_in,
        model.n_hidden,
        model.n_out,
        model.W_in.detach().cpu(),
        model.W_out.detach().cpu(),
        replace(model.config, dt=dt),
    ).to(device)
    clone.omega.copy_(model.omega)
    return clone


def _regrid_inputs(
    inputs: Sequence[InputSpike], dt: float
) -> tuple[InputSpike, ...]:
    return tuple(
        InputSpike(
            exact_time=float(spike.exact_time),
            grid_step=math.floor(float(spike.exact_time) / dt + 0.5),
            channel=spike.channel,
        )
        for spike in inputs
    )


def _loss_signature(loss: DelGradLoss) -> tuple[object, ...]:
    return (
        loss.physical_mask,
        loss.active_wrong_outputs,
        loss.fake_anchor_output,
    )


def _local_boundary_signature(
    *,
    thresholds: Sequence[tuple[float, tuple[object, ...]]],
    arrivals: Sequence[tuple[float, tuple[object, ...]]],
) -> tuple[tuple[tuple[object, ...], tuple[object, ...]], ...]:
    """Record local boundary interleaving, including simultaneous boundaries."""
    grouped: dict[
        float,
        tuple[list[tuple[object, ...]], list[tuple[object, ...]]],
    ] = {}
    for time, identity in thresholds:
        threshold_group, _ = grouped.setdefault(float(time), ([], []))
        threshold_group.append(identity)
    for time, identity in arrivals:
        _, arrival_group = grouped.setdefault(float(time), ([], []))
        arrival_group.append(identity)
    return tuple(
        (
            tuple(sorted(grouped[time][0], key=repr)),
            tuple(sorted(grouped[time][1], key=repr)),
        )
        for time in sorted(grouped)
    )


def _continuous_topology(
    model: TeSNeurons,
    forward: ContinuousOmegaForwardResult,
    loss: DelGradLoss,
) -> TopologySignature:
    hidden_identities: dict[int, tuple[int, int]] = {}
    hidden_thresholds: list[list[tuple[float, tuple[object, ...]]]] = [
        [] for _ in range(model.n_hidden)
    ]
    for hidden in range(model.n_hidden):
        events = sorted(
            (event for event in forward.hidden_spikes if event.neuron == hidden),
            key=lambda event: (event.time, event.event_id),
        )
        for ordinal, event in enumerate(events):
            hidden_identities[event.event_id] = (hidden, ordinal)
            hidden_thresholds[hidden].append(
                (event.time, ("threshold", ordinal))
            )
    input_arrivals = tuple(
        (
            arrival.time,
            ("input", arrival.source, float(arrival.time)),
        )
        for arrival in forward.input_arrivals
    )
    hidden_local = tuple(
        _local_boundary_signature(
            thresholds=hidden_thresholds[hidden],
            arrivals=input_arrivals,
        )
        for hidden in range(model.n_hidden)
    )

    output_identities: dict[int, tuple[int, int]] = {}
    output_thresholds: list[list[tuple[float, tuple[object, ...]]]] = [
        [] for _ in range(model.n_out)
    ]
    for output in range(model.n_out):
        events = sorted(
            (event for event in forward.output_spikes if event.neuron == output),
            key=lambda event: (event.time, event.event_id),
        )
        for ordinal, event in enumerate(events):
            output_identities[event.event_id] = (output, ordinal)
            output_thresholds[output].append(
                (event.time, ("threshold", ordinal))
            )
    hidden_arrivals = tuple(
        (
            arrival.time,
            ("hidden", *hidden_identities[arrival.parent_threshold_id]),
        )
        for arrival in forward.hidden_arrivals
        if arrival.parent_threshold_id in hidden_identities
    )
    output_local = tuple(
        _local_boundary_signature(
            thresholds=output_thresholds[output],
            arrivals=hidden_arrivals,
        )
        for output in range(model.n_out)
    )
    readout_identities = tuple(
        None
        if readout.physical_threshold_id is None
        else output_identities[readout.physical_threshold_id]
        for readout in forward.readouts
    )
    return (
        hidden_local,
        output_local,
        readout_identities,
        _loss_signature(loss),
    )


def _discrete_topology(
    model: TeSNeurons,
    forward: DiscreteForwardResult,
    loss: DelGradLoss,
) -> TopologySignature:
    identities: dict[int, tuple[int, int]] = {}
    thresholds_by_neuron: list[list[tuple[float, tuple[object, ...]]]] = [
        [] for _ in range(model.n_neurons)
    ]
    for neuron in range(model.n_neurons):
        events = sorted(
            (event for event in forward.thresholds if event.neuron == neuron),
            key=lambda event: (event.step, event.event_id),
        )
        for ordinal, event in enumerate(events):
            identities[event.event_id] = (neuron, ordinal)
            thresholds_by_neuron[neuron].append(
                (event.time, ("threshold", ordinal))
            )
    input_arrivals = tuple(
        (
            arrival.time,
            ("input", arrival.source, float(arrival.exact_time)),
        )
        for arrival in forward.arrivals
        if arrival.kind == "input"
    )
    hidden_local = tuple(
        _local_boundary_signature(
            thresholds=thresholds_by_neuron[hidden],
            arrivals=input_arrivals,
        )
        for hidden in range(model.n_hidden)
    )
    hidden_arrivals = tuple(
        (
            arrival.time,
            (
                "hidden",
                *identities[arrival.parent_threshold_id],
            ),
        )
        for arrival in forward.arrivals
        if arrival.kind == "hidden"
        and arrival.parent_threshold_id in identities
    )
    output_local = tuple(
        _local_boundary_signature(
            thresholds=thresholds_by_neuron[model.n_hidden + output],
            arrivals=hidden_arrivals,
        )
        for output in range(model.n_out)
    )
    readout_identities = tuple(
        None
        if readout.physical_threshold_id is None
        else (
            identities[readout.physical_threshold_id][0] - model.n_hidden,
            identities[readout.physical_threshold_id][1],
        )
        for readout in forward.readouts
    )
    return (
        hidden_local,
        output_local,
        readout_identities,
        _loss_signature(loss),
    )


def _evaluate_continuous(
    model: TeSNeurons,
    inputs: Sequence[InputSpike],
    *,
    label: int,
    neuron: int,
    loss_name: TimingLossName,
    margin: float,
    t_end: float,
) -> _ModeEvaluation:
    forward = model.forward_continuous_omega(inputs, t_end, early_stop=False)
    loss = evaluate_timing_loss(
        forward.readouts,
        label=label,
        loss_name=loss_name,
        margin=margin,
        device=model.W_in.device,
    )
    if loss.output_time_grad is None:
        return _ModeEvaluation(
            valid=False,
            missing_reason=loss.nondifferentiable_reason,
            loss=float(loss.loss),
            gradient=None,
            topology_signature=None,
            comparison_signature=None,
            cap_activations=None,
            minimum_causal_slope=None,
        )
    backward = model.backward_continuous_omega(
        forward, loss.output_time_grad
    )
    topology = _continuous_topology(model, forward, loss)
    return _ModeEvaluation(
        valid=True,
        missing_reason=None,
        loss=float(loss.loss),
        gradient=float(backward.grad_omega[neuron]),
        topology_signature=topology,
        comparison_signature=topology,
        cap_activations=backward.threshold_sensitivity_cap_activations,
        minimum_causal_slope=backward.minimum_causal_slope,
    )


def _evaluate_discrete(
    model: TeSNeurons,
    inputs: Sequence[InputSpike],
    *,
    label: int,
    neuron: int,
    loss_name: TimingLossName,
    margin: float,
    t_end: float,
) -> _ModeEvaluation:
    forward = model.forward_discrete_omega(inputs, t_end, early_stop=False)
    loss = evaluate_timing_loss(
        forward.readouts,
        label=label,
        loss_name=loss_name,
        margin=margin,
        device=model.W_in.device,
    )
    if loss.output_time_grad is None:
        return _ModeEvaluation(
            valid=False,
            missing_reason=loss.nondifferentiable_reason,
            loss=float(loss.loss),
            gradient=None,
            topology_signature=None,
            comparison_signature=None,
            cap_activations=None,
            minimum_causal_slope=None,
        )
    backward = model.backward_discrete_omega(forward, loss.output_time_grad)
    topology = _discrete_topology(model, forward, loss)
    return _ModeEvaluation(
        valid=True,
        missing_reason=None,
        loss=float(loss.loss),
        gradient=float(backward.grad_omega[neuron]),
        topology_signature=topology,
        comparison_signature=topology,
        cap_activations=backward.threshold_sensitivity_cap_activations,
        minimum_causal_slope=backward.minimum_causal_slope,
    )


def _add_segments(
    points: tuple[OmegaLandscapePoint, ...],
    mode: Literal["continuous", "discrete"],
) -> tuple[OmegaLandscapePoint, ...]:
    result: list[OmegaLandscapePoint] = []
    previous_signature: TopologySignature | None = None
    segment = -1
    for point in points:
        valid = getattr(point, f"{mode}_valid")
        signature = getattr(point, f"{mode}_topology_signature")
        if not valid or signature is None:
            previous_signature = None
            result.append(point)
            continue
        if signature != previous_signature:
            segment += 1
        result.append(
            replace(point, **{f"{mode}_topology_segment": segment})
        )
        previous_signature = signature
    return tuple(result)


def _add_continuous_centered_secants(
    points: tuple[OmegaLandscapePoint, ...],
) -> tuple[OmegaLandscapePoint, ...]:
    mutable = list(points)
    for index in range(1, len(points) - 1):
        left, center, right = points[index - 1 : index + 2]
        candidates = (left, center, right)
        if not all(point.continuous_valid for point in candidates):
            continue
        if any(point.continuous_cap_activations != 0 for point in candidates):
            continue
        signatures = tuple(
            point.continuous_topology_signature for point in candidates
        )
        if signatures[0] is None or not (
            signatures[0] == signatures[1] == signatures[2]
        ):
            continue
        left_loss = left.continuous_loss
        right_loss = right.continuous_loss
        gradient = center.continuous_gradient
        if any(
            value is None or not math.isfinite(value)
            for value in (left_loss, right_loss, gradient)
        ):
            continue
        denominator = right.omega - left.omega
        if denominator <= 0.0 or not math.isfinite(denominator):
            continue
        secant = (right_loss - left_loss) / denominator
        absolute = abs(gradient - secant)
        relative = absolute / max(abs(gradient), abs(secant), 1e-12)
        mutable[index] = replace(
            center,
            continuous_secant=secant,
            continuous_absolute_error=absolute,
            continuous_relative_error=relative,
        )
    return tuple(mutable)


def run_omega_loss_landscape(
    model: TeSNeurons,
    inputs: Sequence[InputSpike],
    *,
    label: int,
    neuron: int,
    loss_name: TimingLossName,
    margin: float,
    half_width: float,
    num_points: int,
    t_end: float,
    discrete_dt: float = 0.005,
    provenance: dict[str, object] | None = None,
) -> OmegaLandscape:
    """Sweep one neuron's omega and compare continuous tangents with loss secants."""
    if isinstance(neuron, bool) or not isinstance(neuron, int):
        raise TypeError("neuron must be an integer")
    if not 0 <= neuron < model.n_neurons:
        raise IndexError("neuron is outside the learned omega vector")
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("label must be an integer")
    if not 0 <= label < model.n_out:
        raise ValueError("label is outside the output range")
    if not math.isfinite(discrete_dt) or discrete_dt <= 0.0:
        raise ValueError("discrete_dt must be finite and positive")
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width must be finite and positive")
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, int)
        or num_points < 3
    ):
        raise ValueError("num_points must be an integer of at least three")
    if not math.isfinite(t_end) or t_end <= 0.0:
        raise ValueError("t_end must be finite and positive")
    baseline = float(model.omega[neuron])
    if baseline - half_width <= 0.0:
        raise ValueError("the complete omega sweep must remain strictly positive")

    source_omega = model.omega.clone()
    source_w_in = model.W_in.clone()
    source_w_out = model.W_out.clone()
    discrete_inputs = _regrid_inputs(inputs, discrete_dt)
    frequencies = tuple(
        baseline - half_width
        + 2.0 * half_width * index / (num_points - 1)
        for index in range(num_points)
    )
    points: list[OmegaLandscapePoint] = []
    for omega in frequencies:
        continuous_model = _clone_model(model, dt=discrete_dt)
        discrete_model = _clone_model(model, dt=discrete_dt)
        continuous_model.omega[neuron] = omega
        discrete_model.omega[neuron] = omega
        continuous = _evaluate_continuous(
            continuous_model,
            inputs,
            label=label,
            neuron=neuron,
            loss_name=loss_name,
            margin=margin,
            t_end=t_end,
        )
        discrete = _evaluate_discrete(
            discrete_model,
            discrete_inputs,
            label=label,
            neuron=neuron,
            loss_name=loss_name,
            margin=margin,
            t_end=t_end,
        )
        points.append(
            OmegaLandscapePoint(
                omega=float(omega),
                continuous_valid=continuous.valid,
                continuous_missing_reason=continuous.missing_reason,
                continuous_loss=continuous.loss,
                continuous_gradient=continuous.gradient,
                continuous_topology_signature=continuous.topology_signature,
                continuous_topology_segment=None,
                continuous_cap_activations=continuous.cap_activations,
                continuous_minimum_causal_slope=(
                    continuous.minimum_causal_slope
                ),
                discrete_valid=discrete.valid,
                discrete_missing_reason=discrete.missing_reason,
                discrete_loss=discrete.loss,
                discrete_gradient=discrete.gradient,
                discrete_topology_signature=discrete.topology_signature,
                discrete_topology_segment=None,
                discrete_cap_activations=discrete.cap_activations,
                discrete_minimum_causal_slope=discrete.minimum_causal_slope,
                continuous_comparison_signature=(
                    continuous.comparison_signature
                ),
                discrete_comparison_signature=discrete.comparison_signature,
            )
        )

    processed = tuple(points)
    for mode in ("continuous", "discrete"):
        processed = _add_segments(processed, mode)
    processed = _add_continuous_centered_secants(processed)
    if not (
        torch.equal(model.omega, source_omega)
        and torch.equal(model.W_in, source_w_in)
        and torch.equal(model.W_out, source_w_out)
    ):
        raise RuntimeError("omega landscape mutated the source model")
    layer = "hidden" if neuron < model.n_hidden else "output"
    local_neuron = neuron if layer == "hidden" else neuron - model.n_hidden
    return OmegaLandscape(
        settings={
            "neuron": neuron,
            "layer": layer,
            "local_neuron": local_neuron,
            "baseline_omega": baseline,
            "half_width": half_width,
            "num_points": num_points,
            "t_end": t_end,
            "label": label,
            "loss_name": loss_name,
            "margin": margin,
            "discrete_dt": discrete_dt,
            "provenance": (
                dict(provenance)
                if provenance is not None
                else {"source": "in_memory_model"}
            ),
        },
        points=processed,
    )


def _finite_values(
    landscape: OmegaLandscape, field: str
) -> list[float]:
    return [
        value
        for point in landscape.points
        if (value := getattr(point, field)) is not None and math.isfinite(value)
    ]


def _cross_mode_relative_errors(
    landscape: OmegaLandscape,
) -> tuple[list[float], int]:
    errors = []
    excluded_topology = 0
    for point in landscape.points:
        continuous = point.continuous_gradient
        discrete = point.discrete_gradient
        if (
            continuous is None
            or discrete is None
            or not math.isfinite(continuous)
            or not math.isfinite(discrete)
        ):
            continue
        if (
            point.continuous_comparison_signature is None
            or point.continuous_comparison_signature
            != point.discrete_comparison_signature
        ):
            excluded_topology += 1
            continue
        errors.append(
            abs(continuous - discrete)
            / max(abs(continuous), abs(discrete), 1e-12)
        )
    return errors, excluded_topology


def landscape_aggregates(landscape: OmegaLandscape) -> dict[str, object]:
    continuous_errors = _finite_values(
        landscape, "continuous_relative_error"
    )
    cross_mode_errors, excluded_topology = _cross_mode_relative_errors(
        landscape
    )
    return {
        "attempted_points": len(landscape.points),
        "continuous_valid_points": sum(
            point.continuous_valid for point in landscape.points
        ),
        "continuous_cap_active_points": sum(
            (point.continuous_cap_activations or 0) > 0
            for point in landscape.points
        ),
        "discrete_valid_points": sum(
            point.discrete_valid for point in landscape.points
        ),
        "discrete_cap_active_points": sum(
            (point.discrete_cap_activations or 0) > 0
            for point in landscape.points
        ),
        "continuous_stable_secant_points": len(continuous_errors),
        "continuous_topology_segments": len(
            {
                point.continuous_topology_segment
                for point in landscape.points
                if point.continuous_topology_segment is not None
            }
        ),
        "discrete_topology_segments": len(
            {
                point.discrete_topology_segment
                for point in landscape.points
                if point.discrete_topology_segment is not None
            }
        ),
        "continuous_median_relative_error": (
            statistics.median(continuous_errors) if continuous_errors else None
        ),
        "continuous_maximum_relative_error": (
            max(continuous_errors) if continuous_errors else None
        ),
        "continuous_discrete_compared_points": len(cross_mode_errors),
        "continuous_discrete_excluded_topology_points": excluded_topology,
        "continuous_discrete_median_gradient_relative_error": (
            statistics.median(cross_mode_errors) if cross_mode_errors else None
        ),
        "continuous_discrete_maximum_gradient_relative_error": (
            max(cross_mode_errors) if cross_mode_errors else None
        ),
    }


def write_omega_landscape(
    landscape: OmegaLandscape, path: Path | str
) -> Path:
    path = Path(path)
    payload = {
        "scientific_status": SCIENTIFIC_STATUS,
        "settings": landscape.settings,
        "aggregates": landscape_aggregates(landscape),
        "points": [asdict(point) for point in landscape.points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path


def _evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    if len(indices) <= count:
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [
        indices[round(position * (len(indices) - 1) / (count - 1))]
        for position in range(count)
    ]


def _plot_segmented(
    axis,
    points: tuple[OmegaLandscapePoint, ...],
    *,
    mode: Literal["continuous", "discrete"],
    quantity: Literal["loss", "gradient"],
    color: str,
    linestyle: str,
    label: str,
) -> None:
    current: list[OmegaLandscapePoint] = []
    current_segment: int | None = None
    labeled = False

    def flush() -> None:
        nonlocal current, labeled
        if not current:
            return
        axis.plot(
            [point.omega for point in current],
            [getattr(point, f"{mode}_{quantity}") for point in current],
            color=color,
            linestyle=linestyle,
            marker=".",
            markersize=2.5,
            label=None if labeled else label,
            linewidth=1.7,
        )
        labeled = True
        current = []

    for point in points:
        value = getattr(point, f"{mode}_{quantity}")
        segment = getattr(point, f"{mode}_topology_segment")
        if (
            not getattr(point, f"{mode}_valid")
            or value is None
            or not math.isfinite(value)
        ):
            flush()
            current_segment = None
            continue
        if current and segment != current_segment:
            flush()
        current.append(point)
        current_segment = segment
    flush()


def plot_omega_landscape(
    landscape: OmegaLandscape,
    path: Path | str,
    *,
    arrow_count: int = 7,
) -> Path:
    """Plot continuous loss tangents and the fixed-dt discrete overlay."""
    if (
        isinstance(arrow_count, bool)
        or not isinstance(arrow_count, int)
        or not 1 <= arrow_count <= 9
    ):
        raise ValueError("arrow_count must be an integer from one through nine")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    figure, (loss_axis, gradient_axis) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True
    )
    points = landscape.points
    _plot_segmented(
        loss_axis,
        points,
        mode="continuous",
        quantity="loss",
        color="C0",
        linestyle="-",
        label="continuous event loss",
    )
    _plot_segmented(
        loss_axis,
        points,
        mode="discrete",
        quantity="loss",
        color="C1",
        linestyle="--",
        label=f"discrete loss (dt={landscape.settings['discrete_dt']} ms)",
    )
    tangent_indices = [
        index
        for index, point in enumerate(points)
        if point.continuous_loss is not None
        and point.continuous_gradient is not None
        and point.continuous_secant is not None
        and math.isfinite(point.continuous_loss)
        and math.isfinite(point.continuous_gradient)
    ]
    omega_extent = points[-1].omega - points[0].omega
    arrow_extent = 0.035 * omega_extent
    for index in _evenly_spaced_indices(tangent_indices, arrow_count):
        point = points[index]
        assert point.continuous_loss is not None
        assert point.continuous_gradient is not None
        loss_axis.annotate(
            "",
            xy=(
                point.omega + arrow_extent,
                point.continuous_loss
                + point.continuous_gradient * arrow_extent,
            ),
            xytext=(point.omega, point.continuous_loss),
            arrowprops={"arrowstyle": "->", "color": "C2", "lw": 1.4},
        )
    if tangent_indices:
        loss_axis.plot(
            [], [], color="C2", marker=">", linestyle="none",
            label="continuous analytic tangent",
        )

    _plot_segmented(
        gradient_axis,
        points,
        mode="continuous",
        quantity="gradient",
        color="C2",
        linestyle="-",
        label="continuous returned gradient",
    )
    _plot_segmented(
        gradient_axis,
        points,
        mode="discrete",
        quantity="gradient",
        color="C4",
        linestyle="--",
        label=f"discrete gradient (dt={landscape.settings['discrete_dt']} ms)",
    )
    continuous_secants = [
        point for point in points if point.continuous_secant is not None
    ]
    if continuous_secants:
        gradient_axis.scatter(
            [point.omega for point in continuous_secants],
            [point.continuous_secant for point in continuous_secants],
            color="black",
            marker=".",
            s=18,
            label="continuous centered loss secant",
            zorder=4,
        )
    cap_active = [
        point
        for point in points
        if (point.continuous_cap_activations or 0) > 0
        and point.continuous_gradient is not None
    ]
    if cap_active:
        gradient_axis.scatter(
            [point.omega for point in cap_active],
            [point.continuous_gradient for point in cap_active],
            color="C3",
            marker="x",
            s=38,
            label="cap-active approximate gradient",
            zorder=5,
        )
    baseline = float(landscape.settings["baseline_omega"])
    for axis in (loss_axis, gradient_axis):
        axis.axvline(baseline, color="0.35", linestyle=":", linewidth=1.0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    layer = landscape.settings["layer"]
    local = landscape.settings["local_neuron"]
    neuron = landscape.settings["neuron"]
    loss_axis.set_title(
        f"Omega loss-landscape tangent test: {layer} {local} "
        f"(global neuron {neuron})"
    )
    loss_axis.set_ylabel("timing loss")
    gradient_axis.set_xlabel("omega (rad/ms)")
    gradient_axis.set_ylabel("d loss / d omega")
    caption = (
        f"Continuous event solution is primary; discrete overlay uses dt="
        f"{landscape.settings['discrete_dt']} ms. Centered secants are shown "
        f"only when all three continuous points share one causal topology "
        f"and have no threshold-sensitivity cap activation."
    )
    figure.text(0.5, 0.01, caption, ha="center", va="bottom", wrap=True)
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=170,
        metadata={
            "Title": "Omega loss-landscape tangent test",
            "Description": caption,
        },
    )
    plt.close(figure)
    return path


def _model_from_checkpoint(
    config, checkpoint_path: Path, dt: float
) -> tuple[TeSNeurons, dict[str, object]]:
    base_model = initialize_yinyang_model(config)
    state = load_checkpoint(
        checkpoint_path,
        base_model,
        expected_config=config.to_training_config(),
        expected_run_config=config.to_dict(),
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return _clone_model(base_model, dt=dt), {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": state.epoch,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description="Plot continuous omega tangents with a fine-grid overlay"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--sample-index", type=int, default=4)
    parser.add_argument("--neuron", type=int, action="append")
    parser.add_argument("--half-width", type=float, default=0.015)
    parser.add_argument("--num-points", type=int, default=81)
    parser.add_argument("--discrete-dt", type=float, default=0.005)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args(argv)
    config = load_yinyang_config(arguments.config)
    splits = build_yinyang_splits(
        config.data.train_size,
        config.data.validation_size,
        config.data.test_size,
        config.data.split_seeds,
        dt=arguments.discrete_dt,
    )
    split_index = {"train": 0, "validation": 1, "test": 2}[arguments.split]
    selected = splits[split_index]
    if not 0 <= arguments.sample_index < len(selected.samples):
        raise IndexError("sample index is outside the selected split")
    sample = selected.samples[arguments.sample_index]
    model, checkpoint_provenance = _model_from_checkpoint(
        config, arguments.checkpoint, arguments.discrete_dt
    )
    neurons = arguments.neuron
    if neurons is None:
        neurons = [model.n_hidden - 1, model.n_hidden]
    records = []
    for neuron in neurons:
        landscape = run_omega_loss_landscape(
            model,
            sample.events,
            label=sample.label,
            neuron=neuron,
            loss_name=config.loss.name,
            margin=config.loss.margin_ms,
            half_width=arguments.half_width,
            num_points=arguments.num_points,
            t_end=config.model.t_end_ms,
            discrete_dt=arguments.discrete_dt,
            provenance={
                **checkpoint_provenance,
                "config_path": str(arguments.config.resolve()),
                "split": arguments.split,
                "split_seed": selected.seed,
                "sample_index": arguments.sample_index,
                "sample_point": tuple(float(value) for value in sample.point),
                "sample_label": sample.label,
            },
        )
        stem = f"omega_neuron_{neuron:02d}"
        record_path = write_omega_landscape(
            landscape, arguments.output_dir / f"{stem}.json"
        )
        figure_path = plot_omega_landscape(
            landscape, arguments.output_dir / f"{stem}.png"
        )
        records.append(
            {
                "neuron": neuron,
                "record": str(record_path),
                "figure": str(figure_path),
                "aggregates": landscape_aggregates(landscape),
            }
        )
    summary = {
        "split": arguments.split,
        "sample_index": arguments.sample_index,
        "label": sample.label,
        "discrete_dt": arguments.discrete_dt,
        "landscapes": records,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


if __name__ == "__main__":
    main()
