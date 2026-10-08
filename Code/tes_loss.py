"""Readout-timing objectives independent of TeS neuron dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from Code.tes_events import DelGradLoss, EffectiveReadout


TimingLossName = Literal["delgrad", "squared_hinge"]


@dataclass(frozen=True)
class TimingLossSpecification:
    """Stable public description of one selectable timing objective."""

    name: TimingLossName
    default_margin_ms: float
    objective: str


_TIMING_LOSSES: dict[str, TimingLossSpecification] = {
    "delgrad": TimingLossSpecification(
        name="delgrad",
        default_margin_ms=1.0,
        objective="equality_mse",
    ),
    "squared_hinge": TimingLossSpecification(
        name="squared_hinge",
        default_margin_ms=2.0,
        objective="minimum_margin_squared_hinge",
    ),
}


def get_timing_loss(name: str) -> TimingLossSpecification:
    """Resolve a timing objective by its recorded configuration name."""
    try:
        return _TIMING_LOSSES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_TIMING_LOSSES))
        raise ValueError(
            f"unknown timing loss {name!r}; expected one of: {choices}"
        ) from error


def evaluate_timing_loss(
    readouts: Sequence[EffectiveReadout],
    *,
    label: int,
    loss_name: TimingLossName,
    margin: float | None = None,
    device: torch.device | str | None = None,
) -> DelGradLoss:
    """Return a scalar timing error and its physical-readout time sources."""
    specification = get_timing_loss(loss_name)
    resolved_margin = (
        specification.default_margin_ms if margin is None else float(margin)
    )
    if not math.isfinite(resolved_margin) or resolved_margin < 0.0:
        raise ValueError("timing-loss margin must be finite and non-negative")
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("label must be an integer readout index")
    if not 0 <= label < len(readouts):
        raise ValueError("label is outside the readout range")

    effective_times = tuple(float(readout.effective_time) for readout in readouts)
    physical_mask = tuple(
        readout.physical_time is not None for readout in readouts
    )
    gradient = torch.zeros(len(readouts), dtype=torch.float64, device=device)
    if not any(physical_mask):
        return DelGradLoss(
            loss=0.0,
            effective_times=effective_times,
            physical_mask=physical_mask,
            active_wrong_outputs=(),
            fake_anchor_output=None,
            output_time_grad=gradient,
            nondifferentiable_reason=None,
        )

    target_time = effective_times[label]
    physical_wrong = tuple(
        output
        for output, physical in enumerate(physical_mask)
        if physical and output != label
    )
    errors = {
        output: effective_times[output] - target_time - resolved_margin
        for output in physical_wrong
    }
    if specification.name == "squared_hinge":
        errors = {
            output: error for output, error in errors.items() if error < 0.0
        }
    active_wrong = tuple(errors)
    loss = 0.5 * sum(error * error for error in errors.values())

    physical_times = tuple(readout.physical_time for readout in readouts)
    latest = max(time for time in physical_times if time is not None)
    latest_outputs = tuple(
        output
        for output, time in enumerate(physical_times)
        if time == latest
    )
    anchor = readouts[label].anchor_output
    if not physical_mask[label] and len(latest_outputs) > 1:
        return DelGradLoss(
            loss=float(loss),
            effective_times=effective_times,
            physical_mask=physical_mask,
            active_wrong_outputs=active_wrong,
            fake_anchor_output=anchor,
            output_time_grad=None,
            nondifferentiable_reason=(
                "tied latest physical anchor outputs="
                f"{latest_outputs} for fake target"
            ),
        )

    for output, error in errors.items():
        gradient[output] = error
    error_sum = sum(errors.values())
    if physical_mask[label]:
        gradient[label] = -error_sum
    else:
        if anchor is None:
            raise ValueError("silent target readout is missing its physical anchor")
        gradient[anchor] -= error_sum
    return DelGradLoss(
        loss=float(loss),
        effective_times=effective_times,
        physical_mask=physical_mask,
        active_wrong_outputs=active_wrong,
        fake_anchor_output=anchor if not physical_mask[label] else None,
        output_time_grad=gradient,
        nondifferentiable_reason=None,
    )
