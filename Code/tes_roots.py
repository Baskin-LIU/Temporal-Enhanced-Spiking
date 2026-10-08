"""Certified isolation of the first TeS threshold contact."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from scipy.optimize import brentq
import torch


ROOT_TIME_ATOL = 1e-10
MINIMUM_ISOLATION_WIDTH = 1e-8
MAXIMUM_RECURSION_DEPTH = 80
SLOPE_EPSILON = 1e-12


@dataclass(frozen=True)
class ThresholdContact:
    offset: float
    state: torch.Tensor
    slope: float
    residual: float
    time_uncertainty: float
    residual_tolerance: float
    panels_visited: int


class UnresolvedCrossingError(RuntimeError):
    """The certified subdivision reached its numerical resolution floor."""


class GrazingCrossingError(RuntimeError):
    """Compatibility error retained for callers of the former upward API."""


def first_threshold_contact(
    state: torch.Tensor,
    horizon: float,
    *,
    threshold: float,
    flow: Callable[[torch.Tensor, float], torch.Tensor],
    slope: Callable[[torch.Tensor], float],
    curvature_bound: Callable[[torch.Tensor, float], float],
    panel_width: float,
    panel_counter: list[int] | None = None,
    slope_epsilon: float = SLOPE_EPSILON,
) -> ThresholdContact | None:
    """Return the first certified threshold equality in ``[0, horizon]``."""
    epsilon = torch.finfo(torch.float64).eps
    panels_visited = 0

    def build_contact(
        contact_state: torch.Tensor,
        absolute_offset: float,
        curvature: float,
        *,
        uncertainty: float,
    ) -> ThresholdContact:
        contact_slope = slope(contact_state)
        residual = float(contact_state[0]) - threshold
        scale = max(
            1.0,
            abs(threshold),
            torch.linalg.vector_norm(
                contact_state, ord=float("inf")
            ).item(),
        )
        residual_tolerance = (
            64.0 * epsilon * scale
            + abs(contact_slope) * uncertainty
            + 0.5 * curvature * uncertainty * uncertainty
        )
        if abs(residual) > residual_tolerance:
            raise UnresolvedCrossingError(
                f"root residual {residual!r} exceeds {residual_tolerance!r}"
            )
        return ThresholdContact(
            offset=absolute_offset,
            state=contact_state,
            slope=contact_slope,
            residual=residual,
            time_uncertainty=uncertainty,
            residual_tolerance=residual_tolerance,
            panels_visited=panels_visited,
        )

    initial_scale = max(
        1.0,
        abs(threshold),
        torch.linalg.vector_norm(state, ord=float("inf")).item(),
    )
    if abs(float(state[0]) - threshold) <= 64.0 * epsilon * initial_scale:
        return build_contact(state, 0.0, 0.0, uncertainty=0.0)
    if horizon == 0.0:
        return None

    def isolate(
        left_state: torch.Tensor, left: float, width: float, depth: int
    ) -> ThresholdContact | None:
        nonlocal panels_visited
        panels_visited += 1
        if panel_counter is not None:
            panel_counter[0] += 1
        right_state = flow(left_state, width)
        left_voltage = float(left_state[0])
        right_voltage = float(right_state[0])
        curvature = curvature_bound(left_state, width)

        if max(left_voltage, right_voltage) + curvature * width * width / 8.0 < threshold:
            return None

        left_slope = slope(left_state)
        right_slope = slope(right_state)
        right_uncertainty = ROOT_TIME_ATOL + 4.0 * epsilon * abs(left + width)
        right_scale = max(
            1.0,
            abs(threshold),
            torch.linalg.vector_norm(right_state, ord=float("inf")).item(),
        )
        right_residual_tolerance = (
            64.0 * epsilon * right_scale
            + abs(right_slope) * right_uncertainty
            + 0.5 * curvature * right_uncertainty * right_uncertainty
        )
        if abs(right_voltage - threshold) <= right_residual_tolerance:
            return build_contact(
                right_state,
                left + width,
                curvature,
                uncertainty=right_uncertainty,
            )
        slope_low = left_slope - curvature * width
        slope_high = left_slope + curvature * width

        if slope_low > 0.0:
            if left_voltage <= threshold <= right_voltage:
                local_root = brentq(
                    lambda elapsed: float(flow(left_state, elapsed)[0]) - threshold,
                    0.0,
                    width,
                    xtol=ROOT_TIME_ATOL,
                )
                root_state = flow(left_state, local_root)
                absolute_offset = left + local_root
                uncertainty = ROOT_TIME_ATOL + 4.0 * epsilon * abs(absolute_offset)
                return build_contact(
                    root_state,
                    absolute_offset,
                    curvature,
                    uncertainty=uncertainty,
                )
            return None

        if slope_high < 0.0:
            if right_voltage <= threshold <= left_voltage:
                local_root = brentq(
                    lambda elapsed: float(flow(left_state, elapsed)[0]) - threshold,
                    0.0,
                    width,
                    xtol=ROOT_TIME_ATOL,
                )
                root_state = flow(left_state, local_root)
                absolute_offset = left + local_root
                uncertainty = ROOT_TIME_ATOL + 4.0 * epsilon * abs(absolute_offset)
                return build_contact(
                    root_state,
                    absolute_offset,
                    curvature,
                    uncertainty=uncertainty,
                )
            return None

        if width <= MINIMUM_ISOLATION_WIDTH or depth == MAXIMUM_RECURSION_DEPTH:
            if (
                left_slope <= 0.0 <= right_slope
                or right_slope <= 0.0 <= left_slope
            ):
                stationary_offset = brentq(
                    lambda elapsed: slope(flow(left_state, elapsed)),
                    0.0,
                    width,
                    xtol=ROOT_TIME_ATOL,
                )
                stationary_state = flow(left_state, stationary_offset)
                stationary_slope = slope(stationary_state)
                stationary_residual = float(stationary_state[0]) - threshold
                stationary_scale = max(
                    1.0,
                    abs(threshold),
                    torch.linalg.vector_norm(
                        stationary_state, ord=float("inf")
                    ).item(),
                )
                if (
                    abs(stationary_slope) <= slope_epsilon
                    and abs(stationary_residual)
                    <= 64.0 * epsilon * stationary_scale
                ):
                    absolute_offset = left + stationary_offset
                    uncertainty = (
                        ROOT_TIME_ATOL + 4.0 * epsilon * abs(absolute_offset)
                    )
                    return build_contact(
                        stationary_state,
                        absolute_offset,
                        curvature,
                        uncertainty=uncertainty,
                    )
            raise UnresolvedCrossingError(
                f"threshold behavior unresolved at offset {left!r}, width {width!r}"
            )

        half = width / 2.0
        left_result = isolate(left_state, left, half, depth + 1)
        if left_result is not None:
            return left_result
        midpoint = flow(left_state, half)
        return isolate(midpoint, left + half, half, depth + 1)

    left = 0.0
    left_state = state
    while left < horizon:
        width = min(panel_width, horizon - left)
        result = isolate(left_state, left, width, 0)
        if result is not None:
            return result
        left_state = flow(left_state, width)
        left += width
    return None


# Narrow compatibility aliases for existing external imports.
UpwardCrossing = ThresholdContact
first_upward_crossing = first_threshold_contact
