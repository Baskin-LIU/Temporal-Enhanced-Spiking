"""Analytical TeS postsynaptic-voltage kernels and initialization gains."""

from __future__ import annotations

from dataclasses import dataclass
import math


REFERENCE_VANILLA_TAU_V_MS = 20.0
REFERENCE_VANILLA_ALPHA_PER_MS = 0.1
REFERENCE_VANILLA_UNIT_PSP_PEAK = 0.25


@dataclass(frozen=True)
class PSPPeakCalibration:
    period_ms: float
    omega_per_ms: float
    peak_time_ms: float
    unscaled_peak: float
    initialization_gain: float


def tes_unit_psp_voltage(
    time_ms: float,
    *,
    tau_v_ms: float,
    alpha_per_ms: float,
    omega_per_ms: float,
) -> float:
    """Return the voltage caused by one unit negative-quadrature jump."""
    beta = 1.0 / tau_v_ms
    delta = beta - alpha_per_ms
    denominator = delta * delta + omega_per_ms * omega_per_ms
    synaptic = math.exp(-alpha_per_ms * time_ms)
    somatic = math.exp(-beta * time_ms)
    return beta / denominator * (
        synaptic
        * (
            delta * math.sin(omega_per_ms * time_ms)
            - omega_per_ms * math.cos(omega_per_ms * time_ms)
        )
        + omega_per_ms * somatic
    )


def calibrate_first_positive_lobe(
    *,
    tau_v_ms: float,
    alpha_per_ms: float,
    period_ms: float,
    reference_peak: float = REFERENCE_VANILLA_UNIT_PSP_PEAK,
) -> PSPPeakCalibration:
    """Match the first TeS voltage maximum to a positive reference peak."""
    values = {
        "tau_v_ms": tau_v_ms,
        "alpha_per_ms": alpha_per_ms,
        "period_ms": period_ms,
        "reference_peak": reference_peak,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    omega = 2.0 * math.pi / period_ms
    half_lobe = math.pi / omega

    def stationary_equation(time_ms: float) -> float:
        current = math.exp(-alpha_per_ms * time_ms) * math.sin(
            omega * time_ms
        )
        voltage = tes_unit_psp_voltage(
            time_ms,
            tau_v_ms=tau_v_ms,
            alpha_per_ms=alpha_per_ms,
            omega_per_ms=omega,
        )
        return current - voltage

    lower = half_lobe * 1e-12
    upper = half_lobe
    lower_value = stationary_equation(lower)
    upper_value = stationary_equation(upper)
    if lower_value <= 0.0 or upper_value >= 0.0:
        raise RuntimeError("first positive-lobe maximum is not bracketed")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = stationary_equation(midpoint)
        if midpoint_value > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    peak_time = 0.5 * (lower + upper)
    peak = tes_unit_psp_voltage(
        peak_time,
        tau_v_ms=tau_v_ms,
        alpha_per_ms=alpha_per_ms,
        omega_per_ms=omega,
    )
    return PSPPeakCalibration(
        period_ms=period_ms,
        omega_per_ms=omega,
        peak_time_ms=peak_time,
        unscaled_peak=peak,
        initialization_gain=reference_peak / peak,
    )
