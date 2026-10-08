import math

import pytest


def test_reference_vanilla_psp_constants_are_explicit():
    from Code.tes_psp import (
        REFERENCE_VANILLA_ALPHA_PER_MS,
        REFERENCE_VANILLA_TAU_V_MS,
        REFERENCE_VANILLA_UNIT_PSP_PEAK,
    )

    assert REFERENCE_VANILLA_ALPHA_PER_MS == 0.1
    assert REFERENCE_VANILLA_TAU_V_MS == 20.0
    assert REFERENCE_VANILLA_UNIT_PSP_PEAK == 0.25


def test_tes_unit_psp_is_finite_when_alpha_equals_inverse_tau_v():
    from Code.tes_psp import tes_unit_psp_voltage

    assert tes_unit_psp_voltage(
        0.0,
        tau_v_ms=20.0,
        alpha_per_ms=0.05,
        omega_per_ms=2.0 * math.pi / 30.0,
    ) == 0.0
    assert math.isfinite(
        tes_unit_psp_voltage(
            12.7621597069,
            tau_v_ms=20.0,
            alpha_per_ms=0.05,
            omega_per_ms=2.0 * math.pi / 30.0,
        )
    )


@pytest.mark.parametrize(
    ("period_ms", "expected_time", "expected_peak", "expected_gain"),
    (
        (
            8.0 * math.pi,
            11.167633112750769,
            0.25914209904586484,
            0.9647216755613035,
        ),
        (
            12.0 * math.pi,
            15.871227336806228,
            0.3202711116383285,
            0.7805886666491378,
        ),
    ),
)
def test_first_positive_lobe_calibration_matches_hand_checked_values(
    period_ms, expected_time, expected_peak, expected_gain
):
    from Code.tes_psp import calibrate_first_positive_lobe, tes_unit_psp_voltage

    result = calibrate_first_positive_lobe(
        tau_v_ms=20.0,
        alpha_per_ms=1.0 / 40.0,
        period_ms=period_ms,
    )

    assert result.period_ms == period_ms
    assert result.omega_per_ms == pytest.approx(
        2.0 * math.pi / period_ms, abs=1e-15
    )
    assert result.peak_time_ms == pytest.approx(expected_time, abs=1e-10)
    assert result.unscaled_peak == pytest.approx(expected_peak, abs=1e-10)
    assert result.initialization_gain == pytest.approx(expected_gain, abs=1e-10)
    assert tes_unit_psp_voltage(
        result.peak_time_ms,
        tau_v_ms=20.0,
        alpha_per_ms=1.0 / 40.0,
        omega_per_ms=result.omega_per_ms,
    ) == pytest.approx(result.unscaled_peak, abs=1e-14)
    assert result.initialization_gain * result.unscaled_peak == pytest.approx(
        0.25, abs=1e-12
    )


@pytest.mark.parametrize(
    ("field", "value"),
    tuple(
        (field, value)
        for field in (
            "tau_v_ms",
            "alpha_per_ms",
            "period_ms",
            "reference_peak",
        )
        for value in (0.0, -1.0, float("nan"), float("inf"))
    ),
)
def test_first_positive_lobe_calibration_rejects_invalid_inputs(field, value):
    from Code.tes_psp import calibrate_first_positive_lobe

    arguments = {
        "tau_v_ms": 20.0,
        "alpha_per_ms": 0.05,
        "period_ms": 30.0,
        "reference_peak": 0.25,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        calibrate_first_positive_lobe(**arguments)
