import math

import pytest
import torch

from Code.neurons import TeSConfig, TeSNeurons
from Code.tes_roots import first_threshold_contact


@pytest.fixture
def two_group_model() -> TeSNeurons:
    return TeSNeurons(
        1,
        2,
        1,
        torch.ones((2, 1), dtype=torch.float64),
        torch.ones((1, 2), dtype=torch.float64),
        TeSConfig(dt=0.05, reset_drop=5.0),
    )


@pytest.mark.parametrize("neuron", [0, 1])
@pytest.mark.parametrize("elapsed", [0.0, 0.013, 1.7, 12.0])
def test_precomputed_analytical_flow_matches_matrix_exponential(
    two_group_model: TeSNeurons, neuron: int, elapsed: float
):
    """Catches a wrong sign or coefficient in either frequency group's flow."""
    state = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)
    omega = float(two_group_model.omega[neuron])
    expected = torch.matrix_exp(two_group_model.system_matrix(omega) * elapsed) @ state
    actual = two_group_model._flow_single(state, neuron, elapsed)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("neuron", [0, 1])
def test_analytical_adjoint_is_mass_scaled_forward_transpose(
    two_group_model: TeSNeurons, neuron: int
):
    """Catches loss of the EventProp mass scaling in exact reverse flow."""
    value = torch.tensor([0.2, -0.3, 0.9], dtype=torch.float64)
    elapsed = 2.3
    omega = float(two_group_model.omega[neuron])
    phi = torch.matrix_exp(two_group_model.system_matrix(omega) * elapsed)
    mass = torch.diag(
        torch.tensor([two_group_model.config.tau_v, 1.0, 1.0], dtype=torch.float64)
    )
    expected = torch.linalg.solve(mass, phi.T @ mass @ value)
    actual = two_group_model._adjoint_flow_single(value, neuron, elapsed)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_certified_root_finds_first_crossing_with_below_threshold_endpoints(
    two_group_model: TeSNeurons,
):
    """Catches endpoint-only root searches that miss an interior oscillatory spike."""
    from Code.tes_roots import first_upward_crossing

    state = torch.tensor([0.0, 0.0, -10.0], dtype=torch.float64)
    horizon = 20.0
    crossing = first_upward_crossing(
        state,
        horizon,
        threshold=1.0,
        flow=lambda value, delta: two_group_model._flow_single(value, 0, delta),
        slope=lambda value: (
            float(value[1]) - float(value[0])
        ) / two_group_model.config.tau_v,
        curvature_bound=lambda value, width: two_group_model._voltage_curvature_bound(
            value, 0, width
        ),
        panel_width=two_group_model._root_panel_width_fast,
    )
    assert crossing is not None
    voltage = lambda elapsed: float(
        two_group_model._flow_single(state, 0, elapsed)[0]
    )
    assert voltage(0.0) < 1.0
    assert voltage(horizon) < 1.0
    grid = torch.linspace(0.0, horizon, 20_001, dtype=torch.float64)
    bracket = next(
        (float(left), float(right))
        for left, right in zip(grid[:-1], grid[1:], strict=True)
        if voltage(float(left)) < 1.0 <= voltage(float(right))
    )
    left, right = bracket
    for _ in range(60):
        middle = 0.5 * (left + right)
        if voltage(middle) < 1.0:
            left = middle
        else:
            right = middle
    independently_bracketed = 0.5 * (left + right)
    assert crossing.offset == pytest.approx(independently_bracketed, abs=2e-9)
    assert float(crossing.state[0]) == pytest.approx(1.0, abs=1e-9)
    assert crossing.slope > 0.0
    assert crossing.panels_visited > 0


def test_certified_root_returns_none_for_pure_decay(two_group_model: TeSNeurons):
    """Catches a root finder that invents a spike in a decaying state."""
    from Code.tes_roots import first_upward_crossing

    state = torch.tensor([0.8, 0.0, 0.0], dtype=torch.float64)
    panel_counter = [0]
    crossing = first_upward_crossing(
        state,
        20.0,
        threshold=1.0,
        flow=lambda value, delta: two_group_model._flow_single(value, 0, delta),
        slope=lambda value: (
            float(value[1]) - float(value[0])
        ) / two_group_model.config.tau_v,
        curvature_bound=lambda value, width: two_group_model._voltage_curvature_bound(
            value, 0, width
        ),
        panel_width=2.0 * math.pi / float(two_group_model.omega[0]) / 8.0,
        panel_counter=panel_counter,
    )
    assert crossing is None
    assert panel_counter[0] > 0


def test_continuous_contact_uses_configured_slope_tolerance(
    two_group_model: TeSNeurons, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, float] = {}

    def fake_first_threshold_contact(state, horizon, **kwargs):
        captured["slope_epsilon"] = kwargs["slope_epsilon"]
        return None

    monkeypatch.setattr(
        "Code.neurons.first_threshold_contact", fake_first_threshold_contact
    )

    two_group_model._continuous_contact(
        torch.zeros(3, dtype=torch.float64), neuron=0, horizon=1.0
    )

    assert captured["slope_epsilon"] == two_group_model.config.slope_epsilon


def test_left_boundary_threshold_equality_is_contact_with_negative_slope():
    """Catches reinstating a positive-slope acceptance condition."""
    state = torch.tensor([1.0, -1.0], dtype=torch.float64)
    kwargs = {
        "threshold": 1.0,
        "flow": lambda value, elapsed: torch.tensor(
            [value[0] + value[1] * elapsed, value[1]], dtype=torch.float64
        ),
        "slope": lambda value: float(value[1]),
        "curvature_bound": lambda value, width: 0.0,
        "panel_width": 1.0,
    }

    contact = first_threshold_contact(state, 1.0, **kwargs)
    zero_horizon = first_threshold_contact(state, 0.0, **kwargs)

    assert contact is not None
    assert contact.offset == 0.0
    assert contact.slope == -1.0
    assert zero_horizon is not None
    assert zero_horizon.offset == 0.0


def test_interior_descending_threshold_equality_is_a_contact():
    """Catches discarding a bracket solely because its slope is negative."""
    state = torch.tensor([2.0, -1.0], dtype=torch.float64)

    contact = first_threshold_contact(
        state,
        2.0,
        threshold=1.0,
        flow=lambda value, elapsed: torch.tensor(
            [value[0] + value[1] * elapsed, value[1]], dtype=torch.float64
        ),
        slope=lambda value: float(value[1]),
        curvature_bound=lambda value, width: 0.0,
        panel_width=2.0,
    )

    assert contact is not None
    assert contact.offset == pytest.approx(1.0)
    assert contact.slope == -1.0
    assert contact.residual == pytest.approx(0.0, abs=contact.residual_tolerance)


def test_stationary_certified_threshold_contact_is_returned():
    """Catches treating an exactly certified grazing contact as silence."""
    state = torch.tensor([0.0, 2.0], dtype=torch.float64)

    contact = first_threshold_contact(
        state,
        2.0,
        threshold=1.0,
        flow=lambda value, elapsed: torch.tensor(
            [
                value[0] + value[1] * elapsed - elapsed * elapsed,
                value[1] - 2.0 * elapsed,
            ],
            dtype=torch.float64,
        ),
        slope=lambda value: float(value[1]),
        curvature_bound=lambda value, width: 2.0,
        panel_width=2.0,
    )

    assert contact is not None
    assert contact.offset == pytest.approx(1.0, abs=1e-8)
    assert contact.slope == pytest.approx(0.0, abs=2e-8)
