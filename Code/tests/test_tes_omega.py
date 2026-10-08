import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons


def one_path_model(*, dt: float = 0.01) -> TeSNeurons:
    return TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[35.0]], dtype=torch.float64),
        torch.tensor([[4.0]], dtype=torch.float64),
        TeSConfig(dt=dt, reset_drop=5.0),
    )


def continuous_readout_time(neuron: int, perturbation: float) -> float:
    model = one_path_model()
    model.omega[neuron] += perturbation
    forward = model.forward_continuous_omega(
        [InputSpike(0.0, 0, 0)], 50.0
    )
    assert forward.readouts[0].physical_time is not None
    return float(forward.readouts[0].physical_time)


@pytest.mark.parametrize("neuron", [0, 1])
def test_continuous_omega_gradient_matches_per_neuron_centered_difference(neuron):
    """Catches a missing neuron contribution or accidental group reduction."""
    model = one_path_model()
    forward = model.forward_continuous_omega(
        [InputSpike(0.0, 0, 0)], 50.0
    )

    result = model.backward_continuous_omega(
        forward, torch.ones(1, dtype=torch.float64)
    )

    epsilon = 3e-6
    finite = (
        continuous_readout_time(neuron, epsilon)
        - continuous_readout_time(neuron, -epsilon)
    ) / (2.0 * epsilon)
    assert result.grad_omega.shape == (model.n_neurons,)
    assert float(result.grad_omega[neuron]) == pytest.approx(
        finite, rel=3e-5, abs=3e-6
    )


def test_continuous_omega_forward_stores_only_event_boundary_oscillator_states():
    """Catches replacing the sparse event tape with a dense time-grid history."""
    model = one_path_model(dt=0.001)

    forward = model.forward_continuous_omega(
        [InputSpike(0.0, 0, 0)], 50.0
    )

    checkpoints = forward.omega_checkpoints
    assert checkpoints
    for neuron in range(model.n_neurons):
        local = [item for item in checkpoints if item.neuron == neuron]
        assert local[0].time == 0.0
        assert [item.time for item in local] == sorted(
            {item.time for item in local}
        )
    event_bound = (
        model.n_neurons
        + model.n_hidden * len(forward.input_arrivals)
        + len(forward.hidden_spikes)
        + model.n_out * len(forward.hidden_arrivals)
        + len(forward.output_spikes)
    )
    assert len(checkpoints) <= event_bound
    assert all(
        torch.isfinite(torch.tensor((item.i, item.q))).all()
        for item in checkpoints
    )


def test_discrete_omega_gradient_is_consistent_with_continuous_event_gradient():
    """Catches a sign, sidedness, or quadrature error in the dense-grid path."""
    inputs = [InputSpike(0.0, 0, 0)]
    continuous_model = one_path_model(dt=0.005)
    discrete_model = one_path_model(dt=0.005)
    adapted = torch.tensor([0.28, 0.19], dtype=torch.float64)
    continuous_model.omega.copy_(adapted)
    discrete_model.omega.copy_(adapted)
    continuous_forward = continuous_model.forward_continuous_omega(inputs, 50.0)
    discrete_forward = discrete_model.forward_discrete_omega(inputs, 50.0)

    continuous = continuous_model.backward_continuous_omega(
        continuous_forward, torch.ones(1, dtype=torch.float64)
    )
    discrete = discrete_model.backward_discrete_omega(
        discrete_forward, torch.ones(1, dtype=torch.float64)
    )

    assert discrete_forward.I is not None
    assert discrete_forward.Q is not None
    torch.testing.assert_close(
        discrete.grad_omega,
        continuous.grad_omega,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_omega_backward_zero_source_has_exact_zero_frequency_gradient(mode):
    """Catches an uninitialized accumulator or a loss-independent term."""
    model = one_path_model()
    inputs = [InputSpike(0.0, 0, 0)]
    forward = getattr(model, f"forward_{mode}_omega")(inputs, 50.0)

    result = getattr(model, f"backward_{mode}_omega")(
        forward, torch.zeros(1, dtype=torch.float64)
    )

    assert torch.count_nonzero(result.grad_omega) == 0


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_omega_path_preserves_legacy_dynamics_and_weight_gradients_at_initialization(
    mode,
):
    """Catches divergence before any per-neuron frequency update is applied."""
    inputs = [InputSpike(0.0, 0, 0)]
    legacy_model = one_path_model()
    omega_model = one_path_model()
    legacy_forward = getattr(legacy_model, f"forward_{mode}")(
        inputs, 50.0, early_stop=False
    )
    omega_forward = getattr(omega_model, f"forward_{mode}_omega")(
        inputs, 50.0, early_stop=False
    )

    assert [item.physical_threshold_id for item in omega_forward.readouts] == [
        item.physical_threshold_id for item in legacy_forward.readouts
    ]
    assert [item.physical_time for item in omega_forward.readouts] == pytest.approx(
        [item.physical_time for item in legacy_forward.readouts],
        rel=2e-14,
        abs=2e-14,
    )
    legacy_backward = getattr(legacy_model, f"backward_{mode}")(
        legacy_forward, torch.ones(1, dtype=torch.float64)
    )
    omega_backward = getattr(omega_model, f"backward_{mode}_omega")(
        omega_forward, torch.ones(1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        omega_backward.grad_W_in,
        legacy_backward.grad_W_in,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        omega_backward.grad_W_out,
        legacy_backward.grad_W_out,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_omega_forward_rejects_a_nonpositive_adapted_frequency(mode):
    """Catches an update entering an invalid period/root-search regime."""
    model = one_path_model()
    model.omega[0] = 0.0

    with pytest.raises(ValueError, match="positive and finite"):
        getattr(model, f"forward_{mode}_omega")(
            [InputSpike(0.0, 0, 0)], 50.0
        )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_legacy_path_remains_fixed_after_learned_omega_is_adapted(mode):
    """Catches learned omega leaking into the legacy grouped adjoint."""
    inputs = [InputSpike(0.0, 0, 0)]
    reference = one_path_model()
    adapted = one_path_model()
    adapted.omega.copy_(torch.tensor([0.5, 0.4], dtype=torch.float64))

    reference_forward = getattr(reference, f"forward_{mode}")(inputs, 50.0)
    adapted_forward = getattr(adapted, f"forward_{mode}")(inputs, 50.0)
    reference_backward = getattr(reference, f"backward_{mode}")(
        reference_forward, torch.ones(1, dtype=torch.float64)
    )
    adapted_backward = getattr(adapted, f"backward_{mode}")(
        adapted_forward, torch.ones(1, dtype=torch.float64)
    )

    assert adapted_forward.readouts == reference_forward.readouts
    torch.testing.assert_close(
        adapted_backward.grad_W_in, reference_backward.grad_W_in
    )
    torch.testing.assert_close(
        adapted_backward.grad_W_out, reference_backward.grad_W_out
    )


def test_discrete_omega_resolution_warning_uses_fastest_adapted_neuron():
    """Catches a learned high frequency bypassing the grid safety warning."""
    model = one_path_model(dt=1.0)
    model.omega[0] = 2.0

    forward = model.forward_discrete_omega([], 0.0)

    assert forward.warnings == (
        "fewer than 20 grid steps per fastest learned oscillation period",
    )
