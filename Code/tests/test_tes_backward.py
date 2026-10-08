import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_events import (
    DiscreteForwardResult,
    DiscreteThreshold,
    EffectiveReadout,
    StopInfo,
)
from Code.yinyang import YinYang, build_demo, encode_events


def make_backward_model(dt: float = 0.1) -> TeSNeurons:
    return TeSNeurons(
        1,
        2,
        1,
        torch.zeros((2, 1), dtype=torch.float64),
        torch.zeros((1, 2), dtype=torch.float64),
        TeSConfig(dt=dt),
    )


def build_refined_model(*, dt: float = 0.01) -> TeSNeurons:
    """Local frozen fixture formerly supplied by a landscape utility."""
    reference = build_demo(dt=dt, seed=23)
    generator = torch.Generator().manual_seed(413)
    w_in = (
        2.75
        + 1.15
        * torch.randn(
            (reference.n_hidden, reference.n_in),
            generator=generator,
            dtype=torch.float64,
        )
    ).clamp_min(0.0)
    config = TeSConfig(
        dt=dt,
        d_plus=reference.config.d_plus,
        c0=reference.config.c0,
        c1=reference.config.c1,
        reset_drop=5.0,
    )
    scaled_readout = reference.W_out * torch.tensor(
        (0.49, 0.69, 0.66), dtype=torch.float64
    ).unsqueeze(1)
    heterogeneity = torch.Generator().manual_seed(90)
    multipliers = torch.exp(
        0.8
        * torch.randn(
            scaled_readout.shape,
            generator=heterogeneity,
            dtype=torch.float64,
        )
    )
    w_out = scaled_readout * multipliers
    w_out.mul_(scaled_readout.sum(dim=1, keepdim=True) / w_out.sum(dim=1, keepdim=True))
    return TeSNeurons(
        reference.n_in,
        reference.n_hidden,
        reference.n_out,
        w_in,
        w_out,
        config,
    )


def test_adjoint_flow_is_mass_scaled_transpose_of_forward_flow():
    model = make_backward_model()
    delta = 1.7
    source = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)

    for neuron in range(model.n_neurons):
        actual = model._adjoint_flow_single(source, neuron, delta)
        transition = model._transition(float(model.omega[neuron]), delta)
        mass = torch.diag(
            torch.tensor(
                [model.config.tau_v, 1.0, 1.0], dtype=torch.float64
            )
        )
        expected = torch.linalg.solve(mass, transition.T @ mass @ source)
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)


def test_adjoint_flow_has_semigroup_property():
    model = make_backward_model()
    source = torch.tensor([0.4, -0.7, 1.1], dtype=torch.float64)

    for neuron in range(model.n_neurons):
        split = model._adjoint_flow_single(
            model._adjoint_flow_single(source, neuron, 0.7), neuron, 1.2
        )
        direct = model._adjoint_flow_single(source, neuron, 1.9)
        torch.testing.assert_close(split, direct, rtol=2e-11, atol=2e-12)


def test_zero_readout_source_produces_exact_zero_discrete_gradients():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[40.0]], dtype=torch.float64),
        torch.tensor([[16.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    forward = model.forward_discrete(
        [InputSpike(0.0, 0, 0)], 50.0, record_state=False
    )

    result = model.backward_discrete(
        forward, torch.zeros(1, dtype=torch.float64)
    )

    assert torch.count_nonzero(result.hidden_spike_time_grad) == 0
    assert torch.count_nonzero(result.grad_W_in) == 0
    assert torch.count_nonzero(result.grad_W_out) == 0


@pytest.mark.parametrize("neuron", [0, 2])
def test_discrete_threshold_reverse_uses_capped_inverse_tau_slope(neuron):
    model = make_backward_model()
    threshold = DiscreteThreshold(
        event_id=0,
        time=1.0,
        step=10,
        neuron=neuron,
        slope=0.001,
        v_pre=1.0,
        i_pre=0.0,
        q_pre=0.0,
        v_post=0.0,
        continuous_delay=2.5,
        delay_steps=25 if neuron < model.n_hidden else None,
        arrival_time=3.5,
    )
    value = torch.zeros((model.n_neurons, 3), dtype=torch.float64)
    value[neuron] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    timing_source = torch.tensor(0.4, dtype=torch.float64)

    model._reverse_discrete_threshold(value, threshold, timing_source)

    sensitivity = 10.0
    acceleration = -threshold.slope * model._inv_tau_v
    delay_derivative = -model.config.d_plus * model.config.c0 * 0.25
    expected_v = (
        (1.0 + model.config.reset_drop * sensitivity) * 1.0
        + 0.4 * sensitivity
        + 0.4
        * delay_derivative
        * (model._inv_tau_v**2 + acceleration * sensitivity)
    )
    assert float(value[neuron, 0]) == pytest.approx(expected_v)


def test_discrete_telemetry_ignores_disconnected_zero_adjoint_threshold():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.ones((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.1, reset_drop=5.0),
    )
    hidden = DiscreteThreshold(
        0, 0.5, 5, 0, 0.001, 1.0, 1.02, 0.0, -4.0, 2.5, 25, 3.0
    )
    output = DiscreteThreshold(
        1, 1.0, 10, 1, 0.2, 1.0, 5.0, 0.0, -4.0, 1.0, None, 2.0
    )
    forward = DiscreteForwardResult(
        times=None,
        V=None,
        I=None,
        Q=None,
        arrivals=(),
        thresholds=(hidden, output),
        readouts=(EffectiveReadout(0, 1, 1.0, 1.0, False, None),),
        warnings=(),
        stop=StopInfo("requested_horizon", 1.0, 1.0, 0, 0, 0),
    )

    result = model.backward_discrete(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    assert result.threshold_sensitivity_cap_activations == 0
    assert result.minimum_causal_slope == pytest.approx(0.2)


def test_backward_discrete_sums_repeated_shared_weight_events():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[40.0]], dtype=torch.float64),
        torch.tensor([[16.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    forward = model.forward_discrete(
        [InputSpike(0.0, 0, 0)], 50.0, record_state=False
    )

    result = model.backward_discrete(
        forward, torch.ones(1, dtype=torch.float64)
    )

    hidden_events = [
        event for event in forward.thresholds if event.neuron == 0
    ]
    causal_arrivals = [
        arrival
        for arrival in forward.arrivals
        if arrival.kind == "hidden"
        and arrival.time < forward.readouts[0].physical_time
    ]
    assert len(hidden_events) >= 2
    assert len(causal_arrivals) >= 2
    assert result.hidden_spike_time_grad.numel() == len(hidden_events)
    assert torch.count_nonzero(result.hidden_spike_time_grad) >= 2
    assert torch.isfinite(result.grad_W_in).all()
    assert torch.isfinite(result.grad_W_out).all()
    assert result.grad_W_in[0, 0] != 0.0
    assert result.grad_W_out[0, 0] != 0.0


def test_yinyang_discrete_loss_backward_is_deterministic_and_nonmutating():
    coordinates, labels = YinYang().sample_balanced(n_per_class=1, seed=23)
    model = build_refined_model(dt=0.01)
    inputs = encode_events(coordinates[0], 0.01).events
    forward = model.forward_discrete(inputs, 50.0, record_state=False)
    loss = model.squared_hinge_loss(forward, label=int(labels[0]), margin=2.0)
    weights_before = (model.W_in.clone(), model.W_out.clone())

    first = model.backward_discrete(forward, loss.output_time_grad)
    second = model.backward_discrete(forward, loss.output_time_grad)

    torch.testing.assert_close(first.grad_W_in, second.grad_W_in)
    torch.testing.assert_close(first.grad_W_out, second.grad_W_out)
    torch.testing.assert_close(model.W_in, weights_before[0])
    torch.testing.assert_close(model.W_out, weights_before[1])
    torch.testing.assert_close(model.dCdW_in, first.grad_W_in)
    torch.testing.assert_close(model.dCdW_out, first.grad_W_out)
