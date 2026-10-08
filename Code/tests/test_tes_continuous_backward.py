from dataclasses import replace

import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_events import (
    ContinuousArrival,
    ContinuousForwardResult,
    ContinuousThreshold,
    EffectiveReadout,
    StopInfo,
)


def one_path_model(*, w_in: float = 35.0, w_out: float = 4.0) -> TeSNeurons:
    return TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[w_in]], dtype=torch.float64),
        torch.tensor([[w_out]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )


def first_output_time(w_in: float, w_out: float) -> float:
    forward = one_path_model(w_in=w_in, w_out=w_out).forward_continuous(
        [InputSpike(0.0, 0, 0)], 50.0
    )
    assert len(forward.hidden_spikes) >= 2
    assert forward.readouts[0].physical_time is not None
    return float(forward.readouts[0].physical_time)


def readout_only_forward(
    physical_times: tuple[float | None, ...], *, t_end: float = 50.0
) -> ContinuousForwardResult:
    present = [time for time in physical_times if time is not None]
    latest = max(present) if present else None
    latest_outputs = tuple(
        output for output, time in enumerate(physical_times) if time == latest
    )
    anchor = latest_outputs[0] if latest_outputs else None
    fake_time = t_end if latest is None else latest + 2.0
    readouts = tuple(
        EffectiveReadout(
            neuron=output,
            physical_threshold_id=None if time is None else output,
            physical_time=time,
            effective_time=fake_time if time is None else time,
            censored=time is None,
            anchor_output=anchor if time is None else None,
        )
        for output, time in enumerate(physical_times)
    )
    return ContinuousForwardResult(
        input_arrivals=(),
        hidden_spikes=(),
        hidden_arrivals=(),
        output_spikes=(),
        readouts=readouts,
        stop=StopInfo("requested_horizon", t_end, t_end, 0, 0, 0),
    )


@pytest.mark.parametrize(
    ("physical_times", "label", "expected_effective", "expected_loss", "expected_grad"),
    [
        ((10.0, 13.0, None), 0, (10.0, 13.0, 15.0), 0.0, (0.0, 0.0, 0.0)),
        ((None, 10.0, 13.0), 0, (15.0, 10.0, 13.0), 32.5, (0.0, -7.0, 7.0)),
        ((10.0, None, 13.0), 0, (10.0, 15.0, 13.0), 0.0, (0.0, 0.0, 0.0)),
        ((None, None, None), 1, (50.0, 50.0, 50.0), 0.0, (0.0, 0.0, 0.0)),
    ],
)
def test_class_loss_has_hand_derived_value_and_physical_sources(
    physical_times, label, expected_effective, expected_loss, expected_grad
):
    model = TeSNeurons(
        1,
        1,
        3,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((3, 1), dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )

    result = model.squared_hinge_loss(
        readout_only_forward(physical_times), label=label, margin=2.0
    )

    assert result.effective_times == expected_effective
    assert result.loss == pytest.approx(expected_loss)
    expected_active = tuple(
        output
        for output, gradient in enumerate(expected_grad)
        if output != label and gradient != 0.0
    )
    assert result.active_wrong_outputs == expected_active
    assert result.output_time_grad is not None
    torch.testing.assert_close(
        result.output_time_grad,
        torch.tensor(expected_grad, dtype=torch.float64),
    )


def test_class_loss_marks_tied_fake_target_anchor_nondifferentiable():
    model = TeSNeurons(
        1,
        1,
        3,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((3, 1), dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )

    result = model.squared_hinge_loss(
        readout_only_forward((None, 13.0, 13.0)), label=0, margin=2.0
    )

    assert result.loss == pytest.approx(16.0)
    assert result.active_wrong_outputs == (1, 2)
    assert result.output_time_grad is None
    assert "tied latest physical anchor" in result.nondifferentiable_reason


def test_backward_reports_threshold_sensitivity_cap_telemetry():
    model = one_path_model()
    output = ContinuousThreshold(
        event_id=0,
        layer="output",
        neuron=0,
        time=4.0,
        slope=0.001,
        acceleration_pre=0.0,
        v_pre=1.0,
        i_pre=1.02,
        q_pre=0.0,
        v_post=-4.0,
        continuous_delay=2.5,
        arrival_time=6.5,
    )
    forward = ContinuousForwardResult(
        input_arrivals=(),
        hidden_spikes=(),
        hidden_arrivals=(),
        output_spikes=(output,),
        readouts=(EffectiveReadout(0, 0, 6.5, 6.5, False, None),),
        stop=StopInfo("requested_horizon", 4.0, 4.0, 0, 0, 0),
    )

    result = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    assert result.threshold_sensitivity_cap_activations == 1
    assert result.minimum_causal_slope == pytest.approx(0.001)


def test_backward_telemetry_ignores_disconnected_zero_adjoint_threshold():
    model = one_path_model()
    disconnected = ContinuousThreshold(
        0, "hidden", 0, 1.0, 0.001, 0.0, 1.0, 1.02, 0.0, -4.0, 2.5, 3.5
    )
    output = ContinuousThreshold(
        1, "output", 0, 4.0, 0.2, 0.0, 1.0, 5.0, 0.0, -4.0, 1.0, 5.0
    )
    forward = ContinuousForwardResult(
        input_arrivals=(),
        hidden_spikes=(disconnected,),
        hidden_arrivals=(),
        output_spikes=(output,),
        readouts=(EffectiveReadout(0, 1, 5.0, 5.0, False, None),),
        stop=StopInfo("requested_horizon", 4.0, 4.0, 0, 0, 0),
    )

    result = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    assert result.threshold_sensitivity_cap_activations == 0
    assert result.minimum_causal_slope == pytest.approx(0.2)


@pytest.mark.parametrize(("which", "value"), [("in", 35.0), ("out", 4.0)])
def test_repeated_hidden_threshold_adjoint_matches_centered_difference(which, value):
    model = one_path_model(w_in=35.0, w_out=4.0)
    forward = model.forward_continuous([InputSpike(0.0, 0, 0)], 50.0)
    first_output = float(forward.readouts[0].physical_time)

    causal_arrivals = [
        arrival for arrival in forward.hidden_arrivals if arrival.time < first_output
    ]
    assert len(causal_arrivals) >= 2
    assert [arrival.source for arrival in causal_arrivals] == [0] * len(
        causal_arrivals
    )

    result = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    h = 1e-5
    if which == "in":
        finite = (
            first_output_time(value + h, 4.0)
            - first_output_time(value - h, 4.0)
        ) / (2.0 * h)
        analytic = float(result.grad_W_in[0, 0])
    else:
        finite = (
            first_output_time(35.0, value + h)
            - first_output_time(35.0, value - h)
        ) / (2.0 * h)
        analytic = float(result.grad_W_out[0, 0])

    assert analytic == pytest.approx(finite, rel=2e-3, abs=2e-5)
    assert result.hidden_spike_time_grad.numel() == len(forward.hidden_spikes)
    assert torch.count_nonzero(
        result.hidden_spike_time_grad.abs() > 1e-10
    ) == len(causal_arrivals)


def duplicated_input_output_time(w_in: float) -> float:
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[w_in]], dtype=torch.float64),
        torch.tensor([[4.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0), InputSpike(0.0, 0, 0)]
    forward = model.forward_continuous(inputs, 50.0)
    assert forward.readouts[0].physical_time is not None
    return float(forward.readouts[0].physical_time)


def test_repeated_same_channel_arrivals_accumulate_into_one_shared_weight():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[17.5]], dtype=torch.float64),
        torch.tensor([[4.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0), InputSpike(0.0, 0, 0)]
    forward = model.forward_continuous(inputs, 50.0)

    result = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    h = 1e-5
    finite = (
        duplicated_input_output_time(17.5 + h)
        - duplicated_input_output_time(17.5 - h)
    ) / (2.0 * h)
    assert [arrival.source for arrival in forward.input_arrivals] == [0, 0]
    assert float(result.grad_W_in[0, 0]) == pytest.approx(
        finite, rel=2e-3, abs=2e-5
    )


@pytest.mark.parametrize("layer", ["hidden", "output"])
def test_continuous_threshold_reverse_uses_capped_inverse_tau_slope(layer):
    model = one_path_model()
    threshold = ContinuousThreshold(
        event_id=0,
        layer=layer,
        neuron=0,
        time=1.0,
        slope=0.001,
        acceleration_pre=0.2,
        v_pre=1.0,
        i_pre=1.02,
        q_pre=0.0,
        v_post=-4.0,
        continuous_delay=2.5,
        arrival_time=3.5,
    )
    value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    timing_source = torch.tensor(0.4, dtype=torch.float64)

    actual = model._reverse_continuous_threshold(
        value, threshold, timing_source
    )

    sensitivity = 10.0
    delay_derivative = -model.config.d_plus * model.config.c0 * 0.25
    expected_v = (
        (1.0 + model.config.reset_drop * sensitivity) * 1.0
        + 0.4 * sensitivity
        + 0.4
        * delay_derivative
        * (model._inv_tau_v**2 + threshold.acceleration_pre * sensitivity)
    )
    assert float(actual[0]) == pytest.approx(expected_v)


def test_input_at_hidden_threshold_is_reversed_as_a_post_threshold_arrival():
    """A boundary input affects later dynamics but not its coincident threshold."""
    model = one_path_model(w_in=35.0, w_out=4.0)
    first_hidden = ContinuousThreshold(
        0, "hidden", 0, 1.0, 0.2, 0.0, 1.0, 5.0, 0.0, -4.0, 4.0, 5.0
    )
    second_hidden = ContinuousThreshold(
        1, "hidden", 0, 2.0, 0.2, 0.0, 1.0, 5.0, 0.0, -4.0, 1.0, 3.0
    )
    output = ContinuousThreshold(
        2, "output", 0, 4.0, 0.2, 0.0, 1.0, 5.0, 0.0, -4.0, 1.0, 5.0
    )
    forward = ContinuousForwardResult(
        input_arrivals=(ContinuousArrival(0, 0, 1.0, "input", 0, None),),
        hidden_spikes=(first_hidden, second_hidden),
        hidden_arrivals=(ContinuousArrival(1, 0, 3.0, "hidden", 0, 1),),
        output_spikes=(output,),
        readouts=(EffectiveReadout(0, 2, 5.0, 5.0, False, None),),
        stop=StopInfo("requested_horizon", 4.0, 4.0, 0, 0, 0),
    )

    with_boundary_arrival = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )
    without_boundary_arrival = model.backward_continuous(
        replace(forward, input_arrivals=()),
        torch.tensor([1.0], dtype=torch.float64),
    )

    assert with_boundary_arrival.grad_W_in[0, 0] != pytest.approx(
        0.0, abs=1e-12
    )
    assert without_boundary_arrival.grad_W_in[0, 0] == 0.0


def test_hidden_arrival_at_output_threshold_is_not_causal():
    model = one_path_model(w_in=35.0, w_out=4.0)
    hidden = ContinuousThreshold(
        event_id=0,
        layer="hidden",
        neuron=0,
        time=1.0,
        slope=0.2,
        acceleration_pre=0.0,
        v_pre=1.0,
        i_pre=5.0,
        q_pre=0.0,
        v_post=-4.0,
        continuous_delay=4.0,
        arrival_time=5.0,
    )
    output = ContinuousThreshold(
        event_id=1,
        layer="output",
        neuron=0,
        time=5.0,
        slope=0.2,
        acceleration_pre=0.0,
        v_pre=1.0,
        i_pre=5.0,
        q_pre=0.0,
        v_post=-4.0,
        continuous_delay=1.0,
        arrival_time=6.0,
    )
    forward = ContinuousForwardResult(
        input_arrivals=(),
        hidden_spikes=(hidden,),
        hidden_arrivals=(
            ContinuousArrival(0, 0, 5.0, "hidden", 0, hidden.event_id),
        ),
        output_spikes=(output,),
        readouts=(EffectiveReadout(0, 1, 6.0, 6.0, False, None),),
        stop=StopInfo("requested_horizon", 10.0, 10.0, 0, 0, 0),
    )

    result = model.backward_continuous(
        forward, torch.tensor([1.0], dtype=torch.float64)
    )

    assert result.grad_W_out[0, 0] == 0.0
    assert result.hidden_spike_time_grad.tolist() == [0.0]


def test_silent_paths_stay_zero_and_backward_does_not_mutate_forward():
    model = TeSNeurons(
        2,
        2,
        2,
        torch.tensor([[35.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[4.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    forward = model.forward_continuous([InputSpike(0.0, 0, 0)], 50.0)
    snapshot = repr(forward)

    result = model.backward_continuous(
        forward, torch.tensor([1.0, 0.0], dtype=torch.float64)
    )

    assert repr(forward) == snapshot
    assert result.grad_W_in[1].tolist() == [0.0, 0.0]
    assert result.grad_W_in[:, 1].tolist() == [0.0, 0.0]
    assert result.grad_W_out[1].tolist() == [0.0, 0.0]
    assert result.grad_W_out[:, 1].tolist() == [0.0, 0.0]


def test_all_silent_backward_returns_exact_zero_gradients():
    model = one_path_model(w_in=0.0, w_out=0.0)
    forward = model.forward_continuous([], 50.0)

    result = model.backward_continuous(
        forward, torch.tensor([0.0], dtype=torch.float64)
    )

    assert result.hidden_spike_time_grad.numel() == 0
    assert torch.count_nonzero(result.grad_W_in) == 0
    assert torch.count_nonzero(result.grad_W_out) == 0


def test_post_terminal_hidden_events_do_not_change_first_readout_gradient():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[35.0]], dtype=torch.float64),
        torch.tensor([[16.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0)]
    full = model.forward_continuous(inputs, 50.0)
    output_by_id = {event.event_id: event for event in full.output_spikes}
    terminal = output_by_id[
        full.readouts[0].physical_threshold_id
    ].time
    truncated = model.forward_continuous(inputs, terminal)

    assert any(event.time > terminal for event in full.hidden_spikes)
    assert any(arrival.time > terminal for arrival in full.hidden_arrivals)
    assert truncated.readouts[0].physical_time == pytest.approx(
        full.readouts[0].physical_time, abs=3e-12
    )

    source = torch.tensor([1.0], dtype=torch.float64)
    full_backward = model.backward_continuous(full, source)
    truncated_backward = model.backward_continuous(truncated, source)

    torch.testing.assert_close(
        full_backward.grad_W_in, truncated_backward.grad_W_in
    )
    torch.testing.assert_close(
        full_backward.grad_W_out, truncated_backward.grad_W_out
    )
