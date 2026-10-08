import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_roots import first_upward_crossing


def stop_fixture(mode: str, case: str):
    if case == "all_physical":
        w_in = torch.tensor([[35.0]], dtype=torch.float64)
        w_out = torch.tensor([[16.0], [12.0]], dtype=torch.float64)
        label = 0
    elif case == "mixed":
        w_in = torch.tensor([[35.0]], dtype=torch.float64)
        w_out = torch.tensor([[16.0], [0.0]], dtype=torch.float64)
        label = 1
    else:
        w_in = torch.zeros((1, 1), dtype=torch.float64)
        w_out = torch.zeros((2, 1), dtype=torch.float64)
        label = 0
    model = TeSNeurons(
        1,
        1,
        2,
        w_in,
        w_out,
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    return model, [InputSpike(0.0, 0, 0)], 50.0, label


def test_no_future_spike_certificate_is_strict_and_conservative():
    model, _, _, _ = stop_fixture("continuous", "all_physical")

    assert model._no_future_spike(
        torch.tensor([0.2, 0.5, 0.2], dtype=torch.float64)
    )
    assert not model._no_future_spike(
        torch.tensor([0.9, 0.0, 1.0], dtype=torch.float64)
    )
    dangerous = torch.tensor([0.9, 0.0, -5.0], dtype=torch.float64)
    assert not model._no_future_spike(dangerous)
    crossing = first_upward_crossing(
        dangerous,
        25.0,
        threshold=1.0,
        flow=lambda state, delta: model._flow_single(state, 0, delta),
        slope=lambda state: (float(state[1]) - float(state[0]))
        / model.config.tau_v,
        curvature_bound=lambda state, width: model._voltage_curvature_bound(
            state, 0, width
        ),
        panel_width=model._root_panel_width_fast,
    )
    assert crossing is not None


def test_silence_certificate_rejects_an_open_arrival_stream():
    model, _, _, _ = stop_fixture("continuous", "all_physical")

    with pytest.raises(RuntimeError, match="closed causal arrival stream"):
        model._certify_silent(
            torch.zeros(3, dtype=torch.float64), arrival_stream_closed=False
        )


@pytest.mark.parametrize("mode", ["discrete", "continuous"])
@pytest.mark.parametrize("case", ["all_physical", "mixed", "all_silent"])
def test_early_stop_matches_full_loss_and_gradients(mode, case):
    model, inputs, t_end, label = stop_fixture(mode, case)
    forward = getattr(model, f"forward_{mode}")
    backward = getattr(model, f"backward_{mode}")
    kwargs = {"record_state": False} if mode == "discrete" else {}

    full = forward(inputs, t_end, early_stop=False, **kwargs)
    short = forward(inputs, t_end, early_stop=True, **kwargs)
    full_loss = model.squared_hinge_loss(full, label)
    short_loss = model.squared_hinge_loss(short, label)

    assert [readout.neuron for readout in short.readouts] == [
        readout.neuron for readout in full.readouts
    ]
    assert [readout.censored for readout in short.readouts] == [
        readout.censored for readout in full.readouts
    ]
    assert [readout.anchor_output for readout in short.readouts] == [
        readout.anchor_output for readout in full.readouts
    ]
    assert [readout.effective_time for readout in short.readouts] == pytest.approx(
        [readout.effective_time for readout in full.readouts], abs=1e-10
    )
    assert short_loss.loss == pytest.approx(full_loss.loss, abs=1e-10)
    if case != "mixed":
        assert short.stop.actual_t_end < full.stop.actual_t_end
    else:
        assert short.stop.actual_t_end == full.stop.actual_t_end
    expected_reason = {
        "all_physical": "all_readouts_physical",
        "mixed": "requested_horizon",
        "all_silent": "all_readouts_silent",
    }[case]
    assert short.stop.reason == expected_reason
    if case == "all_physical":
        threshold_events = (
            short.output_spikes if mode == "continuous" else short.thresholds
        )
        threshold_by_id = {
            event.event_id: event for event in threshold_events
        }
        assert short.stop.actual_t_end == pytest.approx(
            max(
                threshold_by_id[readout.physical_threshold_id].time
                for readout in short.readouts
            ),
            abs=1e-12,
        )
        assert short.stop.actual_t_end < max(
            float(readout.physical_time) for readout in short.readouts
        )
        assert (
            short.stop.hidden_flow_intervals
            + short.stop.output_flow_intervals
            < full.stop.hidden_flow_intervals + full.stop.output_flow_intervals
            or short.stop.root_panels < full.stop.root_panels
        )

    assert full_loss.output_time_grad is not None
    assert short_loss.output_time_grad is not None
    full_gradient = backward(full, full_loss.output_time_grad)
    short_gradient = backward(short, short_loss.output_time_grad)
    torch.testing.assert_close(short_gradient.grad_W_in, full_gradient.grad_W_in)
    torch.testing.assert_close(short_gradient.grad_W_out, full_gradient.grad_W_out)
    if case == "all_silent":
        assert short_loss.loss == 0.0
        assert torch.count_nonzero(short_gradient.grad_W_in) == 0
        assert torch.count_nonzero(short_gradient.grad_W_out) == 0


def test_discrete_early_stop_trims_recorded_state_to_actual_end():
    model, inputs, t_end, _ = stop_fixture("discrete", "all_physical")

    result = model.forward_discrete(
        inputs, t_end, early_stop=True, record_state=True
    )

    assert result.times is not None
    assert result.V is not None
    assert result.times[-1].item() == pytest.approx(result.stop.actual_t_end)
    assert result.V.shape[0] == result.times.numel()
    assert result.I.shape == result.V.shape
    assert result.Q.shape == result.V.shape


def test_early_stop_false_retains_later_physical_events():
    model, inputs, t_end, _ = stop_fixture("continuous", "all_physical")

    full = model.forward_continuous(inputs, t_end, early_stop=False)
    short = model.forward_continuous(inputs, t_end, early_stop=True)

    terminal = short.stop.actual_t_end
    assert any(event.time > terminal for event in full.hidden_spikes)
    assert any(event.time > terminal for event in full.output_spikes)
    assert all(event.time <= terminal for event in short.output_spikes)
    assert all(event.time <= terminal for event in short.hidden_spikes)


def test_full_silent_continuous_run_counts_unsuccessful_root_panels():
    model, inputs, t_end, _ = stop_fixture("continuous", "all_silent")

    result = model.forward_continuous(inputs, t_end, early_stop=False)

    assert result.stop.root_panels > 0
