import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_training import physical_prediction


def delayed_readout_model(*, n_out: int = 1) -> TeSNeurons:
    output_weights = torch.tensor(
        [[5.0], [6.0]][:n_out], dtype=torch.float64
    )
    return TeSNeurons(
        1,
        1,
        n_out,
        torch.tensor([[40.0]], dtype=torch.float64),
        output_weights,
        TeSConfig(dt=0.01, reset_drop=5.0),
    )


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_output_threshold_records_exact_delay_and_delayed_physical_time(mode: str):
    """Catches using output threshold time as the physical readout time."""
    model = delayed_readout_model()
    inputs = [InputSpike(0.0, 0, 0)]
    kwargs = {"record_state": False} if mode == "discrete" else {}

    full = getattr(model, f"forward_{mode}")(
        inputs, 20.0, early_stop=False, **kwargs
    )
    full_events = (
        full.output_spikes
        if mode == "continuous"
        else tuple(
            event
            for event in full.thresholds
            if event.neuron >= model.n_hidden
        )
    )
    first_full_event = full_events[0]
    horizon = 0.5 * (first_full_event.time + first_full_event.arrival_time)
    result = getattr(model, f"forward_{mode}")(
        inputs, horizon, early_stop=False, **kwargs
    )
    events = (
        result.output_spikes
        if mode == "continuous"
        else tuple(
            event
            for event in result.thresholds
            if event.neuron >= model.n_hidden
        )
    )

    assert len(events) == 1
    event = events[0]
    readout = result.readouts[0]
    assert event.time <= horizon
    assert event.continuous_delay is not None
    assert event.arrival_time == pytest.approx(
        event.time + event.continuous_delay, abs=1e-12
    )
    assert event.arrival_time > horizon
    assert readout.physical_threshold_id == event.event_id
    assert readout.physical_time == pytest.approx(event.arrival_time, abs=1e-12)
    assert readout.effective_time == pytest.approx(event.arrival_time, abs=1e-12)
    assert not readout.censored


@pytest.mark.parametrize("mode", ["continuous", "discrete"])
def test_loss_and_prediction_consume_delayed_readout_times(mode: str):
    """Catches delayed event fields that are ignored by downstream consumers."""
    model = delayed_readout_model(n_out=2)
    kwargs = {"record_state": False} if mode == "discrete" else {}
    result = getattr(model, f"forward_{mode}")(
        [InputSpike(0.0, 0, 0)], 12.0, early_stop=False, **kwargs
    )

    loss = model.squared_hinge_loss(result, label=0, margin=2.0)
    prediction = physical_prediction(result.readouts, label=1)
    delayed_times = tuple(
        float(readout.physical_time) for readout in result.readouts
    )

    assert loss.effective_times == pytest.approx(delayed_times, abs=1e-12)
    assert prediction.predicted == min(
        range(model.n_out), key=delayed_times.__getitem__
    )
    assert all(
        readout.physical_time
        > next(
            event.time
            for event in (
                result.output_spikes
                if mode == "continuous"
                else result.thresholds
            )
            if (
                event.neuron == readout.neuron
                if mode == "continuous"
                else event.neuron == model.n_hidden + readout.neuron
            )
        )
        for readout in result.readouts
    )


def test_hidden_delay_beyond_horizon_is_recorded_but_not_delivered():
    """Catches feeding a post-horizon hidden arrival into output dynamics."""
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[8.0]], dtype=torch.float64),
        torch.tensor([[20.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )

    full = model.forward_continuous([InputSpike(0.0, 0, 0)], 50.0)
    first_hidden = full.hidden_spikes[0]
    horizon = 0.5 * (first_hidden.time + first_hidden.arrival_time)
    result = model.forward_continuous([InputSpike(0.0, 0, 0)], horizon)

    assert len(result.hidden_spikes) == 1
    assert result.hidden_spikes[0].time < horizon
    assert result.hidden_spikes[0].arrival_time is not None
    assert result.hidden_spikes[0].arrival_time > horizon
    assert result.hidden_arrivals == ()
    assert result.output_spikes == ()
    assert result.readouts[0].physical_time is None
