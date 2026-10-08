import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons


def multispike_model(*, w_in: float = 40.0, w_out: float = 16.0) -> TeSNeurons:
    return TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[w_in]], dtype=torch.float64),
        torch.tensor([[w_out]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )


def test_continuous_forward_keeps_repeated_hidden_and_readout_spikes():
    """Catches a threshold handler that deactivates a neuron after one spike."""
    result = multispike_model().forward_continuous(
        [InputSpike(0.0, 0, 0)], 50.0, early_stop=False
    )

    assert len(result.hidden_spikes) >= 2
    assert len(result.output_spikes) >= 2
    assert [event.time for event in result.hidden_spikes] == sorted(
        event.time for event in result.hidden_spikes
    )
    assert [event.time for event in result.output_spikes] == sorted(
        event.time for event in result.output_spikes
    )
    for event in result.hidden_spikes + result.output_spikes:
        assert event.v_post == pytest.approx(event.v_pre - 5.0, abs=1e-12)


def test_every_hidden_arrival_links_to_its_exact_repeated_parent():
    """Catches source-only lineage that collapses multiple spikes from one neuron."""
    result = multispike_model().forward_continuous(
        [InputSpike(0.0, 0, 0)], 50.0
    )
    parents = {event.event_id: event for event in result.hidden_spikes}

    assert len(result.hidden_arrivals) == len(result.hidden_spikes)
    assert len({arrival.parent_threshold_id for arrival in result.hidden_arrivals}) == len(
        result.hidden_spikes
    )
    for arrival in result.hidden_arrivals:
        parent = parents[arrival.parent_threshold_id]
        assert arrival.source == parent.neuron
        assert arrival.time == pytest.approx(parent.arrival_time, abs=1e-12)
        assert arrival.time > parent.time


def test_continuous_events_do_not_depend_on_the_discrete_dt_setting():
    """The analytical solver uses exact input times and frozen dynamics only."""
    fine = multispike_model(w_in=10.0, w_out=10.0)
    coarse = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[10.0]], dtype=torch.float64),
        torch.tensor([[10.0]], dtype=torch.float64),
        TeSConfig(dt=0.37, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0)]
    fine_result = fine.forward_continuous(inputs, 20.0)
    coarse_result = coarse.forward_continuous(inputs, 20.0)

    assert coarse_result.hidden_spikes == fine_result.hidden_spikes
    assert coarse_result.hidden_arrivals == fine_result.hidden_arrivals
    assert coarse_result.output_spikes == fine_result.output_spikes
    assert coarse_result.readouts == fine_result.readouts


def test_silent_readout_is_fake_without_creating_a_threshold():
    """Catches fake report times being inserted into the physical event tape."""
    model = TeSNeurons(
        1,
        1,
        2,
        torch.tensor([[10.0]], dtype=torch.float64),
        torch.tensor([[10.0], [0.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    result = model.forward_continuous([InputSpike(0.0, 0, 0)], 20.0)

    physical, fake = result.readouts
    assert physical.physical_time is not None
    assert fake.physical_time is None
    assert fake.effective_time == pytest.approx(physical.physical_time + 2.0)
    assert fake.anchor_output == 0
    assert all(event.neuron != 1 for event in result.output_spikes)


def test_multispike_discrete_times_converge_toward_continuous_events():
    """Keeps a grid oracle after the transitional continuous solver is removed."""
    inputs = [InputSpike(0.0, 0, 0)]
    exact = multispike_model().forward_continuous(inputs, 50.0)

    def grid_times(dt):
        model = TeSNeurons(
            1,
            1,
            1,
            torch.tensor([[40.0]], dtype=torch.float64),
            torch.tensor([[16.0]], dtype=torch.float64),
            TeSConfig(dt=dt, reset_drop=5.0),
        )
        result = model.forward_discrete(inputs, 50.0, record_state=False)
        hidden = [event.time for event in result.thresholds if event.neuron == 0]
        output = [event.time for event in result.thresholds if event.neuron == 1]
        return hidden, output

    coarse_hidden, coarse_output = grid_times(0.01)
    fine_hidden, fine_output = grid_times(0.005)
    exact_hidden = [event.time for event in exact.hidden_spikes]
    exact_output = [event.time for event in exact.output_spikes]

    assert len(coarse_hidden) == len(fine_hidden) == len(exact_hidden) >= 2
    assert len(coarse_output) == len(fine_output) == len(exact_output) >= 2
    for coarse, fine, continuous in zip(
        coarse_hidden + coarse_output,
        fine_hidden + fine_output,
        exact_hidden + exact_output,
        strict=True,
    ):
        assert abs(fine - continuous) <= abs(coarse - continuous) + 1e-12
    # Repeated causal threshold/arrival quantizations can accumulate several dt.
    assert max(
        abs(fine - continuous)
        for fine, continuous in zip(
            fine_hidden + fine_output,
            exact_hidden + exact_output,
            strict=True,
        )
    ) < 4.0 * 0.005


def test_threshold_at_input_boundary_precedes_the_arrival():
    """Locks the exact-time convention independently of legacy solver classes."""
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[10.0]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    first = model.forward_continuous([InputSpike(0.0, 0, 0)], 20.0)
    boundary = first.hidden_spikes[0].time

    collision = model.forward_continuous(
        [InputSpike(0.0, 0, 0), InputSpike(boundary, 0, 0)], 20.0
    )

    assert collision.input_arrivals[-1].time == boundary
    assert collision.hidden_spikes[0].time == pytest.approx(boundary, abs=1e-12)
    assert collision.hidden_spikes[0].slope == pytest.approx(
        first.hidden_spikes[0].slope, abs=1e-12
    )
    assert collision.hidden_spikes[0].q_pre == pytest.approx(
        first.hidden_spikes[0].q_pre, abs=1e-12
    )


def test_input_at_trial_endpoint_is_recorded_without_post_endpoint_flow():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[40.0]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )

    result = model.forward_continuous([InputSpike(20.0, 2000, 0)], 20.0)

    assert [arrival.time for arrival in result.input_arrivals] == [20.0]
    assert result.hidden_spikes == ()
    assert result.hidden_arrivals == ()


def test_multi_neuron_order_lineage_and_silent_stream_are_deterministic():
    model = TeSNeurons(
        1,
        3,
        1,
        torch.tensor([[40.0], [0.0], [40.0]], dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = [InputSpike(0.0, 0, 0)]

    first = model.forward_continuous(inputs, 20.0)
    second = model.forward_continuous(inputs, 20.0)

    assert first == second
    assert {event.neuron for event in first.hidden_spikes} == {0, 2}
    assert [event.event_id for event in first.hidden_spikes] == list(
        range(len(first.hidden_spikes))
    )
    assert all(event.neuron != 1 for event in first.hidden_spikes)
    assert [arrival.source for arrival in first.hidden_arrivals] == [
        event.neuron for event in first.hidden_spikes
    ]
    assert [arrival.parent_threshold_id for arrival in first.hidden_arrivals] == list(
        range(len(first.hidden_spikes))
    )
