import math

import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons


def make_model(dt: float = 0.1) -> TeSNeurons:
    config = TeSConfig(dt=dt)
    w_in = torch.zeros((4, 4), dtype=torch.float64)
    w_out = torch.zeros((3, 4), dtype=torch.float64)
    return TeSNeurons(4, 4, 3, w_in, w_out, config)


def test_constructor_rejects_inconsistent_weight_shapes():
    with pytest.raises(ValueError, match="weight shapes must be"):
        TeSNeurons(
            2,
            3,
            1,
            torch.zeros((3, 1), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            TeSConfig(),
        )


def test_odd_hidden_width_has_one_fewer_fast_neuron_than_slow_neurons():
    config = TeSConfig()
    model = TeSNeurons(
        1,
        15,
        1,
        torch.zeros((15, 1), dtype=torch.float64),
        torch.zeros((1, 15), dtype=torch.float64),
        config,
    )

    torch.testing.assert_close(
        model.omega[:7],
        torch.full((7,), config.omega_fast, dtype=torch.float64),
    )
    torch.testing.assert_close(
        model.omega[7:15],
        torch.full((8,), config.omega_slow, dtype=torch.float64),
    )


def test_config_exposes_only_fixed_dynamics_constants():
    config = TeSConfig()
    assert config.tau_v == 20.0
    assert config.alpha == 1.0 / 40.0
    assert config.threshold == 1.0
    assert config.fast_period_ms == 8.0 * math.pi
    assert config.slow_period_ms == 12.0 * math.pi
    assert config.omega_fast == 1.0 / 4.0
    assert config.omega_slow == 1.0 / 6.0
    assert config.d_plus == 5.0
    assert config.c0 == 20.0
    assert config.c1 == 2.0


def test_threshold_sensitivity_caps_inverse_tau_slope_at_ten():
    model = make_model()

    assert model.config.threshold_sensitivity_cap == 10.0
    assert model._threshold_sensitivity(0.01) == pytest.approx(5.0)
    assert model._threshold_sensitivity(0.005) == pytest.approx(10.0)
    assert model._threshold_sensitivity(0.001) == pytest.approx(10.0)


def test_threshold_sensitivity_uses_configured_clip_of_two():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.ones((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        TeSConfig(threshold_sensitivity_cap=2.0),
    )

    assert model._threshold_sensitivity(0.1) == pytest.approx(0.5)
    assert model._threshold_sensitivity(0.01) == pytest.approx(2.0)
    assert model._threshold_sensitivity(0.001) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_threshold_sensitivity_rejects_invalid_clip(value):
    with pytest.raises(ValueError, match="threshold_sensitivity_cap"):
        TeSConfig(threshold_sensitivity_cap=value)


def test_config_accepts_root_slope_classification_tolerance():
    assert TeSConfig(slope_epsilon=1e-9).slope_epsilon == 1e-9


def test_config_accepts_experimental_dynamics_overrides():
    config = TeSConfig(
        tau_v=15.0,
        alpha=0.04,
        threshold=1.2,
        omega_fast=0.3,
        omega_slow=0.2,
    )

    assert config.tau_v == 15.0
    assert config.alpha == 0.04
    assert config.threshold == 1.2
    assert config.fast_period_ms == pytest.approx(2.0 * math.pi / 0.3)
    assert config.slow_period_ms == pytest.approx(2.0 * math.pi / 0.2)


@pytest.mark.parametrize("name", ["fast_period_ms", "slow_period_ms"])
def test_config_rejects_overriding_derived_periods(name):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TeSConfig(**{name: 999.0})


def test_config_accepts_explicit_delay_parameters_and_uses_them():
    config = TeSConfig(dt=0.25, d_plus=7.5, c0=8.0, c1=-0.4)
    model = TeSNeurons(
        1,
        1,
        1,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        config,
    )

    assert config.d_plus == 7.5
    assert config.c0 == 8.0
    assert config.c1 == -0.4
    assert model.delay_from_slope(0.2) == pytest.approx(
        7.5 / (1.0 + math.exp(2.0))
    )


def test_frequency_groups_are_fixed_and_deterministic():
    model = make_model()
    expected = torch.tensor([
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
    ], dtype=torch.float64)
    torch.testing.assert_close(model.omega, expected)
    assert model._root_panel_width_fast == pytest.approx(math.pi)
    assert model._root_panel_width_slow == pytest.approx(1.5 * math.pi)
    assert list(model.parameters()) == []


def test_coarse_step_warning_tracks_twenty_steps_per_fast_period():
    boundary = 2.0 * math.pi / 5.0
    assert make_model(dt=boundary).forward_discrete([], t_end=0.0).warnings == ()
    assert make_model(dt=boundary + 1e-6).forward_discrete([], t_end=0.0).warnings == (
        "fewer than 20 grid steps per fast oscillation period",
    )


def test_odd_hidden_width_uses_floor_fast_rows_and_slow_readouts():
    model = TeSNeurons(
        1,
        5,
        2,
        torch.zeros((5, 1), dtype=torch.float64),
        torch.zeros((2, 5), dtype=torch.float64),
        TeSConfig(),
    )

    expected = torch.tensor(
        [
            1.0 / 4.0,
            1.0 / 4.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
            1.0 / 6.0,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(model.omega, expected)


@pytest.mark.parametrize("neuron", [0, 1])
def test_free_flow_matches_damped_rotation(neuron):
    model = make_model()
    state = torch.zeros((model.n_neurons, 3), dtype=torch.float64)
    state[neuron, 2] = -2.0
    delta = 3.25
    flowed = model.flow_state(state, delta)
    omega = model.omega[neuron]
    envelope = math.exp(-model.config.alpha * delta)
    assert flowed[neuron, 1].item() == pytest.approx(
        2.0 * envelope * math.sin(omega.item() * delta), rel=1e-11, abs=1e-12
    )
    assert flowed[neuron, 2].item() == pytest.approx(
        -2.0 * envelope * math.cos(omega.item() * delta), rel=1e-11, abs=1e-12
    )


def test_free_flow_has_semigroup_property():
    model = make_model()
    generator = torch.Generator().manual_seed(9)
    state = torch.randn((model.n_neurons, 3), generator=generator, dtype=torch.float64)
    split = model.flow_state(model.flow_state(state, 1.7), 2.3)
    direct = model.flow_state(state, 4.0)
    torch.testing.assert_close(split, direct, rtol=2e-11, atol=2e-12)


def test_simultaneous_inputs_change_only_hidden_q():
    config = TeSConfig(dt=0.1)
    w_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    model = TeSNeurons(2, 2, 1, w_in, torch.zeros((1, 2)), config)
    before = model.state.clone()
    model.enqueue_input(InputSpike(exact_time=0.0, grid_step=0, channel=0))
    model.enqueue_input(InputSpike(exact_time=0.0, grid_step=0, channel=1))
    model.apply_due_arrivals(step=0)
    torch.testing.assert_close(model.V, before[:, 0])
    torch.testing.assert_close(model.I, before[:, 1])
    torch.testing.assert_close(model.Q[:2], -(w_in[:, 0] + w_in[:, 1]))
    assert model.Q[2].item() == 0.0


def test_simultaneous_hidden_arrivals_change_only_output_q_by_negative_weights():
    w_out = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float64)
    model = TeSNeurons(
        1,
        2,
        2,
        torch.zeros((2, 1), dtype=torch.float64),
        w_out,
        TeSConfig(dt=0.1),
    )
    model.state.copy_(
        torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype=torch.float64,
        )
    )
    before = model.state.clone()
    model._enqueue_hidden(source=0, spike_step=0, continuous_delay=0.11)
    model._enqueue_hidden(source=1, spike_step=0, continuous_delay=0.11)

    model.apply_due_arrivals(step=2)

    expected = before.clone()
    expected[2:, 2] -= torch.tensor([-1.0, 3.5], dtype=torch.float64)
    torch.testing.assert_close(model.state, expected)
    assert [event.source for event in model.arrival_records] == [0, 1]


def test_delay_decreases_with_slope_and_rounds_up():
    model = make_model(dt=0.3)
    slow = model.delay_from_slope(0.05)
    fast = model.delay_from_slope(0.20)
    assert 0.0 < fast < slow < model.config.d_plus
    assert model.quantize_delay(1.01) == 4
    assert model.quantize_delay(1.20) == 4


def test_grid_step_records_positive_slope_and_subtractive_reset():
    model = make_model(dt=0.1)
    model.state[0] = torch.tensor([0.999, 2.0, 0.5], dtype=torch.float64)
    model.step(step=0, delta=0.1)
    event = model.threshold_events[0]
    assert event.neuron == 0
    assert event.slope > 0.0
    assert event.v_post == pytest.approx(event.v_pre - model.config.threshold)
    assert event.i_pre == pytest.approx(model.I[0].item())
    assert event.q_pre == pytest.approx(model.Q[0].item())
    assert abs(event.i_pre) > 0.1
    assert abs(event.q_pre) > 0.1
    assert model.V[0].item() < model.config.threshold


def test_internal_arrivals_use_stable_time_then_sequence_order():
    model = make_model(dt=0.1)
    model._enqueue_hidden(source=1, spike_step=2, continuous_delay=0.31)
    model._enqueue_hidden(source=0, spike_step=2, continuous_delay=0.11)
    queued = sorted(model._queue)
    assert [(item.step, item.source) for item in queued] == [(4, 0), (6, 1)]


def test_equal_step_queue_entries_preserve_insertion_order_by_sequence():
    model = make_model(dt=0.1)
    model._enqueue_hidden(
        source=1, spike_step=2, continuous_delay=0.11, parent_threshold_id=7
    )
    model._enqueue_hidden(
        source=0, spike_step=2, continuous_delay=0.11, parent_threshold_id=8
    )

    queued = sorted(model._queue)

    assert [(item.step, item.sequence, item.source) for item in queued] == [
        (4, 0, 1),
        (4, 1, 0),
    ]
    assert [item.parent_threshold_id for item in queued] == [7, 8]


def test_hidden_arrival_links_to_unique_parent_threshold():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[10.0]], dtype=torch.float64),
        torch.tensor([[2.5]], dtype=torch.float64),
        TeSConfig(dt=0.1),
    )
    trajectory = model.forward_discrete(
        [InputSpike(exact_time=0.0, grid_step=0, channel=0)],
        t_end=6.0,
    )
    hidden_thresholds = [event for event in trajectory.thresholds if event.neuron == 0]
    hidden_arrivals = [event for event in trajectory.arrivals if event.kind == "hidden"]

    assert [event.event_id for event in trajectory.thresholds] == list(
        range(len(trajectory.thresholds))
    )
    assert hidden_arrivals
    assert hidden_arrivals[0].parent_threshold_id == hidden_thresholds[0].event_id
    assert all(
        event.parent_threshold_id is None
        for event in trajectory.arrivals
        if event.kind == "input"
    )


def test_run_delivers_external_arrival_at_grid_aligned_end_time():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[1.75]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.1),
    )

    trajectory = model.forward_discrete(
        [InputSpike(exact_time=0.2, grid_step=2, channel=0)],
        t_end=0.2,
    )

    assert [(event.time, event.kind) for event in trajectory.arrivals] == [(0.2, "input")]
    assert trajectory.Q[-1, 0].item() == pytest.approx(-1.75)


def test_run_treats_float_rounded_grid_multiple_as_aligned_end_time():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[1.75]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.01),
    )

    trajectory = model.forward_discrete(
        [InputSpike(exact_time=0.07, grid_step=7, channel=0)],
        t_end=0.07,
    )

    assert trajectory.times.shape == (8,)
    assert trajectory.times[-1].item() == pytest.approx(0.07)
    assert trajectory.arrivals[-1].time == pytest.approx(0.07)
    assert trajectory.Q[-1, 0].item() == pytest.approx(-1.75)


def test_run_delivers_hidden_arrival_at_grid_aligned_end_time():
    def make_endpoint_model() -> TeSNeurons:
        return TeSNeurons(
            1,
            1,
            1,
            torch.tensor([[10.0]], dtype=torch.float64),
            torch.tensor([[2.5]], dtype=torch.float64),
            TeSConfig(dt=0.1),
        )

    input_spikes = [InputSpike(exact_time=0.0, grid_step=0, channel=0)]
    probe = make_endpoint_model().forward_discrete(input_spikes, t_end=10.0)
    threshold = probe.thresholds[0]
    assert threshold.arrival_time is not None
    assert threshold.delay_steps is not None
    delivery_time = (threshold.step + threshold.delay_steps) * 0.1

    trajectory = make_endpoint_model().forward_discrete(
        input_spikes,
        t_end=delivery_time,
    )

    assert trajectory.thresholds[0].arrival_time == pytest.approx(
        threshold.arrival_time
    )
    assert trajectory.arrivals[-1].kind == "hidden"
    assert trajectory.arrivals[-1].time == pytest.approx(delivery_time)
    assert trajectory.arrivals[-1].exact_time == pytest.approx(
        threshold.arrival_time
    )
    assert trajectory.Q[-1, 1].item() == pytest.approx(-2.5)


def test_run_does_not_deliver_next_grid_arrival_past_shortened_horizon():
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[1.75]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64),
        TeSConfig(dt=0.1),
    )

    trajectory = model.forward_discrete(
        [InputSpike(exact_time=0.3, grid_step=3, channel=0)],
        t_end=0.25,
    )

    assert trajectory.times[-1].item() == pytest.approx(0.25)
    assert trajectory.arrivals == ()
    assert trajectory.Q[-1, 0].item() == 0.0


def test_forward_discrete_compact_mode_preserves_events_and_readouts():
    """Catches training mode changing dynamics while dropping state history."""
    model = TeSNeurons(
        1,
        1,
        1,
        torch.tensor([[40.0]], dtype=torch.float64),
        torch.tensor([[16.0]], dtype=torch.float64),
        TeSConfig(dt=0.01, reset_drop=5.0),
    )
    inputs = (InputSpike(exact_time=0.0, grid_step=0, channel=0),)

    full = model.forward_discrete(inputs, 50.0, record_state=True)
    compact = model.forward_discrete(inputs, 50.0, record_state=False)

    assert full.times is not None
    assert full.V is not None
    assert compact.times is None
    assert compact.V is None
    assert compact.I is None
    assert compact.Q is None
    assert compact.arrivals == full.arrivals
    assert compact.thresholds == full.thresholds
    assert compact.readouts == full.readouts
    assert sum(event.neuron == 0 for event in compact.thresholds) >= 2
