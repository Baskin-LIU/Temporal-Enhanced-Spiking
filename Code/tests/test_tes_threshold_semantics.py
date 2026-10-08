import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons


def zero_model(*, n_hidden: int, n_out: int) -> TeSNeurons:
    return TeSNeurons(
        1,
        n_hidden,
        n_out,
        torch.zeros((n_hidden, 1), dtype=torch.float64),
        torch.zeros((n_out, n_hidden), dtype=torch.float64),
        TeSConfig(dt=0.1, reset_drop=5.0),
    )


@pytest.mark.parametrize(
    ("n_hidden", "expected"),
    [
        (1, ["slow"]),
        (4, ["fast", "fast", "slow", "slow"]),
        (5, ["fast", "fast", "slow", "slow", "slow"]),
    ],
)
def test_hidden_frequency_groups_are_contiguous_and_outputs_are_slow(
    n_hidden: int, expected: list[str]
):
    """Catches alternating hidden groups or a fast readout frequency."""
    model = zero_model(n_hidden=n_hidden, n_out=3)

    actual = [
        "fast" if float(value) == model.config.omega_fast else "slow"
        for value in model.omega[:n_hidden]
    ]

    assert actual == expected
    torch.testing.assert_close(
        model.omega[n_hidden:],
        torch.full((3,), model.config.omega_slow, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("slope", "expected"),
    [(0.02, 2.5), (0.001, 10.0), (0.0, 10.0), (-0.02, -2.5)],
)
def test_threshold_sensitivity_is_clamped_reciprocal(
    slope: float, expected: float
):
    """Catches a Python scalar divide instead of Torch reciprocal clamping."""
    model = zero_model(n_hidden=1, n_out=1)

    assert model._threshold_sensitivity(slope) == pytest.approx(expected)


def test_left_boundary_equality_resets_before_same_time_input_arrival():
    """Catches applying an arrival before an existing threshold equality."""
    model = zero_model(n_hidden=1, n_out=1)
    model.W_in.fill_(3.0)
    model.state[0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    model.enqueue_input(InputSpike(exact_time=0.0, grid_step=0, channel=0))

    model.step(step=0, delta=model.config.dt)

    event = model.threshold_events[0]
    assert event.time == 0.0
    assert event.step == 0
    assert event.v_pre == 1.0
    assert event.q_pre == 0.0
    assert event.v_post == pytest.approx(1.0 - model.config.reset_drop)
    assert model.arrival_records[0].step == 0
