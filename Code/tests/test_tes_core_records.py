from dataclasses import FrozenInstanceError, fields

import pytest


def test_event_records_are_frozen_and_backward_result_is_training_facing():
    """Catches mutable event tape records or leakage of debug-only adjoint traces."""
    from Code.tes_events import BackwardResult, InputSpike

    spike = InputSpike(exact_time=0.2, grid_step=4, channel=1)
    with pytest.raises(FrozenInstanceError):
        spike.channel = 0

    assert {item.name for item in fields(BackwardResult)} == {
        "output_time_grad",
        "hidden_spike_time_grad",
        "grad_W_in",
        "grad_W_out",
        "threshold_sensitivity_cap_activations",
        "minimum_causal_slope",
    }
