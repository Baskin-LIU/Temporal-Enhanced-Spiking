import json
from pathlib import Path

import torch

from Code.tes_loss import evaluate_timing_loss
from Code.tes_training import build_yinyang_splits
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config
from Code.yinyang_experiment import initialize_yinyang_model


GOLDEN_PATH = Path(__file__).parent / "data" / "yinyang_numerical_golden.json"


def _canonical_result() -> dict[str, object]:
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    training = config.to_training_config()
    model = initialize_yinyang_model(config)
    sample = build_yinyang_splits(
        train_size=config.data.train_size,
        validation_size=config.data.validation_size,
        test_size=config.data.test_size,
        seeds=config.data.split_seeds,
        dt=config.model.dt_ms,
    )[0].samples[0]
    forward = model.forward_continuous(
        sample.events,
        config.model.t_end_ms,
        early_stop=False,
    )
    loss = evaluate_timing_loss(
        forward.readouts,
        label=sample.label,
        loss_name=training.loss_name,
        margin=training.margin,
    )
    backward = model.backward_continuous(forward, loss.output_time_grad)
    return {
        "sample": {
            "index": sample.index,
            "point": list(sample.point),
            "label": sample.label,
            "events": [
                {
                    "exact_time": event.exact_time,
                    "grid_step": event.grid_step,
                    "channel": event.channel,
                }
                for event in sample.events
            ],
        },
        "hidden_spikes": [
            {
                "event_id": event.event_id,
                "neuron": event.neuron,
                "time": event.time,
                "slope": event.slope,
            }
            for event in forward.hidden_spikes
        ],
        "output_spikes": [
            {
                "event_id": event.event_id,
                "neuron": event.neuron,
                "time": event.time,
                "slope": event.slope,
            }
            for event in forward.output_spikes
        ],
        "readouts": [
            {
                "neuron": readout.neuron,
                "physical_time": readout.physical_time,
                "effective_time": readout.effective_time,
                "censored": readout.censored,
                "anchor_output": readout.anchor_output,
            }
            for readout in forward.readouts
        ],
        "loss": {
            "value": loss.loss,
            "effective_times": list(loss.effective_times),
            "physical_mask": list(loss.physical_mask),
            "active_wrong_outputs": list(loss.active_wrong_outputs),
            "output_time_grad": loss.output_time_grad.tolist(),
        },
        "backward": {
            "grad_W_in": backward.grad_W_in.tolist(),
            "grad_W_out": backward.grad_W_out.tolist(),
            "cap_activations": backward.threshold_sensitivity_cap_activations,
            "minimum_causal_slope": backward.minimum_causal_slope,
        },
    }


def test_canonical_forward_loss_and_backward_match_golden():
    expected = json.loads(GOLDEN_PATH.read_text())
    actual = _canonical_result()

    assert actual["sample"] == expected["sample"]
    assert [
        (event["event_id"], event["neuron"])
        for event in actual["hidden_spikes"]
    ] == [
        (event["event_id"], event["neuron"])
        for event in expected["hidden_spikes"]
    ]
    assert [
        (event["event_id"], event["neuron"])
        for event in actual["output_spikes"]
    ] == [
        (event["event_id"], event["neuron"])
        for event in expected["output_spikes"]
    ]
    for section in ("hidden_spikes", "output_spikes"):
        torch.testing.assert_close(
            torch.tensor(
                [[event["time"], event["slope"]] for event in actual[section]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[event["time"], event["slope"]] for event in expected[section]],
                dtype=torch.float64,
            ),
            rtol=1e-12,
            atol=1e-12,
        )
    assert actual["readouts"] == expected["readouts"]
    assert actual["loss"]["physical_mask"] == expected["loss"]["physical_mask"]
    assert actual["loss"]["active_wrong_outputs"] == expected["loss"]["active_wrong_outputs"]
    torch.testing.assert_close(
        torch.tensor(
            [actual["loss"]["value"], *actual["loss"]["output_time_grad"]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [expected["loss"]["value"], *expected["loss"]["output_time_grad"]],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual["backward"]["cap_activations"] == expected["backward"]["cap_activations"]
    torch.testing.assert_close(
        torch.tensor(actual["backward"]["grad_W_in"], dtype=torch.float64),
        torch.tensor(expected["backward"]["grad_W_in"], dtype=torch.float64),
        rtol=1e-11,
        atol=1e-11,
    )
    torch.testing.assert_close(
        torch.tensor(actual["backward"]["grad_W_out"], dtype=torch.float64),
        torch.tensor(expected["backward"]["grad_W_out"], dtype=torch.float64),
        rtol=1e-11,
        atol=1e-11,
    )
