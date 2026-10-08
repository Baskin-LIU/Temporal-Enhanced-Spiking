import json

import pytest
import torch

from Code.debug import (
    check_weight_gradient,
    inspect_checkpoint,
    main,
    summarize_result,
)
from Code.tes_training import AdamState, save_checkpoint
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config
from Code.yinyang_experiment import initialize_yinyang_model


def test_weight_gradient_check_is_deterministic_and_nonmutating():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)

    first = check_weight_gradient(
        config,
        split="train",
        sample_index=0,
        parameter="W_out",
        index=(0, 0),
        epsilon=1e-5,
    )
    second = check_weight_gradient(
        config,
        split="train",
        sample_index=0,
        parameter="W_out",
        index=(0, 0),
        epsilon=1e-5,
    )

    assert first == second
    assert first.source_unchanged
    assert first.analytic != 0.0
    assert first.numeric != 0.0
    assert first.relative_error < 1e-5


def test_weight_gradient_check_rejects_an_out_of_range_index():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)

    with pytest.raises(IndexError, match="outside W_in"):
        check_weight_gradient(
            config,
            split="validation",
            sample_index=0,
            parameter="W_in",
            index=(config.model.n_hidden, 0),
            epsilon=1e-5,
        )


def test_result_and_checkpoint_inspectors_return_compact_metadata(tmp_path, capsys):
    result_path = tmp_path / "metrics.json"
    result_path.write_text(
        json.dumps(
            {
                "configuration": {"loss": {"name": "squared_hinge"}},
                "epochs": [{"epoch": 1}, {"epoch": 2}],
                "final_validation": {"accuracy": 0.9},
                "test": {"accuracy": 0.8},
                "runtime_seconds": 12.5,
                "artifacts": {"training_figure": "figures/training.png"},
            }
        )
    )
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    model = initialize_yinyang_model(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        AdamState.zeros_like(model),
        epoch=3,
        config=config.to_training_config(),
        run_config=config.to_dict(),
    )

    result = summarize_result(result_path)
    checkpoint = inspect_checkpoint(checkpoint_path)

    assert result["epoch_count"] == 2
    assert result["final_validation_accuracy"] == 0.9
    assert result["test_accuracy"] == 0.8
    assert checkpoint["epoch"] == 3
    assert checkpoint["w_in_shape"] == [12, 4]
    assert checkpoint["w_out_shape"] == [3, 12]
    assert checkpoint["adam_step"] == 0
    assert all(not isinstance(value, torch.Tensor) for value in checkpoint.values())

    main(["checkpoint", str(checkpoint_path)])
    printed = json.loads(capsys.readouterr().out)
    assert printed["epoch"] == 3
