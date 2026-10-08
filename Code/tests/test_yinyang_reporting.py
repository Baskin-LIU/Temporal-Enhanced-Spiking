import json
from dataclasses import replace

import pytest
import torch

import Code.yinyang_reporting as yinyang_reporting
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config
from Code.yinyang_experiment import run_yinyang_training
from Code.yinyang_reporting import (
    compact_epoch,
    write_training_artifacts,
    write_training_report,
)


def _small_config(*, rasters: bool = False, intervals: int = 0, epochs: int = 1):
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    return replace(
        config,
        data=replace(
            config.data,
            train_size=50,
            validation_size=12,
            test_size=12,
        ),
        optimization=replace(
            config.optimization,
            epochs=epochs,
            cosine_t_max=epochs,
        ),
        artifacts=replace(
            config.artifacts,
            checkpoint_interval=intervals,
            plot_interval=intervals,
            save_rasters=rasters,
        ),
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_compact_report_retains_scientific_metrics_without_hot_path_records(tmp_path):
    config = _small_config()
    artifacts = run_yinyang_training(config, tmp_path, evaluate_test=True)
    destination = write_training_report(
        tmp_path / "metrics.json",
        artifacts,
        config,
        artifact_paths={
            "best_validation_checkpoint": "checkpoints/best_validation.pt",
            "final_tested_checkpoint": "checkpoints/final_tested.pt",
            "training_figure": "figures/training.png",
        },
    )
    report = json.loads(destination.read_text())

    assert report["configuration"] == json.loads(json.dumps(config.to_dict()))
    assert report["omega_training"]["enabled"] is False
    assert report["test_evaluations"] == 1
    assert report["epochs"] == json.loads(
        json.dumps([compact_epoch(artifacts.epochs[0])])
    )
    assert report["epochs"][0]["restarted_hidden"] == list(
        artifacts.epochs[0].restarted_hidden
    )
    assert report["restart_history"] == json.loads(
        json.dumps(artifacts.restart_history)
    )
    assert "confusion_matrix" in report["final_validation"]
    assert "per_class_accuracy" in report["final_validation"]
    assert "hidden_silence_counts" in report["final_validation"]
    assert "output_silence_counts" in report["test"]
    assert set(report["artifacts"]) == {
        "best_validation_checkpoint",
        "final_tested_checkpoint",
        "training_figure",
    }
    forbidden = {"batches", "sample_indices", "event_tape", "encoded_events"}
    assert forbidden.isdisjoint(set(_all_keys(report)))
    assert str(tmp_path.resolve()) not in destination.read_text()


def test_default_artifact_policy_writes_only_selected_outputs(tmp_path):
    config = _small_config()
    artifacts = run_yinyang_training(config, tmp_path, evaluate_test=True)

    manifest = write_training_artifacts(tmp_path, artifacts)

    files = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert files == {
        "config.json",
        "metrics.json",
        "checkpoints/best_validation.pt",
        "checkpoints/final_tested.pt",
        "figures/training.png",
    }
    assert manifest["best_validation_checkpoint"] == (
        "checkpoints/best_validation.pt"
    )
    assert manifest["final_tested_checkpoint"] == "checkpoints/final_tested.pt"
    assert all(not value.startswith("/") for value in manifest.values())


def test_intervals_and_rasters_are_opt_in(tmp_path):
    config = _small_config(rasters=True, intervals=1)
    artifacts = run_yinyang_training(config, tmp_path, evaluate_test=False)

    manifest = write_training_artifacts(tmp_path, artifacts)

    assert (tmp_path / "checkpoints/epoch_001.pt").is_file()
    assert (tmp_path / "figures/epoch_001.png").is_file()
    assert not (tmp_path / "checkpoints/final_tested.pt").exists()
    assert manifest["raster_pre_training"] == (
        "figures/raster_pre_training.png"
    )
    assert (tmp_path / manifest["raster_correct"]).is_file()
    assert (tmp_path / manifest["raster_incorrect"]).is_file()


def test_training_figure_has_requested_six_panel_contract(tmp_path):
    import matplotlib.pyplot as plt

    config = _small_config(epochs=2)
    artifacts = run_yinyang_training(
        config,
        tmp_path,
        evaluate_test=False,
        learn_omega=True,
    )
    artifacts = replace(
        artifacts,
        restart_history=(
            {
                "kind": "in_epoch_bump",
                "epoch": 1,
                "batch": 1,
                "hidden": (0, 2),
            },
            {
                "kind": "validation_restart",
                "epoch": 2,
                "hidden": (1,),
            },
        ),
    )

    figure = yinyang_reporting._build_training_figure(
        artifacts,
        artifacts.epochs,
    )
    titled_axes = {
        axis.get_title(): axis for axis in figure.axes if axis.get_title()
    }

    assert set(titled_axes) == {
        "Timing loss",
        "Physical-spike accuracy",
        "Validation silence (causal H / physical O)",
        "Optimizer diagnostics",
        "Lightweight Rescue",
        "Per-neuron omega",
    }
    accuracy = titled_axes["Physical-spike accuracy"]
    assert accuracy.get_ylim()[0] == pytest.approx(0.5)
    annotation = "\n".join(text.get_text() for text in accuracy.texts)
    assert "Max train" in annotation
    assert "Max validation" in annotation

    silence = titled_axes["Validation silence (causal H / physical O)"]
    matrix = silence.images[0].get_array()
    assert matrix.shape == (
        config.model.n_hidden + config.model.n_outputs,
        config.optimization.epochs,
    )
    labels = [label.get_text() for label in silence.get_yticklabels()]
    assert labels[0] == "H0"
    assert labels[-1] == f"O{config.model.n_outputs - 1}"

    rescue = titled_axes["Lightweight Rescue"]
    assert rescue.lines[0].get_ydata()[-1] == 2
    assert rescue.lines[1].get_ydata()[-1] == 1

    omega = titled_axes["Per-neuron omega"]
    assert [line.get_label() for line in omega.lines] == ["initial", "final"]
    plt.close(figure)


def test_sparse_epoch_ticks_keep_endpoints_for_a_thirty_epoch_run():
    positions, labels = yinyang_reporting._sparse_epoch_ticks(
        tuple(range(1, 31)),
    )

    assert positions[0] == 0
    assert positions[-1] == 29
    assert labels[0] == 1
    assert labels[-1] == 30
    assert len(labels) <= 10


def test_interval_checkpoint_never_embeds_a_future_best_snapshot(tmp_path):
    config = _small_config(intervals=1, epochs=2)
    artifacts = run_yinyang_training(config, tmp_path, evaluate_test=False)
    forced_future_best = replace(
        artifacts,
        best_validation_snapshot=artifacts.final_snapshot,
    )

    write_training_artifacts(tmp_path, forced_future_best)
    first = torch.load(
        tmp_path / "checkpoints/epoch_001.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert first["epoch"] == 1
    assert first["best_validation_snapshot"]["epoch"] <= 1


def test_cli_entry_point_uses_the_typed_config_and_compact_writer(tmp_path, capsys):
    from Code.train_yinyang import main

    report = main(
        [
            "--run-dir",
            str(tmp_path),
            "--train-size",
            "50",
            "--validation-size",
            "12",
            "--test-size",
            "12",
            "--epochs",
            "1",
            "--cosine-t-max",
            "1",
            "--no-save-rasters",
            "--evaluate-test",
        ]
    )

    assert report == tmp_path / "metrics.json"
    assert capsys.readouterr().out.strip() == str(report)
    assert report.is_file()


def test_cli_learn_omega_records_frequency_training_and_checkpoint(tmp_path):
    from Code.train_yinyang import main
    from Code.yinyang_config import parse_yinyang_args

    report_path = main(
        [
            "--run-dir",
            str(tmp_path),
            "--train-size",
            "50",
            "--validation-size",
            "12",
            "--test-size",
            "12",
            "--epochs",
            "1",
            "--cosine-t-max",
            "1",
            "--no-save-rasters",
            "--learn_omega",
            "--omega-learning-rate-scale",
            "0.02",
            "--in-epoch-bump-interval",
            "10",
            "--in-epoch-bump-silence-fraction",
            "0.5",
            "--in-epoch-bump-amount",
            "0.05",
        ]
    )

    report = json.loads(report_path.read_text())
    checkpoint = torch.load(
        tmp_path / "checkpoints/best_validation.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert report["omega_training"]["enabled"] is True
    assert report["omega_training"]["learning_rate_scale"] == 0.02
    assert report["omega_training"]["minimum"] >= 0.1
    assert report["omega_training"]["maximum"] <= 0.35
    assert report["in_epoch_bump"] == {
        "enabled": True,
        "interval_batches": 10,
        "silence_fraction": 0.5,
        "amount": 0.05,
    }
    assert checkpoint["config"]["learn_omega"] is True
    assert checkpoint["omega"].shape == (15,)
    assert checkpoint["adam"]["omega_step"] == 1
    saved_config = json.loads((tmp_path / "config.json").read_text())
    assert saved_config["training_options"]["learn_omega"] is True
    restored = parse_yinyang_args(
        ["--config", str(tmp_path / "config.json")]
    )
    assert restored.learn_omega is True
    assert restored.omega_learning_rate_scale == pytest.approx(0.02)
    assert restored.in_epoch_bump_interval_batches == 10
    assert restored.in_epoch_bump_silence_fraction == pytest.approx(0.5)
    assert restored.in_epoch_bump_amount == pytest.approx(0.05)
    assert parse_yinyang_args(
        ["--config", str(tmp_path / "config.json"), "--no-learn-omega"]
    ).learn_omega is False
