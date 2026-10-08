"""Compact checkpoints, reports, and figures for Yin-Yang experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch

from Code.neurons import TeSNeurons
from Code.tes_training import (
    AdamState,
    EvaluationMetrics,
    TrainingSample,
    YinYangSplit,
    physical_prediction,
    save_checkpoint,
)
from Code.yinyang_config import YinYangConfig
from Code.yinyang_experiment import (
    EpochRecord,
    TrainingArtifacts,
    TrainingSnapshot,
    initialize_yinyang_model,
)


@dataclass(frozen=True)
class SpikeRasterExample:
    sample_index: int
    true_label: int
    predicted_label: int | None
    correct: bool
    input_times: tuple[tuple[float, ...], ...]
    hidden_times: tuple[tuple[float, ...], ...]
    output_times: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RasterExamples:
    pre_training: tuple[SpikeRasterExample | None, ...]
    correct: tuple[SpikeRasterExample | None, ...]
    incorrect: tuple[SpikeRasterExample | None, ...]
    t_end_ms: float


def _plain(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    os.replace(temporary, path)
    return path


def _compact_evaluation(metrics: EvaluationMetrics | None) -> dict | None:
    return None if metrics is None else asdict(metrics)


def compact_epoch(record: EpochRecord) -> dict[str, object]:
    """Return one complete epoch summary without batch or sample histories."""
    return {
        "epoch": record.epoch,
        "learning_rate": record.learning_rate,
        "training": asdict(record.training),
        "validation_pre_restart": asdict(record.validation_pre_restart),
        "validation": asdict(record.validation),
        "restarted_hidden": list(record.restarted_hidden),
    }


def write_training_report(
    path: Path | str,
    artifacts: TrainingArtifacts,
    config: YinYangConfig,
    *,
    artifact_paths: Mapping[str, str] | None = None,
) -> Path:
    """Write the compact, exact experiment record."""
    payload = {
        "configuration": config.to_dict(),
        "omega_training": {
            "enabled": artifacts.training_config.learn_omega,
            "learning_rate_scale": (
                artifacts.training_config.omega_learning_rate_scale
            ),
            "clamp_minimum": artifacts.training_config.omega_min,
            "clamp_maximum": artifacts.training_config.omega_max,
            "minimum": float(artifacts.final_snapshot.omega.min()),
            "maximum": float(artifacts.final_snapshot.omega.max()),
            "values": tuple(
                float(value) for value in artifacts.final_snapshot.omega
            ),
        },
        "in_epoch_bump": {
            "enabled": (
                artifacts.training_config.in_epoch_bump_interval_batches > 0
            ),
            "interval_batches": (
                artifacts.training_config.in_epoch_bump_interval_batches
            ),
            "silence_fraction": (
                artifacts.training_config.in_epoch_bump_silence_fraction
            ),
            "amount": artifacts.training_config.in_epoch_bump_amount,
        },
        "initialization": artifacts.initialization,
        "calibration": asdict(artifacts.calibration),
        "baseline_train": _compact_evaluation(artifacts.baseline_train_metrics),
        "baseline_validation": _compact_evaluation(
            artifacts.baseline_validation_metrics
        ),
        "epochs": [compact_epoch(record) for record in artifacts.epochs],
        "restart_history": artifacts.restart_history,
        "final_train": _compact_evaluation(artifacts.final_train_metrics),
        "final_validation": _compact_evaluation(
            artifacts.final_validation_metrics
        ),
        "test": _compact_evaluation(artifacts.test_metrics),
        "test_evaluations": artifacts.test_evaluations,
        "runtime_seconds": artifacts.runtime_seconds,
        "artifacts": dict(artifact_paths or {}),
    }
    return _write_json(Path(path), payload)


def _model_from_snapshot(
    config: YinYangConfig,
    snapshot: TrainingSnapshot,
) -> TeSNeurons:
    model = initialize_yinyang_model(config)
    model.W_in.copy_(snapshot.w_in)
    model.W_out.copy_(snapshot.w_out)
    model.omega.copy_(snapshot.omega)
    return model


def _adam_from_snapshot(snapshot: TrainingSnapshot) -> AdamState:
    payload = snapshot.adam
    return AdamState(
        m_in=payload["m_in"].clone(),
        v_in=payload["v_in"].clone(),
        m_out=payload["m_out"].clone(),
        v_out=payload["v_out"].clone(),
        m_omega=payload["m_omega"].clone(),
        v_omega=payload["v_omega"].clone(),
        step=int(payload["step"]),
        omega_step=int(payload["omega_step"]),
    )


def _snapshot_payload(snapshot: TrainingSnapshot) -> dict[str, object]:
    return {
        "epoch": snapshot.epoch,
        "w_in": snapshot.w_in.detach().clone(),
        "w_out": snapshot.w_out.detach().clone(),
        "omega": snapshot.omega.detach().clone(),
        "adam": {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for name, value in snapshot.adam.items()
        },
    }


def _save_snapshot(
    path: Path,
    artifacts: TrainingArtifacts,
    snapshot: TrainingSnapshot,
    *,
    best_validation_snapshot: TrainingSnapshot | None = None,
) -> Path:
    records = tuple(
        compact_epoch(record)
        for record in artifacts.epochs
        if record.epoch <= snapshot.epoch
    )
    return save_checkpoint(
        path,
        _model_from_snapshot(artifacts.config, snapshot),
        _adam_from_snapshot(snapshot),
        epoch=snapshot.epoch,
        config=artifacts.training_config,
        calibration_history=tuple(
            asdict(record) for record in artifacts.calibration.rounds
        ),
        bump_history=tuple(
            item
            for item in artifacts.restart_history
            if int(item["epoch"]) <= snapshot.epoch
        ),
        epoch_metrics=records,
        run_config=artifacts.config.to_dict(),
        model_config=asdict(artifacts.config.to_tes_config()),
        baseline_train=asdict(artifacts.baseline_train_metrics),
        baseline_validation=asdict(artifacts.baseline_validation_metrics),
        best_validation_snapshot=_snapshot_payload(
            artifacts.best_validation_snapshot
            if best_validation_snapshot is None
            else best_validation_snapshot
        ),
    )


def _neuron_activity(
    records: Sequence[EpochRecord],
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    n_hidden = len(records[0].validation_pre_restart.hidden_silence_counts)
    n_outputs = len(records[0].validation_pre_restart.output_silence_counts)
    silence = torch.tensor(
        [
            [
                record.validation_pre_restart.hidden_silence_counts[hidden]
                / record.validation_pre_restart.samples
                for record in records
            ]
            for hidden in range(n_hidden)
        ]
        + [
            [
                record.validation_pre_restart.output_silence_counts[output]
                / record.validation_pre_restart.samples
                for record in records
            ]
            for output in range(n_outputs)
        ],
        dtype=torch.float64,
    )
    restarts = torch.zeros(
        (n_hidden + n_outputs, len(records)), dtype=torch.bool
    )
    for column, record in enumerate(records):
        for hidden in record.restarted_hidden:
            restarts[hidden, column] = True
    labels = tuple(f"H{hidden}" for hidden in range(n_hidden)) + tuple(
        f"O{output}" for output in range(n_outputs)
    )
    return silence, restarts, labels


def _sparse_epoch_ticks(
    epochs: Sequence[int],
    *,
    maximum: int = 10,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not epochs:
        return (), ()
    if len(epochs) <= maximum:
        positions = tuple(range(len(epochs)))
    else:
        step = math.ceil((len(epochs) - 1) / (maximum - 1))
        selected = list(range(0, len(epochs), step))
        if selected[-1] != len(epochs) - 1:
            selected.append(len(epochs) - 1)
        positions = tuple(selected)
    return positions, tuple(int(epochs[index]) for index in positions)


def _cumulative_rescue_activity(
    artifacts: TrainingArtifacts,
    records: Sequence[EpochRecord],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    epochs = tuple(record.epoch for record in records)
    epoch_to_index = {epoch: index for index, epoch in enumerate(epochs)}
    lightweight = [0] * len(epochs)
    validation = [0] * len(epochs)
    for event in artifacts.restart_history:
        index = epoch_to_index.get(int(event["epoch"]))
        if index is None:
            continue
        count = len(event["hidden"])
        if event.get("kind") == "in_epoch_bump":
            lightweight[index] += count
        else:
            validation[index] += count
    for index in range(1, len(epochs)):
        lightweight[index] += lightweight[index - 1]
        validation[index] += validation[index - 1]
    return tuple(lightweight), tuple(validation)


def _build_training_figure(
    artifacts: TrainingArtifacts,
    records: Sequence[EpochRecord],
):
    import matplotlib.pyplot as plt

    x = [record.epoch for record in records]
    figure, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].plot(
        x,
        [record.training.mean_loss for record in records],
        "o-",
        label="train",
    )
    axes[0, 0].plot(
        x,
        [record.validation.mean_loss for record in records],
        "o-",
        label="validation",
    )
    axes[0, 0].set(title="Timing loss", xlabel="epoch", ylabel="mean loss")
    axes[0, 0].legend()

    accuracy = axes[0, 1]
    accuracy.plot(
        x,
        [record.training.accuracy for record in records],
        "o-",
        label="train",
    )
    accuracy.plot(
        x,
        [record.validation.accuracy for record in records],
        "o-",
        label="validation",
    )
    if artifacts.test_metrics is not None and records[-1].epoch == artifacts.epochs[-1].epoch:
        accuracy.scatter(
            x[-1],
            artifacts.test_metrics.accuracy,
            marker="*",
            s=130,
            label="test",
        )
    train_peak = max(records, key=lambda record: record.training.accuracy)
    validation_peak = max(
        records, key=lambda record: record.validation.accuracy
    )
    accuracy.text(
        0.03,
        0.05,
        (
            f"Max train: {train_peak.training.accuracy:.1%} "
            f"(epoch {train_peak.epoch})\n"
            f"Max validation: {validation_peak.validation.accuracy:.1%} "
            f"(epoch {validation_peak.epoch})"
        ),
        transform=accuracy.transAxes,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    accuracy.set(
        title="Physical-spike accuracy",
        xlabel="epoch",
        ylabel="accuracy",
        ylim=(0.5, 1.02),
    )
    accuracy.legend()

    silence, restarts, neuron_labels = _neuron_activity(records)
    silence_axis = axes[0, 2]
    image = silence_axis.imshow(
        silence.numpy(),
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )
    locations = torch.nonzero(restarts, as_tuple=False)
    if len(locations):
        silence_axis.scatter(
            locations[:, 1].numpy(),
            locations[:, 0].numpy(),
            marker="x",
            color="cyan",
            label="validation restart",
        )
        silence_axis.legend(loc="upper right")
    n_hidden = artifacts.config.model.n_hidden
    silence_axis.axhline(n_hidden - 0.5, color="white", linewidth=0.8)
    tick_positions, tick_labels = _sparse_epoch_ticks(x)
    silence_axis.set_xticks(tick_positions, tick_labels)
    silence_axis.set_yticks(range(len(neuron_labels)), neuron_labels)
    silence_axis.set(
        title="Validation silence (causal H / physical O)",
        xlabel="epoch",
        ylabel="neuron",
    )
    figure.colorbar(image, ax=silence_axis, fraction=0.046, pad=0.04).set_label(
        "silence ratio"
    )

    optimizer = axes[1, 0]
    optimizer.plot(
        x,
        [record.training.mean_raw_gradient_norm for record in records],
        "o-",
        label="raw norm",
    )
    optimizer.plot(
        x,
        [record.training.mean_clipped_gradient_norm for record in records],
        "o-",
        label="clipped norm",
    )
    optimizer.bar(
        x,
        [record.training.clipped_updates for record in records],
        alpha=0.3,
        label="clipped updates",
    )
    optimizer.set(title="Optimizer diagnostics", xlabel="epoch")
    optimizer.legend(loc="upper left")
    learning_rate_axis = optimizer.twinx()
    learning_rate_axis.step(
        x,
        [record.learning_rate for record in records],
        where="mid",
        color="tab:green",
        linestyle="--",
        label="learning rate",
    )
    learning_rate_axis.set_ylabel("learning rate", color="tab:green")

    lightweight, validation = _cumulative_rescue_activity(artifacts, records)
    rescue = axes[1, 1]
    rescue.step(
        x,
        lightweight,
        where="post",
        marker="o",
        label="lightweight row bumps",
    )
    rescue.step(
        x,
        validation,
        where="post",
        marker="x",
        label="validation restarted rows",
    )
    rescue.set(
        title="Lightweight Rescue",
        xlabel="epoch",
        ylabel="cumulative neuron rows",
    )
    rescue.legend(loc="upper left")

    omega = axes[1, 2]
    initial_omega = artifacts.pre_training_snapshot.omega.detach().cpu()
    final_omega = artifacts.final_snapshot.omega.detach().cpu()
    neurons = list(range(len(initial_omega)))
    omega.plot(neurons, initial_omega, "o-", label="initial")
    omega.plot(neurons, final_omega, "o-", label="final")
    omega.axhspan(
        artifacts.training_config.omega_min,
        artifacts.training_config.omega_max,
        color="grey",
        alpha=0.1,
        label="clamp range",
    )
    omega_labels = [
        *[f"H{hidden}" for hidden in range(artifacts.config.model.n_hidden)],
        *[
            f"O{output}"
            for output in range(artifacts.config.model.n_outputs)
        ],
    ]
    omega.set_xticks(neurons, omega_labels, rotation=45)
    omega.set(
        title="Per-neuron omega",
        xlabel="neuron",
        ylabel=r"$\omega$ (ms$^{-1}$)",
    )
    omega.legend(loc="best")

    figure.suptitle(
        f"Continuous TeS Yin-Yang training through epoch {x[-1]}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def render_training_figure(
    path: Path | str,
    artifacts: TrainingArtifacts,
    *,
    through_epoch: int | None = None,
) -> Path:
    """Render six-panel training, rescue, silence, and omega diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = tuple(
        record
        for record in artifacts.epochs
        if through_epoch is None or record.epoch <= through_epoch
    )
    if not records:
        raise ValueError("a training figure requires at least one epoch")
    figure = _build_training_figure(artifacts, records)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def _spike_example(
    model: TeSNeurons,
    sample: TrainingSample,
    t_end_ms: float,
    *,
    learn_omega: bool,
) -> SpikeRasterExample:
    forward_method = (
        model.forward_continuous_omega
        if learn_omega
        else model.forward_continuous
    )
    forward = forward_method(sample.events, t_end_ms, early_stop=False)
    prediction = physical_prediction(forward.readouts, sample.label)
    inputs = [[] for _ in range(model.n_in)]
    hidden = [[] for _ in range(model.n_hidden)]
    outputs = [[] for _ in range(model.n_out)]
    for event in sample.events:
        inputs[event.channel].append(float(event.exact_time))
    for event in forward.hidden_spikes:
        hidden[event.neuron].append(float(event.time))
    for event in forward.output_spikes:
        outputs[event.neuron].append(float(event.time))
    return SpikeRasterExample(
        sample_index=sample.index,
        true_label=sample.label,
        predicted_label=prediction.predicted,
        correct=prediction.correct,
        input_times=tuple(tuple(values) for values in inputs),
        hidden_times=tuple(tuple(values) for values in hidden),
        output_times=tuple(tuple(values) for values in outputs),
    )


def _select_by_class(
    examples: Sequence[SpikeRasterExample],
    n_classes: int,
    outcome: str,
) -> tuple[SpikeRasterExample | None, ...]:
    selected: list[SpikeRasterExample | None] = [None] * n_classes
    for example in sorted(examples, key=lambda item: item.sample_index):
        qualifies = outcome == "any" or (
            outcome == "correct" and example.correct
        ) or (outcome == "incorrect" and not example.correct)
        if qualifies and selected[example.true_label] is None:
            selected[example.true_label] = example
    return tuple(selected)


def build_raster_examples(artifacts: TrainingArtifacts) -> RasterExamples:
    validation = artifacts.splits[1]
    before = _model_from_snapshot(artifacts.config, artifacts.pre_training_snapshot)
    after = _model_from_snapshot(artifacts.config, artifacts.final_snapshot)
    t_end = artifacts.config.model.t_end_ms
    learn_omega = artifacts.training_config.learn_omega
    pre = tuple(
        _spike_example(
            before, sample, t_end, learn_omega=learn_omega
        )
        for sample in validation.samples
    )
    post = tuple(
        _spike_example(
            after, sample, t_end, learn_omega=learn_omega
        )
        for sample in validation.samples
    )
    n_classes = artifacts.config.model.n_outputs
    return RasterExamples(
        pre_training=_select_by_class(pre, n_classes, "any"),
        correct=_select_by_class(post, n_classes, "correct"),
        incorrect=_select_by_class(post, n_classes, "incorrect"),
        t_end_ms=t_end,
    )


def _render_raster(
    path: Path,
    examples: Sequence[SpikeRasterExample | None],
    t_end_ms: float,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = next((example for example in examples if example is not None), None)
    n_hidden = 12 if present is None else len(present.hidden_times)
    n_outputs = len(examples) if present is None else len(present.output_times)
    input_positions = tuple(range(4))
    hidden_positions = tuple(range(5, 5 + n_hidden))
    output_positions = tuple(range(6 + n_hidden, 6 + n_hidden + n_outputs))
    labels = (
        ("x", "y", "1-x", "1-y")
        + tuple(f"H{index}" for index in range(n_hidden))
        + tuple(f"O{index}" for index in range(n_outputs))
    )
    positions = input_positions + hidden_positions + output_positions
    figure, axes = plt.subplots(1, len(examples), figsize=(16, 8), sharex=True, sharey=True, squeeze=False)
    for class_index, (axis, example) in enumerate(zip(axes[0], examples, strict=True)):
        axis.set(xlim=(0.0, t_end_ms), ylim=(output_positions[-1] + 0.8, -0.8), xlabel="time [ms]")
        axis.set_yticks(positions, labels)
        axis.grid(axis="x", alpha=0.2)
        if example is None:
            axis.set_title(f"true class {class_index}\nno qualifying sample")
            continue
        groups = (
            (example.input_times, input_positions, "black"),
            (example.hidden_times, hidden_positions, "tab:blue"),
            (example.output_times, output_positions, "tab:red"),
        )
        for spike_groups, rows, color in groups:
            times = []
            y_values = []
            for row, spike_times in zip(rows, spike_groups, strict=True):
                times.extend(spike_times)
                y_values.extend([row] * len(spike_times))
            if times:
                axis.scatter(times, y_values, marker="|", s=120, linewidths=1.6, color=color)
        predicted = "none" if example.predicted_label is None else str(example.predicted_label)
        axis.set_title(f"true class {example.true_label}\nsample {example.sample_index}, predicted {predicted}")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render_spike_rasters(
    directory: Path | str,
    examples: RasterExamples,
) -> tuple[Path, ...]:
    """Render deterministic pre-training, correct, and incorrect rasters."""
    destination = Path(directory)
    return (
        _render_raster(destination / "raster_pre_training.png", examples.pre_training, examples.t_end_ms, "Pre-training validation examples"),
        _render_raster(destination / "raster_correct.png", examples.correct, examples.t_end_ms, "Post-training correct validation examples"),
        _render_raster(destination / "raster_incorrect.png", examples.incorrect, examples.t_end_ms, "Post-training incorrect validation examples"),
    )


def write_training_artifacts(
    run_dir: Path | str,
    artifacts: TrainingArtifacts,
) -> dict[str, str]:
    """Apply the selected-checkpoint and compact-output policy."""
    root = Path(run_dir)
    checkpoint_dir = root / "checkpoints"
    figure_dir = root / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    best_path = _save_snapshot(
        checkpoint_dir / "best_validation.pt",
        artifacts,
        artifacts.best_validation_snapshot,
    )
    manifest["best_validation_checkpoint"] = best_path.relative_to(root).as_posix()
    if artifacts.test_metrics is not None:
        tested_path = _save_snapshot(
            checkpoint_dir / "final_tested.pt",
            artifacts,
            artifacts.final_snapshot,
        )
        manifest["final_tested_checkpoint"] = tested_path.relative_to(root).as_posix()

    for interval_snapshot in artifacts.interval_snapshots:
        snapshot = interval_snapshot.state
        interval_path = _save_snapshot(
            checkpoint_dir / f"epoch_{snapshot.epoch:03d}.pt",
            artifacts,
            snapshot,
            best_validation_snapshot=interval_snapshot.best_validation,
        )
        manifest[f"checkpoint_epoch_{snapshot.epoch:03d}"] = interval_path.relative_to(root).as_posix()

    training_figure = render_training_figure(figure_dir / "training.png", artifacts)
    manifest["training_figure"] = training_figure.relative_to(root).as_posix()
    plot_interval = artifacts.config.artifacts.plot_interval
    if plot_interval:
        for record in artifacts.epochs:
            if record.epoch % plot_interval == 0:
                path = render_training_figure(
                    figure_dir / f"epoch_{record.epoch:03d}.png",
                    artifacts,
                    through_epoch=record.epoch,
                )
                manifest[f"plot_epoch_{record.epoch:03d}"] = path.relative_to(root).as_posix()

    if artifacts.config.artifacts.save_rasters:
        raster_paths = render_spike_rasters(
            figure_dir,
            build_raster_examples(artifacts),
        )
        for name, path in zip(
            ("raster_pre_training", "raster_correct", "raster_incorrect"),
            raster_paths,
            strict=True,
        ):
            manifest[name] = path.relative_to(root).as_posix()

    resolved_config = artifacts.config.to_dict()
    resolved_config["training_options"] = {
        "learn_omega": artifacts.training_config.learn_omega,
        "omega_learning_rate_scale": (
            artifacts.training_config.omega_learning_rate_scale
        ),
        "in_epoch_bump_interval_batches": (
            artifacts.training_config.in_epoch_bump_interval_batches
        ),
        "in_epoch_bump_silence_fraction": (
            artifacts.training_config.in_epoch_bump_silence_fraction
        ),
        "in_epoch_bump_amount": artifacts.training_config.in_epoch_bump_amount,
    }
    _write_json(root / "config.json", resolved_config)
    write_training_report(
        root / "metrics.json",
        artifacts,
        artifacts.config,
        artifact_paths=manifest,
    )
    return manifest
