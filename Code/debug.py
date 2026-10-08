"""Small opt-in diagnostics for the production Yin-Yang training path."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import torch

from Code.tes_loss import evaluate_timing_loss
from Code.tes_training import build_yinyang_splits
from Code.yinyang_config import DEFAULT_CONFIG_PATH, YinYangConfig, load_yinyang_config
from Code.yinyang_experiment import initialize_yinyang_model


@dataclass(frozen=True)
class GradientCheckResult:
    split: str
    sample_index: int
    parameter: str
    index: tuple[int, int]
    epsilon: float
    analytic: float
    numeric: float
    absolute_error: float
    relative_error: float
    source_unchanged: bool


def _loss_for_sample(model, sample, config: YinYangConfig) -> float:
    forward = model.forward_continuous(
        sample.events,
        config.model.t_end_ms,
        early_stop=False,
    )
    return float(
        evaluate_timing_loss(
            forward.readouts,
            label=sample.label,
            loss_name=config.loss.name,
            margin=config.loss.margin_ms,
        ).loss
    )


def check_weight_gradient(
    config: YinYangConfig,
    *,
    split: Literal["train", "validation", "test"],
    sample_index: int,
    parameter: Literal["W_in", "W_out"],
    index: tuple[int, int],
    epsilon: float,
) -> GradientCheckResult:
    """Compare one analytic weight gradient with a centered finite difference."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if parameter not in {"W_in", "W_out"}:
        raise ValueError("parameter must be W_in or W_out")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    splits = build_yinyang_splits(
        config.data.train_size,
        config.data.validation_size,
        config.data.test_size,
        config.data.split_seeds,
        dt=config.model.dt_ms,
    )
    selected_split = splits[{"train": 0, "validation": 1, "test": 2}[split]]
    if not 0 <= sample_index < len(selected_split.samples):
        raise IndexError(f"sample index {sample_index} is outside {split}")
    sample = selected_split.samples[sample_index]
    source = initialize_yinyang_model(config)
    source_weights = (source.W_in.clone(), source.W_out.clone())
    weight = getattr(source, parameter)
    if (
        len(index) != 2
        or not 0 <= index[0] < weight.shape[0]
        or not 0 <= index[1] < weight.shape[1]
    ):
        raise IndexError(f"index {index!r} is outside {parameter} shape {tuple(weight.shape)}")

    forward = source.forward_continuous(
        sample.events,
        config.model.t_end_ms,
        early_stop=False,
    )
    loss = evaluate_timing_loss(
        forward.readouts,
        label=sample.label,
        loss_name=config.loss.name,
        margin=config.loss.margin_ms,
    )
    if loss.output_time_grad is None:
        raise RuntimeError(
            loss.nondifferentiable_reason
            or "selected sample has no analytic timing gradient"
        )
    backward = source.backward_continuous(forward, loss.output_time_grad)
    analytic_tensor = (
        backward.grad_W_in if parameter == "W_in" else backward.grad_W_out
    )
    analytic = float(analytic_tensor[index])

    perturbed_losses = []
    for direction in (1.0, -1.0):
        perturbed = initialize_yinyang_model(config)
        perturbed.W_in.copy_(source_weights[0])
        perturbed.W_out.copy_(source_weights[1])
        getattr(perturbed, parameter)[index] += direction * epsilon
        perturbed_losses.append(_loss_for_sample(perturbed, sample, config))
    numeric = (perturbed_losses[0] - perturbed_losses[1]) / (2.0 * epsilon)
    absolute_error = abs(analytic - numeric)
    relative_error = absolute_error / max(abs(analytic), abs(numeric), 1e-12)
    unchanged = torch.equal(source.W_in, source_weights[0]) and torch.equal(
        source.W_out,
        source_weights[1],
    )
    return GradientCheckResult(
        split=split,
        sample_index=sample_index,
        parameter=parameter,
        index=index,
        epsilon=epsilon,
        analytic=analytic,
        numeric=numeric,
        absolute_error=absolute_error,
        relative_error=relative_error,
        source_unchanged=unchanged,
    )


def summarize_result(path: Path | str) -> dict[str, object]:
    """Return headline fields from a compact metrics JSON file."""
    payload = json.loads(Path(path).read_text())
    epochs = payload.get("epochs", [])
    final_validation = payload.get("final_validation") or {}
    test = payload.get("test") or {}
    return {
        "epoch_count": len(epochs),
        "last_epoch": None if not epochs else epochs[-1].get("epoch"),
        "final_validation_accuracy": final_validation.get("accuracy"),
        "test_accuracy": test.get("accuracy"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "artifacts": payload.get("artifacts", {}),
        "configuration": payload.get("configuration"),
    }


def inspect_checkpoint(path: Path | str) -> dict[str, object]:
    """Return checkpoint metadata without exposing stored tensors."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    adam = payload.get("adam") or {}
    w_in = payload["w_in"]
    w_out = payload["w_out"]
    return {
        "schema_version": payload.get("schema_version"),
        "model_semantics_version": payload.get("model_semantics_version"),
        "training_semantics_version": payload.get("training_semantics_version"),
        "hidden_silence_metric_version": payload.get(
            "hidden_silence_metric_version"
        ),
        "epoch": payload.get("epoch"),
        "w_in_shape": list(w_in.shape),
        "w_out_shape": list(w_out.shape),
        "dtype": str(w_in.dtype),
        "adam_step": adam.get("step"),
        "calibration_rounds": len(payload.get("calibration_history", ())),
        "restart_count": len(payload.get("bump_history", ())),
        "epoch_record_count": len(payload.get("epoch_metrics", ())),
        "run_config": payload.get("run_config"),
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Inspect or check a Yin-Yang run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    result_parser = subparsers.add_parser("result")
    result_parser.add_argument("path", type=Path)
    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("path", type=Path)
    gradient_parser = subparsers.add_parser("gradient")
    gradient_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    gradient_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    gradient_parser.add_argument("--sample-index", type=int, default=0)
    gradient_parser.add_argument(
        "--parameter", choices=("W_in", "W_out"), default="W_out"
    )
    gradient_parser.add_argument("--row", type=int, default=0)
    gradient_parser.add_argument("--column", type=int, default=0)
    gradient_parser.add_argument("--epsilon", type=float, default=1e-5)
    arguments = parser.parse_args(argv)
    if arguments.command == "result":
        payload = summarize_result(arguments.path)
    elif arguments.command == "checkpoint":
        payload = inspect_checkpoint(arguments.path)
    else:
        payload = asdict(
            check_weight_gradient(
                load_yinyang_config(arguments.config),
                split=arguments.split,
                sample_index=arguments.sample_index,
                parameter=arguments.parameter,
                index=(arguments.row, arguments.column),
                epsilon=arguments.epsilon,
            )
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
