"""In-memory orchestration for deterministic TeS Yin-Yang training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from Code.neurons import TeSNeurons
from Code.tes_psp import PSPPeakCalibration, calibrate_first_positive_lobe
from Code.tes_training import (
    AdamState,
    BatchMetrics,
    CheckpointMismatchError,
    EvaluationMetrics,
    TrainingConfig,
    YinYangSplit,
    build_yinyang_splits,
    bump_silent_hidden,
    evaluate_split,
    load_checkpoint,
    train_epoch,
)
from Code.yinyang_config import YinYangConfig


@dataclass(frozen=True)
class CalibrationRound:
    round: int
    coverage_counts: tuple[int, ...]
    coverage: tuple[float, ...]
    row_sums: tuple[float, ...]
    scaled_rows: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationHistory:
    rounds: tuple[CalibrationRound, ...]
    target: float
    scale: float
    succeeded: bool


class ReadoutCalibrationError(RuntimeError):
    def __init__(self, history: CalibrationHistory) -> None:
        super().__init__(
            f"readout coverage remained below {history.target:.3f} after "
            f"{len(history.rounds)} rounds"
        )
        self.history = history


@dataclass(frozen=True)
class EpochTrainingMetrics:
    samples: int
    mean_loss: float
    accuracy: float
    correct: int
    all_silent: int
    target_silent: int
    ambiguous_ties: int
    threshold_sensitivity_cap_activations: int
    minimum_causal_slope: float | None
    mean_raw_gradient_norm: float
    mean_clipped_gradient_norm: float
    clipped_updates: int
    negative_weight_entry_observations: int


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    learning_rate: float
    training: EpochTrainingMetrics
    validation_pre_restart: EvaluationMetrics
    validation: EvaluationMetrics
    restarted_hidden: tuple[int, ...]


@dataclass(frozen=True)
class TrainingSnapshot:
    epoch: int
    w_in: torch.Tensor
    w_out: torch.Tensor
    omega: torch.Tensor
    adam: dict[str, torch.Tensor | int]


@dataclass(frozen=True)
class IntervalSnapshot:
    state: TrainingSnapshot
    best_validation: TrainingSnapshot


@dataclass(frozen=True)
class TrainingArtifacts:
    config: YinYangConfig
    training_config: TrainingConfig
    model: TeSNeurons
    adam_state: AdamState
    splits: tuple[YinYangSplit, YinYangSplit, YinYangSplit]
    calibration: CalibrationHistory
    initialization: dict[str, object]
    baseline_train_metrics: EvaluationMetrics
    baseline_validation_metrics: EvaluationMetrics
    epochs: tuple[EpochRecord, ...]
    restart_history: tuple[dict[str, object], ...]
    pre_training_snapshot: TrainingSnapshot
    best_validation_snapshot: TrainingSnapshot
    interval_snapshots: tuple[IntervalSnapshot, ...]
    final_snapshot: TrainingSnapshot
    final_train_metrics: EvaluationMetrics
    final_validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics | None
    test_evaluations: int
    runtime_seconds: float


def _psp_calibrations(config: YinYangConfig) -> dict[str, PSPPeakCalibration]:
    model = config.to_tes_config()
    return {
        "fast": calibrate_first_positive_lobe(
            tau_v_ms=model.tau_v,
            alpha_per_ms=model.alpha,
            period_ms=model.fast_period_ms,
        ),
        "slow": calibrate_first_positive_lobe(
            tau_v_ms=model.tau_v,
            alpha_per_ms=model.alpha,
            period_ms=model.slow_period_ms,
        ),
    }


def initialize_yinyang_model(config: YinYangConfig) -> TeSNeurons:
    """Build a deterministic model from the complete experiment config."""
    config.validate()
    model_config = config.to_tes_config()
    initialization = config.initialization
    calibrations = _psp_calibrations(config)
    n_hidden = config.model.n_hidden
    fast_count = n_hidden // 2

    hidden_generator = torch.Generator().manual_seed(initialization.hidden_seed)
    w_in = initialization.hidden_mean + initialization.hidden_std * torch.randn(
        (n_hidden, config.model.n_inputs),
        generator=hidden_generator,
        dtype=torch.float64,
    )
    w_in.clamp_min_(initialization.hidden_floor)
    w_in[:fast_count].mul_(
        calibrations["fast"].initialization_gain
        * initialization.fast_hidden_gain
    )
    w_in[fast_count:].mul_(calibrations["slow"].initialization_gain)

    readout_generator = torch.Generator().manual_seed(initialization.readout_seed)
    base = initialization.readout_mean + initialization.readout_std * torch.randn(
        (config.model.n_outputs, n_hidden),
        generator=readout_generator,
        dtype=torch.float64,
    )
    base.clamp_min_(initialization.readout_floor)
    scales = torch.tensor(
        initialization.readout_row_scales,
        dtype=torch.float64,
    ).unsqueeze(1)
    scaled = base * scales
    target_sums = (
        scaled.mean(dim=1, keepdim=True)
        * initialization.readout_reference_hidden
        * calibrations["slow"].initialization_gain
        if initialization.readout_target_row_sums is None
        else torch.tensor(
            initialization.readout_target_row_sums,
            dtype=torch.float64,
        ).unsqueeze(1)
    )
    heterogeneity_generator = torch.Generator().manual_seed(
        initialization.heterogeneity_seed
    )
    factors = torch.exp(
        initialization.heterogeneity_sigma
        * torch.randn(
            scaled.shape,
            generator=heterogeneity_generator,
            dtype=torch.float64,
        )
    )
    w_out = scaled * factors
    w_out.mul_(target_sums / w_out.sum(dim=1, keepdim=True))
    return TeSNeurons(
        config.model.n_inputs,
        n_hidden,
        config.model.n_outputs,
        w_in,
        w_out,
        model_config,
    )


def _readout_coverage(
    model: TeSNeurons,
    split: YinYangSplit,
    config: YinYangConfig,
) -> tuple[int, ...]:
    coverage = [0] * model.n_out
    for sample in split.samples:
        forward = model.forward_continuous(
            sample.events,
            config.model.t_end_ms,
            early_stop=False,
        )
        for output, readout in enumerate(forward.readouts):
            coverage[output] += int(readout.physical_time is not None)
    return tuple(coverage)


def calibrate_readouts(
    model: TeSNeurons,
    train_split: YinYangSplit,
    config: YinYangConfig,
) -> CalibrationHistory:
    if train_split.name != "train":
        raise ValueError("readout calibration requires the training split")
    records = []
    sample_count = len(train_split.samples)
    for round_number in range(1, config.calibration.max_rounds + 1):
        counts = _readout_coverage(model, train_split, config)
        coverage = tuple(count / sample_count for count in counts)
        failing = tuple(
            output
            for output, fraction in enumerate(coverage)
            if fraction < config.calibration.target_fraction
        )
        records.append(
            CalibrationRound(
                round=round_number,
                coverage_counts=counts,
                coverage=coverage,
                row_sums=tuple(float(value) for value in model.W_out.sum(dim=1)),
                scaled_rows=failing,
            )
        )
        if not failing:
            return CalibrationHistory(
                rounds=tuple(records),
                target=config.calibration.target_fraction,
                scale=config.calibration.scale,
                succeeded=True,
            )
        for output in failing:
            model.W_out[output].mul_(config.calibration.scale)
    history = CalibrationHistory(
        rounds=tuple(records),
        target=config.calibration.target_fraction,
        scale=config.calibration.scale,
        succeeded=False,
    )
    raise ReadoutCalibrationError(history)


def ensure_finite_weights(
    model: TeSNeurons, *, epoch: int, include_omega: bool = False
) -> None:
    """Perform the sole production finite check at the epoch boundary."""
    if not torch.isfinite(model.W_in).all() or not torch.isfinite(model.W_out).all():
        raise FloatingPointError(f"non-finite model weight after epoch {epoch}")
    if include_omega and (
        not torch.isfinite(model.omega).all() or torch.any(model.omega <= 0.0)
    ):
        raise FloatingPointError(f"invalid model omega after epoch {epoch}")


def _summarize_batches(batches: Sequence[BatchMetrics]) -> EpochTrainingMetrics:
    samples = sum(batch.samples for batch in batches)
    slopes = tuple(
        batch.minimum_causal_slope
        for batch in batches
        if batch.minimum_causal_slope is not None
    )
    return EpochTrainingMetrics(
        samples=samples,
        mean_loss=sum(batch.mean_loss * batch.samples for batch in batches) / samples,
        accuracy=sum(batch.correct for batch in batches) / samples,
        correct=sum(batch.correct for batch in batches),
        all_silent=sum(batch.all_silent for batch in batches),
        target_silent=sum(batch.target_silent for batch in batches),
        ambiguous_ties=sum(batch.ambiguous_ties for batch in batches),
        threshold_sensitivity_cap_activations=sum(
            batch.threshold_sensitivity_cap_activations for batch in batches
        ),
        minimum_causal_slope=min(slopes, default=None),
        mean_raw_gradient_norm=sum(batch.update.raw_gradient_norm for batch in batches)
        / len(batches),
        mean_clipped_gradient_norm=sum(
            batch.update.clipped_gradient_norm for batch in batches
        )
        / len(batches),
        clipped_updates=sum(batch.update.clipping_factor < 1.0 for batch in batches),
        negative_weight_entry_observations=sum(
            batch.update.negative_w_in + batch.update.negative_w_out
            for batch in batches
        ),
    )


def _validation_improves(
    accuracy: float,
    loss: float,
    best_accuracy: float,
    best_loss: float,
) -> bool:
    """Prefer higher accuracy, then lower loss for deterministic ties."""
    return accuracy > best_accuracy or (
        accuracy == best_accuracy and loss < best_loss
    )


def _snapshot(epoch: int, model: TeSNeurons, state: AdamState) -> TrainingSnapshot:
    return TrainingSnapshot(
        epoch=epoch,
        w_in=model.W_in.detach().clone(),
        w_out=model.W_out.detach().clone(),
        omega=model.omega.detach().clone(),
        adam=state.to_payload(),
    )


def _snapshot_from_payload(
    payload: Mapping[str, object], *, default_omega: torch.Tensor
) -> TrainingSnapshot:
    adam = payload["adam"]
    if not isinstance(adam, dict):
        raise CheckpointMismatchError("checkpoint best-validation Adam state is malformed")
    normalized_adam = {
        name: value.clone() if isinstance(value, torch.Tensor) else int(value)
        for name, value in adam.items()
    }
    normalized_adam.setdefault("m_omega", torch.zeros_like(default_omega))
    normalized_adam.setdefault("v_omega", torch.zeros_like(default_omega))
    normalized_adam.setdefault("omega_step", 0)
    return TrainingSnapshot(
        epoch=int(payload["epoch"]),
        w_in=payload["w_in"].clone(),
        w_out=payload["w_out"].clone(),
        omega=payload.get("omega", default_omega).clone(),
        adam=normalized_adam,
    )


def _evaluation_from_payload(payload: Mapping[str, object]) -> EvaluationMetrics:
    values = dict(payload)
    values["confusion_matrix"] = tuple(
        tuple(row) for row in values["confusion_matrix"]
    )
    for name in (
        "per_class_accuracy",
        "output_silence_counts",
        "hidden_silence_counts",
        "hidden_full_horizon_silence_counts",
    ):
        values[name] = tuple(values[name])
    return EvaluationMetrics(**values)


def _epoch_from_payload(payload: Mapping[str, object]) -> EpochRecord:
    return EpochRecord(
        epoch=int(payload["epoch"]),
        learning_rate=float(payload["learning_rate"]),
        training=EpochTrainingMetrics(**payload["training"]),
        validation_pre_restart=_evaluation_from_payload(
            payload["validation_pre_restart"]
        ),
        validation=_evaluation_from_payload(payload["validation"]),
        restarted_hidden=tuple(payload["restarted_hidden"]),
    )


def _calibrated_model(
    config: YinYangConfig,
    calibration: CalibrationHistory,
) -> TeSNeurons:
    model = initialize_yinyang_model(config)
    target = torch.tensor(
        calibration.rounds[-1].row_sums,
        dtype=torch.float64,
    ).unsqueeze(1)
    model.W_out.mul_(target / model.W_out.sum(dim=1, keepdim=True))
    return model


def _initialization_metadata(
    initial_w_in: torch.Tensor,
    initial_w_out: torch.Tensor,
    calibrated_model: TeSNeurons,
) -> dict[str, object]:
    return {
        "w_in_row_sums": tuple(float(value) for value in initial_w_in.sum(dim=1)),
        "w_out_row_sums_before_calibration": tuple(
            float(value) for value in initial_w_out.sum(dim=1)
        ),
        "w_out_row_sums_after_calibration": tuple(
            float(value) for value in calibrated_model.W_out.sum(dim=1)
        ),
    }


def run_yinyang_training(
    config: YinYangConfig,
    run_dir: Path,
    *,
    resume_checkpoint: Path | None = None,
    evaluate_test: bool = False,
    learn_omega: bool = False,
    omega_learning_rate_scale: float | None = None,
    in_epoch_bump_interval_batches: int = 0,
    in_epoch_bump_silence_fraction: float = 0.5,
    in_epoch_bump_amount: float = 0.05,
) -> TrainingArtifacts:
    """Run training without writing artifacts; reporting owns filesystem output."""
    del run_dir
    started = time.perf_counter()
    config.validate()
    training = config.to_training_config(
        learn_omega=learn_omega,
        omega_learning_rate_scale=omega_learning_rate_scale,
        in_epoch_bump_interval_batches=in_epoch_bump_interval_batches,
        in_epoch_bump_silence_fraction=in_epoch_bump_silence_fraction,
        in_epoch_bump_amount=in_epoch_bump_amount,
    )
    splits = build_yinyang_splits(
        train_size=config.data.train_size,
        validation_size=config.data.validation_size,
        test_size=config.data.test_size,
        seeds=config.data.split_seeds,
        dt=config.model.dt_ms,
    )
    train_split, validation_split, test_split = splits
    model = initialize_yinyang_model(config)
    initial_w_in = model.W_in.clone()
    initial_w_out = model.W_out.clone()
    epoch_records: list[EpochRecord] = []
    restart_history: list[dict[str, object]] = []
    interval_snapshots: list[IntervalSnapshot] = []
    first_epoch = 1
    restored_best_snapshot: TrainingSnapshot | None = None

    if resume_checkpoint is None:
        calibration = calibrate_readouts(model, train_split, config)
        adam_state = AdamState.zeros_like(model)
    else:
        restored = load_checkpoint(
            resume_checkpoint,
            model,
            expected_config=training,
            expected_run_config=config.to_dict(),
            allow_epoch_extension=True,
            allow_omega_learning_upgrade=training.learn_omega,
        )
        calibration = CalibrationHistory(
            rounds=tuple(CalibrationRound(**record) for record in restored.calibration_history),
            target=config.calibration.target_fraction,
            scale=config.calibration.scale,
            succeeded=True,
        )
        adam_state = restored.adam
        restart_history.extend(dict(record) for record in restored.bump_history)
        epoch_records.extend(
            _epoch_from_payload(record) for record in restored.epoch_metrics
        )
        if restored.best_validation_snapshot is not None:
            restored_best_snapshot = _snapshot_from_payload(
                restored.best_validation_snapshot,
                default_omega=model.omega,
            )
            if restored_best_snapshot.epoch > restored.epoch:
                raise CheckpointMismatchError(
                    "checkpoint best-validation snapshot is from a future epoch"
                )
        first_epoch = restored.epoch + 1
        if first_epoch > config.optimization.epochs:
            raise CheckpointMismatchError("checkpoint has no remaining epochs")

    calibrated_model = _calibrated_model(config, calibration)
    pre_training_snapshot = _snapshot(0, calibrated_model, AdamState.zeros_like(calibrated_model))
    baseline_train = evaluate_split(
        calibrated_model,
        train_split,
        training.t_end,
        training.margin,
        loss_name=training.loss_name,
        learn_omega=training.learn_omega,
    )
    baseline_validation = evaluate_split(
        calibrated_model,
        validation_split,
        training.t_end,
        training.margin,
        loss_name=training.loss_name,
        learn_omega=training.learn_omega,
    )
    best_snapshot = (
        restored_best_snapshot
        if restored_best_snapshot is not None
        else _snapshot(first_epoch - 1, model, adam_state)
    )
    prior_best = max(
        epoch_records,
        key=lambda record: (
            record.validation.accuracy,
            -record.validation.mean_loss,
        ),
        default=None,
    )
    best_accuracy = (
        float("-inf")
        if prior_best is None
        else prior_best.validation.accuracy
    )
    best_loss = (
        float("inf") if prior_best is None else prior_best.validation.mean_loss
    )

    for epoch in range(first_epoch, config.optimization.epochs + 1):
        batches = train_epoch(model, train_split, adam_state, training, epoch=epoch)
        for batch_metrics in batches:
            if batch_metrics.in_epoch_bumped_hidden:
                restart_history.append(
                    {
                        "kind": "in_epoch_bump",
                        "epoch": epoch,
                        "batch": batch_metrics.batch,
                        "hidden": batch_metrics.in_epoch_bumped_hidden,
                        "interval_batches": (
                            training.in_epoch_bump_interval_batches
                        ),
                        "silence_fraction": (
                            training.in_epoch_bump_silence_fraction
                        ),
                        "amount": training.in_epoch_bump_amount,
                    }
                )
        ensure_finite_weights(
            model, epoch=epoch, include_omega=training.learn_omega
        )
        training_metrics = _summarize_batches(batches)
        validation_pre = evaluate_split(
            model,
            validation_split,
            training.t_end,
            training.margin,
            loss_name=training.loss_name,
            phase="validation",
            epoch=epoch,
            learn_omega=training.learn_omega,
        )
        before_restart = model.W_in.clone()
        restart_seed = config.restart.seed + epoch
        restarted = bump_silent_hidden(
            model,
            validation_pre.hidden_silence_counts,
            adam_state,
            validation_pre.samples,
            config.restart.silence_fraction,
            config.restart.offset,
            seed=restart_seed,
            noise_std=config.restart.noise_std,
        )
        validation = (
            evaluate_split(
                model,
                validation_split,
                training.t_end,
                training.margin,
                loss_name=training.loss_name,
                phase="post_restart_validation",
                epoch=epoch,
                learn_omega=training.learn_omega,
            )
            if restarted
            else validation_pre
        )
        if restarted:
            restart_history.append(
                {
                    "kind": "validation_restart",
                    "epoch": epoch,
                    "hidden": restarted,
                    "pre_silence_counts": validation_pre.hidden_silence_counts,
                    "post_silence_counts": validation.hidden_silence_counts,
                    "offset": config.restart.offset,
                    "noise_std": config.restart.noise_std,
                    "seed": restart_seed,
                    "deltas": {
                        hidden: tuple(
                            float(value)
                            for value in model.W_in[hidden] - before_restart[hidden]
                        )
                        for hidden in restarted
                    },
                }
            )
        record = EpochRecord(
            epoch=epoch,
            learning_rate=training.adam_for_epoch(epoch).learning_rate,
            training=training_metrics,
            validation_pre_restart=validation_pre,
            validation=validation,
            restarted_hidden=restarted,
        )
        epoch_records.append(record)
        if _validation_improves(
            validation.accuracy,
            validation.mean_loss,
            best_accuracy,
            best_loss,
        ):
            best_accuracy = validation.accuracy
            best_loss = validation.mean_loss
            best_snapshot = _snapshot(epoch, model, adam_state)
        interval = config.artifacts.checkpoint_interval
        if interval and epoch % interval == 0:
            interval_snapshots.append(
                IntervalSnapshot(
                    state=_snapshot(epoch, model, adam_state),
                    best_validation=best_snapshot,
                )
            )

    final_train = evaluate_split(
        model,
        train_split,
        training.t_end,
        training.margin,
        loss_name=training.loss_name,
        phase="final_train",
        epoch=config.optimization.epochs,
        learn_omega=training.learn_omega,
    )
    test_metrics = (
        evaluate_split(
            model,
            test_split,
            training.t_end,
            training.margin,
            loss_name=training.loss_name,
            phase="test",
            epoch=config.optimization.epochs,
            learn_omega=training.learn_omega,
        )
        if evaluate_test
        else None
    )
    final_snapshot = _snapshot(config.optimization.epochs, model, adam_state)
    return TrainingArtifacts(
        config=config,
        training_config=training,
        model=model,
        adam_state=adam_state,
        splits=splits,
        calibration=calibration,
        initialization=_initialization_metadata(
            initial_w_in,
            initial_w_out,
            calibrated_model,
        ),
        baseline_train_metrics=baseline_train,
        baseline_validation_metrics=baseline_validation,
        epochs=tuple(epoch_records),
        restart_history=tuple(restart_history),
        pre_training_snapshot=pre_training_snapshot,
        best_validation_snapshot=best_snapshot,
        interval_snapshots=tuple(interval_snapshots),
        final_snapshot=final_snapshot,
        final_train_metrics=final_train,
        final_validation_metrics=epoch_records[-1].validation,
        test_metrics=test_metrics,
        test_evaluations=int(evaluate_test),
        runtime_seconds=time.perf_counter() - started,
    )
