"""Typed configuration and command-line overrides for Yin-Yang training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "yinyang_best.json"


@dataclass(frozen=True)
class DataConfig:
    train_size: int
    validation_size: int
    test_size: int
    split_seeds: tuple[int, int, int]


@dataclass(frozen=True)
class ModelConfig:
    n_inputs: int
    n_hidden: int
    n_outputs: int
    dt_ms: float
    t_end_ms: float
    tau_v_ms: float
    alpha_per_ms: float
    threshold: float
    reset_drop: float
    omega_fast_per_ms: float
    omega_slow_per_ms: float
    d_max_ms: float
    c0: float
    c1: float
    slope_epsilon: float
    v_dot_clip: float
    silent_target_gap_ms: float


@dataclass(frozen=True)
class InitializationConfig:
    hidden_mean: float
    hidden_std: float
    hidden_floor: float
    hidden_seed: int
    fast_hidden_gain: float
    readout_mean: float
    readout_std: float
    readout_floor: float
    readout_row_scales: tuple[float, ...]
    readout_target_row_sums: tuple[float, ...] | None
    readout_reference_hidden: int
    readout_seed: int
    heterogeneity_sigma: float
    heterogeneity_seed: int


@dataclass(frozen=True)
class CalibrationConfig:
    target_fraction: float
    scale: float
    max_rounds: int


@dataclass(frozen=True)
class LossConfig:
    name: str
    margin_ms: float


@dataclass(frozen=True)
class AdamSettings:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    gradient_norm_cap: float


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int
    batch_size: int
    shuffle_seed: int
    cosine_t_max: int
    cosine_eta_min: float
    adam: AdamSettings


@dataclass(frozen=True)
class RestartConfig:
    silence_fraction: float
    noise_std: float
    offset: float
    seed: int


@dataclass(frozen=True)
class ArtifactConfig:
    checkpoint_interval: int
    plot_interval: int
    save_rasters: bool


def _section(cls, values: object, name: str):
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a JSON object")
    expected = {field.name for field in fields(cls)}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise ValueError(f"unknown {name} field: {sorted(unknown)[0]}")
    if missing:
        raise ValueError(f"missing {name} field: {sorted(missing)[0]}")
    return cls(**values)


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")


@dataclass(frozen=True)
class YinYangConfig:
    data: DataConfig
    model: ModelConfig
    initialization: InitializationConfig
    calibration: CalibrationConfig
    loss: LossConfig
    optimization: OptimizationConfig
    restart: RestartConfig
    artifacts: ArtifactConfig

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> YinYangConfig:
        expected = {field.name for field in fields(cls)}
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown:
            raise ValueError(f"unknown configuration section: {sorted(unknown)[0]}")
        if missing:
            raise ValueError(f"missing configuration section: {sorted(missing)[0]}")
        for name in expected:
            if not isinstance(values[name], dict):
                raise ValueError(f"{name} must be a JSON object")

        data_values = dict(values["data"])
        if not isinstance(data_values.get("split_seeds"), (tuple, list)):
            raise ValueError("data.split_seeds must be an array")
        data_values["split_seeds"] = tuple(data_values["split_seeds"])
        init_values = dict(values["initialization"])
        init_values.setdefault("readout_reference_hidden", 12)
        if not isinstance(init_values.get("readout_row_scales"), (tuple, list)):
            raise ValueError("initialization.readout_row_scales must be an array")
        init_values["readout_row_scales"] = tuple(init_values["readout_row_scales"])
        if init_values["readout_target_row_sums"] is not None:
            if not isinstance(
                init_values["readout_target_row_sums"],
                (tuple, list),
            ):
                raise ValueError(
                    "initialization.readout_target_row_sums must be an array or null"
                )
            init_values["readout_target_row_sums"] = tuple(
                init_values["readout_target_row_sums"]
            )
        optimization_values = dict(values["optimization"])
        optimization_values["adam"] = _section(
            AdamSettings,
            optimization_values["adam"],
            "optimization.adam",
        )
        config = cls(
            data=_section(DataConfig, data_values, "data"),
            model=_section(ModelConfig, values["model"], "model"),
            initialization=_section(
                InitializationConfig,
                init_values,
                "initialization",
            ),
            calibration=_section(
                CalibrationConfig,
                values["calibration"],
                "calibration",
            ),
            loss=_section(LossConfig, values["loss"], "loss"),
            optimization=_section(
                OptimizationConfig,
                optimization_values,
                "optimization",
            ),
            restart=_section(RestartConfig, values["restart"], "restart"),
            artifacts=_section(ArtifactConfig, values["artifacts"], "artifacts"),
        )
        config.validate()
        return config

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        if self.initialization.readout_reference_hidden == 12:
            values["initialization"].pop("readout_reference_hidden")
        return values

    def validate(self) -> None:
        integer_values = {
            "train_size": self.data.train_size,
            "validation_size": self.data.validation_size,
            "test_size": self.data.test_size,
            "n_inputs": self.model.n_inputs,
            "n_hidden": self.model.n_hidden,
            "n_outputs": self.model.n_outputs,
            "epochs": self.optimization.epochs,
            "batch_size": self.optimization.batch_size,
            "cosine_t_max": self.optimization.cosine_t_max,
            "calibration.max_rounds": self.calibration.max_rounds,
            "readout_reference_hidden": (
                self.initialization.readout_reference_hidden
            ),
        }
        for name, value in integer_values.items():
            _require_integer(name, value)
        for name, value in (
            ("shuffle_seed", self.optimization.shuffle_seed),
            ("hidden_seed", self.initialization.hidden_seed),
            ("readout_seed", self.initialization.readout_seed),
            ("heterogeneity_seed", self.initialization.heterogeneity_seed),
            ("restart.seed", self.restart.seed),
            ("checkpoint_interval", self.artifacts.checkpoint_interval),
            ("plot_interval", self.artifacts.plot_interval),
        ):
            _require_integer(name, value)
        if not isinstance(self.artifacts.save_rasters, bool):
            raise ValueError("artifacts.save_rasters must be a boolean")
        numeric_values = {
            "dt_ms": self.model.dt_ms,
            "t_end_ms": self.model.t_end_ms,
            "tau_v_ms": self.model.tau_v_ms,
            "alpha_per_ms": self.model.alpha_per_ms,
            "threshold": self.model.threshold,
            "reset_drop": self.model.reset_drop,
            "omega_fast_per_ms": self.model.omega_fast_per_ms,
            "omega_slow_per_ms": self.model.omega_slow_per_ms,
            "d_max_ms": self.model.d_max_ms,
            "c0": self.model.c0,
            "c1": self.model.c1,
            "slope_epsilon": self.model.slope_epsilon,
            "v_dot_clip": self.model.v_dot_clip,
            "silent_target_gap_ms": self.model.silent_target_gap_ms,
            "hidden_mean": self.initialization.hidden_mean,
            "hidden_std": self.initialization.hidden_std,
            "hidden_floor": self.initialization.hidden_floor,
            "fast_hidden_gain": self.initialization.fast_hidden_gain,
            "readout_mean": self.initialization.readout_mean,
            "readout_std": self.initialization.readout_std,
            "readout_floor": self.initialization.readout_floor,
            "heterogeneity_sigma": self.initialization.heterogeneity_sigma,
            "calibration.target_fraction": self.calibration.target_fraction,
            "calibration.scale": self.calibration.scale,
            "loss.margin_ms": self.loss.margin_ms,
            "cosine_eta_min": self.optimization.cosine_eta_min,
            "adam.learning_rate": self.optimization.adam.learning_rate,
            "adam.beta1": self.optimization.adam.beta1,
            "adam.beta2": self.optimization.adam.beta2,
            "adam.epsilon": self.optimization.adam.epsilon,
            "adam.gradient_norm_cap": self.optimization.adam.gradient_norm_cap,
            "restart.silence_fraction": self.restart.silence_fraction,
            "restart.noise_std": self.restart.noise_std,
            "restart.offset": self.restart.offset,
        }
        for name, value in numeric_values.items():
            _require_number(name, value)
        for index, value in enumerate(self.data.split_seeds):
            _require_integer(f"split_seeds[{index}]", value)
        for name, sequence in (
            ("readout_row_scales", self.initialization.readout_row_scales),
            ("readout_target_row_sums", self.initialization.readout_target_row_sums),
        ):
            if sequence is not None:
                for index, value in enumerate(sequence):
                    _require_number(f"{name}[{index}]", value)
        if any(value <= 0 for value in integer_values.values()):
            raise ValueError("sizes, epochs, batch size, and cosine T_max must be positive")
        if self.model.n_inputs != 4 or self.model.n_outputs != 3:
            raise ValueError("Yin-Yang encoding requires four inputs and three outputs")
        if self.data.train_size % self.optimization.batch_size:
            raise ValueError("training size must divide into complete batches")
        if len(self.data.split_seeds) != 3 or len(set(self.data.split_seeds)) != 3:
            raise ValueError("split seeds must contain three distinct values")
        if len(self.initialization.readout_row_scales) != self.model.n_outputs:
            raise ValueError("one readout row scale is required per output")
        targets = self.initialization.readout_target_row_sums
        if targets is not None and len(targets) != self.model.n_outputs:
            raise ValueError("one target row sum is required per output")
        row_values = self.initialization.readout_row_scales + (() if targets is None else targets)
        if any(not math.isfinite(value) or value <= 0.0 for value in row_values):
            raise ValueError("readout row scales and sums must be positive and finite")
        if not isinstance(self.loss.name, str) or self.loss.name not in {
            "delgrad",
            "squared_hinge",
        }:
            raise ValueError("unknown timing loss")
        for name, value in (
            ("dt_ms", self.model.dt_ms),
            ("t_end_ms", self.model.t_end_ms),
            ("tau_v_ms", self.model.tau_v_ms),
            ("alpha_per_ms", self.model.alpha_per_ms),
            ("threshold", self.model.threshold),
            ("reset_drop", self.model.reset_drop),
            ("omega_fast_per_ms", self.model.omega_fast_per_ms),
            ("omega_slow_per_ms", self.model.omega_slow_per_ms),
            ("d_max_ms", self.model.d_max_ms),
            ("c0", self.model.c0),
            ("v_dot_clip", self.model.v_dot_clip),
            ("learning_rate", self.optimization.adam.learning_rate),
            ("gradient_norm_cap", self.optimization.adam.gradient_norm_cap),
            ("calibration.scale", self.calibration.scale),
            ("adam.epsilon", self.optimization.adam.epsilon),
            ("fast_hidden_gain", self.initialization.fast_hidden_gain),
        ):
            _finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("c1", self.model.c1),
            ("slope_epsilon", self.model.slope_epsilon),
            ("silent_target_gap_ms", self.model.silent_target_gap_ms),
            ("loss.margin_ms", self.loss.margin_ms),
            ("cosine_eta_min", self.optimization.cosine_eta_min),
            ("restart.noise_std", self.restart.noise_std),
            ("restart.offset", self.restart.offset),
            ("hidden_mean", self.initialization.hidden_mean),
            ("hidden_floor", self.initialization.hidden_floor),
            ("readout_mean", self.initialization.readout_mean),
            ("readout_floor", self.initialization.readout_floor),
        ):
            _finite(name, value)
        for name, value in (
            ("hidden_std", self.initialization.hidden_std),
            ("readout_std", self.initialization.readout_std),
            ("heterogeneity_sigma", self.initialization.heterogeneity_sigma),
        ):
            _finite(name, value)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("adam.beta1", self.optimization.adam.beta1),
            ("adam.beta2", self.optimization.adam.beta2),
        ):
            _finite(name, value)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.loss.margin_ms < 0.0:
            raise ValueError("loss margin must be non-negative")
        if self.model.slope_epsilon < 0.0 or self.model.silent_target_gap_ms < 0.0:
            raise ValueError("slope epsilon and silent-target gap must be non-negative")
        seed_values = (
            *self.data.split_seeds,
            self.initialization.hidden_seed,
            self.initialization.readout_seed,
            self.initialization.heterogeneity_seed,
            self.optimization.shuffle_seed,
            self.restart.seed,
        )
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values):
            raise ValueError("seeds must be non-negative integers")
        if not 0.0 <= self.calibration.target_fraction <= 1.0:
            raise ValueError("calibration target must be between zero and one")
        if not 0.0 <= self.restart.silence_fraction <= 1.0:
            raise ValueError("restart silence fraction must be between zero and one")
        if self.restart.noise_std < 0.0:
            raise ValueError("restart noise standard deviation must be non-negative")
        if self.restart.offset <= 0.0:
            raise ValueError("restart offset must be positive")
        if self.optimization.cosine_eta_min < 0.0:
            raise ValueError("cosine eta minimum must be non-negative")
        if self.optimization.cosine_eta_min > self.optimization.adam.learning_rate:
            raise ValueError("cosine eta minimum cannot exceed the initial rate")
        if self.artifacts.checkpoint_interval < 0 or self.artifacts.plot_interval < 0:
            raise ValueError("artifact intervals must be non-negative")

    def to_tes_config(self):
        from Code.neurons import TeSConfig

        return TeSConfig(
            dt=self.model.dt_ms,
            tau_v=self.model.tau_v_ms,
            alpha=self.model.alpha_per_ms,
            threshold=self.model.threshold,
            reset_drop=self.model.reset_drop,
            omega_fast=self.model.omega_fast_per_ms,
            omega_slow=self.model.omega_slow_per_ms,
            d_plus=self.model.d_max_ms,
            c0=self.model.c0,
            c1=self.model.c1,
            slope_epsilon=self.model.slope_epsilon,
            threshold_sensitivity_cap=self.model.v_dot_clip,
            silent_target_gap_ms=self.model.silent_target_gap_ms,
        )

    def to_training_config(
        self,
        *,
        learn_omega: bool = False,
        omega_learning_rate_scale: float | None = None,
        in_epoch_bump_interval_batches: int = 0,
        in_epoch_bump_silence_fraction: float = 0.5,
        in_epoch_bump_amount: float = 0.05,
    ):
        from Code.tes_training import (
            AdamConfig,
            OMEGA_LEARNING_RATE_SCALE,
            TrainingConfig,
        )

        return TrainingConfig(
            epochs=self.optimization.epochs,
            batch_size=self.optimization.batch_size,
            t_end=self.model.t_end_ms,
            loss_name=self.loss.name,
            margin=self.loss.margin_ms,
            shuffle_seed=self.optimization.shuffle_seed,
            adam=AdamConfig(
                learning_rate=self.optimization.adam.learning_rate,
                beta1=self.optimization.adam.beta1,
                beta2=self.optimization.adam.beta2,
                epsilon=self.optimization.adam.epsilon,
                gradient_norm_cap=self.optimization.adam.gradient_norm_cap,
            ),
            cosine_annealing_t_max=self.optimization.cosine_t_max,
            cosine_annealing_eta_min=self.optimization.cosine_eta_min,
            learn_omega=learn_omega,
            omega_learning_rate_scale=(
                OMEGA_LEARNING_RATE_SCALE
                if omega_learning_rate_scale is None
                else omega_learning_rate_scale
            ),
            in_epoch_bump_interval_batches=(
                in_epoch_bump_interval_batches
            ),
            in_epoch_bump_silence_fraction=(
                in_epoch_bump_silence_fraction
            ),
            in_epoch_bump_amount=in_epoch_bump_amount,
        )


@dataclass(frozen=True)
class RunRequest:
    config: YinYangConfig
    run_dir: Path
    resume_checkpoint: Path | None
    evaluate_test: bool
    learn_omega: bool
    omega_learning_rate_scale: float
    in_epoch_bump_interval_batches: int
    in_epoch_bump_silence_fraction: float
    in_epoch_bump_amount: float


CLI_FIELD_PATHS = {
    ("data", "train_size"): "--train-size",
    ("data", "validation_size"): "--validation-size",
    ("data", "test_size"): "--test-size",
    ("data", "split_seeds"): "--split-seeds",
    ("model", "n_inputs"): "--n-inputs",
    ("model", "n_hidden"): "--n-hidden",
    ("model", "n_outputs"): "--n-outputs",
    ("model", "dt_ms"): "--dt",
    ("model", "t_end_ms"): "--t-end",
    ("model", "tau_v_ms"): "--tau-v",
    ("model", "alpha_per_ms"): "--alpha",
    ("model", "threshold"): "--threshold",
    ("model", "reset_drop"): "--reset-drop",
    ("model", "omega_fast_per_ms"): "--omega-fast",
    ("model", "omega_slow_per_ms"): "--omega-slow",
    ("model", "d_max_ms"): "--d-max",
    ("model", "c0"): "--c0",
    ("model", "c1"): "--c1",
    ("model", "slope_epsilon"): "--slope-epsilon",
    ("model", "v_dot_clip"): "--v-dot-clip",
    ("model", "silent_target_gap_ms"): "--silent-target-gap",
    ("initialization", "hidden_mean"): "--hidden-init-mean",
    ("initialization", "hidden_std"): "--hidden-init-std",
    ("initialization", "hidden_floor"): "--hidden-init-floor",
    ("initialization", "hidden_seed"): "--hidden-seed",
    ("initialization", "fast_hidden_gain"): "--fast-hidden-gain",
    ("initialization", "readout_mean"): "--readout-init-mean",
    ("initialization", "readout_std"): "--readout-init-std",
    ("initialization", "readout_floor"): "--readout-init-floor",
    ("initialization", "readout_row_scales"): "--readout-row-scales",
    ("initialization", "readout_target_row_sums"): "--readout-target-row-sums",
    ("initialization", "readout_reference_hidden"): "--readout-reference-hidden",
    ("initialization", "readout_seed"): "--readout-seed",
    ("initialization", "heterogeneity_sigma"): "--heterogeneity-sigma",
    ("initialization", "heterogeneity_seed"): "--heterogeneity-seed",
    ("calibration", "target_fraction"): "--calibration-target",
    ("calibration", "scale"): "--calibration-scale",
    ("calibration", "max_rounds"): "--calibration-max-rounds",
    ("loss", "name"): "--loss",
    ("loss", "margin_ms"): "--margin",
    ("optimization", "epochs"): "--epochs",
    ("optimization", "batch_size"): "--batch-size",
    ("optimization", "shuffle_seed"): "--shuffle-seed",
    ("optimization", "cosine_t_max"): "--cosine-t-max",
    ("optimization", "cosine_eta_min"): "--cosine-eta-min",
    ("optimization", "adam", "learning_rate"): "--learning-rate",
    ("optimization", "adam", "beta1"): "--adam-beta1",
    ("optimization", "adam", "beta2"): "--adam-beta2",
    ("optimization", "adam", "epsilon"): "--adam-epsilon",
    ("optimization", "adam", "gradient_norm_cap"): "--gradient-cap",
    ("restart", "silence_fraction"): "--hidden-silence-fraction",
    ("restart", "noise_std"): "--hidden-restart-noise-std",
    ("restart", "offset"): "--hidden-bump-amount",
    ("restart", "seed"): "--hidden-restart-seed",
    ("artifacts", "checkpoint_interval"): "--checkpoint-interval",
    ("artifacts", "plot_interval"): "--plot-interval",
    ("artifacts", "save_rasters"): "--save-rasters",
}

_INTEGER_PATHS = {
    ("data", "train_size"),
    ("data", "validation_size"),
    ("data", "test_size"),
    ("model", "n_inputs"),
    ("model", "n_hidden"),
    ("model", "n_outputs"),
    ("initialization", "hidden_seed"),
    ("initialization", "readout_reference_hidden"),
    ("initialization", "readout_seed"),
    ("initialization", "heterogeneity_seed"),
    ("calibration", "max_rounds"),
    ("optimization", "epochs"),
    ("optimization", "batch_size"),
    ("optimization", "shuffle_seed"),
    ("optimization", "cosine_t_max"),
    ("restart", "seed"),
    ("artifacts", "checkpoint_interval"),
    ("artifacts", "plot_interval"),
}
_TUPLE_PATHS = {
    ("data", "split_seeds"): (int, 3),
    ("initialization", "readout_row_scales"): (float, 3),
    ("initialization", "readout_target_row_sums"): (float, 3),
}
_BOOLEAN_PATHS = {
    ("artifacts", "save_rasters"),
}


def load_yinyang_config(path: Path | str) -> YinYangConfig:
    config, *_ = _load_yinyang_config_and_training_options(path)
    return config


def _load_yinyang_config_and_training_options(
    path: Path | str,
) -> tuple[YinYangConfig, bool, float, int, float, float]:
    from Code.tes_training import OMEGA_LEARNING_RATE_SCALE

    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a JSON object")
    training_options = values.pop("training_options", {})
    if not isinstance(training_options, dict):
        raise ValueError("training_options must be a JSON object")
    unknown_options = set(training_options) - {
        "learn_omega",
        "omega_learning_rate_scale",
        "in_epoch_bump_interval_batches",
        "in_epoch_bump_silence_fraction",
        "in_epoch_bump_amount",
    }
    if unknown_options:
        raise ValueError(
            f"unknown training option: {sorted(unknown_options)[0]}"
        )
    learn_omega = training_options.get("learn_omega", False)
    if not isinstance(learn_omega, bool):
        raise ValueError("training_options.learn_omega must be boolean")
    omega_learning_rate_scale = training_options.get(
        "omega_learning_rate_scale",
        OMEGA_LEARNING_RATE_SCALE,
    )
    if (
        isinstance(omega_learning_rate_scale, bool)
        or not isinstance(omega_learning_rate_scale, (int, float))
        or not math.isfinite(omega_learning_rate_scale)
        or omega_learning_rate_scale <= 0.0
    ):
        raise ValueError(
            "training_options.omega_learning_rate_scale must be positive and finite"
        )
    in_epoch_bump_interval_batches = training_options.get(
        "in_epoch_bump_interval_batches", 0
    )
    in_epoch_bump_silence_fraction = training_options.get(
        "in_epoch_bump_silence_fraction", 0.5
    )
    in_epoch_bump_amount = training_options.get(
        "in_epoch_bump_amount", 0.05
    )
    if (
        isinstance(in_epoch_bump_interval_batches, bool)
        or not isinstance(in_epoch_bump_interval_batches, int)
        or in_epoch_bump_interval_batches < 0
        or isinstance(in_epoch_bump_silence_fraction, bool)
        or not isinstance(in_epoch_bump_silence_fraction, (int, float))
        or not math.isfinite(in_epoch_bump_silence_fraction)
        or not 0.0 <= in_epoch_bump_silence_fraction <= 1.0
        or isinstance(in_epoch_bump_amount, bool)
        or not isinstance(in_epoch_bump_amount, (int, float))
        or not math.isfinite(in_epoch_bump_amount)
        or in_epoch_bump_amount <= 0.0
    ):
        raise ValueError("invalid in-epoch hidden bump training options")
    return (
        YinYangConfig.from_dict(values),
        learn_omega,
        float(omega_learning_rate_scale),
        in_epoch_bump_interval_batches,
        float(in_epoch_bump_silence_fraction),
        float(in_epoch_bump_amount),
    )


def apply_cli_overrides(
    config: YinYangConfig,
    overrides: Mapping[tuple[str, ...], object],
) -> YinYangConfig:
    values = config.to_dict()
    for path, value in overrides.items():
        target = values
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return YinYangConfig.from_dict(values)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the canonical TeS Yin-Yang model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-dir", type=Path, default=Path("results/yinyang_run"))
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    omega_group = parser.add_mutually_exclusive_group()
    omega_group.add_argument(
        "--learn_omega",
        "--learn-omega",
        dest="learn_omega",
        action="store_true",
        default=None,
        help="learn one independently adapted oscillator frequency per neuron",
    )
    omega_group.add_argument(
        "--no-learn_omega",
        "--no-learn-omega",
        dest="learn_omega",
        action="store_false",
        help="keep the configured oscillator frequencies fixed",
    )
    parser.add_argument(
        "--omega-learning-rate-scale",
        type=float,
        default=None,
        help="omega learning rate as a multiplier of the weight learning rate",
    )
    parser.add_argument(
        "--in-epoch-bump-interval",
        type=int,
        default=None,
        help="training batches per deterministic hidden-silence window; zero disables",
    )
    parser.add_argument(
        "--in-epoch-bump-silence-fraction",
        type=float,
        default=None,
        help="strict training-silence threshold for deterministic hidden bumps",
    )
    parser.add_argument(
        "--in-epoch-bump-amount",
        type=float,
        default=None,
        help="positive offset added to a silent hidden neuron's input weights",
    )
    for path, flag in CLI_FIELD_PATHS.items():
        destination = "__".join(path)
        if path in _BOOLEAN_PATHS:
            parser.add_argument(
                flag,
                dest=destination,
                action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
            )
        elif path in _TUPLE_PATHS:
            value_type, count = _TUPLE_PATHS[path]
            parser.add_argument(
                flag,
                dest=destination,
                type=value_type,
                nargs=count,
                default=argparse.SUPPRESS,
            )
        elif path == ("loss", "name"):
            parser.add_argument(
                flag,
                dest=destination,
                choices=("delgrad", "squared_hinge"),
                default=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                flag,
                dest=destination,
                type=int if path in _INTEGER_PATHS else float,
                default=argparse.SUPPRESS,
            )
    return parser


def parse_yinyang_args(argv: Sequence[str] | None = None) -> RunRequest:
    namespace = _argument_parser().parse_args(argv)
    values = vars(namespace)
    (
        config,
        saved_learn_omega,
        saved_omega_learning_rate_scale,
        saved_in_epoch_bump_interval_batches,
        saved_in_epoch_bump_silence_fraction,
        saved_in_epoch_bump_amount,
    ) = _load_yinyang_config_and_training_options(values.pop("config"))
    run_dir = values.pop("run_dir")
    resume_checkpoint = values.pop("resume_checkpoint")
    evaluate_test = values.pop("evaluate_test")
    requested_learn_omega = values.pop("learn_omega")
    requested_omega_learning_rate_scale = values.pop(
        "omega_learning_rate_scale"
    )
    requested_in_epoch_bump_interval_batches = values.pop(
        "in_epoch_bump_interval"
    )
    requested_in_epoch_bump_silence_fraction = values.pop(
        "in_epoch_bump_silence_fraction"
    )
    requested_in_epoch_bump_amount = values.pop("in_epoch_bump_amount")
    learn_omega = (
        saved_learn_omega
        if requested_learn_omega is None
        else requested_learn_omega
    )
    omega_learning_rate_scale = (
        saved_omega_learning_rate_scale
        if requested_omega_learning_rate_scale is None
        else requested_omega_learning_rate_scale
    )
    in_epoch_bump_interval_batches = (
        saved_in_epoch_bump_interval_batches
        if requested_in_epoch_bump_interval_batches is None
        else requested_in_epoch_bump_interval_batches
    )
    in_epoch_bump_silence_fraction = (
        saved_in_epoch_bump_silence_fraction
        if requested_in_epoch_bump_silence_fraction is None
        else requested_in_epoch_bump_silence_fraction
    )
    in_epoch_bump_amount = (
        saved_in_epoch_bump_amount
        if requested_in_epoch_bump_amount is None
        else requested_in_epoch_bump_amount
    )
    overrides = {
        tuple(name.split("__")): tuple(value) if isinstance(value, list) else value
        for name, value in values.items()
    }
    return RunRequest(
        config=apply_cli_overrides(config, overrides),
        run_dir=run_dir,
        resume_checkpoint=resume_checkpoint,
        evaluate_test=evaluate_test,
        learn_omega=learn_omega,
        omega_learning_rate_scale=omega_learning_rate_scale,
        in_epoch_bump_interval_batches=in_epoch_bump_interval_batches,
        in_epoch_bump_silence_fraction=in_epoch_bump_silence_fraction,
        in_epoch_bump_amount=in_epoch_bump_amount,
    )
