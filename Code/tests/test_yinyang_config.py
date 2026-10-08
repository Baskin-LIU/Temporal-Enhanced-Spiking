import json

import pytest

from Code.yinyang_config import (
    CLI_FIELD_PATHS,
    DEFAULT_CONFIG_PATH,
    YinYangConfig,
    load_yinyang_config,
    parse_yinyang_args,
)


def _leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        result = set()
        for key, item in value.items():
            result.update(_leaf_paths(item, (*prefix, key)))
        return result
    return {prefix}


def test_canonical_json_matches_the_best_test_experiment():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)

    assert config.loss.name == "squared_hinge"
    assert config.loss.margin_ms == 2.0
    assert config.optimization.epochs == 40
    assert config.optimization.batch_size == 50
    assert config.optimization.adam.learning_rate == 0.01
    assert config.optimization.cosine_t_max == 40
    assert config.restart.offset == 0.1
    assert config.initialization.readout_reference_hidden == 12
    assert config.model.v_dot_clip == 2.0
    assert config.model.alpha_per_ms == pytest.approx(1.0 / 40.0)
    assert config.model.omega_fast_per_ms == pytest.approx(1.0 / 4.0)
    assert config.model.omega_slow_per_ms == pytest.approx(1.0 / 6.0)
    assert config.artifacts.checkpoint_interval == 0
    assert config.artifacts.plot_interval == 0
    assert config.artifacts.save_rasters is True
    assert YinYangConfig.from_dict(config.to_dict()) == config


def test_every_json_leaf_has_a_cli_override():
    values = json.loads(DEFAULT_CONFIG_PATH.read_text())

    assert set(CLI_FIELD_PATHS) == _leaf_paths(values)


def test_cli_applies_only_explicit_overrides(tmp_path):
    request = parse_yinyang_args(
        [
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--run-dir",
            str(tmp_path),
            "--learning-rate",
            "0.02",
            "--cosine-eta-min",
            "0.0",
            "--checkpoint-interval",
            "0",
            "--no-save-rasters",
        ]
    )

    assert request.run_dir == tmp_path
    assert request.config.optimization.adam.learning_rate == 0.02
    assert request.config.optimization.cosine_eta_min == 0.0
    assert request.config.artifacts.checkpoint_interval == 0
    assert request.config.artifacts.save_rasters is False
    assert request.config.restart.offset == 0.1
    assert request.evaluate_test is False
    assert request.learn_omega is False
    assert request.omega_learning_rate_scale == pytest.approx(0.01)


def test_cli_overrides_readout_excitation_reference_width(tmp_path):
    request = parse_yinyang_args(
        [
            "--run-dir",
            str(tmp_path),
            "--readout-reference-hidden",
            "20",
        ]
    )

    assert request.config.initialization.readout_reference_hidden == 20


@pytest.mark.parametrize("flag", ("--learn_omega", "--learn-omega"))
def test_cli_enables_per_neuron_omega_learning(flag, tmp_path):
    request = parse_yinyang_args(
        ["--run-dir", str(tmp_path), flag]
    )

    assert request.learn_omega is True


def test_cli_configures_omega_rate_independently_from_weight_rate(tmp_path):
    request = parse_yinyang_args(
        [
            "--run-dir",
            str(tmp_path),
            "--learning-rate",
            "0.005",
            "--omega-learning-rate-scale",
            "0.02",
            "--learn-omega",
        ]
    )

    training = request.config.to_training_config(
        learn_omega=request.learn_omega,
        omega_learning_rate_scale=request.omega_learning_rate_scale,
    )
    assert training.adam.learning_rate == pytest.approx(0.005)
    assert training.omega_learning_rate_scale == pytest.approx(0.02)
    assert (
        training.adam.learning_rate * training.omega_learning_rate_scale
        == pytest.approx(0.0001)
    )


def test_cli_configures_lightweight_in_epoch_hidden_bumps(tmp_path):
    request = parse_yinyang_args(
        [
            "--run-dir",
            str(tmp_path),
            "--in-epoch-bump-interval",
            "10",
            "--in-epoch-bump-silence-fraction",
            "0.5",
            "--in-epoch-bump-amount",
            "0.05",
        ]
    )

    training = request.config.to_training_config(
        in_epoch_bump_interval_batches=(
            request.in_epoch_bump_interval_batches
        ),
        in_epoch_bump_silence_fraction=(
            request.in_epoch_bump_silence_fraction
        ),
        in_epoch_bump_amount=request.in_epoch_bump_amount,
    )
    assert training.in_epoch_bump_interval_batches == 10
    assert training.in_epoch_bump_silence_fraction == pytest.approx(0.5)
    assert training.in_epoch_bump_amount == pytest.approx(0.05)


def test_cli_accepts_tuple_and_nullable_overrides(tmp_path):
    request = parse_yinyang_args(
        [
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--run-dir",
            str(tmp_path),
            "--split-seeds",
            "7",
            "8",
            "9",
            "--readout-target-row-sums",
            "20",
            "21",
            "22",
            "--evaluate-test",
        ]
    )

    assert request.config.data.split_seeds == (7, 8, 9)
    assert request.config.initialization.readout_target_row_sums == (
        20.0,
        21.0,
        22.0,
    )
    assert request.evaluate_test is True


def test_unknown_json_field_is_rejected():
    values = json.loads(DEFAULT_CONFIG_PATH.read_text())
    values["model"]["typo"] = 1

    with pytest.raises(ValueError, match="unknown model field"):
        YinYangConfig.from_dict(values)


def test_config_builds_active_numerical_and_training_configs():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    model = config.to_tes_config()
    training = config.to_training_config()

    assert model.tau_v == 20.0
    assert model.alpha == 0.025
    assert model.omega_fast == 0.25
    assert model.omega_slow == pytest.approx(1.0 / 6.0)
    assert model.threshold_sensitivity_cap == 2.0
    assert model.silent_target_gap_ms == 2.0
    assert training.loss_name == "squared_hinge"
    assert training.margin == 2.0
    assert training.batch_size == 50
    assert training.cosine_annealing_t_max == 40
    assert not hasattr(training, "learning_rate_decay_start_epoch")


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("initialization", "hidden_std", -1.0),
        ("initialization", "heterogeneity_sigma", -0.1),
        ("initialization", "fast_hidden_gain", 0.0),
        ("optimization", "adam.beta1", 1.0),
        ("optimization", "adam.epsilon", 0.0),
        ("loss", "margin_ms", -0.1),
        ("model", "c1", float("nan")),
        ("model", "slope_epsilon", -1.0),
        ("restart", "seed", -1),
    ),
)
def test_invalid_scientific_values_are_rejected(section, field, value):
    values = json.loads(DEFAULT_CONFIG_PATH.read_text())
    if field.startswith("adam."):
        values[section]["adam"][field.removeprefix("adam.")] = value
    else:
        values[section][field] = value

    with pytest.raises(ValueError):
        YinYangConfig.from_dict(values)


@pytest.mark.parametrize("field", ("readout_row_scales", "readout_target_row_sums"))
def test_readout_row_values_must_be_positive(field):
    values = json.loads(DEFAULT_CONFIG_PATH.read_text())
    values["initialization"]["readout_target_row_sums"] = [20.0, 21.0, 22.0]
    values["initialization"][field][1] = 0.0

    with pytest.raises(ValueError, match="readout row"):
        YinYangConfig.from_dict(values)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("model", "n_hidden"), 12.0),
        (("artifacts", "save_rasters"), "false"),
        (("data", "split_seeds"), [1, 2, 3, 3]),
        (("restart", "offset"), 0.0),
        (("artifacts", "checkpoint_interval"), 1.5),
        (("optimization", "adam", "learning_rate"), True),
    ),
)
def test_malformed_leaf_types_and_domains_fail_at_startup(path, value):
    values = json.loads(DEFAULT_CONFIG_PATH.read_text())
    target = values
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value

    with pytest.raises(ValueError):
        YinYangConfig.from_dict(values)
