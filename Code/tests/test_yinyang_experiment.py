from dataclasses import asdict, replace

import pytest
import torch

from Code.tes_training import (
    AdamConfig,
    AdamState,
    CheckpointMismatchError,
    adam_update,
    build_yinyang_splits,
    save_checkpoint,
)
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config
from Code.yinyang_experiment import (
    _snapshot_from_payload,
    _validation_improves,
    calibrate_readouts,
    ensure_finite_weights,
    initialize_yinyang_model,
    run_yinyang_training,
)


def _small_config():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    return replace(
        config,
        data=replace(
            config.data,
            train_size=50,
            validation_size=30,
            test_size=30,
        ),
        optimization=replace(
            config.optimization,
            epochs=1,
            cosine_t_max=1,
        ),
        artifacts=replace(
            config.artifacts,
            checkpoint_interval=0,
            plot_interval=0,
            save_rasters=False,
        ),
    )


def test_initializer_uses_every_canonical_model_dimension_and_seed():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)

    first = initialize_yinyang_model(config)
    second = initialize_yinyang_model(config)

    assert first.W_in.shape == (config.model.n_hidden, config.model.n_inputs)
    assert first.W_out.shape == (config.model.n_outputs, config.model.n_hidden)
    assert torch.equal(first.W_in, second.W_in)
    assert torch.equal(first.W_out, second.W_out)
    assert first.config == config.to_tes_config()


def test_readout_excitation_is_width_normalized_to_the_reference_network():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    initialization = replace(
        config.initialization,
        readout_mean=1.0,
        readout_std=0.0,
        readout_floor=0.0,
        readout_row_scales=(1.0, 1.0, 1.0),
        readout_target_row_sums=None,
        heterogeneity_sigma=0.0,
    )
    reference = initialize_yinyang_model(
        replace(config, initialization=initialization)
    )
    narrow = initialize_yinyang_model(
        replace(
            config,
            model=replace(config.model, n_hidden=5),
            initialization=initialization,
        )
    )

    torch.testing.assert_close(
        narrow.W_out.sum(dim=1),
        reference.W_out.sum(dim=1),
    )


def test_calibration_mutates_failing_rows_until_real_coverage_succeeds():
    config = _small_config()
    config = replace(
        config,
        initialization=replace(
            config.initialization,
            readout_target_row_sums=(0.1, 0.1, 0.1),
        ),
        calibration=replace(
            config.calibration,
            target_fraction=0.9,
            scale=2.0,
            max_rounds=10,
        ),
    )
    train_split = build_yinyang_splits(
        train_size=config.data.train_size,
        validation_size=config.data.validation_size,
        test_size=config.data.test_size,
        seeds=config.data.split_seeds,
        dt=config.model.dt_ms,
    )[0]
    model = initialize_yinyang_model(config)
    initial_row_sums = model.W_out.sum(dim=1).clone()

    history = calibrate_readouts(model, train_split, config)

    assert history.succeeded
    assert len(history.rounds) > 1
    assert torch.all(model.W_out.sum(dim=1) > initial_row_sums)


def test_initializer_uses_floor_half_fast_rows_for_an_odd_hidden_width():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    config = replace(
        config,
        model=replace(config.model, n_hidden=15),
        initialization=replace(
            config.initialization,
            hidden_mean=1.0,
            hidden_std=0.0,
            hidden_floor=0.0,
            fast_hidden_gain=1.0,
        ),
    )

    model = initialize_yinyang_model(config)

    torch.testing.assert_close(model.W_in[:7], model.W_in[0].expand(7, -1))
    torch.testing.assert_close(model.W_in[7:], model.W_in[7].expand(8, -1))
    assert not torch.equal(model.W_in[6], model.W_in[7])


def test_legacy_best_snapshot_gains_default_omega_optimizer_state():
    model = initialize_yinyang_model(load_yinyang_config(DEFAULT_CONFIG_PATH))
    legacy_adam = {
        "m_in": torch.zeros_like(model.W_in),
        "v_in": torch.zeros_like(model.W_in),
        "m_out": torch.zeros_like(model.W_out),
        "v_out": torch.zeros_like(model.W_out),
        "step": 7,
    }

    snapshot = _snapshot_from_payload(
        {
            "epoch": 3,
            "w_in": model.W_in,
            "w_out": model.W_out,
            "adam": legacy_adam,
        },
        default_omega=model.omega,
    )

    torch.testing.assert_close(snapshot.omega, model.omega)
    assert torch.count_nonzero(snapshot.adam["m_omega"]) == 0
    assert torch.count_nonzero(snapshot.adam["v_omega"]) == 0
    assert snapshot.adam["omega_step"] == 0


def test_adam_update_defers_nonfinite_detection_to_epoch_boundary():
    model = initialize_yinyang_model(load_yinyang_config(DEFAULT_CONFIG_PATH))
    state = AdamState.zeros_like(model)
    gradient_in = torch.zeros_like(model.W_in)
    gradient_out = torch.zeros_like(model.W_out)
    gradient_in[0, 0] = float("nan")

    adam_update(model, gradient_in, gradient_out, state, AdamConfig())

    assert not torch.isfinite(model.W_in).all()
    with pytest.raises(FloatingPointError, match="epoch 3"):
        ensure_finite_weights(model, epoch=3)


def test_small_training_run_is_path_free_and_does_not_evaluate_test(tmp_path):
    artifacts = run_yinyang_training(
        _small_config(),
        tmp_path,
        evaluate_test=False,
    )

    assert len(artifacts.epochs) == 1
    assert artifacts.test_metrics is None
    assert artifacts.test_evaluations == 0
    assert artifacts.final_validation_metrics.samples == 30
    assert torch.isfinite(artifacts.model.W_in).all()
    assert torch.isfinite(artifacts.model.W_out).all()
    assert list(tmp_path.iterdir()) == []


def test_small_training_run_can_learn_and_snapshot_per_neuron_omega(tmp_path):
    artifacts = run_yinyang_training(
        _small_config(),
        tmp_path,
        learn_omega=True,
    )

    assert artifacts.training_config.learn_omega
    assert artifacts.adam_state.omega_step == 1
    assert not torch.equal(
        artifacts.final_snapshot.omega,
        artifacts.pre_training_snapshot.omega,
    )
    torch.testing.assert_close(
        artifacts.final_snapshot.omega,
        artifacts.model.omega,
    )
    assert torch.all(artifacts.model.omega >= artifacts.training_config.omega_min)
    assert torch.all(artifacts.model.omega <= artifacts.training_config.omega_max)


def test_epoch_boundary_check_runs_before_post_epoch_validation(monkeypatch, tmp_path):
    from Code import yinyang_experiment

    evaluation_calls = 0
    real_evaluate = yinyang_experiment.evaluate_split

    def record_evaluation(*args, **kwargs):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return real_evaluate(*args, **kwargs)

    def inject_nonfinite(model, split, state, config, *, epoch):
        model.W_in[0, 0] = float("nan")
        return ()

    monkeypatch.setattr(yinyang_experiment, "evaluate_split", record_evaluation)
    monkeypatch.setattr(yinyang_experiment, "train_epoch", inject_nonfinite)

    with pytest.raises(FloatingPointError, match="after epoch 1"):
        run_yinyang_training(_small_config(), tmp_path)

    assert evaluation_calls == 2


@pytest.mark.parametrize("learn_omega", [False, True])
def test_resume_accepts_only_an_epoch_budget_extension(tmp_path, learn_omega):
    base = _small_config()
    first_config = replace(
        base,
        optimization=replace(base.optimization, cosine_t_max=2),
    )
    first = run_yinyang_training(
        first_config, tmp_path, learn_omega=learn_omega
    )
    first_epoch = first.epochs[0]
    epoch_payload = {
        "epoch": first_epoch.epoch,
        "learning_rate": first_epoch.learning_rate,
        "training": asdict(first_epoch.training),
        "validation_pre_restart": asdict(first_epoch.validation_pre_restart),
        "validation": asdict(first_epoch.validation),
        "restarted_hidden": first_epoch.restarted_hidden,
    }
    best = first.best_validation_snapshot
    best_payload = {
        "epoch": best.epoch,
        "w_in": best.w_in,
        "w_out": best.w_out,
        "omega": best.omega,
        "adam": best.adam,
    }
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        first.model,
        first.adam_state,
        epoch=1,
        config=first.training_config,
        calibration_history=tuple(asdict(item) for item in first.calibration.rounds),
        bump_history=first.restart_history,
        epoch_metrics=(epoch_payload,),
        run_config=first_config.to_dict(),
        best_validation_snapshot=best_payload,
    )
    incompatible = replace(
        first_config,
        restart=replace(first_config.restart, offset=0.25),
    )

    with pytest.raises(CheckpointMismatchError, match="configuration mismatch"):
        run_yinyang_training(
            incompatible,
            tmp_path,
            resume_checkpoint=checkpoint,
            learn_omega=learn_omega,
        )

    compatible = replace(
        first_config,
        optimization=replace(first_config.optimization, epochs=2),
    )
    resumed = run_yinyang_training(
        compatible,
        tmp_path,
        resume_checkpoint=checkpoint,
        learn_omega=learn_omega,
    )
    uninterrupted = run_yinyang_training(
        compatible, tmp_path, learn_omega=learn_omega
    )

    assert tuple(record.epoch for record in resumed.epochs) == (1, 2)
    assert resumed.adam_state.step == first.adam_state.step + 1
    assert resumed.best_validation_snapshot.epoch in (1, 2)
    assert torch.equal(resumed.model.W_in, uninterrupted.model.W_in)
    assert torch.equal(resumed.model.W_out, uninterrupted.model.W_out)
    for name in ("m_in", "v_in", "m_out", "v_out", "m_omega", "v_omega"):
        assert torch.equal(
            getattr(resumed.adam_state, name),
            getattr(uninterrupted.adam_state, name),
        )
    assert resumed.adam_state.step == uninterrupted.adam_state.step
    assert resumed.adam_state.omega_step == uninterrupted.adam_state.omega_step
    assert torch.equal(resumed.model.omega, uninterrupted.model.omega)
    assert resumed.restart_history == uninterrupted.restart_history
    assert resumed.best_validation_snapshot.epoch == (
        uninterrupted.best_validation_snapshot.epoch
    )
    assert torch.equal(
        resumed.best_validation_snapshot.w_in,
        uninterrupted.best_validation_snapshot.w_in,
    )
    assert torch.equal(
        resumed.best_validation_snapshot.w_out,
        uninterrupted.best_validation_snapshot.w_out,
    )
    assert torch.equal(
        resumed.best_validation_snapshot.omega,
        uninterrupted.best_validation_snapshot.omega,
    )


def test_best_validation_ties_are_broken_by_lower_loss():
    assert _validation_improves(0.947, 0.33, 0.947, 0.42)
    assert not _validation_improves(0.947, 0.42, 0.947, 0.33)
    assert _validation_improves(0.948, 99.0, 0.947, 0.1)
