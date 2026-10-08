from dataclasses import replace
import pytest
import torch

import Code.tes_training as tes_training
from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_events import (
    ContinuousForwardResult,
    ContinuousThreshold,
    EffectiveReadout,
    StopInfo,
)
from Code.tes_training import (
    AdamConfig,
    AdamState,
    BatchMetrics,
    CheckpointMismatchError,
    TrainingConfig,
    TrainingSample,
    UpdateMetrics,
    YinYangSplit,
    adam_update,
    build_yinyang_splits,
    bump_silent_hidden,
    evaluate_split,
    load_checkpoint,
    physical_prediction,
    save_checkpoint,
    train_batch,
    train_epoch,
)
from Code.yinyang import YinYang


def _model() -> TeSNeurons:
    return TeSNeurons(
        2,
        2,
        3,
        torch.zeros((2, 2), dtype=torch.float64),
        torch.zeros((3, 2), dtype=torch.float64),
        TeSConfig(),
    )


def _readouts(*times: float | None) -> tuple[EffectiveReadout, ...]:
    return tuple(
        EffectiveReadout(
            neuron=output,
            physical_threshold_id=None if spike_time is None else output,
            physical_time=spike_time,
            effective_time=50.0 if spike_time is None else spike_time,
            censored=spike_time is None,
            anchor_output=None,
        )
        for output, spike_time in enumerate(times)
    )


def test_splits_are_balanced_deterministic_and_disjoint():
    first = build_yinyang_splits(
        train_size=50,
        validation_size=31,
        test_size=29,
        seeds=(1001, 1002, 1003),
    )
    second = build_yinyang_splits(
        train_size=50,
        validation_size=31,
        test_size=29,
        seeds=(1001, 1002, 1003),
    )

    assert first == second
    assert tuple(len(split.samples) for split in first) == (50, 31, 29)
    for split in first:
        assert max(split.class_counts) - min(split.class_counts) <= 1
        assert all(len(sample.events) == 4 for sample in split.samples)
        assert all(
            YinYang().which_class(*sample.point) == sample.label
            for sample in split.samples
        )
    points = [{sample.point for sample in split.samples} for split in first]
    assert points[0].isdisjoint(points[1])
    assert points[0].isdisjoint(points[2])
    assert points[1].isdisjoint(points[2])


def test_physical_prediction_treats_silence_and_ties_as_errors():
    assert physical_prediction(_readouts(2.0, 1.0, 3.0), 1).correct
    assert physical_prediction(_readouts(None, 1.0, None), 1).correct

    silent_target = physical_prediction(_readouts(1.0, None, 2.0), 1)
    assert silent_target.predicted == 0
    assert silent_target.target_silent
    assert not silent_target.correct

    all_silent = physical_prediction(_readouts(None, None, None), 0)
    assert all_silent.predicted is None
    assert all_silent.all_silent

    tied = physical_prediction(_readouts(1.0, 1.0, 2.0), 0)
    assert tied.predicted is None
    assert tied.ambiguous_tie


def test_evaluation_always_uses_the_full_time_horizon():
    early_stop_values: list[bool] = []

    class FakeModel:
        n_hidden = 2
        n_out = 3

        def forward_continuous(self, events, t_end, *, early_stop):
            early_stop_values.append(early_stop)
            return ContinuousForwardResult(
                input_arrivals=(),
                hidden_spikes=(),
                hidden_arrivals=(),
                output_spikes=(),
                readouts=_readouts(1.0, 2.0, None),
                stop=StopInfo("requested_horizon", t_end, t_end, 0, 0, 0),
            )

    samples = tuple(
        TrainingSample(
            index=index,
            point=(0.5, 0.5),
            label=0,
            events=(InputSpike(5.0, 100, 0),),
        )
        for index in range(4)
    )
    split = YinYangSplit("validation", 7, (4, 0, 0), samples)

    metrics = evaluate_split(
        FakeModel(),
        split,
        t_end=50.0,
        margin=2.0,
    )

    assert early_stop_values == [False] * 4
    assert metrics.samples == 4
    assert metrics.accuracy == 1.0


def _threshold(layer: str, neuron: int, time: float) -> ContinuousThreshold:
    return ContinuousThreshold(
        event_id=neuron,
        layer=layer,
        neuron=neuron,
        time=time,
        slope=0.1,
        acceleration_pre=0.0,
        v_pre=1.0,
        i_pre=1.0,
        q_pre=0.0,
        v_post=0.0,
        continuous_delay=1.0,
        arrival_time=time + 1.0,
    )


def test_hidden_activity_after_the_last_first_output_spike_is_silent():
    class FakeModel:
        n_hidden = 2
        n_out = 3

        def forward_continuous(self, events, t_end, *, early_stop):
            return ContinuousForwardResult(
                input_arrivals=(),
                hidden_spikes=(
                    _threshold("hidden", 0, 4.5),
                    _threshold("hidden", 1, 5.1),
                ),
                hidden_arrivals=(),
                output_spikes=(
                    _threshold("output", 0, 3.0),
                    _threshold("output", 1, 4.0),
                    _threshold("output", 2, 5.0),
                ),
                readouts=_readouts(3.0, 4.0, 5.0),
                stop=StopInfo("requested_horizon", t_end, t_end, 0, 0, 0),
            )

    sample = TrainingSample(0, (0.5, 0.5), 0, ())
    split = YinYangSplit("validation", 7, (1, 0, 0), (sample,))

    metrics = evaluate_split(FakeModel(), split, 50.0, 2.0)

    assert metrics.hidden_silence_counts == (0, 1)
    assert metrics.hidden_full_horizon_silence_counts == (0, 0)


def test_each_loss_resolves_its_own_default_margin():
    assert TrainingConfig(loss_name="delgrad").resolved_margin == 1.0
    assert TrainingConfig(loss_name="squared_hinge").resolved_margin == 2.0
    assert TrainingConfig(loss_name="delgrad", margin=0.16).resolved_margin == 0.16


def test_adam_first_step_matches_the_closed_form_update():
    model = _model()
    state = AdamState.zeros_like(model)
    grad_in = torch.tensor([[1.0, -2.0], [0.0, 3.0]], dtype=torch.float64)
    grad_out = torch.tensor(
        [[-1.0, 0.0], [2.0, -3.0], [0.0, 4.0]], dtype=torch.float64
    )
    config = AdamConfig(learning_rate=0.01, gradient_norm_cap=100.0)

    metrics = adam_update(model, grad_in, grad_out, state, config)

    expected_in = -0.01 * grad_in / (grad_in.abs() + config.epsilon)
    expected_out = -0.01 * grad_out / (grad_out.abs() + config.epsilon)
    assert torch.allclose(model.W_in, expected_in)
    assert torch.allclose(model.W_out, expected_out)
    assert state.step == 1
    assert metrics.clipping_factor == 1.0
    assert metrics.negative_w_in > 0
    assert metrics.negative_w_out > 0


def test_adam_uses_one_combined_gradient_clip_factor():
    model = _model()
    state = AdamState.zeros_like(model)
    grad_in = torch.zeros_like(model.W_in)
    grad_out = torch.zeros_like(model.W_out)
    grad_in[0, 0] = 6.0
    grad_out[0, 0] = 8.0

    metrics = adam_update(
        model,
        grad_in,
        grad_out,
        state,
        AdamConfig(gradient_norm_cap=5.0),
    )

    assert metrics.raw_gradient_norm == pytest.approx(10.0)
    assert metrics.clipped_gradient_norm == pytest.approx(5.0)
    assert metrics.clipping_factor == pytest.approx(0.5)
    assert state.m_in[0, 0] == pytest.approx(0.3)
    assert state.m_out[0, 0] == pytest.approx(0.4)


def test_adam_omega_stream_uses_one_hundredth_learning_rate_per_neuron():
    model = _model()
    state = AdamState.zeros_like(model)
    initial = model.omega.clone()
    grad_omega = torch.tensor(
        [1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float64
    )

    metrics = adam_update(
        model,
        torch.zeros_like(model.W_in),
        torch.zeros_like(model.W_out),
        state,
        AdamConfig(learning_rate=0.01, gradient_norm_cap=100.0),
        grad_omega=grad_omega,
        omega_learning_rate_scale=0.01,
        omega_min=0.1,
        omega_max=0.35,
    )

    expected_update = 0.0001 * grad_omega / (grad_omega.abs() + 1e-8)
    torch.testing.assert_close(model.omega, initial - expected_update)
    assert state.omega_step == 1
    assert state.step == 1
    assert metrics.omega_max_absolute_update == pytest.approx(0.0001)
    assert metrics.omega_clamped_entries == 0


def test_adam_omega_stream_clamps_to_the_safe_frequency_range():
    model = _model()
    model.omega[0] = 0.1002
    model.omega[1] = 0.3498
    state = AdamState.zeros_like(model)
    gradient = torch.zeros_like(model.omega)
    gradient[0] = 1.0
    gradient[1] = -1.0

    metrics = adam_update(
        model,
        torch.zeros_like(model.W_in),
        torch.zeros_like(model.W_out),
        state,
        AdamConfig(learning_rate=0.01),
        grad_omega=gradient,
        omega_learning_rate_scale=0.1,
        omega_min=0.1,
        omega_max=0.35,
    )

    assert float(model.omega[0]) == pytest.approx(0.1)
    assert float(model.omega[1]) == pytest.approx(0.35)
    assert metrics.omega_clamped_entries == 2
    assert metrics.omega_min == pytest.approx(0.1)
    assert metrics.omega_max == pytest.approx(0.35)


def test_adam_rejects_bad_omega_shape_before_mutating_weights_or_state():
    model = _model()
    state = AdamState.zeros_like(model)
    initial_w_in = model.W_in.clone()
    initial_w_out = model.W_out.clone()
    initial_state = state.to_payload()

    with pytest.raises(ValueError, match="omega gradient shape"):
        adam_update(
            model,
            torch.ones_like(model.W_in),
            torch.ones_like(model.W_out),
            state,
            AdamConfig(),
            grad_omega=torch.ones(model.n_neurons - 1, dtype=torch.float64),
        )

    torch.testing.assert_close(model.W_in, initial_w_in)
    torch.testing.assert_close(model.W_out, initial_w_out)
    assert state.step == 0
    assert state.omega_step == 0
    for name, value in initial_state.items():
        current = state.to_payload()[name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(current, value)
        else:
            assert current == value


def test_hidden_restart_is_strictly_over_half_and_preserves_adam_state():
    model = _model()
    state = AdamState.zeros_like(model)
    state.m_in.fill_(3.0)
    state.v_in.fill_(4.0)
    before = state.to_payload()

    restarted = bump_silent_hidden(
        model,
        silent_counts=(5, 6),
        state=state,
        sample_count=10,
        threshold_fraction=0.5,
        amount=0.25,
        seed=9,
        noise_std=0.0,
    )

    assert restarted == (1,)
    assert torch.equal(model.W_in[0], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(model.W_in[1], torch.full((2,), 0.25, dtype=torch.float64))
    assert all(
        state.to_payload()[name].equal(value)
        for name, value in before.items()
        if isinstance(value, torch.Tensor)
    )
    assert state.step == before["step"]


def test_deterministic_bump_adds_exact_offset_only_above_silence_threshold():
    model = _model()

    bumped = tes_training.bump_silent_hidden_deterministic(
        model,
        silent_counts=(6, 5),
        sample_count=10,
        threshold_fraction=0.5,
        amount=0.05,
    )

    assert bumped == (0,)
    torch.testing.assert_close(
        model.W_in[0], torch.full((2,), 0.05, dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.W_in[1], torch.zeros(2, dtype=torch.float64)
    )


def test_train_epoch_bumps_after_each_complete_ten_batch_window(monkeypatch):
    model = _model()
    state = AdamState.zeros_like(model)
    samples = tuple(
        TrainingSample(index, (0.5, 0.5), 0, ())
        for index in range(20)
    )
    split = YinYangSplit("train", 7, (20, 0, 0), samples)
    update = UpdateMetrics(
        raw_gradient_norm=0.0,
        clipped_gradient_norm=0.0,
        clipping_factor=1.0,
        negative_w_in=0,
        negative_w_out=0,
        w_in_min=0.0,
        w_in_max=0.0,
        w_out_min=0.0,
        w_out_max=0.0,
    )

    def fake_train_batch(model, samples, state, config, *, epoch, batch):
        del model, samples, state, config
        silence = (1, int(batch <= 5)) if batch <= 10 else (0, 0)
        return BatchMetrics(
            epoch=epoch,
            batch=batch,
            samples=1,
            mean_loss=0.0,
            accuracy=0.0,
            correct=0,
            all_silent=0,
            target_silent=0,
            ambiguous_ties=0,
            hidden_thresholds=0,
            output_thresholds=0,
            threshold_sensitivity_cap_activations=0,
            minimum_causal_slope=None,
            update=update,
            runtime_seconds=0.0,
            hidden_silence_counts=silence,
        )

    monkeypatch.setattr(tes_training, "train_batch", fake_train_batch)
    batches = train_epoch(
        model,
        split,
        state,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            in_epoch_bump_interval_batches=10,
            in_epoch_bump_silence_fraction=0.5,
            in_epoch_bump_amount=0.05,
        ),
        epoch=1,
    )

    assert batches[9].in_epoch_bumped_hidden == (0,)
    assert batches[19].in_epoch_bumped_hidden == ()
    torch.testing.assert_close(
        model.W_in[0], torch.full((2,), 0.05, dtype=torch.float64)
    )
    torch.testing.assert_close(
        model.W_in[1], torch.zeros(2, dtype=torch.float64)
    )


def test_cosine_learning_rate_starts_at_maximum_and_reaches_halfway_value():
    config = TrainingConfig(
        epochs=40,
        adam=AdamConfig(learning_rate=0.01),
        cosine_annealing_t_max=40,
    )

    assert config.adam_for_epoch(1).learning_rate == pytest.approx(0.01)
    assert config.adam_for_epoch(21).learning_rate == pytest.approx(0.005)
    assert config.adam_for_epoch(40).learning_rate < 2e-5


def _omega_training_model() -> TeSNeurons:
    return TeSNeurons(
        1,
        1,
        2,
        torch.tensor([[35.0]], dtype=torch.float64),
        torch.tensor([[4.0], [3.5]], dtype=torch.float64),
        TeSConfig(dt=0.05, reset_drop=5.0),
    )


def test_train_batch_learns_one_clamped_omega_per_neuron_slowly():
    model = _omega_training_model()
    state = AdamState.zeros_like(model)
    initial = model.omega.clone()
    sample = TrainingSample(
        index=0,
        point=(0.5, 0.5),
        label=0,
        events=(InputSpike(0.0, 0, 0),),
    )
    config = TrainingConfig(
        epochs=1,
        batch_size=1,
        t_end=50.0,
        loss_name="delgrad",
        margin=0.0,
        learn_omega=True,
    )

    metrics = train_batch(
        model, (sample,), state, config, epoch=1, batch=1
    )

    assert state.omega_step == 1
    assert not torch.equal(model.omega, initial)
    assert torch.unique(model.omega - initial).numel() > 1
    assert metrics.update.omega_max_absolute_update == pytest.approx(0.0001)
    assert torch.all(model.omega >= config.omega_min)
    assert torch.all(model.omega <= config.omega_max)


def test_train_batch_keeps_legacy_omega_fixed_when_learning_is_disabled():
    model = _omega_training_model()
    state = AdamState.zeros_like(model)
    initial = model.omega.clone()
    sample = TrainingSample(
        index=0,
        point=(0.5, 0.5),
        label=0,
        events=(InputSpike(0.0, 0, 0),),
    )

    metrics = train_batch(
        model,
        (sample,),
        state,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            t_end=50.0,
            loss_name="delgrad",
            margin=0.0,
        ),
        epoch=1,
        batch=1,
    )

    torch.testing.assert_close(model.omega, initial, rtol=0.0, atol=0.0)
    assert state.omega_step == 0
    assert metrics.update.raw_omega_gradient_norm is None
    assert metrics.hidden_silence_counts == ()


def test_checkpoint_round_trip_and_explicit_epoch_extension(tmp_path):
    model = _model()
    model.W_in.copy_(torch.arange(4, dtype=torch.float64).reshape(2, 2))
    model.W_out.copy_(torch.arange(6, dtype=torch.float64).reshape(3, 2))
    model.omega.copy_(torch.linspace(0.12, 0.32, model.n_neurons))
    state = AdamState.zeros_like(model)
    state.step = 7
    state.m_omega.copy_(torch.arange(model.n_neurons, dtype=torch.float64))
    state.v_omega.copy_(torch.arange(model.n_neurons, dtype=torch.float64) + 1.0)
    state.omega_step = 5
    config = TrainingConfig(epochs=2, batch_size=10)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, state, epoch=2, config=config)

    restored = _model()
    loaded = load_checkpoint(checkpoint, restored, expected_config=config)

    assert loaded.epoch == 2
    assert loaded.adam.step == 7
    assert loaded.adam.omega_step == 5
    assert torch.equal(restored.W_in, model.W_in)
    assert torch.equal(restored.W_out, model.W_out)
    assert torch.equal(restored.omega, model.omega)
    assert torch.equal(loaded.adam.m_omega, state.m_omega)
    assert torch.equal(loaded.adam.v_omega, state.v_omega)

    extended = replace(config, epochs=4)
    load_checkpoint(
        checkpoint,
        _model(),
        expected_config=extended,
        allow_epoch_extension=True,
    )


def test_checkpoint_rejects_scientific_config_before_mutating_model(tmp_path):
    source = _model()
    source.W_in.fill_(2.0)
    source.W_out.fill_(3.0)
    checkpoint = tmp_path / "model.pt"
    config = TrainingConfig(epochs=2, batch_size=10)
    saved_run_config = {
        "optimization": {"epochs": 2},
        "model": {"alpha_per_ms": 0.025},
    }
    save_checkpoint(
        checkpoint,
        source,
        AdamState.zeros_like(source),
        epoch=2,
        config=config,
        run_config=saved_run_config,
    )
    target = _model()
    target.W_in.fill_(91.0)
    target.W_out.fill_(92.0)
    requested = {
        "optimization": {"epochs": 2},
        "model": {"alpha_per_ms": 0.05},
    }

    with pytest.raises(CheckpointMismatchError, match="resolved configuration"):
        load_checkpoint(
            checkpoint,
            target,
            expected_config=config,
            expected_run_config=requested,
        )

    assert torch.all(target.W_in == 91.0)
    assert torch.all(target.W_out == 92.0)


def test_weight_only_checkpoint_can_explicitly_start_omega_learning(tmp_path):
    source = _model()
    checkpoint = tmp_path / "legacy.pt"
    weight_only = TrainingConfig(epochs=2, batch_size=10)
    save_checkpoint(
        checkpoint,
        source,
        AdamState.zeros_like(source),
        epoch=1,
        config=weight_only,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["schema_version"] = 4
    payload.pop("omega")
    for name in ("m_omega", "v_omega", "omega_step"):
        payload["adam"].pop(name)
    for name in (
        "learn_omega",
        "omega_learning_rate_scale",
        "omega_min",
        "omega_max",
    ):
        payload["config"].pop(name)
    torch.save(payload, checkpoint)

    target = _model()
    loaded = load_checkpoint(
        checkpoint,
        target,
        expected_config=replace(weight_only, learn_omega=True),
        allow_omega_learning_upgrade=True,
    )

    assert loaded.config.learn_omega is False
    assert loaded.adam.omega_step == 0
    assert torch.count_nonzero(loaded.adam.m_omega) == 0
    assert torch.count_nonzero(loaded.adam.v_omega) == 0
    torch.testing.assert_close(target.omega, source.omega)
