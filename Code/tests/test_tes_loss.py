import pytest
import torch
from types import SimpleNamespace

from Code.tes_events import EffectiveReadout
from Code.tes_loss import evaluate_timing_loss, get_timing_loss
from Code.neurons import TeSConfig, TeSNeurons


def _readouts(*physical_times: float | None) -> tuple[EffectiveReadout, ...]:
    present = [time for time in physical_times if time is not None]
    latest = max(present) if present else None
    anchor = (
        next(index for index, time in enumerate(physical_times) if time == latest)
        if latest is not None
        else None
    )
    fake_time = 50.0 if latest is None else latest + 2.0
    return tuple(
        EffectiveReadout(
            neuron=output,
            physical_threshold_id=None if time is None else output,
            physical_time=time,
            effective_time=fake_time if time is None else time,
            censored=time is None,
            anchor_output=anchor if time is None else None,
        )
        for output, time in enumerate(physical_times)
    )


def test_original_delgrad_defaults_to_one_millisecond_and_penalizes_both_sides():
    specification = get_timing_loss("delgrad")

    result = evaluate_timing_loss(
        _readouts(10.0, 13.0, 14.0),
        label=0,
        loss_name="delgrad",
    )

    assert specification.default_margin_ms == 1.0
    assert result.loss == pytest.approx(6.5)
    assert result.active_wrong_outputs == (1, 2)
    assert result.output_time_grad is not None
    torch.testing.assert_close(
        result.output_time_grad,
        torch.tensor([-5.0, 2.0, 3.0], dtype=torch.float64),
    )


def test_hinge_remains_a_distinct_selectable_timing_loss():
    specification = get_timing_loss("squared_hinge")

    result = evaluate_timing_loss(
        _readouts(10.0, 13.0, 14.0),
        label=0,
        loss_name="squared_hinge",
    )

    assert specification.default_margin_ms == 2.0
    assert result.loss == 0.0
    assert result.active_wrong_outputs == ()
    assert result.output_time_grad is not None
    torch.testing.assert_close(
        result.output_time_grad,
        torch.zeros(3, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_original_delgrad_excludes_silent_wrong_readouts():
    result = evaluate_timing_loss(
        _readouts(10.0, None, 13.0),
        label=0,
        loss_name="delgrad",
    )

    assert result.loss == pytest.approx(2.0)
    assert result.active_wrong_outputs == (2,)
    assert result.output_time_grad is not None
    torch.testing.assert_close(
        result.output_time_grad,
        torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64),
    )


def test_original_delgrad_routes_a_silent_target_through_its_unique_anchor():
    result = evaluate_timing_loss(
        _readouts(None, 10.0, 13.0),
        label=0,
        loss_name="delgrad",
    )

    assert result.effective_times == (15.0, 10.0, 13.0)
    assert result.loss == pytest.approx(22.5)
    assert result.fake_anchor_output == 2
    assert result.output_time_grad is not None
    torch.testing.assert_close(
        result.output_time_grad,
        torch.tensor([0.0, -6.0, 6.0], dtype=torch.float64),
    )
    assert result.output_time_grad.sum().item() == pytest.approx(0.0)


def test_original_delgrad_reports_a_tied_silent_target_anchor_as_nondifferentiable():
    result = evaluate_timing_loss(
        _readouts(None, 13.0, 13.0),
        label=0,
        loss_name="delgrad",
    )

    assert result.loss == pytest.approx(9.0)
    assert result.output_time_grad is None
    assert "tied latest physical anchor" in result.nondifferentiable_reason


def test_timing_loss_rejects_unknown_names_at_the_module_boundary():
    with pytest.raises(ValueError, match="unknown timing loss"):
        get_timing_loss("not-a-loss")


def test_model_compatibility_wrappers_keep_delgrad_and_hinge_names_honest():
    model = TeSNeurons(
        1,
        1,
        3,
        torch.zeros((1, 1), dtype=torch.float64),
        torch.zeros((3, 1), dtype=torch.float64),
        TeSConfig(),
    )
    forward = SimpleNamespace(readouts=_readouts(10.0, 13.0, 14.0))

    delgrad = model.delgrad_loss(forward, label=0)
    hinge = model.squared_hinge_loss(forward, label=0)

    assert delgrad.loss == pytest.approx(6.5)
    assert delgrad.output_time_grad.tolist() == pytest.approx([-5.0, 2.0, 3.0])
    assert hinge.loss == 0.0
    assert hinge.output_time_grad.tolist() == pytest.approx([0.0, 0.0, 0.0])
