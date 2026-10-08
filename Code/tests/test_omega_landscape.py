import json
from dataclasses import replace

import pytest
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.omega_landscape import (
    DEFAULT_CHECKPOINT_PATH,
    _add_continuous_centered_secants,
    _local_boundary_signature,
    _model_from_checkpoint,
    landscape_aggregates,
    plot_omega_landscape,
    run_omega_loss_landscape,
    write_omega_landscape,
)
from Code.tes_training import CheckpointMismatchError
from Code.yinyang_config import DEFAULT_CONFIG_PATH, load_yinyang_config


def two_readout_model() -> TeSNeurons:
    return TeSNeurons(
        1,
        1,
        2,
        torch.tensor([[35.0]], dtype=torch.float64),
        torch.tensor([[4.0], [3.5]], dtype=torch.float64),
        TeSConfig(dt=0.05, reset_drop=5.0),
    )


def test_continuous_omega_tangent_matches_topology_stable_loss_secant():
    model = two_readout_model()
    source_omega = model.omega.clone()
    source_weights = (model.W_in.clone(), model.W_out.clone())
    num_points = 9

    landscape = run_omega_loss_landscape(
        model,
        [InputSpike(0.0, 0, 0)],
        label=0,
        neuron=2,
        loss_name="delgrad",
        margin=0.0,
        half_width=8e-4,
        num_points=num_points,
        t_end=50.0,
        discrete_dt=0.005,
    )

    assert len(landscape.points) == num_points
    center = landscape.points[num_points // 2]
    assert center.continuous_valid
    assert center.continuous_secant is not None
    assert center.continuous_gradient != 0.0
    assert center.continuous_relative_error < 2e-4
    assert center.continuous_cap_activations == 0
    assert center.discrete_valid
    assert center.discrete_gradient == pytest.approx(
        center.continuous_gradient, rel=2e-2, abs=2e-2
    )
    assert landscape.settings["discrete_dt"] == 0.005
    torch.testing.assert_close(model.omega, source_omega, rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.W_in, source_weights[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(model.W_out, source_weights[1], rtol=0.0, atol=0.0)


def test_omega_landscape_outputs_record_and_joint_plot(tmp_path):
    landscape = run_omega_loss_landscape(
        two_readout_model(),
        [InputSpike(0.0, 0, 0)],
        label=0,
        neuron=2,
        loss_name="delgrad",
        margin=0.0,
        half_width=4e-4,
        num_points=5,
        t_end=50.0,
        discrete_dt=0.005,
    )
    record_path = tmp_path / "omega_landscape.json"
    figure_path = tmp_path / "omega_landscape.png"

    write_omega_landscape(landscape, record_path)
    plot_omega_landscape(landscape, figure_path, arrow_count=3)

    payload = json.loads(record_path.read_text())
    assert payload["scientific_status"].startswith(
        "continuous event-forward tangent test"
    )
    assert payload["settings"]["discrete_dt"] == 0.005
    assert payload["aggregates"]["continuous_stable_secant_points"] > 0
    assert payload["aggregates"]["continuous_maximum_relative_error"] < 1e-3
    assert "discrete_stable_secant_points" not in payload["aggregates"]
    assert figure_path.stat().st_size > 1_000


def test_omega_landscape_rejects_a_sweep_crossing_zero():
    with pytest.raises(ValueError, match="strictly positive"):
        run_omega_loss_landscape(
            two_readout_model(),
            [InputSpike(0.0, 0, 0)],
            label=0,
            neuron=2,
            loss_name="delgrad",
            margin=0.0,
            half_width=1.0,
            num_points=5,
            t_end=50.0,
            discrete_dt=0.005,
        )


def test_local_topology_signature_detects_event_arrival_interleaving():
    before = _local_boundary_signature(
        thresholds=((0.5, ("threshold", 0)),),
        arrivals=((1.0, ("input", 0)),),
    )
    simultaneous = _local_boundary_signature(
        thresholds=((1.0, ("threshold", 0)),),
        arrivals=((1.0, ("input", 0)),),
    )
    after = _local_boundary_signature(
        thresholds=((1.5, ("threshold", 0)),),
        arrivals=((1.0, ("input", 0)),),
    )

    assert len({before, simultaneous, after}) == 3


def test_cross_mode_gradient_aggregates_exclude_mismatched_topology():
    landscape = run_omega_loss_landscape(
        two_readout_model(),
        [InputSpike(0.0, 0, 0)],
        label=0,
        neuron=2,
        loss_name="delgrad",
        margin=0.0,
        half_width=4e-4,
        num_points=5,
        t_end=50.0,
        discrete_dt=0.005,
    )
    baseline = landscape_aggregates(landscape)
    changed = replace(
        landscape,
        points=(
            replace(
                landscape.points[0],
                discrete_comparison_signature=("different",),
            ),
            *landscape.points[1:],
        ),
    )
    aggregates = landscape_aggregates(changed)

    assert aggregates["continuous_discrete_compared_points"] == (
        baseline["continuous_discrete_compared_points"] - 1
    )
    assert aggregates["continuous_discrete_excluded_topology_points"] == (
        baseline["continuous_discrete_excluded_topology_points"] + 1
    )


def test_cap_active_point_is_excluded_from_exact_continuous_secants():
    landscape = run_omega_loss_landscape(
        two_readout_model(),
        [InputSpike(0.0, 0, 0)],
        label=0,
        neuron=2,
        loss_name="delgrad",
        margin=0.0,
        half_width=4e-4,
        num_points=5,
        t_end=50.0,
        discrete_dt=0.005,
    )
    center_index = len(landscape.points) // 2
    points = list(landscape.points)
    points[center_index] = replace(
        points[center_index],
        continuous_cap_activations=1,
        continuous_secant=None,
        continuous_absolute_error=None,
        continuous_relative_error=None,
    )

    checked = _add_continuous_centered_secants(tuple(points))

    assert checked[center_index].continuous_secant is None


def test_checkpoint_loader_rejects_a_different_resolved_model_config():
    config = load_yinyang_config(DEFAULT_CONFIG_PATH)
    incompatible = replace(
        config,
        model=replace(
            config.model,
            alpha_per_ms=config.model.alpha_per_ms * 1.01,
        ),
    )

    with pytest.raises(CheckpointMismatchError, match="configuration mismatch"):
        _model_from_checkpoint(incompatible, DEFAULT_CHECKPOINT_PATH, 0.005)
