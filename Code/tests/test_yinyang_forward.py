from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from Code.yinyang import (
    YinYang,
    compare_step_sizes,
    encode_events,
    evaluate_health,
    plot_trajectories,
    run_demo_data,
    run_demo,
)


def test_code_tree_has_no_legacy_module_or_bias_symbol():
    code_root = Path(__file__).parents[1]
    assert not (code_root / "otherneurons.py").exists()
    source = "\n".join(path.read_text() for path in code_root.glob("*.py"))
    assert "bias" not in source.lower()


def test_method_uses_negative_quadrature_jump_and_positive_weight_gradient():
    method = (Path(__file__).parents[2] / "Latex" / "tes_method.tex").read_text()
    assert r"+\mathrm{i}\sum_j" not in method
    assert r"-\mathrm{i}\sum_j" in method
    assert r"Q_i\!\left((t^s_{jk})^+\right)" in method
    assert r"-w_{ij}" in method
    assert r"=+\sum_{k:\,j\ \mathrm{spikes}}" in method


def test_run_demo_writes_plot_and_machine_readable_diagnostics(tmp_path):
    result = run_demo(dt=0.1, seed=23, output_dir=tmp_path)
    assert result.figure_path.stat().st_size > 20_000
    payload = json.loads(result.diagnostics_path.read_text())
    assert payload["dt"] == 0.1
    assert payload["frequency_periods_ms"] == [8.0 * np.pi, 12.0 * np.pi]
    assert payload["four_input_channels"] is True
    assert payload["health"]["passed"] is True


def test_plot_png_metadata_reports_dynamics_health_and_event_activity(tmp_path):
    result = run_demo(dt=0.1, seed=23, output_dir=tmp_path)

    with Image.open(result.figure_path) as figure:
        title = figure.info["Title"]
        description = figure.info["Description"]

    for expected in (
        "dt=0.1 ms",
        "alpha=0.025 /ms",
        "fast=0.250000 rad/ms (period 25.1327 ms)",
        "slow=0.166667 rad/ms (period 37.6991 ms)",
        "health=healthy",
        "warnings=none",
    ):
        assert expected in title
    assert "readout activity" in description
    metrics = result.demo.report.metrics
    assert (
        f"events: input={metrics['external_arrivals']}, "
        f"hidden thresholds={metrics['hidden_thresholds']}, "
        f"delayed={metrics['delayed_arrivals']}"
    ) in description


def test_trajectory_panels_give_iq_a_separate_zero_centered_axis(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    right_axes = []
    original_twinx = Axes.twinx

    def record_twin_axis(left_axis):
        right_axis = original_twinx(left_axis)
        right_axes.append(right_axis)
        return right_axis

    monkeypatch.setattr(Axes, "twinx", record_twin_axis)
    plot_trajectories(run_demo_data(dt=0.1, seed=23), tmp_path / "trajectories.png")

    assert len(right_axes) == 6
    for right_axis in right_axes:
        labeled = {line.get_label(): line for line in right_axis.lines}
        assert set(labeled) >= {"I", "Q"}
        zero_guides = [
            line
            for line in right_axis.lines
            if np.allclose(np.asarray(line.get_ydata(), dtype=float), 0.0)
        ]
        assert len(zero_guides) == 1
        assert zero_guides[0].get_linestyle() == "--"
        assert zero_guides[0].get_alpha() <= 0.35

    left_axes = [
        axis
        for axis in right_axes[0].figure.axes
        if any(line.get_label() == "V" for line in axis.lines)
    ]
    assert len(left_axes) == 6
    for left_axis in left_axes:
        labeled = {line.get_label(): line for line in left_axis.lines}
        assert labeled["V"].get_linewidth() >= 2.0
        assert "I" not in labeled
        assert "Q" not in labeled


def test_balanced_sampling_is_exact_and_deterministic():
    dataset = YinYang()
    coords_a, labels_a = dataset.sample_balanced(n_per_class=4, seed=17)
    coords_b, labels_b = dataset.sample_balanced(n_per_class=4, seed=17)
    np.testing.assert_array_equal(coords_a, coords_b)
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_array_equal(np.bincount(labels_a, minlength=3), [4, 4, 4])


def test_encoder_has_exactly_four_latency_channels():
    encoded = encode_events(np.array([0.2, 0.7]), dt=0.1, t_min=5.0, t_max=25.0)
    np.testing.assert_allclose(encoded.exact_times, [21.0, 11.0, 9.0, 19.0])
    np.testing.assert_array_equal(encoded.grid_steps, [210, 110, 90, 190])
    assert [event.channel for event in encoded.events] == [0, 1, 2, 3]
    assert len(encoded.events) == 4


def test_input_grid_rounding_error_is_at_most_half_a_step():
    encoded = encode_events(np.array([0.123, 0.456]), dt=0.3, t_min=5.0, t_max=25.0)
    assert np.max(np.abs(encoded.grid_times - encoded.exact_times)) <= 0.15 + 1e-12


def test_demo_is_deterministic_and_exercises_required_events():
    first = run_demo_data(dt=0.1, seed=23)
    second = run_demo_data(dt=0.1, seed=23)
    assert first.report.passed, first.report
    assert first.report.checks["external_arrival"]
    assert first.report.checks["hidden_threshold"]
    assert first.report.checks["delayed_arrival"]
    assert first.report.checks["finite_readout"]
    for a, b in zip(first.trajectories, second.trajectories, strict=True):
        torch.testing.assert_close(a.V, b.V, rtol=0.0, atol=0.0)
        assert a.arrivals == b.arrivals
        assert a.thresholds == b.thresholds


def test_demo_supports_multiple_step_sizes_and_reports_topology():
    comparison = compare_step_sizes((0.2, 0.1, 0.05), seed=23)
    assert comparison.step_sizes == (0.2, 0.1, 0.05)
    assert all(comparison.finite)
    if not comparison.topology_changed:
        assert comparison.state_errors[1] <= comparison.state_errors[0] + 1e-10
    assert isinstance(comparison.topology_changed, bool)


def test_frequencies_and_weights_are_not_trainable():
    demo = run_demo_data(dt=0.1, seed=23)
    assert list(demo.model.parameters()) == []


def test_demo_health_requires_hidden_subtractive_reset():
    demo = run_demo_data(dt=0.1, seed=23)
    assert demo.report.checks["hidden_subtractive_reset"]


def test_health_rejects_incorrect_hidden_subtractive_reset_record():
    demo = run_demo_data(dt=0.1, seed=23)
    trajectory_index, threshold_index = next(
        (trajectory_index, threshold_index)
        for trajectory_index, trajectory in enumerate(demo.trajectories)
        for threshold_index, threshold in enumerate(trajectory.thresholds)
        if threshold.neuron < demo.model.n_hidden
    )
    trajectory = demo.trajectories[trajectory_index]
    bad_threshold = replace(
        trajectory.thresholds[threshold_index],
        v_post=trajectory.thresholds[threshold_index].v_pre,
    )
    bad_trajectory = replace(
        trajectory,
        thresholds=(
            trajectory.thresholds[:threshold_index]
            + (bad_threshold,)
            + trajectory.thresholds[threshold_index + 1 :]
        ),
    )
    report = evaluate_health(
        demo.model,
        tuple(
            bad_trajectory if index == trajectory_index else item
            for index, item in enumerate(demo.trajectories)
        ),
    )
    assert report.checks["hidden_subtractive_reset"] is False
    assert report.passed is False
