from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

from Code.neurons import InputSpike, TeSConfig, TeSNeurons
from Code.tes_events import DiscreteForwardResult


@dataclass(frozen=True)
class EncodedInput:
    exact_times: np.ndarray
    grid_steps: np.ndarray
    grid_times: np.ndarray
    events: tuple[InputSpike, ...]


@dataclass(frozen=True)
class HealthReport:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float | int | None]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DemoRun:
    model: TeSNeurons
    coords: np.ndarray
    labels: np.ndarray
    encoded: tuple[EncodedInput, ...]
    trajectories: tuple[DiscreteForwardResult, ...]
    report: HealthReport


@dataclass(frozen=True)
class StepComparison:
    step_sizes: tuple[float, ...]
    state_errors: tuple[float, ...]
    finite: tuple[bool, ...]
    topology_changed: bool


@dataclass(frozen=True)
class DemoArtifacts:
    demo: DemoRun
    figure_path: Path
    diagnostics_path: Path


class YinYang:
    def __init__(self, r_small: float = 0.1, r_big: float = 0.5) -> None:
        self.r_small = r_small
        self.r_big = r_big

    def which_class(self, x: float, y: float) -> int:
        d_right = math.hypot(x - 1.5 * self.r_big, y - self.r_big)
        d_left = math.hypot(x - 0.5 * self.r_big, y - self.r_big)
        is_dot = d_right < self.r_small or d_left < self.r_small
        if is_dot:
            return 2
        is_yin = (
            d_right <= self.r_small
            or (self.r_small < d_left <= 0.5 * self.r_big)
            or (y > self.r_big and d_right > 0.5 * self.r_big)
        )
        return 1 if is_yin else 0

    def sample_balanced(self, n_per_class: int, seed: int = 0):
        if n_per_class <= 0:
            raise ValueError("n_per_class must be positive")
        rng = np.random.default_rng(seed)
        points = [[] for _ in range(3)]
        while min(map(len, points)) < n_per_class:
            x, y = rng.uniform(0.0, 2.0 * self.r_big, size=2)
            if math.hypot(x - self.r_big, y - self.r_big) > self.r_big:
                continue
            label = self.which_class(float(x), float(y))
            if len(points[label]) < n_per_class:
                points[label].append((x, y))
        coords = np.asarray([point for group in points for point in group], dtype=np.float64)
        labels = np.repeat(np.arange(3, dtype=np.int64), n_per_class)
        return coords, labels


def encode_events(point, dt: float, t_min: float = 5.0, t_max: float = 25.0) -> EncodedInput:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (2,) or np.any(point < 0.0) or np.any(point > 1.0):
        raise ValueError("point must have shape (2,) with values in [0, 1]")
    x, y = point
    values = np.asarray([x, y, 1.0 - x, 1.0 - y])
    exact_times = t_min + (1.0 - values) * (t_max - t_min)
    grid_steps = np.floor(exact_times / dt + 0.5).astype(np.int64)
    grid_times = grid_steps * dt
    events = tuple(
        InputSpike(float(exact_times[channel]), int(grid_steps[channel]), channel)
        for channel in range(4)
    )
    return EncodedInput(exact_times, grid_steps, grid_times, events)


def build_demo(dt: float, seed: int = 23) -> TeSNeurons:
    generator = torch.Generator().manual_seed(seed)
    w_in = (
        3.0 + 0.5 * torch.randn((12, 4), generator=generator, dtype=torch.float64)
    ).clamp_min(0.0)
    w_out = (
        1.5 + 0.3 * torch.randn((3, 12), generator=generator, dtype=torch.float64)
    ).clamp_min(0.0)
    return TeSNeurons(4, 12, 3, w_in, w_out, TeSConfig(dt=dt))


def evaluate_health(
    model: TeSNeurons,
    trajectories: tuple[DiscreteForwardResult, ...],
) -> HealthReport:
    thresholds = [event for trajectory in trajectories for event in trajectory.thresholds]
    hidden_thresholds = [event for event in thresholds if event.neuron < model.n_hidden]
    arrivals = [event for trajectory in trajectories for event in trajectory.arrivals]
    delays = [event.continuous_delay for event in hidden_thresholds]
    quantization_errors = [
        event.arrival_time - (event.time + event.continuous_delay)
        for event in hidden_thresholds
    ]
    finite_readout = all(
        torch.isfinite(trajectory.V[:, model.n_hidden :]).all().item()
        and torch.isfinite(trajectory.I[:, model.n_hidden :]).all().item()
        and torch.isfinite(trajectory.Q[:, model.n_hidden :]).all().item()
        for trajectory in trajectories
    )
    maximum_readout_amplitude = max(
        (
            torch.stack(
                (
                    trajectory.V[:, model.n_hidden :].abs().max(),
                    trajectory.I[:, model.n_hidden :].abs().max(),
                    trajectory.Q[:, model.n_hidden :].abs().max(),
                )
            ).max().item()
            for trajectory in trajectories
        ),
        default=0.0,
    )
    monotonic = all(
        all(a.time <= b.time for a, b in zip(trajectory.arrivals, trajectory.arrivals[1:]))
        and all(a.time <= b.time for a, b in zip(trajectory.thresholds, trajectory.thresholds[1:]))
        for trajectory in trajectories
    )
    all_low = bool(delays) and all(delay <= 0.05 * model.config.d_plus for delay in delays)
    all_high = bool(delays) and all(delay >= 0.95 * model.config.d_plus for delay in delays)
    hidden_subtractive_reset = bool(hidden_thresholds) and all(
        event.v_post == event.v_pre - model.config.reset_drop
        for event in hidden_thresholds
    )
    checks = {
        "external_arrival": any(event.kind == "input" for event in arrivals),
        "hidden_threshold": bool(hidden_thresholds),
        "hidden_subtractive_reset": hidden_subtractive_reset,
        "delayed_arrival": any(event.kind == "hidden" for event in arrivals),
        "finite_readout": finite_readout,
        "non_flat_readout": maximum_readout_amplitude > 1e-9,
        "monotonic_event_times": monotonic,
        "positive_crossing_slopes": bool(thresholds) and all(event.slope > 0.0 for event in thresholds),
        "delays_not_saturated": bool(delays) and not all_low and not all_high,
        "resolved_fast_period": all(not trajectory.warnings for trajectory in trajectories),
    }
    warnings = tuple(
        dict.fromkeys(warning for trajectory in trajectories for warning in trajectory.warnings)
    )
    if not checks["delays_not_saturated"]:
        warnings += ("all observed hidden delays are absent or saturated near one bound",)
    metrics = {
        "dt": model.config.dt,
        "external_arrivals": sum(event.kind == "input" for event in arrivals),
        "hidden_thresholds": len(hidden_thresholds),
        "delayed_arrivals": sum(event.kind == "hidden" for event in arrivals),
        "maximum_readout_amplitude": maximum_readout_amplitude,
        "minimum_slope": min((event.slope for event in thresholds), default=None),
        "maximum_slope": max((event.slope for event in thresholds), default=None),
        "minimum_delay": min(delays, default=None),
        "maximum_delay": max(delays, default=None),
        "maximum_delay_quantization_error": max(quantization_errors, default=None),
    }
    return HealthReport(all(checks.values()), checks, metrics, warnings)


def run_demo_data(dt: float = 0.1, seed: int = 23, t_end: float = 50.0) -> DemoRun:
    coords, labels = YinYang().sample_balanced(n_per_class=1, seed=seed)
    encoded = tuple(encode_events(point, dt) for point in coords)
    model = build_demo(dt, seed)
    trajectories = tuple(
        model.forward_discrete(item.events, t_end) for item in encoded
    )
    report = evaluate_health(model, trajectories)
    return DemoRun(model, coords, labels, encoded, trajectories, report)


def _stack_state(trajectory: DiscreteForwardResult) -> np.ndarray:
    return np.stack(
        (trajectory.V.numpy(), trajectory.I.numpy(), trajectory.Q.numpy()),
        axis=-1,
    )


def compare_step_sizes(
    step_sizes: tuple[float, ...],
    seed: int = 23,
) -> StepComparison:
    if len(step_sizes) < 2 or any(a <= b for a, b in zip(step_sizes, step_sizes[1:])):
        raise ValueError("step_sizes must contain at least two values in decreasing order")
    demos = tuple(run_demo_data(dt=dt, seed=seed) for dt in step_sizes)
    reference = demos[-1]
    errors = []
    for coarse in demos[:-1]:
        maximum = 0.0
        for coarse_trajectory, fine_trajectory in zip(
            coarse.trajectories, reference.trajectories, strict=True
        ):
            coarse_state = _stack_state(coarse_trajectory)
            fine_state = _stack_state(fine_trajectory)
            interpolated = np.empty_like(coarse_state)
            for neuron in range(coarse_state.shape[1]):
                for component in range(3):
                    interpolated[:, neuron, component] = np.interp(
                        coarse_trajectory.times.numpy(),
                        fine_trajectory.times.numpy(),
                        fine_state[:, neuron, component],
                    )
            maximum = max(maximum, float(np.max(np.abs(coarse_state - interpolated))))
        errors.append(maximum)
    reference_topology = tuple(
        tuple(event.neuron for event in trajectory.thresholds)
        for trajectory in reference.trajectories
    )
    topology_changed = any(
        tuple(tuple(event.neuron for event in trajectory.thresholds) for trajectory in demo.trajectories)
        != reference_topology
        for demo in demos[:-1]
    )
    finite = tuple(
        all(
            torch.isfinite(trajectory.V).all().item()
            and torch.isfinite(trajectory.I).all().item()
            and torch.isfinite(trajectory.Q).all().item()
            for trajectory in demo.trajectories
        )
        for demo in demos
    )
    return StepComparison(step_sizes, tuple(errors), finite, topology_changed)


def plot_trajectories(demo: DemoRun, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = demo.model.config
    status = "healthy" if demo.report.passed else "diagnostic failure"
    warning_state = "none" if not demo.report.warnings else "present"
    title = (
        f"TeS Yin-Yang forward trajectories — dt={config.dt:g} ms, "
        f"alpha={config.alpha:g} /ms\n"
        f"fast={config.omega_fast:.6f} rad/ms (period {config.fast_period_ms:g} ms), "
        f"slow={config.omega_slow:.6f} rad/ms (period {config.slow_period_ms:g} ms) — "
        f"health={status}, warnings={warning_state}"
    )
    readout_activity = [
        torch.stack(
            (
                trajectory.V[:, demo.model.n_hidden :].abs().max(),
                trajectory.I[:, demo.model.n_hidden :].abs().max(),
                trajectory.Q[:, demo.model.n_hidden :].abs().max(),
            )
        ).max().item()
        for trajectory in demo.trajectories
    ]
    caption = (
        "readout activity max |V,I,Q| by class: "
        + ", ".join(
            f"{int(label)}={activity:.3f}"
            for label, activity in zip(demo.labels, readout_activity, strict=True)
        )
        + " | events: "
        f"input={demo.report.metrics['external_arrivals']}, "
        f"hidden thresholds={demo.report.metrics['hidden_thresholds']}, "
        f"delayed={demo.report.metrics['delayed_arrivals']}"
    )

    fig, axes = plt.subplots(4, 3, figsize=(15, 11), sharex="col")
    for column, (label, encoded, trajectory) in enumerate(
        zip(demo.labels, demo.encoded, demo.trajectories, strict=True)
    ):
        axes[0, column].eventplot(
            [[time] for time in encoded.grid_times],
            lineoffsets=np.arange(4),
            colors="black",
        )
        axes[0, column].set_yticks(range(4), ["x", "y", "1-x", "1-y"])
        hidden = [event for event in trajectory.thresholds if event.neuron < 12]
        axes[1, column].scatter(
            [event.time for event in hidden],
            [event.neuron for event in hidden],
            marker="|",
            color="0.25",
        )
        for event in hidden:
            axes[1, column].plot(
                [event.time, event.arrival_time],
                [event.neuron, event.neuron],
                color="tab:orange",
                alpha=0.45,
            )
        for row, neuron, panel_title in ((2, 0, "fast hidden"), (3, 1, "slow hidden")):
            voltage_axis = axes[row, column]
            iq_axis = voltage_axis.twinx()
            voltage_line = voltage_axis.plot(
                trajectory.times,
                trajectory.V[:, neuron],
                label="V",
                color="tab:blue",
                linewidth=2.2,
            )
            iq_lines = iq_axis.plot(
                trajectory.times,
                trajectory.I[:, neuron],
                label="I",
                color="tab:orange",
            ) + iq_axis.plot(
                trajectory.times,
                trajectory.Q[:, neuron],
                label="Q",
                color="tab:green",
            )
            voltage_axis.axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
            )
            iq_axis.axhline(
                0.0,
                color="0.65",
                linestyle="--",
                linewidth=0.8,
                alpha=0.3,
            )
            voltage_axis.set_ylabel(f"{panel_title}\nV")
            iq_axis.set_ylabel("I, Q", color="0.35")
            iq_axis.tick_params(axis="y", colors="0.35")
            if row == 2 and column == 0:
                voltage_axis.legend(
                    voltage_line + iq_lines,
                    [line.get_label() for line in voltage_line + iq_lines],
                    loc="upper right",
                )
        axes[0, column].set_title(f"class {int(label)}")
        axes[3, column].set_xlabel("time [ms]")
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.92))
    fig.savefig(
        path,
        dpi=150,
        metadata={"Title": title, "Description": caption},
    )
    plt.close(fig)


def write_diagnostics(demo: DemoRun, path: Path) -> None:
    payload = {
        "dt": demo.model.config.dt,
        "alpha_per_ms": demo.model.config.alpha,
        "frequency_periods_ms": [
            demo.model.config.fast_period_ms,
            demo.model.config.slow_period_ms,
        ],
        "four_input_channels": demo.model.n_in == 4,
        "health": asdict(demo.report),
        "samples": [
            {
                "label": int(label),
                "point": point.tolist(),
                "input_times_exact": encoded.exact_times.tolist(),
                "input_times_grid": encoded.grid_times.tolist(),
                "threshold_count": len(trajectory.thresholds),
                "arrival_count": len(trajectory.arrivals),
            }
            for point, label, encoded, trajectory in zip(
                demo.coords,
                demo.labels,
                demo.encoded,
                demo.trajectories,
                strict=True,
            )
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_demo(
    dt: float = 0.1,
    seed: int = 23,
    output_dir: Path | str = Path("Code/output"),
) -> DemoArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo = run_demo_data(dt=dt, seed=seed)
    figure_path = output_dir / "yinyang_tes_trajectories.png"
    diagnostics_path = output_dir / "yinyang_tes_diagnostics.json"
    plot_trajectories(demo, figure_path)
    write_diagnostics(demo, diagnostics_path)
    return DemoArtifacts(demo, figure_path, diagnostics_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TeS Yin-Yang forward demonstration")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=Path("Code/output"))
    args = parser.parse_args()
    artifacts = run_demo(args.dt, args.seed, args.output_dir)
    print(json.dumps(asdict(artifacts.demo.report), indent=2, sort_keys=True))
    print(f"trajectory plot: {artifacts.figure_path}")
    print(f"diagnostics: {artifacts.diagnostics_path}")
    raise SystemExit(0 if artifacts.demo.report.passed else 1)


if __name__ == "__main__":
    main()
