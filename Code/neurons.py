from __future__ import annotations

from dataclasses import dataclass, field, replace
import cmath
import heapq
import math
from typing import Literal, Sequence

import torch

from Code.tes_events import (
    BackwardResult,
    ContinuousArrival,
    ContinuousForwardResult,
    ContinuousOmegaCheckpoint,
    ContinuousOmegaForwardResult,
    ContinuousThreshold,
    DelGradLoss,
    DiscreteArrival,
    DiscreteForwardResult,
    DiscreteThreshold,
    EffectiveReadout,
    InputSpike,
    OmegaBackwardResult,
    StopInfo,
)
from Code.tes_roots import ThresholdContact, first_threshold_contact
from Code.tes_loss import TimingLossName, evaluate_timing_loss


@dataclass(frozen=True)
class TeSConfig:
    dt: float = 0.1
    tau_v: float = 20.0
    alpha: float = 1.0 / 40.0
    threshold: float = 1.0
    reset_drop: float = 1.0
    omega_fast: float = 1.0 / 4.0
    omega_slow: float = 1.0 / 6.0
    d_plus: float = 5.0
    c0: float = 20.0
    c1: float = 2.0
    slope_epsilon: float = 1e-12
    threshold_sensitivity_cap: float = 10.0
    silent_target_gap_ms: float = 2.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.threshold_sensitivity_cap)
            or self.threshold_sensitivity_cap <= 0.0
        ):
            raise ValueError(
                "threshold_sensitivity_cap must be positive and finite"
            )
        if self.omega_fast <= 0.0 or self.omega_slow <= 0.0:
            raise ValueError("oscillator frequencies must be positive")

    @property
    def fast_period_ms(self) -> float:
        return 2.0 * math.pi / self.omega_fast

    @property
    def slow_period_ms(self) -> float:
        return 2.0 * math.pi / self.omega_slow


@dataclass(order=True, frozen=True)
class ScheduledArrival:
    step: int
    sequence: int
    kind: Literal["input", "hidden"] = field(compare=False)
    source: int = field(compare=False)
    exact_time: float = field(compare=False)
    parent_threshold_id: int | None = field(compare=False, default=None)


class TeSNeurons(torch.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, w_in: torch.Tensor,
                 w_out: torch.Tensor, config: TeSConfig) -> None:
        super().__init__()
        if (
            w_in.shape != (n_hidden, n_in)
            or w_out.shape != (n_out, n_hidden)
        ):
            raise ValueError(
                "weight shapes must be "
                f"W_in={(n_hidden, n_in)} and W_out={(n_out, n_hidden)}"
            )
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.n_neurons = n_hidden + n_out
        self.config = config
        self._fast_hidden_count = n_hidden // 2
        hidden_omega = torch.cat(
            (
                torch.full(
                    (self._fast_hidden_count,),
                    config.omega_fast,
                    dtype=torch.float64,
                ),
                torch.full(
                    (n_hidden - self._fast_hidden_count,),
                    config.omega_slow,
                    dtype=torch.float64,
                ),
            )
        )
        out_omega = torch.full(
            (n_out,), config.omega_slow, dtype=torch.float64
        )
        self.register_buffer("omega", torch.cat((hidden_omega, out_omega)))
        self.register_buffer("W_in", w_in.detach().to(dtype=torch.float64).clone())
        self.register_buffer("W_out", w_out.detach().to(dtype=torch.float64).clone())
        self.register_buffer("dCdW_in", torch.zeros_like(self.W_in))
        self.register_buffer("dCdW_out", torch.zeros_like(self.W_out))
        self.register_buffer("state", torch.zeros((self.n_neurons, 3), dtype=torch.float64))
        self._inv_tau_v = 1.0 / config.tau_v
        self._inv_dt = 1.0 / config.dt
        self._hidden_slice = slice(0, n_hidden)
        self._output_slice = slice(n_hidden, self.n_neurons)
        self._frequency_slices = (
            ("fast", slice(0, self._fast_hidden_count)),
            ("slow", slice(self._fast_hidden_count, n_hidden)),
            ("slow", self._output_slice),
        )
        beta_minus_alpha = self._inv_tau_v - config.alpha
        self._flow_coefficients = {
            "fast": self._make_flow_coefficients(config.omega_fast, beta_minus_alpha),
            "slow": self._make_flow_coefficients(config.omega_slow, beta_minus_alpha),
        }
        self._curvature_constants = {
            group: self._make_curvature_constants(omega)
            for group, omega in (
                ("fast", config.omega_fast),
                ("slow", config.omega_slow),
            )
        }
        self._root_panel_width_fast = config.fast_period_ms / 8.0
        self._root_panel_width_slow = config.slow_period_ms / 8.0
        mass = torch.diag(
            torch.tensor([config.tau_v, 1.0, 1.0], dtype=torch.float64)
        )
        self.register_buffer("_adjoint_mass", mass)
        self.register_buffer("_adjoint_mass_inverse", torch.linalg.inv(mass))
        self.register_buffer("_phi_fast", self._analytical_transition("fast", config.dt))
        self.register_buffer("_phi_slow", self._analytical_transition("slow", config.dt))
        self.register_buffer(
            "_psi_fast", self._mass_scaled_transpose(self._phi_fast)
        )
        self.register_buffer(
            "_psi_slow", self._mass_scaled_transpose(self._phi_slow)
        )
        self.reset()

    @property
    def V(self) -> torch.Tensor:
        return self.state[:, 0]

    @property
    def I(self) -> torch.Tensor:
        return self.state[:, 1]

    @property
    def Q(self) -> torch.Tensor:
        return self.state[:, 2]

    def system_matrix(self, omega: float) -> torch.Tensor:
        return torch.tensor(
            [
                [-1.0 / self.config.tau_v, 1.0 / self.config.tau_v, 0.0],
                [0.0, -self.config.alpha, -omega],
                [0.0, omega, -self.config.alpha],
            ],
            dtype=torch.float64,
            device=self.state.device,
        )

    def _transition(self, omega: float, delta: float) -> torch.Tensor:
        return torch.matrix_exp(self.system_matrix(omega) * delta)

    def _make_flow_coefficients(
        self, omega: float, beta_minus_alpha: float
    ) -> tuple[float, float, float, float]:
        denominator = beta_minus_alpha * beta_minus_alpha + omega * omega
        return (
            self._inv_tau_v * beta_minus_alpha / denominator,
            self._inv_tau_v * omega / denominator,
            self._inv_tau_v * omega / denominator,
            -self._inv_tau_v * beta_minus_alpha / denominator,
        )

    def _make_curvature_constants(self, omega: float) -> tuple[float, float]:
        matrix = self.system_matrix(omega)
        acceleration_row_norm = torch.linalg.vector_norm(
            (matrix @ matrix)[0], ord=1
        ).item()
        matrix_norm = torch.linalg.matrix_norm(matrix, ord=float("inf")).item()
        return acceleration_row_norm, matrix_norm

    def _analytical_transition(self, group: str, delta: float) -> torch.Tensor:
        omega = (
            self.config.omega_fast if group == "fast" else self.config.omega_slow
        )
        b_i, b_q, c_i, c_q = self._flow_coefficients[group]
        soma_decay = math.exp(-self._inv_tau_v * delta)
        synapse_decay = math.exp(-self.config.alpha * delta)
        cosine = math.cos(omega * delta)
        sine = math.sin(omega * delta)
        return torch.tensor(
            [
                [
                    soma_decay,
                    -b_i * soma_decay
                    + synapse_decay * (b_i * cosine + c_i * sine),
                    -b_q * soma_decay
                    + synapse_decay * (b_q * cosine + c_q * sine),
                ],
                [0.0, synapse_decay * cosine, -synapse_decay * sine],
                [0.0, synapse_decay * sine, synapse_decay * cosine],
            ],
            dtype=torch.float64,
            device=self.state.device,
        )

    def _mass_scaled_transpose(self, transition: torch.Tensor) -> torch.Tensor:
        return (
            self._adjoint_mass_inverse
            @ transition.T
            @ self._adjoint_mass
        )

    def _flow_single(
        self, state: torch.Tensor, neuron: int, delta: float
    ) -> torch.Tensor:
        group = self._frequency_group(neuron)
        if delta == self.config.dt:
            transition = self._phi_fast if group == "fast" else self._phi_slow
        else:
            transition = self._analytical_transition(group, delta)
        return transition @ state

    def _adjoint_flow_single(
        self, adjoint: torch.Tensor, neuron: int, delta: float
    ) -> torch.Tensor:
        group = self._frequency_group(neuron)
        if delta == self.config.dt:
            transition = self._psi_fast if group == "fast" else self._psi_slow
        else:
            transition = self._mass_scaled_transpose(
                self._analytical_transition(group, delta)
            )
        return transition @ adjoint

    def _validate_learned_omega(self) -> None:
        if not torch.isfinite(self.omega).all() or torch.any(self.omega <= 0.0):
            raise ValueError(
                "learned oscillator frequencies must be positive and finite"
            )

    def _omega_transition(self, neuron: int, delta: float) -> torch.Tensor:
        """Exact transition using the current frequency of one neuron."""
        omega = float(self.omega[neuron])
        beta_minus_alpha = self._inv_tau_v - self.config.alpha
        denominator = beta_minus_alpha * beta_minus_alpha + omega * omega
        b_i = self._inv_tau_v * beta_minus_alpha / denominator
        b_q = self._inv_tau_v * omega / denominator
        c_i = b_q
        c_q = -b_i
        soma_decay = math.exp(-self._inv_tau_v * delta)
        synapse_decay = math.exp(-self.config.alpha * delta)
        cosine = math.cos(omega * delta)
        sine = math.sin(omega * delta)
        return torch.tensor(
            [
                [
                    soma_decay,
                    -b_i * soma_decay
                    + synapse_decay * (b_i * cosine + c_i * sine),
                    -b_q * soma_decay
                    + synapse_decay * (b_q * cosine + c_q * sine),
                ],
                [0.0, synapse_decay * cosine, -synapse_decay * sine],
                [0.0, synapse_decay * sine, synapse_decay * cosine],
            ],
            dtype=torch.float64,
            device=self.state.device,
        )

    def _omega_flow_single(
        self, state: torch.Tensor, neuron: int, delta: float
    ) -> torch.Tensor:
        return self._omega_transition(neuron, delta) @ state

    def _omega_adjoint_flow_single(
        self, adjoint: torch.Tensor, neuron: int, delta: float
    ) -> torch.Tensor:
        transition = self._mass_scaled_transpose(
            self._omega_transition(neuron, delta)
        )
        return transition @ adjoint

    def _omega_voltage_curvature_bound(
        self, state: torch.Tensor, neuron: int, width: float
    ) -> float:
        matrix = self.system_matrix(float(self.omega[neuron]))
        row_norm = torch.linalg.vector_norm(
            (matrix @ matrix)[0], ord=1
        ).item()
        matrix_norm = torch.linalg.matrix_norm(
            matrix, ord=float("inf")
        ).item()
        state_norm = torch.linalg.vector_norm(state, ord=float("inf")).item()
        return row_norm * math.exp(matrix_norm * width) * state_norm

    def _continuous_contact_omega(
        self, state: torch.Tensor, neuron: int, horizon: float
    ) -> tuple[ThresholdContact | None, int]:
        panel_counter = [0]
        crossing = first_threshold_contact(
            state,
            horizon,
            threshold=self.config.threshold,
            flow=lambda value, delta: self._omega_flow_single(
                value, neuron, delta
            ),
            slope=lambda value: (
                float(value[1]) - float(value[0])
            ) * self._inv_tau_v,
            curvature_bound=lambda value, width: self._omega_voltage_curvature_bound(
                value, neuron, width
            ),
            panel_width=(2.0 * math.pi / float(self.omega[neuron])) / 8.0,
            panel_counter=panel_counter,
            slope_epsilon=self.config.slope_epsilon,
        )
        return crossing, panel_counter[0]

    def _continuous_omega_interval_gradient(
        self,
        neuron: int,
        left_i: float,
        left_q: float,
        right_adjoint: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        """Exact integral of lambda_I Q - lambda_Q I on one event interval."""
        if delta == 0.0:
            return torch.zeros((), dtype=torch.float64, device=self.state.device)
        omega = float(self.omega[neuron])
        exponent = complex(-self.config.alpha, omega)
        e = cmath.exp(exponent * delta)
        d = math.exp(-self._inv_tau_v * delta)
        r = complex(self._inv_tau_v - self.config.alpha, omega)
        h = (e - d) / r
        h_omega = 1j * (delta * e - h) / r
        z_left = complex(left_i, left_q)
        z_right = e * z_left
        return (
            delta
            * (
                right_adjoint[1] * z_right.imag
                - right_adjoint[2] * z_right.real
            )
            - right_adjoint[0] * (h_omega * z_left).real
        )

    def _voltage_curvature_bound(
        self, state: torch.Tensor, neuron: int, width: float
    ) -> float:
        row_norm, matrix_norm = self._curvature_constants[
            self._frequency_group(neuron)
        ]
        state_norm = torch.linalg.vector_norm(state, ord=float("inf")).item()
        return row_norm * math.exp(matrix_norm * width) * state_norm

    def _no_future_spike(self, state: torch.Tensor) -> bool:
        radius = torch.hypot(state[1], state[2])
        return max(float(state[0]), float(radius)) < self.config.threshold

    def _certify_silent(
        self, state: torch.Tensor, *, arrival_stream_closed: bool
    ) -> bool:
        if not arrival_stream_closed:
            raise RuntimeError(
                "silence certification requires a closed causal arrival stream"
            )
        return self._no_future_spike(state)

    def flow_state(self, state: torch.Tensor, delta: float) -> torch.Tensor:
        result = torch.empty_like(state)
        for group, neuron_slice in self._frequency_slices:
            if neuron_slice.start == neuron_slice.stop:
                continue
            if delta == self.config.dt:
                transition = self._phi_fast if group == "fast" else self._phi_slow
            else:
                transition = self._analytical_transition(group, delta)
            result[neuron_slice] = state[neuron_slice] @ transition.T
        return result

    def _frequency_group(self, neuron: int) -> Literal["fast", "slow"]:
        return "fast" if neuron < self._fast_hidden_count else "slow"

    def _configured_omega(self, neuron: int) -> float:
        return (
            self.config.omega_fast
            if self._frequency_group(neuron) == "fast"
            else self.config.omega_slow
        )

    def reset(self) -> None:
        self.state.zero_()
        self.dCdW_in.zero_()
        self.dCdW_out.zero_()
        self._queue: list[ScheduledArrival] = []
        self._sequence = 0
        self.arrival_records: list[DiscreteArrival] = []
        self.threshold_events: list[DiscreteThreshold] = []

    def delay_from_slope(self, slope: float) -> float:
        argument = self.config.c1 - self.config.c0 * slope
        return self.config.d_plus * torch.sigmoid(
            torch.tensor(argument, dtype=torch.float64)
        ).item()

    def _threshold_sensitivity(self, slope: float) -> float:
        value = torch.tensor(
            slope, dtype=torch.float64, device=self.state.device
        )
        return float(
            torch.clamp(
                self._inv_tau_v / value,
                max=self.config.threshold_sensitivity_cap,
            )
        )

    def _threshold_sensitivity_is_capped(self, slope: float) -> bool:
        value = torch.tensor(
            slope, dtype=torch.float64, device=self.state.device
        )
        return bool(
            self._inv_tau_v / value > self.config.threshold_sensitivity_cap
        )

    def quantize_delay(self, delay: float) -> int:
        return math.ceil(delay / self.config.dt)

    def _push_arrival(
        self,
        step: int,
        kind: Literal["input", "hidden"],
        source: int,
        exact_time: float,
        parent_threshold_id: int | None = None,
    ) -> None:
        heapq.heappush(
            self._queue,
            ScheduledArrival(
                step, self._sequence, kind, source, exact_time, parent_threshold_id
            ),
        )
        self._sequence += 1

    def _enqueue_hidden(
        self,
        source: int,
        spike_step: int,
        continuous_delay: float,
        spike_time: float | None = None,
        parent_threshold_id: int | None = None,
    ) -> tuple[int, int]:
        delay_steps = self.quantize_delay(continuous_delay)
        arrival_step = spike_step + delay_steps
        emitted_at = spike_step * self.config.dt if spike_time is None else spike_time
        self._push_arrival(
            arrival_step,
            "hidden",
            source,
            emitted_at + continuous_delay,
            parent_threshold_id,
        )
        return delay_steps, arrival_step

    def enqueue_input(self, spike: InputSpike) -> None:
        self._push_arrival(spike.grid_step, "input", spike.channel, spike.exact_time)

    def apply_due_arrivals(self, step: int) -> None:
        due = []
        while self._queue and self._queue[0].step == step:
            due.append(heapq.heappop(self._queue))
        input_sources = [item.source for item in due if item.kind == "input"]
        hidden_sources = [item.source for item in due if item.kind == "hidden"]
        if input_sources:
            self.Q[: self.n_hidden] -= self.W_in[:, input_sources].sum(dim=1)
        if hidden_sources:
            self.Q[self.n_hidden :] -= self.W_out[:, hidden_sources].sum(dim=1)
        for item in due:
            self.arrival_records.append(
                DiscreteArrival(
                    event_id=len(self.arrival_records),
                    time=step * self.config.dt,
                    step=step,
                    kind=item.kind,
                    source=item.source,
                    exact_time=item.exact_time,
                    parent_threshold_id=item.parent_threshold_id,
                )
            )

    def _emit_discrete_thresholds(
        self,
        neurons: Sequence[int],
        *,
        event_time: float,
        event_step: int,
    ) -> None:
        for neuron in neurons:
            event_id = len(self.threshold_events)
            v_pre = self.V[neuron].item()
            i_pre = self.I[neuron].item()
            q_pre = self.Q[neuron].item()
            slope = ((-self.V[neuron] + self.I[neuron]) / self.config.tau_v).item()
            self.V[neuron] -= self.config.reset_drop
            continuous_delay = self.delay_from_slope(slope)
            arrival_time = event_time + continuous_delay
            delay_steps = None
            if neuron < self.n_hidden:
                delay_steps, _ = self._enqueue_hidden(
                    neuron,
                    event_step,
                    continuous_delay,
                    spike_time=event_time,
                    parent_threshold_id=event_id,
                )
            self.threshold_events.append(
                DiscreteThreshold(
                    event_id=event_id,
                    time=event_time,
                    step=event_step,
                    neuron=neuron,
                    slope=slope,
                    v_pre=v_pre,
                    i_pre=i_pre,
                    q_pre=q_pre,
                    v_post=self.V[neuron].item(),
                    continuous_delay=continuous_delay,
                    delay_steps=delay_steps,
                    arrival_time=arrival_time,
                )
            )

    def step(self, step: int, delta: float | None = None) -> torch.Tensor:
        delta = self.config.dt if delta is None else delta
        boundary_indices = torch.nonzero(
            self.V == self.config.threshold
        ).flatten().tolist()
        self._emit_discrete_thresholds(
            boundary_indices,
            event_time=step * self.config.dt,
            event_step=step,
        )
        self.apply_due_arrivals(step)
        post_arrival_state = self.state.clone()
        v_start = self.V.clone()
        self.state.copy_(self.flow_state(self.state, delta))
        crossing = (v_start < self.config.threshold) & (self.V >= self.config.threshold)
        crossing_indices = torch.nonzero(crossing).flatten().tolist()
        self._emit_discrete_thresholds(
            crossing_indices,
            event_time=step * self.config.dt + delta,
            event_step=step + 1,
        )
        return post_arrival_state

    def _omega_transition_batch(self, delta: float) -> torch.Tensor:
        return torch.stack(
            tuple(
                self._omega_transition(neuron, delta)
                for neuron in range(self.n_neurons)
            )
        )

    def _step_omega(
        self,
        step: int,
        delta: float,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        boundary_indices = torch.nonzero(
            self.V == self.config.threshold
        ).flatten().tolist()
        self._emit_discrete_thresholds(
            boundary_indices,
            event_time=step * self.config.dt,
            event_step=step,
        )
        self.apply_due_arrivals(step)
        post_arrival_state = self.state.clone()
        v_start = self.V.clone()
        flowed = torch.bmm(
            transitions, self.state.unsqueeze(-1)
        ).squeeze(-1)
        self.state.copy_(flowed)
        crossing = (v_start < self.config.threshold) & (
            self.V >= self.config.threshold
        )
        self._emit_discrete_thresholds(
            torch.nonzero(crossing).flatten().tolist(),
            event_time=step * self.config.dt + delta,
            event_step=step + 1,
        )
        return post_arrival_state

    def _continuous_contact(
        self, state: torch.Tensor, neuron: int, horizon: float
    ) -> tuple[ThresholdContact | None, int]:
        panel_width = (
            self._root_panel_width_fast
            if self._frequency_group(neuron) == "fast"
            else self._root_panel_width_slow
        )
        panel_counter = [0]
        crossing = first_threshold_contact(
            state,
            horizon,
            threshold=self.config.threshold,
            flow=lambda value, delta: self._flow_single(value, neuron, delta),
            slope=lambda value: (
                float(value[1]) - float(value[0])
            ) * self._inv_tau_v,
            curvature_bound=lambda value, width: self._voltage_curvature_bound(
                value, neuron, width
            ),
            panel_width=panel_width,
            panel_counter=panel_counter,
            slope_epsilon=self.config.slope_epsilon,
        )
        return crossing, panel_counter[0]

    def _simulate_continuous_neuron(
        self,
        *,
        layer: Literal["hidden", "output"],
        neuron: int,
        global_neuron: int,
        weighted_arrivals: Sequence[tuple[float, int, float]],
        t_end: float,
        first_event_id: int,
    ) -> tuple[list[ContinuousThreshold], int, int]:
        grouped: list[tuple[float, float]] = []
        for time, sequence, weight in sorted(
            weighted_arrivals, key=lambda item: (item[0], item[1])
        ):
            if time > t_end:
                break
            if grouped and time == grouped[-1][0]:
                grouped[-1] = (time, grouped[-1][1] + weight)
            else:
                grouped.append((time, weight))

        state = torch.zeros(3, dtype=torch.float64, device=self.state.device)
        current_time = 0.0
        events: list[ContinuousThreshold] = []
        flow_intervals = 0
        root_panels = 0

        def advance(boundary: float) -> None:
            nonlocal state, current_time, flow_intervals, root_panels
            while current_time < boundary:
                crossing, visited_panels = self._continuous_contact(
                    state, global_neuron, boundary - current_time
                )
                flow_intervals += 1
                root_panels += visited_panels
                if crossing is None:
                    state = self._flow_single(
                        state, global_neuron, boundary - current_time
                    )
                    current_time = boundary
                    return

                absolute_time = current_time + crossing.offset
                state_pre = crossing.state
                slope = crossing.slope
                omega = self._configured_omega(global_neuron)
                acceleration = (
                    - slope
                    - self.config.alpha * float(state_pre[1])
                    - omega * float(state_pre[2])
                ) * self._inv_tau_v
                state = state_pre.clone()
                state[0] -= self.config.reset_drop
                delay = self.delay_from_slope(slope)
                arrival_time = absolute_time + delay
                events.append(
                    ContinuousThreshold(
                        event_id=first_event_id + len(events),
                        layer=layer,
                        neuron=neuron,
                        time=absolute_time,
                        slope=slope,
                        acceleration_pre=acceleration,
                        v_pre=float(state_pre[0]),
                        i_pre=float(state_pre[1]),
                        q_pre=float(state_pre[2]),
                        v_post=float(state[0]),
                        continuous_delay=delay,
                        arrival_time=arrival_time,
                    )
                )
                current_time = absolute_time

        for arrival_time, summed_weight in grouped:
            advance(arrival_time)
            state[2] -= summed_weight
        advance(t_end)
        return events, flow_intervals, root_panels

    def _simulate_continuous_neuron_omega(
        self,
        *,
        layer: Literal["hidden", "output"],
        neuron: int,
        global_neuron: int,
        weighted_arrivals: Sequence[tuple[float, int, float]],
        t_end: float,
        first_event_id: int,
    ) -> tuple[
        list[ContinuousThreshold],
        int,
        int,
        list[ContinuousOmegaCheckpoint],
    ]:
        grouped: list[tuple[float, float]] = []
        for time, sequence, weight in sorted(
            weighted_arrivals, key=lambda item: (item[0], item[1])
        ):
            if time > t_end:
                break
            if grouped and time == grouped[-1][0]:
                grouped[-1] = (time, grouped[-1][1] + weight)
            else:
                grouped.append((time, weight))

        state = torch.zeros(3, dtype=torch.float64, device=self.state.device)
        current_time = 0.0
        events: list[ContinuousThreshold] = []
        checkpoints: list[ContinuousOmegaCheckpoint] = []
        flow_intervals = 0
        root_panels = 0

        def record_checkpoint() -> None:
            checkpoint = ContinuousOmegaCheckpoint(
                neuron=global_neuron,
                time=current_time,
                i=float(state[1]),
                q=float(state[2]),
            )
            if checkpoints and checkpoints[-1].time == current_time:
                checkpoints[-1] = checkpoint
            else:
                checkpoints.append(checkpoint)

        def advance(boundary: float) -> None:
            nonlocal state, current_time, flow_intervals, root_panels
            while current_time < boundary:
                crossing, visited_panels = self._continuous_contact_omega(
                    state, global_neuron, boundary - current_time
                )
                flow_intervals += 1
                root_panels += visited_panels
                if crossing is None:
                    state = self._omega_flow_single(
                        state, global_neuron, boundary - current_time
                    )
                    current_time = boundary
                    return

                absolute_time = current_time + crossing.offset
                state_pre = crossing.state
                slope = crossing.slope
                omega = float(self.omega[global_neuron])
                acceleration = (
                    -slope
                    - self.config.alpha * float(state_pre[1])
                    - omega * float(state_pre[2])
                ) * self._inv_tau_v
                state = state_pre.clone()
                state[0] -= self.config.reset_drop
                delay = self.delay_from_slope(slope)
                events.append(
                    ContinuousThreshold(
                        event_id=first_event_id + len(events),
                        layer=layer,
                        neuron=neuron,
                        time=absolute_time,
                        slope=slope,
                        acceleration_pre=acceleration,
                        v_pre=float(state_pre[0]),
                        i_pre=float(state_pre[1]),
                        q_pre=float(state_pre[2]),
                        v_post=float(state[0]),
                        continuous_delay=delay,
                        arrival_time=absolute_time + delay,
                    )
                )
                current_time = absolute_time
                record_checkpoint()

        record_checkpoint()
        for arrival_time, summed_weight in grouped:
            advance(arrival_time)
            state[2] -= summed_weight
            record_checkpoint()
        advance(t_end)
        return events, flow_intervals, root_panels, checkpoints

    def _advance_continuous_stream(
        self,
        *,
        layer: Literal["hidden", "output"],
        neuron: int,
        global_neuron: int,
        state: torch.Tensor,
        start_time: float,
        boundary: float,
        weighted_arrivals: Sequence[tuple[float, int, float]],
        first_event_id: int,
        stop_after_first: bool = False,
    ) -> tuple[
        torch.Tensor,
        float,
        torch.Tensor,
        list[ContinuousThreshold],
        int,
        int,
    ]:
        grouped: list[tuple[float, float]] = []
        for time, sequence, weight in sorted(
            weighted_arrivals, key=lambda item: (item[0], item[1])
        ):
            if grouped and time == grouped[-1][0]:
                grouped[-1] = (time, grouped[-1][1] + weight)
            else:
                grouped.append((time, weight))

        value = state.clone()
        current_time = start_time
        events: list[ContinuousThreshold] = []
        flow_intervals = 0
        root_panels = 0

        def advance(
            target: float, *, commit_tail: bool, stop_on_crossing: bool
        ) -> tuple[bool, torch.Tensor]:
            nonlocal value, current_time, flow_intervals, root_panels
            while current_time < target:
                crossing, visited_panels = self._continuous_contact(
                    value, global_neuron, target - current_time
                )
                flow_intervals += 1
                root_panels += visited_panels
                if crossing is None:
                    target_state = self._flow_single(
                        value, global_neuron, target - current_time
                    )
                    if commit_tail:
                        value = target_state
                        current_time = target
                    return False, target_state

                absolute_time = current_time + crossing.offset
                state_pre = crossing.state
                slope = crossing.slope
                omega = self._configured_omega(global_neuron)
                acceleration = (
                    -slope
                    - self.config.alpha * float(state_pre[1])
                    - omega * float(state_pre[2])
                ) * self._inv_tau_v
                value = state_pre.clone()
                value[0] -= self.config.reset_drop
                delay = self.delay_from_slope(slope)
                arrival_time = absolute_time + delay
                events.append(
                    ContinuousThreshold(
                        event_id=first_event_id + len(events),
                        layer=layer,
                        neuron=neuron,
                        time=absolute_time,
                        slope=slope,
                        acceleration_pre=acceleration,
                        v_pre=float(state_pre[0]),
                        i_pre=float(state_pre[1]),
                        q_pre=float(state_pre[2]),
                        v_post=float(value[0]),
                        continuous_delay=delay,
                        arrival_time=arrival_time,
                    )
                )
                current_time = absolute_time
                if stop_on_crossing:
                    return True, value
            return False, value

        for arrival_time, summed_weight in grouped:
            stopped, frontier_state = advance(
                arrival_time,
                commit_tail=True,
                stop_on_crossing=stop_after_first,
            )
            if stopped:
                return (
                    value,
                    current_time,
                    frontier_state,
                    events,
                    flow_intervals,
                    root_panels,
                )
            value[2] -= summed_weight
        _, frontier_state = advance(
            boundary, commit_tail=True, stop_on_crossing=True
        )
        return (
            value,
            current_time,
            frontier_state,
            events,
            flow_intervals,
            root_panels,
        )

    def _forward_continuous_early(
        self,
        inputs: Sequence[InputSpike],
        t_end: float,
    ) -> ContinuousForwardResult:
        delivered = [
            (sequence, spike)
            for sequence, spike in enumerate(inputs)
            if spike.exact_time <= t_end
        ]
        delivered.sort(key=lambda item: (item[1].exact_time, item[0]))
        all_inputs = tuple(
            ContinuousArrival(
                event_id=index,
                sequence=sequence,
                time=float(spike.exact_time),
                kind="input",
                source=spike.channel,
                parent_threshold_id=None,
            )
            for index, (sequence, spike) in enumerate(delivered)
        )
        hidden_states = torch.zeros(
            (self.n_hidden, 3), dtype=torch.float64, device=self.state.device
        )
        output_states = torch.zeros(
            (self.n_out, 3), dtype=torch.float64, device=self.state.device
        )
        hidden_times = [0.0] * self.n_hidden
        output_times = [0.0] * self.n_out
        hidden_frontier_states = hidden_states.clone()
        output_frontier_states = output_states.clone()
        hidden_closed = [False] * self.n_hidden
        resolved_outputs = [False] * self.n_out
        hidden_events: list[ContinuousThreshold] = []
        output_events: list[ContinuousThreshold] = []
        scheduled_arrivals: list[ContinuousArrival] = []
        first_readout_events: list[ContinuousThreshold | None] = [
            None
        ] * self.n_out
        next_event_id = 0
        next_arrival_sequence = 0
        hidden_intervals = 0
        output_intervals = 0
        root_panels = 0
        frontier = 0.0

        def next_panel_boundary(
            committed_time: float, panel_width: float, after: float
        ) -> float:
            panels = math.floor((after - committed_time) / panel_width) + 1
            candidate = committed_time + panels * panel_width
            if candidate <= after:
                candidate += panel_width
            return candidate

        initial_inputs = tuple(
            arrival for arrival in all_inputs if arrival.time == 0.0
        )
        for hidden in range(self.n_hidden):
            if initial_inputs:
                hidden_states[hidden, 2] -= sum(
                    self.W_in[hidden, arrival.source]
                    for arrival in initial_inputs
                )

        future_inputs = any(arrival.time > frontier for arrival in all_inputs)
        if not future_inputs:
            for hidden in range(self.n_hidden):
                hidden_closed[hidden] = self._certify_silent(
                    hidden_states[hidden], arrival_stream_closed=not future_inputs
                )
        if all(hidden_closed) and all(
            self._certify_silent(
                output_states[output],
                arrival_stream_closed=all(hidden_closed),
            )
            for output in range(self.n_out)
        ):
            actual_t_end = 0.0
            stop_reason = "all_readouts_silent"
        else:
            actual_t_end = t_end
            stop_reason = "requested_horizon"
            while frontier < t_end:
                hidden_panel_frontiers = {
                    hidden: next_panel_boundary(
                        hidden_times[hidden],
                        (
                            self._root_panel_width_fast
                            if self._frequency_group(hidden) == "fast"
                            else self._root_panel_width_slow
                        ),
                        frontier,
                    )
                    for hidden in range(self.n_hidden)
                    if not hidden_closed[hidden]
                }
                output_panel_frontiers = {
                    output: next_panel_boundary(
                        output_times[output],
                        (
                            self._root_panel_width_fast
                            if self._frequency_group(self.n_hidden + output)
                            == "fast"
                            else self._root_panel_width_slow
                        ),
                        frontier,
                    )
                    for output in range(self.n_out)
                    if not resolved_outputs[output]
                }
                candidates = [t_end]
                candidates.extend(hidden_panel_frontiers.values())
                candidates.extend(output_panel_frontiers.values())
                candidates.extend(
                    arrival.time
                    for arrival in all_inputs
                    if arrival.time > frontier
                )
                candidates.extend(
                    arrival.time
                    for arrival in scheduled_arrivals
                    if arrival.time > frontier
                )
                next_frontier = min(candidates)

                for hidden in range(self.n_hidden):
                    if hidden_closed[hidden]:
                        continue
                    new_inputs = tuple(
                        arrival
                        for arrival in all_inputs
                        if hidden_times[hidden]
                        < arrival.time
                        <= next_frontier
                    )
                    weighted = tuple(
                        (
                            arrival.time,
                            arrival.sequence,
                            float(self.W_in[hidden, arrival.source]),
                        )
                        for arrival in new_inputs
                    )
                    (
                        state,
                        committed_time,
                        frontier_state,
                        events,
                        intervals,
                        panels,
                    ) = (
                        self._advance_continuous_stream(
                            layer="hidden",
                            neuron=hidden,
                            global_neuron=hidden,
                            state=hidden_states[hidden],
                            start_time=hidden_times[hidden],
                            boundary=next_frontier,
                            weighted_arrivals=weighted,
                            first_event_id=next_event_id,
                        )
                    )
                    hidden_states[hidden] = state
                    hidden_times[hidden] = committed_time
                    hidden_frontier_states[hidden] = frontier_state
                    hidden_events.extend(events)
                    next_event_id += len(events)
                    hidden_intervals += intervals
                    root_panels += panels
                    for event in events:
                        if event.arrival_time <= t_end:
                            scheduled_arrivals.append(
                                ContinuousArrival(
                                    event_id=next_arrival_sequence,
                                    sequence=next_arrival_sequence,
                                    time=float(event.arrival_time),
                                    kind="hidden",
                                    source=event.neuron,
                                    parent_threshold_id=event.event_id,
                                )
                            )
                            next_arrival_sequence += 1

                no_more_inputs = not any(
                    arrival.time > next_frontier for arrival in all_inputs
                )
                if no_more_inputs:
                    for hidden in range(self.n_hidden):
                        if not hidden_closed[hidden]:
                            hidden_closed[hidden] = self._certify_silent(
                                hidden_frontier_states[hidden],
                                arrival_stream_closed=no_more_inputs,
                            )

                new_hidden_arrivals = tuple(
                    arrival
                    for arrival in scheduled_arrivals
                    if arrival.time <= next_frontier
                )
                for output in range(self.n_out):
                    if resolved_outputs[output]:
                        continue
                    output_arrivals = tuple(
                        arrival
                        for arrival in new_hidden_arrivals
                        if arrival.time > output_times[output]
                    )
                    weighted = tuple(
                        (
                            arrival.time,
                            arrival.sequence,
                            float(self.W_out[output, arrival.source]),
                        )
                        for arrival in output_arrivals
                    )
                    (
                        state,
                        committed_time,
                        frontier_state,
                        events,
                        intervals,
                        panels,
                    ) = (
                        self._advance_continuous_stream(
                            layer="output",
                            neuron=output,
                            global_neuron=self.n_hidden + output,
                            state=output_states[output],
                            start_time=output_times[output],
                            boundary=next_frontier,
                            weighted_arrivals=weighted,
                            first_event_id=next_event_id,
                            stop_after_first=True,
                        )
                    )
                    output_states[output] = state
                    output_times[output] = committed_time
                    output_frontier_states[output] = frontier_state
                    output_intervals += intervals
                    root_panels += panels
                    if events:
                        event = events[0]
                        output_events.append(event)
                        first_readout_events[output] = event
                        resolved_outputs[output] = True
                        next_event_id += 1

                if all(resolved_outputs):
                    actual_t_end = max(
                        event.time for event in first_readout_events
                    )
                    stop_reason = "all_readouts_physical"
                    break

                hidden_streams_closed = all(hidden_closed)
                pending_arrivals = any(
                    arrival.time > next_frontier
                    for arrival in scheduled_arrivals
                )
                if hidden_streams_closed and not pending_arrivals:
                    unresolved_silent = all(
                        resolved_outputs[output]
                        or self._certify_silent(
                            output_frontier_states[output],
                            arrival_stream_closed=(
                                hidden_streams_closed and not pending_arrivals
                            ),
                        )
                        for output in range(self.n_out)
                    )
                    if unresolved_silent:
                        actual_t_end = next_frontier
                        stop_reason = (
                            "all_readouts_resolved"
                            if any(resolved_outputs)
                            else "all_readouts_silent"
                        )
                        break
                frontier = next_frontier

        retained_inputs = tuple(
            replace(arrival, event_id=index)
            for index, arrival in enumerate(
                arrival for arrival in all_inputs if arrival.time <= actual_t_end
            )
        )
        retained_hidden = [
            event for event in hidden_events if event.time <= actual_t_end
        ]
        retained_outputs = [
            event for event in output_events if event.time <= actual_t_end
        ]
        ordered = sorted(
            retained_hidden + retained_outputs,
            key=lambda event: (
                event.time,
                0 if event.layer == "hidden" else 1,
                event.neuron,
                event.event_id,
            ),
        )
        event_id_map = {
            event.event_id: final_id for final_id, event in enumerate(ordered)
        }
        finalized = {
            event.event_id: replace(event, event_id=event_id_map[event.event_id])
            for event in ordered
        }
        hidden_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in retained_hidden),
                key=lambda event: event.event_id,
            )
        )
        output_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in retained_outputs),
                key=lambda event: event.event_id,
            )
        )
        hidden_arrivals = tuple(
            replace(
                arrival,
                event_id=index,
                sequence=index,
                parent_threshold_id=event_id_map[arrival.parent_threshold_id],
            )
            for index, arrival in enumerate(
                sorted(
                    (
                        arrival
                        for arrival in scheduled_arrivals
                        if arrival.time <= actual_t_end
                    ),
                    key=lambda arrival: (
                        arrival.time,
                        arrival.source,
                        arrival.parent_threshold_id,
                    ),
                )
            )
        )
        first_readouts: list[tuple[int, float] | None] = [None] * self.n_out
        for event in output_spikes:
            if first_readouts[event.neuron] is None:
                first_readouts[event.neuron] = (
                    event.event_id,
                    float(event.arrival_time),
                )
        return ContinuousForwardResult(
            input_arrivals=retained_inputs,
            hidden_spikes=hidden_spikes,
            hidden_arrivals=hidden_arrivals,
            output_spikes=output_spikes,
            readouts=self._effective_readouts(first_readouts, t_end),
            stop=StopInfo(
                reason=stop_reason,
                requested_t_end=t_end,
                actual_t_end=actual_t_end,
                hidden_flow_intervals=hidden_intervals,
                output_flow_intervals=output_intervals,
                root_panels=root_panels,
            ),
        )

    def forward_continuous(
        self,
        inputs: Sequence[InputSpike],
        t_end: float,
        *,
        early_stop: bool = False,
    ) -> ContinuousForwardResult:
        if early_stop:
            return self._forward_continuous_early(inputs, t_end)
        delivered = [
            (sequence, spike)
            for sequence, spike in enumerate(inputs)
            if spike.exact_time <= t_end
        ]
        delivered.sort(key=lambda item: (item[1].exact_time, item[0]))
        input_arrivals = tuple(
            ContinuousArrival(
                event_id=index,
                sequence=sequence,
                time=float(spike.exact_time),
                kind="input",
                source=spike.channel,
                parent_threshold_id=None,
            )
            for index, (sequence, spike) in enumerate(delivered)
        )

        hidden_events: list[ContinuousThreshold] = []
        hidden_intervals = 0
        root_panels = 0
        next_event_id = 0
        #iterating through hidden neurons and simulate them separately
        for hidden in range(self.n_hidden):  
            weighted = tuple(
                (
                    arrival.time,
                    arrival.sequence,
                    float(self.W_in[hidden, arrival.source]),
                )
                for arrival in input_arrivals
            )
            events, intervals, panels = self._simulate_continuous_neuron(
                layer="hidden",
                neuron=hidden,
                global_neuron=hidden,
                weighted_arrivals=weighted,
                t_end=t_end,
                first_event_id=next_event_id,
            )
            hidden_events.extend(events)
            next_event_id += len(events)
            hidden_intervals += intervals
            root_panels += panels

        temporary_hidden_arrivals = [
            ContinuousArrival(
                event_id=index,
                sequence=index,
                time=float(event.arrival_time),
                kind="hidden",
                source=event.neuron,
                parent_threshold_id=event.event_id,
            )
            for index, event in enumerate(hidden_events)
            if event.arrival_time is not None and event.arrival_time <= t_end
        ]
        temporary_hidden_arrivals.sort(
            key=lambda arrival: (
                arrival.time,
                arrival.source,
                arrival.parent_threshold_id,
            )
        )

        output_events: list[ContinuousThreshold] = []
        output_intervals = 0
        for output in range(self.n_out):
            weighted = tuple(
                (
                    arrival.time,
                    arrival.sequence,
                    float(self.W_out[output, arrival.source]),
                )
                for arrival in temporary_hidden_arrivals
            )
            events, intervals, panels = self._simulate_continuous_neuron(
                layer="output",
                neuron=output,
                global_neuron=self.n_hidden + output,
                weighted_arrivals=weighted,
                t_end=t_end,
                first_event_id=next_event_id,
            )
            output_events.extend(events)
            next_event_id += len(events)
            output_intervals += intervals
            root_panels += panels

        ordered = sorted(
            hidden_events + output_events,
            key=lambda event: (
                event.time,
                0 if event.layer == "hidden" else 1,
                event.neuron,
                event.event_id,
            ),
        )
        event_id_map = {
            event.event_id: final_id for final_id, event in enumerate(ordered)
        }
        finalized = {
            event.event_id: replace(event, event_id=event_id_map[event.event_id])
            for event in ordered
        }
        hidden_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in hidden_events),
                key=lambda event: event.event_id,
            )
        )
        output_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in output_events),
                key=lambda event: event.event_id,
            )
        )
        hidden_arrivals = tuple(
            replace(
                arrival,
                event_id=index,
                sequence=index,
                parent_threshold_id=event_id_map[arrival.parent_threshold_id],
            )
            for index, arrival in enumerate(temporary_hidden_arrivals)
        )

        first_readouts: list[tuple[int, float] | None] = [None] * self.n_out
        for event in output_spikes:
            if first_readouts[event.neuron] is None:
                first_readouts[event.neuron] = (
                    event.event_id,
                    float(event.arrival_time),
                )
        return ContinuousForwardResult(
            input_arrivals=input_arrivals,
            hidden_spikes=hidden_spikes,
            hidden_arrivals=hidden_arrivals,
            output_spikes=output_spikes,
            readouts=self._effective_readouts(first_readouts, t_end),
            stop=StopInfo(
                reason="requested_horizon",
                requested_t_end=t_end,
                actual_t_end=t_end,
                hidden_flow_intervals=hidden_intervals,
                output_flow_intervals=output_intervals,
                root_panels=root_panels,
            ),
        )

    def forward_continuous_omega(
        self,
        inputs: Sequence[InputSpike],
        t_end: float,
        *,
        early_stop: bool = False,
    ) -> ContinuousOmegaForwardResult:
        """Continuous forward pass with sparse states for learning each omega."""
        if early_stop:
            raise ValueError(
                "forward_continuous_omega requires early_stop=False so its "
                "frequency-gradient tape has a fixed requested horizon"
            )
        self._validate_learned_omega()
        delivered = [
            (sequence, spike)
            for sequence, spike in enumerate(inputs)
            if spike.exact_time <= t_end
        ]
        delivered.sort(key=lambda item: (item[1].exact_time, item[0]))
        input_arrivals = tuple(
            ContinuousArrival(
                event_id=index,
                sequence=sequence,
                time=float(spike.exact_time),
                kind="input",
                source=spike.channel,
                parent_threshold_id=None,
            )
            for index, (sequence, spike) in enumerate(delivered)
        )

        hidden_events: list[ContinuousThreshold] = []
        checkpoints: list[ContinuousOmegaCheckpoint] = []
        hidden_intervals = 0
        root_panels = 0
        next_event_id = 0
        for hidden in range(self.n_hidden):
            weighted = tuple(
                (
                    arrival.time,
                    arrival.sequence,
                    float(self.W_in[hidden, arrival.source]),
                )
                for arrival in input_arrivals
            )
            events, intervals, panels, local_checkpoints = (
                self._simulate_continuous_neuron_omega(
                    layer="hidden",
                    neuron=hidden,
                    global_neuron=hidden,
                    weighted_arrivals=weighted,
                    t_end=t_end,
                    first_event_id=next_event_id,
                )
            )
            hidden_events.extend(events)
            checkpoints.extend(local_checkpoints)
            next_event_id += len(events)
            hidden_intervals += intervals
            root_panels += panels

        temporary_hidden_arrivals = [
            ContinuousArrival(
                event_id=index,
                sequence=index,
                time=float(event.arrival_time),
                kind="hidden",
                source=event.neuron,
                parent_threshold_id=event.event_id,
            )
            for index, event in enumerate(hidden_events)
            if event.arrival_time is not None and event.arrival_time <= t_end
        ]
        temporary_hidden_arrivals.sort(
            key=lambda arrival: (
                arrival.time,
                arrival.source,
                arrival.parent_threshold_id,
            )
        )

        output_events: list[ContinuousThreshold] = []
        output_intervals = 0
        for output in range(self.n_out):
            weighted = tuple(
                (
                    arrival.time,
                    arrival.sequence,
                    float(self.W_out[output, arrival.source]),
                )
                for arrival in temporary_hidden_arrivals
            )
            events, intervals, panels, local_checkpoints = (
                self._simulate_continuous_neuron_omega(
                    layer="output",
                    neuron=output,
                    global_neuron=self.n_hidden + output,
                    weighted_arrivals=weighted,
                    t_end=t_end,
                    first_event_id=next_event_id,
                )
            )
            output_events.extend(events)
            checkpoints.extend(local_checkpoints)
            next_event_id += len(events)
            output_intervals += intervals
            root_panels += panels

        ordered = sorted(
            hidden_events + output_events,
            key=lambda event: (
                event.time,
                0 if event.layer == "hidden" else 1,
                event.neuron,
                event.event_id,
            ),
        )
        event_id_map = {
            event.event_id: final_id for final_id, event in enumerate(ordered)
        }
        finalized = {
            event.event_id: replace(event, event_id=event_id_map[event.event_id])
            for event in ordered
        }
        hidden_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in hidden_events),
                key=lambda event: event.event_id,
            )
        )
        output_spikes = tuple(
            sorted(
                (finalized[event.event_id] for event in output_events),
                key=lambda event: event.event_id,
            )
        )
        hidden_arrivals = tuple(
            replace(
                arrival,
                event_id=index,
                sequence=index,
                parent_threshold_id=event_id_map[arrival.parent_threshold_id],
            )
            for index, arrival in enumerate(temporary_hidden_arrivals)
        )
        first_readouts: list[tuple[int, float] | None] = [None] * self.n_out
        for event in output_spikes:
            if first_readouts[event.neuron] is None:
                first_readouts[event.neuron] = (
                    event.event_id,
                    float(event.arrival_time),
                )

        return ContinuousOmegaForwardResult(
            input_arrivals=input_arrivals,
            hidden_spikes=hidden_spikes,
            hidden_arrivals=hidden_arrivals,
            output_spikes=output_spikes,
            readouts=self._effective_readouts(first_readouts, t_end),
            stop=StopInfo(
                reason="requested_horizon",
                requested_t_end=t_end,
                actual_t_end=t_end,
                hidden_flow_intervals=hidden_intervals,
                output_flow_intervals=output_intervals,
                root_panels=root_panels,
            ),
            omega_checkpoints=tuple(
                sorted(checkpoints, key=lambda item: (item.neuron, item.time))
            ),
        )

    def _effective_readouts(
        self,
        physical: Sequence[tuple[int, float] | None],
        t_end: float,
    ) -> tuple[EffectiveReadout, ...]:
        physical_times = [None if item is None else item[1] for item in physical]
        present = [time for time in physical_times if time is not None]
        latest = max(present) if present else None
        anchor = (
            next(index for index, time in enumerate(physical_times) if time == latest)
            if latest is not None
            else None
        )
        fake_time = (
            t_end
            if latest is None
            else latest + self.config.silent_target_gap_ms
        )
        return tuple(
            EffectiveReadout(
                neuron=output,
                physical_threshold_id=None if item is None else item[0],
                physical_time=None if item is None else item[1],
                effective_time=fake_time if item is None else item[1],
                censored=item is None,
                anchor_output=anchor if item is None else None,
            )
            for output, item in enumerate(physical)
        )

    def timing_loss(
        self,
        forward: DiscreteForwardResult | ContinuousForwardResult,
        label: int,
        *,
        loss_name: TimingLossName,
        margin: float | None = None,
    ) -> DelGradLoss:
        return evaluate_timing_loss(
            forward.readouts,
            label=label,
            loss_name=loss_name,
            margin=margin,
            device=self.state.device,
        )

    def delgrad_loss(
        self,
        forward: DiscreteForwardResult | ContinuousForwardResult,
        label: int,
        margin: float = 1.0,
    ) -> DelGradLoss:
        """Evaluate the original equality-to-margin DelGrad objective."""
        return self.timing_loss(
            forward,
            label,
            loss_name="delgrad",
            margin=margin,
        )

    def squared_hinge_loss(
        self,
        forward: DiscreteForwardResult | ContinuousForwardResult,
        label: int,
        margin: float = 2.0,
    ) -> DelGradLoss:
        """Evaluate the minimum-margin squared-hinge timing objective."""
        return self.timing_loss(
            forward,
            label,
            loss_name="squared_hinge",
            margin=margin,
        )

    def _reverse_continuous_threshold(
        self,
        value: torch.Tensor,
        event: ContinuousThreshold,
        timing_source: torch.Tensor,
    ) -> torch.Tensor:
        lambda_plus = value.clone()
        ratio = float(event.continuous_delay) / self.config.d_plus
        delay_derivative = (
            -self.config.d_plus * self.config.c0 * ratio * (1.0 - ratio)
        )
        sensitivity = self._threshold_sensitivity(event.slope)
        value[0] = (
            (1.0 + self.config.reset_drop * sensitivity) * lambda_plus[0]
            + timing_source * sensitivity
            + timing_source
            * delay_derivative
            * (self._inv_tau_v**2 + event.acceleration_pre * sensitivity)
        )
        value[1] = (
            lambda_plus[1]
            - timing_source * delay_derivative * self._inv_tau_v
        )
        value[2] = lambda_plus[2]
        return value

    def backward_continuous(
        self,
        forward: ContinuousForwardResult,
        output_time_grad: torch.Tensor,
    ) -> BackwardResult:
        source = torch.as_tensor(
            output_time_grad, dtype=torch.float64, device=self.state.device
        ).clone()
        grad_w_in = torch.zeros_like(self.W_in)
        grad_w_out = torch.zeros_like(self.W_out)
        causal_slopes: list[float] = []
        hidden_mu = {
            event.event_id: torch.zeros(
                (), dtype=torch.float64, device=self.state.device
            )
            for event in forward.hidden_spikes
        }
        physical_readouts = tuple(
            readout
            for readout in forward.readouts
            if readout.physical_threshold_id is not None
        )
        if not physical_readouts:
            result = BackwardResult(
                output_time_grad=source,
                hidden_spike_time_grad=torch.zeros(
                    len(forward.hidden_spikes),
                    dtype=torch.float64,
                    device=self.state.device,
                ),
                grad_W_in=grad_w_in,
                grad_W_out=grad_w_out,
                threshold_sensitivity_cap_activations=0,
                minimum_causal_slope=None,
            )
            self.dCdW_in.zero_()
            self.dCdW_out.zero_()
            return result

        output_by_id = {
            event.event_id: event for event in forward.output_spikes
        }
        for output, readout in enumerate(forward.readouts):
            if readout.physical_threshold_id is None:
                continue
            threshold = output_by_id[readout.physical_threshold_id]
            if source[output] != 0.0:
                causal_slopes.append(threshold.slope)
            value = torch.zeros(
                3, dtype=torch.float64, device=self.state.device
            )
            value = self._reverse_continuous_threshold(
                value, threshold, source[output]
            )
            cursor = threshold.time
            arrivals_by_time: dict[float, list[ContinuousArrival]] = {}
            for arrival in forward.hidden_arrivals:
                if arrival.time < threshold.time:
                    arrivals_by_time.setdefault(arrival.time, []).append(arrival)
            global_output = self.n_hidden + output
            for arrival_time in sorted(arrivals_by_time, reverse=True):
                value = self._adjoint_flow_single(
                    value, global_output, cursor - arrival_time
                )
                for arrival in arrivals_by_time[arrival_time]:
                    hidden = arrival.source
                    grad_w_out[output, hidden] += value[2]
                    hidden_mu[arrival.parent_threshold_id] += self.W_out[
                        output, hidden
                    ] * (
                        self._configured_omega(global_output) * value[1]
                        + self.config.alpha * value[2]
                    )
                cursor = arrival_time

        terminal_time = max(
            output_by_id[readout.physical_threshold_id].time
            for readout in physical_readouts
        )
        for hidden in range(self.n_hidden):
            arrivals_by_time: dict[float, list[ContinuousArrival]] = {}
            thresholds_by_time: dict[float, list[ContinuousThreshold]] = {}
            for arrival in forward.input_arrivals:
                if arrival.time < terminal_time:
                    arrivals_by_time.setdefault(arrival.time, []).append(arrival)
            for threshold in forward.hidden_spikes:
                if threshold.neuron == hidden and threshold.time < terminal_time:
                    thresholds_by_time.setdefault(threshold.time, []).append(
                        threshold
                    )
            event_times = sorted(
                set(arrivals_by_time) | set(thresholds_by_time), reverse=True
            )
            value = torch.zeros(
                3, dtype=torch.float64, device=self.state.device
            )
            cursor = terminal_time
            for event_time in event_times:
                value = self._adjoint_flow_single(
                    value, hidden, cursor - event_time
                )
                for arrival in arrivals_by_time.get(event_time, ()):
                    grad_w_in[hidden, arrival.source] += value[2]
                for threshold in reversed(
                    thresholds_by_time.get(event_time, ())
                ):
                    if (
                        value[0] != 0.0
                        or hidden_mu[threshold.event_id] != 0.0
                    ):
                        causal_slopes.append(threshold.slope)
                    value = self._reverse_continuous_threshold(
                        value, threshold, hidden_mu[threshold.event_id]
                    )
                cursor = event_time

        hidden_time_grad = torch.stack(
            tuple(hidden_mu[event.event_id] for event in forward.hidden_spikes)
        ) if forward.hidden_spikes else torch.zeros(
            0, dtype=torch.float64, device=self.state.device
        )
        result = BackwardResult(
            output_time_grad=source,
            hidden_spike_time_grad=hidden_time_grad,
            grad_W_in=grad_w_in.clone(),
            grad_W_out=grad_w_out.clone(),
            threshold_sensitivity_cap_activations=sum(
                self._threshold_sensitivity_is_capped(slope)
                for slope in causal_slopes
            ),
            minimum_causal_slope=min(causal_slopes) if causal_slopes else None,
        )
        self.dCdW_in.copy_(result.grad_W_in)
        self.dCdW_out.copy_(result.grad_W_out)
        return result

    def backward_continuous_omega(
        self,
        forward: ContinuousOmegaForwardResult,
        output_time_grad: torch.Tensor,
    ) -> OmegaBackwardResult:
        """Continuous adjoint with exact event-interval omega gradients."""
        self._validate_learned_omega()
        source = torch.as_tensor(
            output_time_grad, dtype=torch.float64, device=self.state.device
        ).clone()
        grad_w_in = torch.zeros_like(self.W_in)
        grad_w_out = torch.zeros_like(self.W_out)
        grad_omega = torch.zeros_like(self.omega)
        causal_slopes: list[float] = []
        hidden_mu = {
            event.event_id: torch.zeros(
                (), dtype=torch.float64, device=self.state.device
            )
            for event in forward.hidden_spikes
        }
        checkpoint_by_boundary = {
            (checkpoint.neuron, checkpoint.time): checkpoint
            for checkpoint in forward.omega_checkpoints
        }
        physical_readouts = tuple(
            readout
            for readout in forward.readouts
            if readout.physical_threshold_id is not None
        )
        if not physical_readouts:
            result = OmegaBackwardResult(
                output_time_grad=source,
                hidden_spike_time_grad=torch.zeros(
                    len(forward.hidden_spikes),
                    dtype=torch.float64,
                    device=self.state.device,
                ),
                grad_W_in=grad_w_in,
                grad_W_out=grad_w_out,
                threshold_sensitivity_cap_activations=0,
                minimum_causal_slope=None,
                grad_omega=grad_omega,
            )
            self.dCdW_in.zero_()
            self.dCdW_out.zero_()
            return result

        output_by_id = {
            event.event_id: event for event in forward.output_spikes
        }
        for output, readout in enumerate(forward.readouts):
            if readout.physical_threshold_id is None:
                continue
            threshold = output_by_id[readout.physical_threshold_id]
            if source[output] != 0.0:
                causal_slopes.append(threshold.slope)
            value = torch.zeros(
                3, dtype=torch.float64, device=self.state.device
            )
            value = self._reverse_continuous_threshold(
                value, threshold, source[output]
            )
            cursor = threshold.time
            arrivals_by_time: dict[float, list[ContinuousArrival]] = {}
            for arrival in forward.hidden_arrivals:
                if arrival.time < threshold.time:
                    arrivals_by_time.setdefault(arrival.time, []).append(arrival)
            global_output = self.n_hidden + output
            for arrival_time in sorted(arrivals_by_time, reverse=True):
                checkpoint = checkpoint_by_boundary[
                    (global_output, arrival_time)
                ]
                grad_omega[global_output] += (
                    self._continuous_omega_interval_gradient(
                        global_output,
                        checkpoint.i,
                        checkpoint.q,
                        value,
                        cursor - arrival_time,
                    )
                )
                value = self._omega_adjoint_flow_single(
                    value, global_output, cursor - arrival_time
                )
                for arrival in arrivals_by_time[arrival_time]:
                    hidden = arrival.source
                    grad_w_out[output, hidden] += value[2]
                    hidden_mu[arrival.parent_threshold_id] += self.W_out[
                        output, hidden
                    ] * (
                        self.omega[global_output] * value[1]
                        + self.config.alpha * value[2]
                    )
                cursor = arrival_time

        terminal_time = max(
            output_by_id[readout.physical_threshold_id].time
            for readout in physical_readouts
        )
        for hidden in range(self.n_hidden):
            arrivals_by_time: dict[float, list[ContinuousArrival]] = {}
            thresholds_by_time: dict[float, list[ContinuousThreshold]] = {}
            for arrival in forward.input_arrivals:
                if arrival.time < terminal_time:
                    arrivals_by_time.setdefault(arrival.time, []).append(arrival)
            for threshold in forward.hidden_spikes:
                if threshold.neuron == hidden and threshold.time < terminal_time:
                    thresholds_by_time.setdefault(threshold.time, []).append(
                        threshold
                    )
            event_times = sorted(
                set(arrivals_by_time) | set(thresholds_by_time), reverse=True
            )
            value = torch.zeros(
                3, dtype=torch.float64, device=self.state.device
            )
            cursor = terminal_time
            for event_time in event_times:
                checkpoint = checkpoint_by_boundary[(hidden, event_time)]
                grad_omega[hidden] += self._continuous_omega_interval_gradient(
                    hidden,
                    checkpoint.i,
                    checkpoint.q,
                    value,
                    cursor - event_time,
                )
                value = self._omega_adjoint_flow_single(
                    value, hidden, cursor - event_time
                )
                for arrival in arrivals_by_time.get(event_time, ()):
                    grad_w_in[hidden, arrival.source] += value[2]
                for threshold in reversed(
                    thresholds_by_time.get(event_time, ())
                ):
                    if (
                        value[0] != 0.0
                        or hidden_mu[threshold.event_id] != 0.0
                    ):
                        causal_slopes.append(threshold.slope)
                    value = self._reverse_continuous_threshold(
                        value, threshold, hidden_mu[threshold.event_id]
                    )
                cursor = event_time

        hidden_time_grad = torch.stack(
            tuple(hidden_mu[event.event_id] for event in forward.hidden_spikes)
        ) if forward.hidden_spikes else torch.zeros(
            0, dtype=torch.float64, device=self.state.device
        )
        result = OmegaBackwardResult(
            output_time_grad=source,
            hidden_spike_time_grad=hidden_time_grad,
            grad_W_in=grad_w_in.clone(),
            grad_W_out=grad_w_out.clone(),
            threshold_sensitivity_cap_activations=sum(
                self._threshold_sensitivity_is_capped(slope)
                for slope in causal_slopes
            ),
            minimum_causal_slope=min(causal_slopes) if causal_slopes else None,
            grad_omega=grad_omega.clone(),
        )
        self.dCdW_in.copy_(result.grad_W_in)
        self.dCdW_out.copy_(result.grad_W_out)
        return result

    def forward_discrete(
        self,
        inputs: Sequence[InputSpike],
        t_end: float,
        *,
        early_stop: bool = False,
        record_state: bool = True,
    ) -> DiscreteForwardResult:
        self.reset()
        for spike in inputs:
            self.enqueue_input(spike)
        step_ratio = t_end * self._inv_dt
        nearest_step = round(step_ratio)
        aligned_end = math.isclose(
            step_ratio,
            nearest_step,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(step_ratio)),
        )
        n_steps = int(nearest_step) if aligned_end else math.ceil(step_ratio)
        times = (
            torch.tensor(
                [min(step * self.config.dt, t_end) for step in range(n_steps + 1)],
                dtype=torch.float64,
                device=self.state.device,
            )
            if record_state
            else None
        )
        history = (
            torch.zeros(
                (n_steps + 1, self.n_neurons, 3),
                dtype=torch.float64,
                device=self.state.device,
            )
            if record_state
            else None
        )
        first_readouts: list[tuple[int, float] | None] = [None] * self.n_out
        first_readout_threshold_times: list[float | None] = [None] * self.n_out
        stop_reason = "requested_horizon"
        actual_t_end = t_end
        completed_steps = 0
        stopped = False

        if history is not None:
            history[0] = self.state

        if early_stop:
            future_inputs = any(item.kind == "input" for item in self._queue)
            hidden_closed = not future_inputs and all(
                self._certify_silent(
                    self.state[hidden], arrival_stream_closed=not future_inputs
                )
                for hidden in range(self.n_hidden)
            )
            pending_hidden = any(item.kind == "hidden" for item in self._queue)
            if hidden_closed and not pending_hidden and all(
                self._certify_silent(
                    self.state[self.n_hidden + output],
                    arrival_stream_closed=(hidden_closed and not pending_hidden),
                )
                for output in range(self.n_out)
            ):
                stop_reason = "all_readouts_silent"
                actual_t_end = 0.0
                stopped = True

        threshold_cursor = 0
        if not stopped:
            for step in range(n_steps):
                delta = min(self.config.dt, t_end - step * self.config.dt)
                post_arrival = self.step(step, delta)
                completed_steps = step + 1
                if history is not None:
                    history[step] = post_arrival
                    history[step + 1] = self.state
                for event in self.threshold_events[threshold_cursor:]:
                    output = event.neuron - self.n_hidden
                    if 0 <= output < self.n_out and first_readouts[output] is None:
                        first_readouts[output] = (
                            event.event_id,
                            float(event.arrival_time),
                        )
                        first_readout_threshold_times[output] = event.time
                threshold_cursor = len(self.threshold_events)

                if not early_stop:
                    continue
                if all(item is not None for item in first_readouts):
                    stop_reason = "all_readouts_physical"
                    actual_t_end = max(
                        time
                        for time in first_readout_threshold_times
                        if time is not None
                    )
                    stopped = True
                    break
                future_inputs = any(item.kind == "input" for item in self._queue)
                hidden_closed = not future_inputs and all(
                    self._certify_silent(
                        self.state[hidden],
                        arrival_stream_closed=not future_inputs,
                    )
                    for hidden in range(self.n_hidden)
                )
                pending_hidden = any(
                    item.kind == "hidden" for item in self._queue
                )
                unresolved_silent = hidden_closed and not pending_hidden and all(
                    first_readouts[output] is not None
                    or self._certify_silent(
                        self.state[self.n_hidden + output],
                        arrival_stream_closed=True,
                    )
                    for output in range(self.n_out)
                )
                if hidden_closed and not pending_hidden and unresolved_silent:
                    stop_reason = (
                        "all_readouts_resolved"
                        if any(item is not None for item in first_readouts)
                        else "all_readouts_silent"
                    )
                    actual_t_end = min(
                        (step + 1) * self.config.dt, t_end
                    )
                    stopped = True
                    break

        if not stopped and aligned_end:
            self.apply_due_arrivals(n_steps)
            if history is not None:
                history[completed_steps] = self.state

        if stopped and history is not None:
            history = history[: completed_steps + 1]
            times = times[: completed_steps + 1]

        arrivals = tuple(self.arrival_records)
        thresholds = tuple(self.threshold_events)
        if not early_stop:
            for event in thresholds:
                output = event.neuron - self.n_hidden
                if 0 <= output < self.n_out and first_readouts[output] is None:
                    first_readouts[output] = (
                        event.event_id,
                        float(event.arrival_time),
                    )
                    first_readout_threshold_times[output] = event.time
        warnings = (
            ()
            if self.config.dt <= self.config.fast_period_ms / 20.0
            else ("fewer than 20 grid steps per fast oscillation period",)
        )
        return DiscreteForwardResult(
            times=None if times is None else times.clone(),
            V=None if history is None else history[:, :, 0].clone(),
            I=None if history is None else history[:, :, 1].clone(),
            Q=None if history is None else history[:, :, 2].clone(),
            arrivals=arrivals,
            thresholds=thresholds,
            readouts=self._effective_readouts(first_readouts, t_end),
            warnings=warnings,
            stop=StopInfo(
                reason=stop_reason,
                requested_t_end=t_end,
                actual_t_end=actual_t_end,
                hidden_flow_intervals=completed_steps * self.n_hidden,
                output_flow_intervals=completed_steps * self.n_out,
                root_panels=0,
            ),
        )

    def forward_discrete_omega(
        self,
        inputs: Sequence[InputSpike],
        t_end: float,
        *,
        early_stop: bool = False,
    ) -> DiscreteForwardResult:
        """Grid forward pass retaining dense I,Q histories for omega learning."""
        self._validate_learned_omega()
        self.reset()
        for spike in inputs:
            self.enqueue_input(spike)
        step_ratio = t_end * self._inv_dt
        nearest_step = round(step_ratio)
        aligned_end = math.isclose(
            step_ratio,
            nearest_step,
            rel_tol=0.0,
            abs_tol=1e-12 * max(1.0, abs(step_ratio)),
        )
        n_steps = int(nearest_step) if aligned_end else math.ceil(step_ratio)
        times = torch.tensor(
            [min(step * self.config.dt, t_end) for step in range(n_steps + 1)],
            dtype=torch.float64,
            device=self.state.device,
        )
        oscillator_history = torch.zeros(
            (n_steps + 1, self.n_neurons, 2),
            dtype=torch.float64,
            device=self.state.device,
        )
        first_readouts: list[tuple[int, float] | None] = [None] * self.n_out
        first_readout_threshold_times: list[float | None] = [None] * self.n_out
        stop_reason = "requested_horizon"
        actual_t_end = t_end
        completed_steps = 0
        stopped = False
        oscillator_history[0] = self.state[:, 1:]
        full_transitions = self._omega_transition_batch(self.config.dt)

        if early_stop:
            future_inputs = any(item.kind == "input" for item in self._queue)
            hidden_closed = not future_inputs and all(
                self._certify_silent(
                    self.state[hidden], arrival_stream_closed=not future_inputs
                )
                for hidden in range(self.n_hidden)
            )
            pending_hidden = any(item.kind == "hidden" for item in self._queue)
            if hidden_closed and not pending_hidden and all(
                self._certify_silent(
                    self.state[self.n_hidden + output],
                    arrival_stream_closed=(hidden_closed and not pending_hidden),
                )
                for output in range(self.n_out)
            ):
                stop_reason = "all_readouts_silent"
                actual_t_end = 0.0
                stopped = True

        threshold_cursor = 0
        if not stopped:
            for step in range(n_steps):
                delta = min(self.config.dt, t_end - step * self.config.dt)
                transitions = (
                    full_transitions
                    if delta == self.config.dt
                    else self._omega_transition_batch(delta)
                )
                post_arrival = self._step_omega(
                    step, delta, transitions
                )
                completed_steps = step + 1
                oscillator_history[step] = post_arrival[:, 1:]
                oscillator_history[step + 1] = self.state[:, 1:]
                for event in self.threshold_events[threshold_cursor:]:
                    output = event.neuron - self.n_hidden
                    if 0 <= output < self.n_out and first_readouts[output] is None:
                        first_readouts[output] = (
                            event.event_id,
                            float(event.arrival_time),
                        )
                        first_readout_threshold_times[output] = event.time
                threshold_cursor = len(self.threshold_events)

                if not early_stop:
                    continue
                if all(item is not None for item in first_readouts):
                    stop_reason = "all_readouts_physical"
                    actual_t_end = max(
                        time
                        for time in first_readout_threshold_times
                        if time is not None
                    )
                    stopped = True
                    break
                future_inputs = any(item.kind == "input" for item in self._queue)
                hidden_closed = not future_inputs and all(
                    self._certify_silent(
                        self.state[hidden],
                        arrival_stream_closed=not future_inputs,
                    )
                    for hidden in range(self.n_hidden)
                )
                pending_hidden = any(
                    item.kind == "hidden" for item in self._queue
                )
                unresolved_silent = hidden_closed and not pending_hidden and all(
                    first_readouts[output] is not None
                    or self._certify_silent(
                        self.state[self.n_hidden + output],
                        arrival_stream_closed=True,
                    )
                    for output in range(self.n_out)
                )
                if hidden_closed and not pending_hidden and unresolved_silent:
                    stop_reason = (
                        "all_readouts_resolved"
                        if any(item is not None for item in first_readouts)
                        else "all_readouts_silent"
                    )
                    actual_t_end = min((step + 1) * self.config.dt, t_end)
                    stopped = True
                    break

        if not stopped and aligned_end:
            self.apply_due_arrivals(n_steps)
            oscillator_history[completed_steps] = self.state[:, 1:]

        if stopped:
            oscillator_history = oscillator_history[: completed_steps + 1]
            times = times[: completed_steps + 1]

        arrivals = tuple(self.arrival_records)
        thresholds = tuple(self.threshold_events)
        if not early_stop:
            for event in thresholds:
                output = event.neuron - self.n_hidden
                if 0 <= output < self.n_out and first_readouts[output] is None:
                    first_readouts[output] = (
                        event.event_id,
                        float(event.arrival_time),
                    )
                    first_readout_threshold_times[output] = event.time
        fastest_period = 2.0 * math.pi / float(torch.max(self.omega))
        warnings = (
            ()
            if self.config.dt <= fastest_period / 20.0
            else (
                "fewer than 20 grid steps per fastest learned oscillation period",
            )
        )
        return DiscreteForwardResult(
            times=times.clone(),
            V=None,
            I=oscillator_history[:, :, 0].clone(),
            Q=oscillator_history[:, :, 1].clone(),
            arrivals=arrivals,
            thresholds=thresholds,
            readouts=self._effective_readouts(first_readouts, t_end),
            warnings=warnings,
            stop=StopInfo(
                reason=stop_reason,
                requested_t_end=t_end,
                actual_t_end=actual_t_end,
                hidden_flow_intervals=completed_steps * self.n_hidden,
                output_flow_intervals=completed_steps * self.n_out,
                root_panels=0,
            ),
        )

    def _reverse_discrete_threshold(
        self,
        value: torch.Tensor,
        event: DiscreteThreshold,
        timing_source: torch.Tensor,
        *,
        omega: float | None = None,
    ) -> None:
        neuron = event.neuron
        if omega is None:
            omega = self._configured_omega(neuron)
        acceleration = (
            -event.slope - self.config.alpha * event.i_pre - omega * event.q_pre
        ) * self._inv_tau_v
        ratio = float(event.continuous_delay) / self.config.d_plus
        delay_derivative = (
            -self.config.d_plus * self.config.c0 * ratio * (1.0 - ratio)
        )
        lambda_plus = value[neuron].clone()
        sensitivity = self._threshold_sensitivity(event.slope)
        value[neuron, 0] = (
            (1.0 + self.config.reset_drop * sensitivity) * lambda_plus[0]
            + timing_source * sensitivity
            + timing_source
            * delay_derivative
            * (self._inv_tau_v**2 + acceleration * sensitivity)
        )
        value[neuron, 1] = (
            lambda_plus[1]
            - timing_source * delay_derivative * self._inv_tau_v
        )
        value[neuron, 2] = lambda_plus[2]

    def _flow_discrete_adjoint(
        self, value: torch.Tensor, delta: float
    ) -> torch.Tensor:
        result = torch.empty_like(value)
        for group, neuron_slice in self._frequency_slices:
            if neuron_slice.start == neuron_slice.stop:
                continue
            if delta == self.config.dt:
                transition = self._psi_fast if group == "fast" else self._psi_slow
            else:
                transition = self._mass_scaled_transpose(
                    self._analytical_transition(group, delta)
                )
            result[neuron_slice] = value[neuron_slice] @ transition.T
        return result

    def _omega_adjoint_transition_batch(self, delta: float) -> torch.Tensor:
        return torch.stack(
            tuple(
                self._mass_scaled_transpose(
                    self._omega_transition(neuron, delta)
                )
                for neuron in range(self.n_neurons)
            )
        )

    def _discrete_omega_midpoint_integrand(
        self,
        left_i: torch.Tensor,
        left_q: torch.Tensor,
        midpoint_adjoint: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        half_delta = 0.5 * delta
        decay = math.exp(-self.config.alpha * half_delta)
        angle = self.omega * half_delta
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        midpoint_i = decay * (cosine * left_i - sine * left_q)
        midpoint_q = decay * (sine * left_i + cosine * left_q)
        return (
            midpoint_adjoint[:, 1] * midpoint_q
            - midpoint_adjoint[:, 2] * midpoint_i
        )

    def backward_discrete(
        self,
        forward: DiscreteForwardResult,
        output_time_grad: torch.Tensor,
    ) -> BackwardResult:
        source = torch.as_tensor(
            output_time_grad, dtype=torch.float64, device=self.state.device
        ).clone()
        timing = torch.zeros(
            len(forward.thresholds), dtype=torch.float64, device=self.state.device
        )
        for output, readout in enumerate(forward.readouts):
            if readout.physical_threshold_id is not None:
                timing[readout.physical_threshold_id] = source[output]

        physical = [
            readout for readout in forward.readouts if readout.physical_time is not None
        ]
        grad_w_in = torch.zeros_like(self.W_in)
        grad_w_out = torch.zeros_like(self.W_out)
        causal_slopes: list[float] = []
        if not physical:
            hidden_ids = [
                event.event_id
                for event in forward.thresholds
                if event.neuron < self.n_hidden
            ]
            result = BackwardResult(
                output_time_grad=source,
                hidden_spike_time_grad=timing[hidden_ids],
                grad_W_in=grad_w_in,
                grad_W_out=grad_w_out,
                threshold_sensitivity_cap_activations=0,
                minimum_causal_slope=None,
            )
            self.dCdW_in.zero_()
            self.dCdW_out.zero_()
            return result

        physical_threshold_ids = {
            readout.physical_threshold_id for readout in physical
        }
        terminal_step = max(
            event.step
            for event in forward.thresholds
            if event.event_id in physical_threshold_ids
        )
        arrivals_by_step: dict[int, list[DiscreteArrival]] = {}
        thresholds_by_step: dict[int, list[DiscreteThreshold]] = {}
        for arrival in forward.arrivals:
            if arrival.step <= terminal_step:
                arrivals_by_step.setdefault(arrival.step, []).append(arrival)
        for event in forward.thresholds:
            if event.step <= terminal_step:
                thresholds_by_step.setdefault(event.step, []).append(event)

        value = torch.zeros(
            (self.n_neurons, 3), dtype=torch.float64, device=self.state.device
        )
        for step in range(terminal_step, -1, -1):
            for arrival in reversed(arrivals_by_step.get(step, ())):
                if arrival.kind == "input":
                    grad_w_in[:, arrival.source] += value[: self.n_hidden, 2]
                else:
                    grad_w_out[:, arrival.source] += value[self._output_slice, 2]
                    timing[arrival.parent_threshold_id] += torch.sum(
                        self.W_out[:, arrival.source]
                        * (
                            self.config.omega_slow
                            * value[self._output_slice, 1]
                            + self.config.alpha * value[self._output_slice, 2]
                        )
                    )
            for event in reversed(thresholds_by_step.get(step, ())):
                if value[event.neuron, 0] != 0.0 or timing[event.event_id] != 0.0:
                    causal_slopes.append(event.slope)
                self._reverse_discrete_threshold(
                    value, event, timing[event.event_id]
                )
            if step > 0:
                right_time = min(step * self.config.dt, forward.stop.actual_t_end)
                left_time = min((step - 1) * self.config.dt, forward.stop.actual_t_end)
                value = self._flow_discrete_adjoint(value, right_time - left_time)

        hidden_ids = [
            event.event_id
            for event in forward.thresholds
            if event.neuron < self.n_hidden
        ]
        result = BackwardResult(
            output_time_grad=source,
            hidden_spike_time_grad=timing[hidden_ids].clone(),
            grad_W_in=grad_w_in.clone(),
            grad_W_out=grad_w_out.clone(),
            threshold_sensitivity_cap_activations=sum(
                self._threshold_sensitivity_is_capped(slope)
                for slope in causal_slopes
            ),
            minimum_causal_slope=min(causal_slopes) if causal_slopes else None,
        )
        self.dCdW_in.copy_(result.grad_W_in)
        self.dCdW_out.copy_(result.grad_W_out)
        return result

    def backward_discrete_omega(
        self,
        forward: DiscreteForwardResult,
        output_time_grad: torch.Tensor,
    ) -> OmegaBackwardResult:
        """Grid adjoint with dense midpoint integration of the omega gradient."""
        self._validate_learned_omega()
        if forward.I is None or forward.Q is None:
            raise ValueError(
                "backward_discrete_omega requires the I,Q history returned by "
                "forward_discrete_omega"
            )
        source = torch.as_tensor(
            output_time_grad, dtype=torch.float64, device=self.state.device
        ).clone()
        timing = torch.zeros(
            len(forward.thresholds), dtype=torch.float64, device=self.state.device
        )
        for output, readout in enumerate(forward.readouts):
            if readout.physical_threshold_id is not None:
                timing[readout.physical_threshold_id] = source[output]

        physical = [
            readout
            for readout in forward.readouts
            if readout.physical_time is not None
        ]
        grad_w_in = torch.zeros_like(self.W_in)
        grad_w_out = torch.zeros_like(self.W_out)
        grad_omega = torch.zeros_like(self.omega)
        causal_slopes: list[float] = []
        if not physical:
            hidden_ids = [
                event.event_id
                for event in forward.thresholds
                if event.neuron < self.n_hidden
            ]
            result = OmegaBackwardResult(
                output_time_grad=source,
                hidden_spike_time_grad=timing[hidden_ids],
                grad_W_in=grad_w_in,
                grad_W_out=grad_w_out,
                threshold_sensitivity_cap_activations=0,
                minimum_causal_slope=None,
                grad_omega=grad_omega,
            )
            self.dCdW_in.zero_()
            self.dCdW_out.zero_()
            return result

        physical_threshold_ids = {
            readout.physical_threshold_id for readout in physical
        }
        terminal_step = max(
            event.step
            for event in forward.thresholds
            if event.event_id in physical_threshold_ids
        )
        arrivals_by_step: dict[int, list[DiscreteArrival]] = {}
        thresholds_by_step: dict[int, list[DiscreteThreshold]] = {}
        for arrival in forward.arrivals:
            if arrival.step <= terminal_step:
                arrivals_by_step.setdefault(arrival.step, []).append(arrival)
        for event in forward.thresholds:
            if event.step <= terminal_step:
                thresholds_by_step.setdefault(event.step, []).append(event)

        value = torch.zeros(
            (self.n_neurons, 3), dtype=torch.float64, device=self.state.device
        )
        adjoint_transition_cache: dict[float, torch.Tensor] = {}

        def flow_adjoint(adjoint: torch.Tensor, delta: float) -> torch.Tensor:
            if delta not in adjoint_transition_cache:
                adjoint_transition_cache[delta] = (
                    self._omega_adjoint_transition_batch(delta)
                )
            return torch.bmm(
                adjoint_transition_cache[delta], adjoint.unsqueeze(-1)
            ).squeeze(-1)

        for step in range(terminal_step, -1, -1):
            for arrival in reversed(arrivals_by_step.get(step, ())):
                if arrival.kind == "input":
                    grad_w_in[:, arrival.source] += value[: self.n_hidden, 2]
                else:
                    grad_w_out[:, arrival.source] += value[
                        self._output_slice, 2
                    ]
                    timing[arrival.parent_threshold_id] += torch.sum(
                        self.W_out[:, arrival.source]
                        * (
                            self.omega[self._output_slice]
                            * value[self._output_slice, 1]
                            + self.config.alpha * value[self._output_slice, 2]
                        )
                    )
            for event in reversed(thresholds_by_step.get(step, ())):
                if value[event.neuron, 0] != 0.0 or timing[event.event_id] != 0.0:
                    causal_slopes.append(event.slope)
                self._reverse_discrete_threshold(
                    value,
                    event,
                    timing[event.event_id],
                    omega=float(self.omega[event.neuron]),
                )
            if step > 0:
                right_time = min(
                    step * self.config.dt, forward.stop.actual_t_end
                )
                left_time = min(
                    (step - 1) * self.config.dt,
                    forward.stop.actual_t_end,
                )
                delta = right_time - left_time
                if math.isclose(
                    delta,
                    self.config.dt,
                    rel_tol=0.0,
                    abs_tol=1e-12 * self.config.dt,
                ):
                    delta = self.config.dt
                midpoint_adjoint = flow_adjoint(value, 0.5 * delta)
                grad_omega += delta * self._discrete_omega_midpoint_integrand(
                    forward.I[step - 1],
                    forward.Q[step - 1],
                    midpoint_adjoint,
                    delta,
                )
                value = flow_adjoint(value, delta)

        hidden_ids = [
            event.event_id
            for event in forward.thresholds
            if event.neuron < self.n_hidden
        ]
        result = OmegaBackwardResult(
            output_time_grad=source,
            hidden_spike_time_grad=timing[hidden_ids].clone(),
            grad_W_in=grad_w_in.clone(),
            grad_W_out=grad_w_out.clone(),
            threshold_sensitivity_cap_activations=sum(
                self._threshold_sensitivity_is_capped(slope)
                for slope in causal_slopes
            ),
            minimum_causal_slope=min(causal_slopes) if causal_slopes else None,
            grad_omega=grad_omega.clone(),
        )
        self.dCdW_in.copy_(result.grad_W_in)
        self.dCdW_out.copy_(result.grad_W_out)
        return result
