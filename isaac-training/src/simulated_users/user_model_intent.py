from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse

from src.simulated_users.pilot_modes import IntentMode, ReactMode
from src.simulated_users.pilot_perception import PilotPerceptionModel, PerceptionConfig


@dataclass
class IntentPilotConfig:
    alpha_range: tuple[float, float] = (0.1, 0.95)
    beta_range: tuple[float, float] = (0.25, 1.0)
    psi_range: tuple[float, float] = (0.2, 1.0)
    phi_range: tuple[float, float] = (0.35, 1.0)
    eta_range: tuple[float, float] = (0.0, 1.0)

    tau_perc_range: tuple[float, float] = (0.05, 0.35)
    sigma_perc_range: tuple[float, float] = (0.0, 0.4)
    d_react_range: tuple[float, float] = (0.7, 2.2)
    wrong_direction_prob_range: tuple[float, float] = (0.0, 0.45)

    max_speed: float = 2.0
    max_speed_z_scale: float = 0.4
    horizon_sec: float = 2.0
    waypoint_mix: float = 0.10
    react_feedback_gain: float = 0.12
    p_gain_scale: float = 0.55
    i_gain: float = 0.15
    i_decay: float = 0.92

    heading_ou_tau: float = 1.5
    heading_ou_sigma: float = 0.35
    maneuver_lateral_scale: float = 0.8
    maneuver_vertical_scale: float = 0.35
    station_keep_speed_scale: float = 0.45
    idle_speed_scale: float = 0.15

    conservative_mode_prior: tuple[float, float, float, float] = (0.30, 0.10, 0.35, 0.25)
    aggressive_mode_prior: tuple[float, float, float, float] = (0.62, 0.25, 0.08, 0.05)
    tunnel_mode_prior: tuple[float, float, float, float] = (0.74, 0.18, 0.04, 0.04)

    dwell_lognormal: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "cruise": (1.6, 0.6),
            "maneuver": (0.4, 0.5),
            "station_keep": (1.1, 0.7),
            "idle": (0.0, 0.5),
        }
    )

    react_trigger_threshold: float = 0.22
    late_react_threshold_scale: float = 1.35
    spontaneous_react_rate_hz: float = 0.03
    react_min_dwell_sec: float = 0.45
    react_max_dwell_sec: float = 1.1
    overcorrect_osc_hz: float = 1.8
    surge_scale: float = 1.5
    evade_scale: float = 1.1
    sim_dt: float = 0.02
    max_delay_sec: float = 0.5

    @classmethod
    def from_cfg(cls, cfg) -> "IntentPilotConfig":
        intent_cfg = cfg.user_model.get("intent", {})
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in intent_cfg:
                value = intent_cfg[field_name]
                if isinstance(value, list):
                    value = tuple(value)
                elif isinstance(value, dict):
                    value = dict(value)
                kwargs[field_name] = value
        kwargs.setdefault("max_speed", cfg.algo.actor.action_limit)
        kwargs.setdefault("sim_dt", cfg.sim.dt)
        return cls(**kwargs)


class UserModelIntent:
    """Intent/reactive pilot model with joystick dynamics and privileged state."""

    INTENT_DIM = IntentMode.count()
    REACT_DIM = ReactMode.count()
    PRIVILEGED_DIM = 9 + INTENT_DIM + REACT_DIM + 3 + 3 + 1 + 1 + 1 + 3

    def __init__(self, num_envs: int, cfg, logger=None):
        self.num_envs = num_envs
        self.cfg = IntentPilotConfig.from_cfg(cfg)
        self.device = torch.device(cfg.device)
        self.logger = logger
        self.dt = self.cfg.sim_dt
        self.max_speed = self.cfg.max_speed
        self.max_speed_z = self.cfg.max_speed * self.cfg.max_speed_z_scale
        self.zero_flags = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

        self.perception = PilotPerceptionModel(
            num_envs,
            self.device,
            PerceptionConfig(sim_dt=self.dt, max_delay_sec=self.cfg.max_delay_sec),
        )

        self.alpha = torch.zeros(num_envs, device=self.device)
        self.beta = torch.zeros(num_envs, device=self.device)
        self.psi = torch.zeros(num_envs, device=self.device)
        self.phi = torch.zeros(num_envs, device=self.device)
        self.eta = torch.zeros(num_envs, device=self.device)
        self.tau_perc = torch.zeros(num_envs, device=self.device)
        self.sigma_perc = torch.zeros(num_envs, device=self.device)
        self.d_react = torch.zeros(num_envs, device=self.device)
        self.wrong_direction_prob = torch.zeros(num_envs, device=self.device)

        self.intent_mode = torch.full(
            (num_envs,), int(IntentMode.CRUISE), dtype=torch.long, device=self.device
        )
        self.react_mode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.dwell_steps = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.react_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.react_total_steps = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.mode_prior_override = None

        self.W_t = torch.zeros(num_envs, 3, device=self.device)
        self.anchor = torch.zeros(num_envs, 3, device=self.device)
        self.heading_dir = torch.zeros(num_envs, 3, device=self.device)
        self.J = torch.zeros(num_envs, 3, device=self.device)
        self.I = torch.zeros(num_envs, 3, device=self.device)
        self.last_pilot_action = torch.zeros(num_envs, 3, device=self.device)

        self.intent_velocity_body = torch.zeros(num_envs, 3, device=self.device)
        self.final_velocity_body = torch.zeros(num_envs, 3, device=self.device)
        self.perceived_dist = torch.full((num_envs,), float("inf"), device=self.device)
        self.perceived_normal = torch.zeros(num_envs, 3, device=self.device)
        self.threat = torch.zeros(num_envs, device=self.device)
        self.step_counter = torch.zeros(num_envs, device=self.device)

    @property
    def privileged_dim(self) -> int:
        return self.PRIVILEGED_DIM

    def reset(
        self,
        pos,
        quat,
        env_ids=None,
        seed: Optional[int] = None,
        mode_prior_override=None,
        anchor=None,
    ):
        ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        if pos.ndim == 3:
            pos = pos.squeeze(1)
        if quat.ndim == 3:
            quat = quat.squeeze(1)
        if seed is not None:
            self.rng.manual_seed(int(seed))

        self._sample_profile(ids)
        self.mode_prior_override = (
            torch.as_tensor(mode_prior_override, device=self.device, dtype=torch.float32)
            if mode_prior_override is not None
            else None
        )
        self.anchor[ids] = pos if anchor is None else anchor
        self.heading_dir[ids] = self._forward_world(quat)
        self.J[ids] = 0.0
        self.I[ids] = 0.0
        self.last_pilot_action[ids] = 0.0
        self.intent_velocity_body[ids] = 0.0
        self.final_velocity_body[ids] = 0.0
        self.threat[ids] = 0.0
        self.perceived_dist[ids] = float("inf")
        self.perceived_normal[ids] = 0.0
        self.react_mode[ids] = int(ReactMode.NONE)
        self.react_steps[ids] = 0
        self.react_total_steps[ids] = 1
        self.step_counter[ids] = 0.0
        self.perception.reset(ids)
        self._sample_intent_mode(ids)
        self._sample_dwell(ids)
        self._refresh_waypoint(ids, pos, quat)

    def step(
        self,
        drone_state,
        drone_pos_w,
        assistant_action=None,
        env_geom=None,
    ):
        if drone_pos_w.ndim == 3:
            drone_pos_w = drone_pos_w.squeeze(1)
        quat = drone_state[..., 6:10]
        if quat.ndim == 3:
            quat = quat.squeeze(1)
        if assistant_action is None:
            assistant_action = torch.zeros_like(self.J)
        else:
            assistant_action = torch.nan_to_num(
                assistant_action.to(device=self.device),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        nearest_dist = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
        nearest_normal = torch.zeros(self.num_envs, 3, device=self.device)
        if env_geom is not None:
            nearest_dist = env_geom.get("nearest_obstacle_dist", nearest_dist)
            nearest_normal = env_geom.get("nearest_obstacle_normal", nearest_normal)
        nearest_dist = self._sanitize_distance(nearest_dist)
        nearest_normal = self._sanitize_normal(nearest_normal)

        self.perception.update(nearest_dist, nearest_normal)
        threat, perceived_dist, perceived_normal = self.perception.perceive(
            self.tau_perc,
            self.sigma_perc,
            self.d_react,
            generator=self.rng,
        )
        self.threat = torch.nan_to_num(threat, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        self.perceived_dist = self._sanitize_distance(perceived_dist)
        self.perceived_normal = self._sanitize_normal(perceived_normal)

        self._update_reactive_layer()
        self._tick_intent_layer(drone_pos_w, quat, assistant_action)

        intent_velocity = self._intent_to_velocity(drone_pos_w, quat)
        self.intent_velocity_body = intent_velocity
        final_velocity = self._apply_reactive_overlay(intent_velocity)
        self.final_velocity_body = final_velocity

        p_gain = (self.psi * self.cfg.p_gain_scale).unsqueeze(-1)
        self.J = self.J + (final_velocity - self.J) * p_gain
        self.I = self.cfg.i_decay * self.I + (
            self.last_pilot_action - assistant_action
        ) * (1.0 - self.alpha).unsqueeze(-1)
        pilot_action = self.J + self.I * self.cfg.i_gain
        self._raise_if_nonfinite_pilot_action(
            pilot_action,
            assistant_action,
            nearest_dist,
            nearest_normal,
            "pre_clamp",
        )
        pilot_action[:, 0:2] = pilot_action[:, 0:2].clamp(-self.max_speed, self.max_speed)
        pilot_action[:, 2] = pilot_action[:, 2].clamp(-self.max_speed_z, self.max_speed_z)
        self._raise_if_nonfinite_pilot_action(
            pilot_action,
            assistant_action,
            nearest_dist,
            nearest_normal,
            "post_clamp",
        )

        self.last_pilot_action = pilot_action
        self.step_counter += 1.0
        return pilot_action, self.zero_flags

    def get_privileged_obs(self):
        intent_one_hot = F.one_hot(
            self.intent_mode, num_classes=IntentMode.count()
        ).float()
        react_one_hot = F.one_hot(
            self.react_mode, num_classes=ReactMode.count()
        ).float()
        dwell_norm = self.dwell_steps.float().unsqueeze(-1) * self.dt / 5.0
        react_norm = self.react_steps.float().unsqueeze(-1) * self.dt / 2.0
        threat = self.threat.unsqueeze(-1)
        profile = torch.stack(
            [
                self.alpha,
                self.beta,
                self.psi,
                self.phi,
                self.eta,
                self.tau_perc,
                self.sigma_perc,
                self.d_react,
                self.wrong_direction_prob,
            ],
            dim=-1,
        )
        return torch.cat(
            [
                profile,
                intent_one_hot,
                react_one_hot,
                self.W_t,
                self.intent_velocity_body,
                dwell_norm,
                react_norm,
                threat,
                self.last_pilot_action,
            ],
            dim=-1,
        )

    def debug_state(self):
        return {
            "intent_mode": self.intent_mode.clone(),
            "react_mode": self.react_mode.clone(),
            "threat": self.threat.clone(),
            "intent_velocity_body": self.intent_velocity_body.clone(),
            "final_velocity_body": self.final_velocity_body.clone(),
            "perceived_dist": self.perceived_dist.clone(),
        }

    def _sample_profile(self, ids: torch.Tensor):
        self.alpha[ids] = self._uniform(len(ids), *self.cfg.alpha_range)
        self.beta[ids] = self._uniform(len(ids), *self.cfg.beta_range)
        self.psi[ids] = self._uniform(len(ids), *self.cfg.psi_range)
        self.phi[ids] = self._uniform(len(ids), *self.cfg.phi_range)
        self.eta[ids] = self._uniform(len(ids), *self.cfg.eta_range)
        self.tau_perc[ids] = self._uniform(len(ids), *self.cfg.tau_perc_range)
        self.sigma_perc[ids] = self._uniform(len(ids), *self.cfg.sigma_perc_range)
        self.d_react[ids] = self._uniform(len(ids), *self.cfg.d_react_range)
        self.wrong_direction_prob[ids] = self._uniform(
            len(ids), *self.cfg.wrong_direction_prob_range
        )

    def _sample_intent_mode(self, ids: torch.Tensor):
        conservative = torch.tensor(
            self.cfg.conservative_mode_prior, device=self.device
        )
        aggressive = torch.tensor(self.cfg.aggressive_mode_prior, device=self.device)
        priors = torch.lerp(
            conservative.unsqueeze(0),
            aggressive.unsqueeze(0),
            self.eta[ids].unsqueeze(-1),
        )
        if self.mode_prior_override is not None:
            priors = priors * self.mode_prior_override.unsqueeze(0)
        priors = priors / priors.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        self.intent_mode[ids] = torch.multinomial(
            priors, 1, generator=self.rng
        ).squeeze(-1)

    def _sample_dwell(self, ids: torch.Tensor):
        mode_names = {
            IntentMode.CRUISE: "cruise",
            IntentMode.MANEUVER: "maneuver",
            IntentMode.STATION_KEEP: "station_keep",
            IntentMode.IDLE: "idle",
        }
        dwell = torch.empty(len(ids), device=self.device)
        for mode, name in mode_names.items():
            mask = self.intent_mode[ids] == int(mode)
            if not mask.any():
                continue
            local_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            mu, sigma = self.cfg.dwell_lognormal[name]
            normal = torch.randn(len(local_ids), device=self.device, generator=self.rng)
            dwell_sec = torch.exp(normal * sigma + mu)
            dwell_sec = dwell_sec * (0.75 + 0.75 * self.beta[ids[local_ids]])
            dwell[local_ids] = dwell_sec.clamp_min(0.25)
        self.dwell_steps[ids] = torch.ceil(dwell / self.dt).long().clamp_min(1)

    def _refresh_waypoint(self, ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor):
        forward_w = self._forward_world(quat)
        side_w = self._right_world(quat)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward_w)
        horizon = self.cfg.horizon_sec
        targets = pos.clone()
        for local_idx, env_id in enumerate(ids.tolist()):
            current_mode = IntentMode(int(self.intent_mode[env_id].item()))
            if current_mode == IntentMode.CRUISE:
                speed = self.phi[env_id] * self.max_speed
                targets[local_idx] = pos[local_idx] + self.heading_dir[env_id] * speed * horizon
            elif current_mode == IntentMode.MANEUVER:
                lateral = (
                    (torch.rand(1, device=self.device, generator=self.rng).item() * 2.0 - 1.0)
                    * self.cfg.maneuver_lateral_scale
                    * self.max_speed
                )
                vertical = (
                    (torch.rand(1, device=self.device, generator=self.rng).item() * 2.0 - 1.0)
                    * self.cfg.maneuver_vertical_scale
                    * self.max_speed
                )
                targets[local_idx] = (
                    pos[local_idx]
                    + forward_w[local_idx] * self.max_speed * 0.9
                    + side_w[local_idx] * lateral
                    + up[local_idx] * vertical
                )
            elif current_mode == IntentMode.STATION_KEEP:
                targets[local_idx] = self.anchor[env_id]
            else:
                targets[local_idx] = pos[local_idx]
        self.W_t[ids] = targets

    def _tick_intent_layer(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        assistant_action: torch.Tensor,
    ):
        drift_scale = (self.dt / max(self.cfg.heading_ou_tau, self.dt)) * self.cfg.heading_ou_sigma
        ou_noise = torch.randn(
            self.num_envs, 3, device=self.device, generator=self.rng
        ) * drift_scale
        self.heading_dir = F.normalize(
            self.heading_dir + ou_noise, dim=-1, eps=1e-6
        )
        self.dwell_steps = torch.clamp(self.dwell_steps - 1, min=0)
        resample_ids = (self.dwell_steps == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._sample_intent_mode(resample_ids)
            self._sample_dwell(resample_ids)

        desired_target = self._sample_targets_from_mode(pos, quat)
        assistant_world = quat_rotate(quat, assistant_action)
        pilot_world = quat_rotate(quat, self.last_pilot_action)
        feedback = (assistant_world - pilot_world) * self.cfg.react_feedback_gain
        mix = self.cfg.waypoint_mix
        self.W_t = (1.0 - mix) * self.W_t + mix * desired_target + self.alpha.unsqueeze(-1) * feedback

    def _sample_targets_from_mode(self, pos: torch.Tensor, quat: torch.Tensor):
        forward_w = self._forward_world(quat)
        side_w = self._right_world(quat)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward_w)
        targets = pos.clone()
        horizon = self.cfg.horizon_sec

        cruise_mask = self.intent_mode == int(IntentMode.CRUISE)
        if cruise_mask.any():
            speed = self.phi[cruise_mask].unsqueeze(-1) * self.max_speed
            targets[cruise_mask] = pos[cruise_mask] + self.heading_dir[cruise_mask] * speed * horizon

        maneuver_mask = self.intent_mode == int(IntentMode.MANEUVER)
        if maneuver_mask.any():
            lat = (
                torch.rand(maneuver_mask.sum(), 1, device=self.device, generator=self.rng) * 2.0
                - 1.0
            ) * self.cfg.maneuver_lateral_scale * self.max_speed
            vert = (
                torch.rand(maneuver_mask.sum(), 1, device=self.device, generator=self.rng) * 2.0
                - 1.0
            ) * self.cfg.maneuver_vertical_scale * self.max_speed
            targets[maneuver_mask] = (
                pos[maneuver_mask]
                + forward_w[maneuver_mask] * self.max_speed * 0.8
                + side_w[maneuver_mask] * lat
                + up[maneuver_mask] * vert
            )

        station_mask = self.intent_mode == int(IntentMode.STATION_KEEP)
        if station_mask.any():
            targets[station_mask] = self.anchor[station_mask]

        idle_mask = self.intent_mode == int(IntentMode.IDLE)
        if idle_mask.any():
            targets[idle_mask] = pos[idle_mask]
        return targets

    def _update_reactive_layer(self):
        self.react_steps = torch.clamp(self.react_steps - 1, min=0)
        self.react_mode = torch.where(
            self.react_steps > 0,
            self.react_mode,
            torch.full_like(self.react_mode, int(ReactMode.NONE)),
        )
        spontaneous = torch.rand(
            self.num_envs, device=self.device, generator=self.rng
        ) < (self.cfg.spontaneous_react_rate_hz * self.dt)
        should_trigger = (self.threat > self.cfg.react_trigger_threshold) | spontaneous
        trigger_ids = should_trigger.nonzero(as_tuple=False).squeeze(-1)
        if len(trigger_ids) == 0:
            return

        novice_score = (
            0.35 * self._normalize_range(self.tau_perc, self.cfg.tau_perc_range)
            + 0.30 * self._normalize_range(self.sigma_perc, self.cfg.sigma_perc_range)
            + 0.20 * (1.0 - self._normalize_range(self.d_react, self.cfg.d_react_range))
            + 0.15 * self.wrong_direction_prob
        )
        logits = torch.stack(
            [
                torch.full_like(novice_score, -8.0),
                0.15 + novice_score,
                0.35 + novice_score,
                0.55 + (1.0 - self.alpha),
                0.30 + (1.0 - self.eta),
                0.70 + self.eta,
                0.20 + novice_score + (1.0 - self.alpha),
                0.20 + self.eta + (1.0 - self.alpha),
            ],
            dim=-1,
        )
        probs = torch.softmax(logits[trigger_ids], dim=-1)
        sampled = torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)
        sampled = torch.clamp(sampled, min=int(ReactMode.NO_REACT))
        self.react_mode[trigger_ids] = sampled

        dwell_sec = self._uniform(
            len(trigger_ids),
            self.cfg.react_min_dwell_sec,
            self.cfg.react_max_dwell_sec,
        )
        dwell_steps = torch.ceil(dwell_sec / self.dt).long().clamp_min(1)
        self.react_steps[trigger_ids] = dwell_steps
        self.react_total_steps[trigger_ids] = dwell_steps

    def _intent_to_velocity(self, pos: torch.Tensor, quat: torch.Tensor):
        error_w = self.W_t - pos
        dist = error_w.norm(dim=-1, keepdim=True)
        desired_speed = self.phi.unsqueeze(-1) * self.max_speed
        ramp = (dist / max(self.cfg.horizon_sec, self.dt)).clamp(0.0, 1.0)
        desired_world = F.normalize(error_w, dim=-1, eps=1e-6) * desired_speed * ramp

        station_mask = self.intent_mode == int(IntentMode.STATION_KEEP)
        if station_mask.any():
            desired_world[station_mask] *= self.cfg.station_keep_speed_scale
        idle_mask = self.intent_mode == int(IntentMode.IDLE)
        if idle_mask.any():
            desired_world[idle_mask] *= self.cfg.idle_speed_scale

        desired_body = quat_rotate_inverse(quat, desired_world)
        desired_body[:, 0:2] = desired_body[:, 0:2].clamp(-self.max_speed, self.max_speed)
        desired_body[:, 2] = desired_body[:, 2].clamp(-self.max_speed_z, self.max_speed_z)
        return desired_body

    def _apply_reactive_overlay(self, intent_velocity: torch.Tensor):
        velocity = intent_velocity.clone()
        react_progress = 1.0 - (
            self.react_steps.float() / self.react_total_steps.float().clamp_min(1.0)
        )
        normal = self._sanitize_normal(self.perceived_normal)

        emergency_mask = self.react_mode == int(ReactMode.EMERGENCY_STOP)
        if emergency_mask.any():
            fade = (1.0 - react_progress[emergency_mask]).unsqueeze(-1)
            velocity[emergency_mask] *= fade

        freeze_mask = self.react_mode == int(ReactMode.FREEZE)
        if freeze_mask.any():
            velocity[freeze_mask] = 0.0

        late_mask = self.react_mode == int(ReactMode.LATE_REACT)
        if late_mask.any():
            late_threshold = self.cfg.react_trigger_threshold * self.cfg.late_react_threshold_scale
            active = self.threat[late_mask] > late_threshold
            if active.any():
                local_ids = late_mask.nonzero(as_tuple=False).squeeze(-1)[active]
                velocity[local_ids] += normal[local_ids] * (0.55 * self.max_speed)

        evade_mask = self.react_mode == int(ReactMode.EVADE)
        if evade_mask.any():
            direction = normal[evade_mask]
            wrong = (
                torch.rand(evade_mask.sum(), device=self.device, generator=self.rng)
                < self.wrong_direction_prob[evade_mask]
            )
            direction[wrong] = -direction[wrong]
            velocity[evade_mask] += direction * (
                self.cfg.evade_scale * self.threat[evade_mask].unsqueeze(-1) * self.max_speed
            )

        overcorrect_mask = self.react_mode == int(ReactMode.OVERCORRECT)
        if overcorrect_mask.any():
            phase = (
                self.step_counter[overcorrect_mask] * self.dt * self.cfg.overcorrect_osc_hz * 2.0 * torch.pi
            )
            oscillation = torch.stack(
                [torch.zeros_like(phase), torch.sin(phase), 0.2 * torch.cos(phase)],
                dim=-1,
            )
            velocity[overcorrect_mask] = (
                -0.55 * intent_velocity[overcorrect_mask] + oscillation * self.max_speed
            )

        surge_mask = self.react_mode == int(ReactMode.SURGE)
        if surge_mask.any():
            velocity[surge_mask] *= self.cfg.surge_scale

        velocity[:, 0:2] = velocity[:, 0:2].clamp(-self.max_speed, self.max_speed)
        velocity[:, 2] = velocity[:, 2].clamp(-self.max_speed_z, self.max_speed_z)
        return velocity

    def _forward_world(self, quat: torch.Tensor):
        forward = torch.zeros(quat.shape[0], 3, device=self.device)
        forward[:, 0] = 1.0
        return F.normalize(quat_rotate(quat, forward), dim=-1, eps=1e-6)

    def _right_world(self, quat: torch.Tensor):
        right = torch.zeros(quat.shape[0], 3, device=self.device)
        right[:, 1] = 1.0
        return F.normalize(quat_rotate(quat, right), dim=-1, eps=1e-6)

    def _uniform(self, size: int, low: float, high: float):
        return torch.rand(size, device=self.device, generator=self.rng) * (high - low) + low

    @staticmethod
    def _normalize_range(value: torch.Tensor, value_range: tuple[float, float]):
        low, high = value_range
        return ((value - low) / max(high - low, 1e-6)).clamp(0.0, 1.0)

    @staticmethod
    def _sanitize_distance(distance: torch.Tensor):
        return torch.nan_to_num(
            distance,
            nan=float("inf"),
            posinf=float("inf"),
            neginf=0.0,
        ).clamp_min(0.0)

    @staticmethod
    def _sanitize_normal(normal: torch.Tensor):
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        normal_norm = normal.norm(dim=-1, keepdim=True)
        return torch.where(
            normal_norm > 1e-6,
            normal / normal_norm.clamp_min(1e-6),
            torch.zeros_like(normal),
        )

    def _raise_if_nonfinite_pilot_action(
        self,
        pilot_action: torch.Tensor,
        assistant_action: torch.Tensor,
        nearest_dist: torch.Tensor,
        nearest_normal: torch.Tensor,
        stage: str,
    ):
        if torch.isfinite(pilot_action).all():
            return
        bad_ids = (~torch.isfinite(pilot_action)).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
        sample_ids = bad_ids[:5]

        def sample(tensor: torch.Tensor):
            return tensor.detach()[sample_ids].cpu().tolist()

        context = {
            "stage": stage,
            "env_ids": sample_ids.detach().cpu().tolist(),
            "pilot_action": sample(pilot_action),
            "assistant_action": sample(assistant_action),
            "nearest_dist": sample(nearest_dist),
            "nearest_normal": sample(nearest_normal),
            "perceived_dist": sample(self.perceived_dist),
            "perceived_normal": sample(self.perceived_normal),
            "intent_velocity_body": sample(self.intent_velocity_body),
            "final_velocity_body": sample(self.final_velocity_body),
            "react_mode": sample(self.react_mode),
            "threat": sample(self.threat),
        }
        raise FloatingPointError(f"Non-finite pilot_action in UserModelIntent.step: {context}")
