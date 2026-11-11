"""Opt-in residual around a frozen actor; no reference or CPG is duplicated."""

import copy

import torch
from torch import nn
from torch.distributions import Normal


class FrozenBalanceResidual(nn.Module):
    """Actor inputs are the existing 59 externally scaled deployable features.

    Gate uses only IMU gravity/angular rate and measured joint velocities.
    Limits are radians. The inherited reference is added solely by Go2.
    """

    def __init__(self, balance_actor, max_correction=0.04):
        super().__init__()
        if not 0 < max_correction <= 0.08:
            raise ValueError("Correction limit must be in (0, 0.08] radians")
        self.balance = copy.deepcopy(balance_actor).eval().requires_grad_(False)
        self.recovery = nn.Sequential(
            nn.Linear(59, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
        )
        nn.init.zeros_(self.recovery[-1].weight)
        nn.init.zeros_(self.recovery[-1].bias)
        # Fixed exploration is over latent corrections, never baseline actions.
        self.register_buffer("latent_std", torch.full((12,), 0.15))
        self.max_correction = float(max_correction)

    def train(self, mode=True):
        super().train(mode)
        self.balance.eval()
        return self

    @staticmethod
    def gate(obs):
        tilt = torch.acos(obs[:, 3].clamp(-1, 1))
        omega = (obs[:, :3] * 0.3).norm(dim=-1)
        scales = obs.new_tensor([2.0, 2.0, 4.0] * 4)
        joint_speed = (obs[:, 22:34] * scales).abs().amax(-1)
        # Exactly zero inside the confidence region; smoothstep outside it.
        severity = (
            torch.stack(
                (
                    (tilt - 0.05236) / 0.20944,
                    (omega - 0.8) / 1.2,
                    (joint_speed - 6.0) / 10.0,
                ),
                dim=-1,
            )
            .amax(-1)
            .clamp(0, 1)
        )
        return severity.square() * (3 - 2 * severity)

    def distribution(self, obs):
        return Normal(self.recovery(obs), self.latent_std)

    def correction(self, obs, latent=None):
        latent = self.recovery(obs) if latent is None else latent
        return self.max_correction * self.gate(obs)[:, None] * latent.tanh()

    def combined(self, obs, correction):
        with torch.no_grad():
            baseline = self.balance.act_inference(obs)
        # Explicit bypass ensures bitwise command parity at zero correction.
        return torch.where(correction == 0, baseline, baseline + correction / 0.25)


class ResidualAdapter:
    """Attach to an existing Policy without changing its environment/reference.

    Rate-limit *additional* displacement using native velocity bounds. Baseline
    balance has no persistent joint-target slew limiter, and adding one would
    break parity. Existing downstream transition target limits remain in place.
    Nonzero corrections never worsen baseline joint-target/rate violations.
    """

    def __init__(self, policy, model):
        self.policy, self.model = policy, model
        self.previous = torch.zeros(policy.env.num_envs, 12)
        self.last = self.previous.clone()
        self.last_gate = torch.zeros(policy.env.num_envs)
        self.maximum_target_error = 0.0
        self.telemetry_calls = 0
        self.correction_square_sum = 0.0
        self.activation_sum = 0.0

    def reset(self, mask=None):
        if mask is None:
            self.previous.zero_()
        else:
            self.previous[mask] = 0

    @torch.no_grad()
    def apply(self, latent=None):
        p, e = self.policy, self.policy.env
        obs = p.runner.get_obs(p.runner.actor_cfg["obs"])
        delta = self.model.correction(obs, latent)
        velocity = e.dof_vel_limits[e.actuated_dof_indices] * e.dt
        delta = self.previous + (delta - self.previous).clamp(-velocity, velocity)
        baseline = self.model.balance.act_inference(obs)
        target = e._standing_posture() + 0.25 * baseline
        limits = e.dof_pos_limits[e.actuated_dof_indices]
        lo = torch.minimum(limits[:, 0] - target, torch.zeros_like(target))
        hi = torch.maximum(limits[:, 1] - target, torch.zeros_like(target))
        delta = delta.clamp(lo, hi)
        # Stable gate is a hard protection; zero head and zero state bypass all
        # additional limiting, preserving original targets at initialization.
        delta = torch.where(self.model.gate(obs)[:, None] == 0, 0.0, delta)
        command = torch.where(delta == 0, baseline, baseline + delta / 0.25)
        p.runner.set_actions(p.runner.actor_cfg["actions"], command, False)
        self.previous.copy_(delta)
        self.last.copy_(delta)
        self.last_gate.copy_(self.model.gate(obs))
        self.telemetry_calls += 1
        self.correction_square_sum += float(delta.square().mean())
        self.activation_sum += float((delta.abs().amax(-1) > 1e-6).float().mean())
        if not delta.any():
            error = (e._standing_posture() + e.dof_pos_target - target).abs().max()
            self.maximum_target_error = max(self.maximum_target_error, float(error))
        return obs, delta
