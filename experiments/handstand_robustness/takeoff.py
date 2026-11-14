"""Takeoff-only, bounded residual feedback and instrumented native simulation.

A phase-indexed nominal demonstration is explicitly retained as a prior. Only
state-dependent residuals are learned. This is not evidence of learning takeoff
from scratch, nor a full-maneuver controller. Critic/reward may use simulator state.
"""

from collections import defaultdict

import mujoco
import numpy as np
import torch
from torch import nn

from experiments.handstand_robustness.core import Environment
from gym.envs.go2.handstand_replay import CONTROL_DT


class ResidualPolicy(nn.Module):
    def __init__(self, actions, centers, bound=0.05):
        super().__init__()
        self.register_buffer("actions", torch.as_tensor(actions, dtype=torch.float32))
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.register_buffer(
            "scale",
            torch.tensor(
                [0.02] * 3 + [0.1] * 3 + [0.02] * 12 + [0.02] * 12 + [0.02] * 12
            ),
        )
        self.register_buffer("bound", torch.tensor(bound))
        self.gains = nn.Parameter(torch.zeros(len(actions), 12, 42))
        self.log_std = nn.Parameter(torch.full((12,), -3.0))
        self.critic = nn.Sequential(nn.Linear(43, 64), nn.Tanh(), nn.Linear(64, 1))

    def inputs(self, obs):
        k = (
            (obs[..., 42] * 8 / CONTROL_DT)
            .round()
            .long()
            .clamp(0, min(48, len(self.actions)) - 1)
        )
        delta = ((obs[..., :42] - self.centers[k, :42]) / self.scale).clamp(-20, 20)
        return k, delta

    def latent(self, obs):
        k, delta = self.inputs(obs)
        return torch.einsum("...ij,...j->...i", self.gains[k], delta)

    def forward(self, obs, feedback=True):
        k, _ = self.inputs(obs)
        correction = self.bound * self.latent(obs).tanh() if feedback else 0.0
        return (self.actions[k] + correction).clamp(-1, 1)

    def distribution(self, obs):
        return torch.distributions.Normal(
            self.latent(obs), self.log_std.clamp(-5, -1).exp()
        )

    def action(self, obs, latent):
        k, _ = self.inputs(obs)
        return (self.actions[k] + self.bound * latent.tanh()).clamp(-1, 1)

    def value(self, obs):
        _, delta = self.inputs(obs)
        return self.critic(torch.cat([delta, obs[..., 42:43]], -1)).squeeze(-1)


class TakeoffEnvironment(Environment):
    def __init__(self, model, trajectory, seed=0, condition=None, horizon=48):
        self.horizon = horizon
        self.max_rear_clearance = -float("inf")
        self.max_torque_ratio = self.max_speed_ratio = 0.0
        self.motor_peak_Nm = np.zeros(12)
        self.saturation_steps = self.physics_steps = 0
        self.rear_contact_loss = False
        self.collision_bodies = set()
        self.reward_totals = defaultdict(float)
        self.finite = True
        super().__init__(model, trajectory, seed, condition)

    def record_metrics(self):
        # The strict gate is still evaluated at every physics step by the base
        # environment; expensive hold-only KE metrics are unnecessary for this pilot.
        if self.horizon > 48:
            super().record_metrics()
        rear = self.d.geom_xpos[self.foot_geoms[2:], 2] - 0.022
        self.max_rear_clearance = max(self.max_rear_clearance, float(rear.min()))
        tau = abs(self.d.qfrc_applied[self.vids])
        self.motor_peak_Nm = np.maximum(self.motor_peak_Nm, tau)
        self.max_torque_ratio = max(
            self.max_torque_ratio, float((tau / self.effort).max())
        )
        self.max_speed_ratio = max(
            self.max_speed_ratio, float(abs(self.d.qvel[self.vids] / self.speed).max())
        )
        self.saturation_steps += int((tau >= self.effort * 0.999).any())
        self.physics_steps += 1
        grounded = set()
        for contact in self.d.contact:
            if contact.dist > 0:
                continue
            bodies = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 in bodies:
                grounded.update(int(b) for b in bodies if b)
            if all(bodies) or any(b and b not in self.feet for b in bodies):
                self.collision_bodies.update(
                    self.m.body(int(b)).name for b in bodies if b
                )
        self.rear_contact_loss |= not any(b in grounded for b in self.feet[2:])

    def step(self, action):
        before = self.takeoff
        obs, old_reward, done = super().step(action)
        # Dense nominal tracking prevents rewarding a static standing strategy.
        ref = mujoco.MjData(self.m)
        from gym.envs.go2.handstand_replay import STATE_SPEC

        mujoco.mj_setState(
            self.m, ref, self.trajectory["reference_states"][self.index], STATE_SPEC
        )
        joint_error = np.mean((self.d.qpos[self.qids] - ref.qpos[self.qids]) ** 2)
        tilt_error = 1 - abs(float(self.d.qpos[3:7] @ ref.qpos[3:7]))
        tracking = float(np.exp(-20 * joint_error - 20 * tilt_error))
        clear = max(
            0.0, float((self.d.geom_xpos[self.foot_geoms[2:], 2] - 0.022).min())
        )
        terms = dict(
            tracking=0.02 * tracking,
            clearance=0.02
            * min(clear / 0.15, 1.0)
            * int(self.captured or self.takeoff),
            takeoff_event=2.0 * int(self.takeoff and not before),
            collision=-2.0 * int(not self.safe),
            residual_cost=-0.001
            * float(
                np.mean(
                    (np.asarray(action) - self.trajectory["actions"][self.index - 1])
                    ** 2
                )
            ),
            legacy_reward=old_reward,
        )
        self.last_terms = terms
        for key, value in terms.items():
            self.reward_totals[key] += value
        reward = sum(v for k, v in terms.items() if k != "legacy_reward")
        self.finite &= bool(
            np.isfinite(obs).all()
            and np.isfinite(reward)
            and np.isfinite(self.d.qfrc_applied).all()
        )
        return obs, reward, done or self.index >= self.horizon

    def result(self):
        return super().result() | dict(
            standing_start=True,
            horizon_steps=self.horizon,
            control_steps=self.index,
            takeoff_success=bool(self.takeoff and self.safe),
            max_simultaneous_rear_clearance_m=self.max_rear_clearance,
            rear_contact_loss=bool(self.rear_contact_loss),
            torque_ratio_max=self.max_torque_ratio,
            joint_speed_ratio_max=self.max_speed_ratio,
            motor_peak_Nm=self.motor_peak_Nm.tolist(),
            torque_saturation_fraction=self.saturation_steps
            / max(1, self.physics_steps),
            collision_bodies=sorted(self.collision_bodies),
            finite=self.finite,
            reward_components=dict(self.reward_totals),
        )
