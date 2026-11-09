"""Standing-start kick reference and ground-relative, front-anchored shaping."""

import math
import torch

from gym.envs.go2.go2handstand import Go2Handstand
from gym.envs.go2.go2balance import BalanceDiagnostics
from gym.envs.go2.go2_config import FOOT_RADIUS


class KickupDiagnostics(BalanceDiagnostics):
    def __init__(self, device):
        self.names = (
            "rear_clearance_m",
            "front_slip_speed_m_s",
            "front_displacement_m",
            "success",
        )
        self.sums = torch.zeros(len(self.names), device=device)
        self.reset()
        self.finish()


class Go2Kickup(Go2Handstand):
    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.terrain.mesh_type != "plane":
            raise ValueError(
                "kick-up clearance currently requires the z=0 ground plane"
            )
        c = self.cfg.kick_curriculum
        n = len(self.cfg.handstand_curriculum.pitch_degrees)
        if any(
            len(getattr(c, name)) != n
            for name in (
                "clearance",
                "amplitude",
                "pitch_degrees",
                "front_drift",
                "front_speed",
            )
        ):
            raise ValueError("kick curriculum tables must match reset curriculum")
        m = self.cfg.kick_motion
        if m.crouch_seconds <= 0 or m.extend_seconds <= 0:
            raise ValueError("kick motion durations must be positive")
        self._kick_clock = torch.zeros(self.num_envs, device=self.device)
        self._front_anchor = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._front_anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._support_loss_time = torch.zeros_like(self._kick_clock)
        self._support_failed = torch.zeros_like(self._front_anchor_valid)
        self.training_diagnostics = KickupDiagnostics(self.device)
        self.diagnostics_category = "kickup"

    def _goal(self, name):
        table = torch.tensor(
            getattr(self.cfg.kick_curriculum, name), device=self.device
        )
        # Episodes in flight retain their own level when a batch promotes.
        return table[self._reset_level.clamp(min=0)]

    @property
    def kick_support_obs(self):
        offset = self._rigid_body_pos[:, self.feet_indices[:2], :2] - self._front_anchor
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[:2], :2]
        return torch.cat(
            (
                offset.flatten(1) / 0.1,
                velocity.flatten(1),
                self._rear_clearance() / 0.16,
            ),
            dim=1,
        )

    @property
    def kick_goal_obs(self):
        return torch.stack(
            (self._goal("clearance") / 0.16, self._goal("pitch_degrees") / 90), dim=1
        )

    @property
    def kick_motion_obs(self):
        m = self.cfg.kick_motion
        return (
            self._kick_clock / (m.settle_seconds + m.crouch_seconds + m.extend_seconds)
        ).clamp(0, 1)[:, None]

    def _standing_posture(self):
        standing = super()._standing_posture()
        m = self.cfg.kick_motion
        t = self._kick_clock - m.settle_seconds
        crouch = (t / m.crouch_seconds).clamp(0, 1)
        extend = ((t - m.crouch_seconds) / m.extend_seconds).clamp(0, 1)
        # Smoothstep gives zero endpoint velocity, including at the phase join.
        crouch = (crouch.square() * (3 - 2 * crouch))[:, None]
        extend = (extend.square() * (3 - 2 * extend))[:, None]
        folded = standing.clone()
        folded[:, [7, 10]] = m.rear_thigh
        folded[:, [8, 11]] = m.rear_knee
        thrust = folded.clone()
        thrust[:, [7, 10]] = m.rear_extend_thigh
        thrust[:, [8, 11]] = m.rear_extend_knee
        reference = standing.lerp(folded, crouch).lerp(thrust, extend)
        return standing + self._goal("amplitude")[:, None] * (reference - standing)

    def set_state(self, name, value):
        if name == "dof_pos_target":
            # Scale normalized actions once, then bound and slew motor residuals.
            limit = self.scales[name]
            desired = value.clamp(-1, 1) * limit
            delta = (desired - self.dof_pos_target).clamp(-0.04, 0.04)
            self.dof_pos_target = self.dof_pos_target + delta
        else:
            super().set_state(name, value)

    def _reset_idx(self, reset_mask):
        self._kick_clock[reset_mask] = 0
        super()._reset_idx(reset_mask)
        # MuJoCo reset_state performs mj_forward and refreshes link positions.
        self._front_anchor[reset_mask] = self._rigid_body_pos[
            :, self.feet_indices[:2], :2
        ][reset_mask]
        self._front_anchor_valid[reset_mask] = True
        self._support_loss_time[reset_mask] = 0
        self._support_failed[reset_mask] = False

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._kick_clock += self.dt
        contact = (
            self._handstand_loads()[:, :2] >= self.cfg.reward_settings.min_front_load
        ).all(dim=1)
        self._support_loss_time[:] = torch.where(
            (
                contact
                & (self._front_speed() <= self._goal("front_speed")[:, None]).all(dim=1)
            ),
            0.0,
            self._support_loss_time + self.dt,
        )
        self._support_failed |= (
            self._support_loss_time > self.cfg.kick_curriculum.support_grace_s
        ) | (self._front_displacement() > self._goal("front_drift")[:, None]).any(dim=1)

    def _advance_handstand_stage(self):
        # No handstand stages or handoff: this task learns only the kick.
        pass

    def _rear_clearance(self):
        # Backend ground is a world z=0 plane; body origins are foot sphere centres.
        return (self._rigid_body_pos[:, self.feet_indices[2:], 2] - FOOT_RADIUS).clamp(
            min=0
        )

    def _front_displacement(self):
        delta = self._rigid_body_pos[:, self.feet_indices[:2], :2] - self._front_anchor
        return delta.norm(dim=-1) * self._front_anchor_valid[:, None]

    def _front_speed(self):
        return self._rigid_body_lin_vel[:, self.feet_indices[:2], :2].norm(dim=-1)

    def _valid_kick_support(self):
        return (
            (
                self._handstand_loads()[:, :2]
                >= self.cfg.reward_settings.min_front_load
            ).all(dim=1)
            & (self._front_speed() <= self._goal("front_speed")[:, None]).all(dim=1)
            & (self._front_displacement() <= self._goal("front_drift")[:, None]).all(
                dim=1
            )
            & ~self._support_failed
        )

    def _update_handstand_hold(self):
        held = (
            self._valid_kick_support()
            & (self._rear_clearance() >= self._goal("clearance")[:, None]).all(dim=1)
            & ~self.terminated
        )
        self._hold_time[:] = torch.where(held, self._hold_time + self.dt, 0.0)
        self._best_hold[:] = torch.maximum(self._best_hold, self._hold_time)

    def _record_handstand_episodes(self, reset_mask):
        self._best_hold[reset_mask & (self.terminated | self._support_failed)] = 0
        super()._record_handstand_episodes(reset_mask)

    def _reward_kick_front_support(self):
        return (
            (self._handstand_loads()[:, :2] / self.cfg.reward_settings.min_front_load)
            .clamp(0, 1)
            .mean(dim=1)
        )

    def _reward_kick_clearance(self):
        lift = (self._rear_clearance() / self._goal("clearance")[:, None]).clamp(0, 1)
        # Minimum requires both feet; mean supplies a gradient to the first mover.
        quality = torch.exp(
            -self._front_speed().square().sum(dim=1) / 0.04
            - self._front_displacement().square().sum(dim=1) / 0.01
        )
        return (
            (0.75 * lift.amin(dim=1) + 0.25 * lift.mean(dim=1))
            * quality
            * self._reward_kick_front_support()
        )

    def _reward_kick_pitch(self):
        target = torch.sin(self._goal("pitch_degrees") * math.pi / 180)
        progress = (self.projected_gravity[:, 0] / target).clamp(0, 1)
        overshoot = (self.projected_gravity[:, 0] - target).clamp(min=0)
        return (
            progress
            * torch.exp(-overshoot.square() / 0.04)
            * self._reward_kick_front_support()
        )

    def _reward_kick_motion(self):
        error = (
            self.dof_pos.index_select(1, self.actuated_dof_indices)
            - self._standing_posture()
        )
        return torch.exp(
            -error.square().mean(dim=1) / self.cfg.reward_settings.motion_sigma**2
        )

    def _reward_kick_front_displacement(self):
        return -self._front_displacement().square().sum(dim=1)

    def _reward_kick_instability(self):
        # Permit forward pitch velocity while suppressing roll, yaw and violent spin.
        return -(
            self.base_ang_vel[:, [0, 2]].square().sum(dim=1)
            + (self.base_ang_vel[:, 1].abs() - 3).clamp(min=0).square()
        )

    def update_training_diagnostics(self):
        self.training_diagnostics.update(
            dict(
                rear_clearance_m=self._rear_clearance(),
                front_slip_speed_m_s=self._front_speed(),
                front_displacement_m=self._front_displacement(),
                success=self._best_hold >= self.cfg.handstand_curriculum.hold_seconds,
            )
        )
