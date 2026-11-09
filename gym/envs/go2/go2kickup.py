"""Crouch/extension reference with strict front-support kick-up success."""

import torch

from gym.envs.go2.go2handstand import Go2Handstand
from gym.envs.go2.go2balance import BalanceDiagnostics


class KickupDiagnostics(BalanceDiagnostics):
    def __init__(self, device):
        self.names = (
            "rear_foot_height_m",
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
        m = self.cfg.kick_motion
        if m.crouch_seconds <= 0 or m.extend_seconds <= 0:
            raise ValueError("kick motion durations must be positive")
        self._front_anchor = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._front_anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.training_diagnostics = KickupDiagnostics(self.device)
        self.diagnostics_category = "kickup"

    @property
    def kick_motion_obs(self):
        m = self.cfg.kick_motion
        return (
            (
                self.episode_length_buf.float()
                * self.dt
                / (m.crouch_seconds + m.extend_seconds)
            )
            .clamp(0, 1)
            .unsqueeze(1)
        )

    def _standing_posture(self):
        # The CPG supplies contact clocks, not absolute motor angles. PD targets
        # are this absolute reference plus the learned motor-position residual.
        standing = super()._standing_posture()
        m = self.cfg.kick_motion
        t = self.episode_length_buf.float() * self.dt
        crouch = (t / m.crouch_seconds).clamp(0, 1)[:, None]
        extend = ((t - m.crouch_seconds) / m.extend_seconds).clamp(0, 1)[:, None]
        folded = standing.new_tensor(
            [0, m.front_thigh, m.front_knee] * 2 + [0, m.rear_thigh, m.rear_knee] * 2
        )
        thrust = folded.clone()
        thrust[[7, 10]] = m.rear_extend_thigh
        thrust[[8, 11]] = m.rear_extend_knee
        reference = standing.lerp(folded.expand_as(standing), crouch)
        reference = reference.lerp(thrust.expand_as(standing), extend)
        # Tilted curriculum resets retain their sampled absolute pose as the
        # reference. Standing starts execute the crouch/extension primitive.
        if hasattr(self, "_kick_reset_reference"):
            reference = torch.where(
                (self._reset_stage > 0)[:, None], self._kick_reset_reference, reference
            )
        return reference

    def _reset_idx(self, reset_mask):
        # Clear stale pose references before the parent computes reset residuals.
        if not hasattr(self, "_kick_reset_reference"):
            self._kick_reset_reference = self.dof_pos.index_select(
                1, self.actuated_dof_indices
            ).clone()
        super()._reset_idx(reset_mask)
        self._kick_reset_reference[reset_mask] = self.dof_pos.index_select(
            1, self.actuated_dof_indices
        )[reset_mask]
        target = (
            self.dof_pos.index_select(1, self.actuated_dof_indices)
            - self._standing_posture()
        )
        self.dof_pos_target[reset_mask] = target[reset_mask]
        self.dof_pos_history[reset_mask] = target[reset_mask].repeat(1, 3)
        # Backend link positions may still belong to the previous episode here.
        self._front_anchor_valid[reset_mask] = False

    def _post_decimation_step(self):
        super()._post_decimation_step()
        new = ~self._front_anchor_valid
        self._front_anchor[new] = self._rigid_body_pos[new][
            :, self.feet_indices[:2], :2
        ]
        self._front_anchor_valid[new] = True

    def _front_displacement(self):
        delta = self._rigid_body_pos[:, self.feet_indices[:2], :2] - self._front_anchor
        return delta.norm(dim=-1) * self._front_anchor_valid[:, None]

    def _valid_kick_support(self):
        s = self.cfg.reward_settings
        speed = self._rigid_body_lin_vel[:, self.feet_indices[:2], :2].norm(dim=-1)
        return (
            (self._handstand_loads()[:, :2] >= s.min_front_load).all(dim=1)
            & (speed <= s.valid_front_speed).all(dim=1)
            & (self._front_displacement() <= s.max_front_displacement).all(dim=1)
            & (self._rear_clearance() >= s.min_rear_clearance).all(dim=1)
        )

    def _reward_handstand_balance(self):
        return super()._reward_handstand_balance() * self._valid_kick_support()

    def _reward_kick_clearance(self):
        clearance = (
            self._rear_clearance().amin(dim=1)
            / self.cfg.reward_settings.balance_clearance
        ).clamp(0, 1)
        return clearance * self._reward_handstand_lean()

    def _reward_kick_motion(self):
        error = (
            self.dof_pos.index_select(1, self.actuated_dof_indices)
            - self._standing_posture()
        )
        return torch.exp(
            -error.square().mean(dim=1) / self.cfg.reward_settings.motion_sigma**2
        ) * (self.handstand_stage < 2)

    def _reward_kick_front_displacement(self):
        return -self._front_displacement().square().sum(dim=1)

    def _record_handstand_episodes(self, reset_mask):
        # A kick followed by a fall cannot promote the curriculum.
        self._best_hold[reset_mask & self.terminated] = 0
        super()._record_handstand_episodes(reset_mask)

    def update_training_diagnostics(self):
        self.training_diagnostics.update(
            dict(
                rear_foot_height_m=self._rigid_body_pos[:, self.feet_indices[2:], 2],
                rear_clearance_m=self._rear_clearance(),
                front_slip_speed_m_s=self._rigid_body_lin_vel[
                    :, self.feet_indices[:2], :2
                ].norm(dim=-1),
                front_displacement_m=self._front_displacement(),
                success=self._best_hold >= self.cfg.handstand_curriculum.hold_seconds,
            )
        )
