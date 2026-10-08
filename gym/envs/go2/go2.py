import torch

from gym.envs.base.legged_robot import LeggedRobot
# from learning.utils.logger.SaveStates import (
#     init_env_log_buffers,
# )


class Go2(LeggedRobot):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)

    def _standing_posture(self):
        from gym.envs.go2.go2_config import LEG_LENGTH, FOOT_RADIUS

        extension = (self.commands[:, 3:4] - FOOT_RADIUS) / (2 * LEG_LENGTH)
        thigh = torch.acos(extension.clamp(0.0, 1.0))
        return torch.cat((torch.zeros_like(thigh), thigh, -2 * thigh), dim=1).repeat(
            1, 4
        )

    def _compute_torques(self):
        # The learned action is a balance residual around the requested height.
        pos = self.dof_pos.index_select(1, self.actuated_dof_indices)
        vel = self.dof_vel.index_select(1, self.actuated_dof_indices)
        target = self._standing_posture() + self.dof_pos_target
        torques = self.p_gains * (target - pos)
        torques += self.d_gains * (self.dof_vel_target - vel) + self.tau_ff
        return torques.clamp(-self.actuated_torque_limits, self.actuated_torque_limits)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self.dof_pos_target[reset_mask] = 0.0
        self.dof_pos_history[reset_mask] = 0.0

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        stopped = command_mask & (
            torch.rand(self.num_envs, device=self.device)
            < getattr(self.cfg.commands, "standing_probability", 0.0)
        )
        self.commands[stopped, :3] = 0.0

    def _reward_unwanted_motion(self):
        error = (self.base_lin_vel[:, :2] - self.commands[:, :2]).square().sum(dim=1)
        return -error - (self.base_ang_vel[:, 2] - self.commands[:, 2]).square()

    def _reward_residual_motion(self):
        return -self.dof_pos_target.square().mean(dim=1)

    def _reward_lin_vel_z(self):
        """Penalize z axis base linear velocity with squared exp"""
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """Penalize xy axes base angular velocity"""
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """Penalize non-flat base orientation"""
        error = (
            torch.square(self.projected_gravity[:, :2])
            / self.cfg.reward_settings.tracking_sigma
        )
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """Squared exponential saturating at base_height target"""
        error = self.base_height - self.cfg.reward_settings.base_height_target
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """Tracking of linear velocity commands (xy axes)"""
        # just use lin_vel?
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # * scale by (1+|cmd|): if cmd=0, no scaling.
        error *= 1.0 / (1.0 + torch.abs(self.commands[:, :2]))
        error = torch.sum(torch.square(error), dim=1)
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracking of angular velocity commands (yaw)"""
        ang_vel_error = torch.square(
            (self.commands[:, 2] - self.base_ang_vel[:, 2]) / 2.5
        )
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """Penalize dof velocities"""
        return torch.sum(self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]
            ),
            dim=1,
        )
