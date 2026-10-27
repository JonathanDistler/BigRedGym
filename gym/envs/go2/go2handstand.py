"""Staged front-foot handstand primitives; all positive rewards are bounded."""

import torch

from gym.envs.go2.go2 import Go2
from gym.envs.go2.handstand_curriculum import HandstandCurriculum


class Go2Handstand(HandstandCurriculum, Go2):
    @property
    def handstand_stage_obs(self):
        return self.handstand_stage.unsqueeze(1) / 2

    def _init_buffers(self):
        super()._init_buffers()
        self.handstand_stage = torch.zeros(self.num_envs, device=self.device)
        self._lean_time = torch.zeros_like(self.handstand_stage)
        self._kick_time = torch.zeros_like(self.handstand_stage)
        self._init_handstand_curriculum()

    def _reset_idx(self, reset_mask):
        self._record_handstand_episodes(reset_mask)
        self._reset_stage[reset_mask] = 0
        super()._reset_idx(reset_mask)
        self.handstand_stage[reset_mask] = self._reset_stage[reset_mask]
        self._hold_time[reset_mask] = 0
        self._best_hold[reset_mask] = 0
        self._episode_active[reset_mask] = True
        # Match PD targets to the sampled pose; avoid snapping to standing.
        target = (
            self.dof_pos.index_select(1, self.actuated_dof_indices)
            - self._standing_posture()
        )
        self.dof_pos_target[reset_mask] = target[reset_mask]
        self.dof_pos_history[reset_mask] = target[reset_mask].repeat(1, 3)
        self._lean_time[reset_mask] = 0
        self._kick_time[reset_mask] = 0

    def _resample_commands(self, command_mask):
        super()._resample_commands(command_mask)
        self.commands[command_mask, :3] = 0
        self.commands[command_mask, 3] = self.cfg.reward_settings.base_height_target

    def _handstand_loads(self):
        # Layout contract: FL, FR, RL, RR. Forces are world-frame upward loads.
        weight = 9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)
        return self.contact_forces[:, self.feet_indices, 2].clamp(min=0) / weight

    def _reward_handstand_lean(self):
        s = self.cfg.reward_settings
        load = self._handstand_loads()
        total = load.sum(dim=1)
        front_share = load[:, :2].sum(dim=1) / total.clamp(min=1e-6)
        share = torch.exp(-((1 - front_share) / s.lean_share_sigma).square())
        # Positive projected gravity x means nose down; don't reward a rearward lean.
        pitch = (self.projected_gravity[:, 0] / s.lean_gravity_x).clamp(0, 1)
        support = torch.exp(-((total - 1) / s.support_sigma).square())
        both = (load[:, :2].amin(dim=1) / s.min_front_load).clamp(0, 1)
        return share * pitch * support * both

    def _rear_extension(self):
        s = self.cfg.reward_settings
        joints = self.dof_pos.index_select(1, self.actuated_dof_indices)
        # -0.95 rad is inside the URDF knee limit (~89% of full reach).
        return torch.exp(
            -((joints[:, [8, 11]] - s.kick_knee_target) / s.kick_knee_sigma).square()
        )

    def _rear_clearance(self):
        feet = self._rigid_body_pos[:, self.feet_indices]
        return feet[:, 2:, 2] - feet[:, :2, 2].mean(dim=1, keepdim=True)

    def _reward_handstand_kick(self):
        s = self.cfg.reward_settings
        clearance = (self._rear_clearance() / s.kick_clearance).clamp(0, 1)
        unloaded = (1 - self._handstand_loads()[:, 2:] / s.rear_load_tolerance).clamp(
            0, 1
        )
        # Either rear leg may initiate, but extension alone on the floor cannot win.
        kick = (self._rear_extension() * clearance * unloaded).amax(dim=1)
        return (self.handstand_stage >= 1) * kick

    def _reward_handstand_balance(self):
        s = self.cfg.reward_settings
        load = self._handstand_loads()
        front = load[:, :2]
        support = torch.exp(-((front.sum(dim=1) - 1) / s.support_sigma).square())
        sharing = 2 * front.amin(dim=1) / front.sum(dim=1).clamp(min=1e-6)
        rear_off = torch.exp(-(load[:, 2:].sum(dim=1) / s.rear_load_tolerance).square())
        # Nose-down vertical body: gravity in body coordinates is (+1, 0, 0).
        target = self.projected_gravity.new_tensor([1.0, 0.0, 0.0])
        vertical = torch.exp(
            -(
                (self.projected_gravity - target).square().sum(dim=1)
                / s.vertical_sigma**2
            )
        )
        clearance = (self._rear_clearance().amin(dim=1) / s.balance_clearance).clamp(
            0, 1
        )
        stable = torch.exp(
            -self.base_ang_vel.square().sum(dim=1) / s.angular_sigma**2
            - self.base_lin_vel.square().sum(dim=1) / s.linear_sigma**2
        )
        return (
            (self.handstand_stage >= 2)
            * support
            * sharing
            * rear_off
            * vertical
            * clearance
            * self._rear_extension().amin(dim=1)
            * stable
        )

    def _reward_handstand_slip(self):
        contact = self._handstand_loads()[:, :2] > 0.02
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[:2], :2]
        return -(velocity.square().sum(dim=2) * contact).sum(dim=1)

    def step(self):
        super().step()
        self._update_handstand_hold()

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._advance_handstand_stage()

    def _advance_handstand_stage(self):
        s = self.cfg.reward_settings
        lean = self._reward_handstand_lean() >= s.lean_threshold
        self._lean_time[:] = torch.where(lean, self._lean_time + self.dt, 0.0)
        # Evaluate kick against the previous stage: no skipping both gates in one step.
        kick = self._reward_handstand_kick() >= s.kick_threshold
        self._kick_time[:] = torch.where(kick, self._kick_time + self.dt, 0.0)
        self.handstand_stage[:] = torch.where(
            (self.handstand_stage == 1) & (self._kick_time >= s.kick_hold_s),
            2.0,
            self.handstand_stage,
        )
        self.handstand_stage[:] = torch.where(
            (self.handstand_stage == 0) & (self._lean_time >= s.lean_hold_s),
            1.0,
            self.handstand_stage,
        )
