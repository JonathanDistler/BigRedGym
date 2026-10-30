"""Empirical backward reachable-set curriculum around front-foot balance."""

import mujoco
import numpy as np
import torch

from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.reachability_physics import ContactSampler


class Go2Reachability(Go2Balance):
    def _init_buffers(self):
        super()._init_buffers()
        self.contact_failed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.reset_accepted = torch.zeros_like(self.contact_failed)
        self.reset_attempts = torch.zeros(self.num_envs, device=self.device)
        self.initial_joints = torch.zeros_like(self.dof_pos)
        self.initial_velocity = torch.zeros_like(self.dof_vel)
        self.velocity_envelope = torch.zeros(self.num_envs, device=self.device)
        self._sampler = ContactSampler(self)

    def reset_to_handstand(self, reset_mask):
        super().reset_to_handstand(reset_mask)
        s = self._sampler
        c = self.cfg.reachability
        for i in reset_mask.nonzero().flatten().tolist():
            root = self.root_states[i].cpu().numpy().copy()
            reference = self.dof_pos[i].cpu().numpy().copy()
            level = int(self._reset_level[i].clamp(min=0))
            self.reset_accepted[i] = False
            self.velocity_envelope[i] = 0
            self.reset_attempts[i] = 0
            for attempt in range(c.attempts):
                self.reset_attempts[i] = attempt + 1
                draw = torch.rand(6).numpy()
                target = s.limits[6:, 0] + draw * np.diff(s.limits[6:], axis=1).ravel()
                joints = reference.copy()
                joints[6:] += c.position_fraction[level] * (target - joints[6:])
                s.load(root, joints)
                direction = 2 * torch.rand(6).numpy() - 1
                direction /= max(np.linalg.norm(direction), 1e-10)
                velocity, inertia = s.direction(direction)
                bound = s.bound(joints, direction, inertia)
                amplitude = c.velocity_fraction[level] * float(torch.rand(())) * bound
                velocity *= amplitude
                # Include compensating planted-leg motors in the speed check.
                ratio = np.max(np.abs(velocity[s.v]) / s.speed)
                velocity /= max(1.0, ratio)
                if not s.probe(velocity, joints):
                    continue
                self.dof_pos[i] = torch.as_tensor(joints, device=self.device)
                self.dof_vel[i] = torch.as_tensor(velocity[s.v], device=self.device)
                self.root_states[i, 7:10] = torch.as_tensor(
                    velocity[:3], device=self.device
                )
                # MuJoCo base angular velocity is local; task root uses world frame.
                rotation = np.empty(9)

                mujoco.mju_quat2Mat(rotation, root[[6, 3, 4, 5]])
                self.root_states[i, 10:13] = torch.as_tensor(
                    rotation.reshape(3, 3) @ velocity[3:6], device=self.device
                )
                self.velocity_envelope[i] = bound
                self.reset_accepted[i] = True
                break
            # Rejected proposals fall back to goal replay and cannot promote curriculum.
            self._primary_reset[i] &= self.reset_accepted[i]
            self.initial_joints[i] = self.dof_pos[i]
            self.initial_velocity[i] = self.dof_vel[i]
            self.contact_failed[i] = False

    def _post_physics_step(self):
        super()._post_physics_step()
        c = self.cfg.reachability
        active = self.episode_length_buf * self.dt >= c.contact_grace_seconds
        loads = self.contact_forces[:, self.feet_indices[:2], 2]
        speed = self._rigid_body_lin_vel[:, self.feet_indices[:2]].norm(dim=2)
        self.contact_failed |= active & (
            (loads < c.min_normal_force).any(dim=1)
            | (speed > c.max_slip_speed).any(dim=1)
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated |= self.contact_failed

    def _record_handstand_episodes(self, reset_mask):
        if hasattr(self, "contact_failed"):
            self._best_hold[self.contact_failed] = 0
        super()._record_handstand_episodes(reset_mask)
