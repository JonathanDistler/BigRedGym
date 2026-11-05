"""Balance around a handstand reference with success-gated perturbations."""

import math

import torch

from gym.envs.go2.go2handstand import Go2Handstand


class BalanceDiagnostics:
    """Raw means over all robots and control steps in the current rollout."""

    def __init__(self, device):
        self.names = (
            "support",
            "sharing",
            "rear_off",
            "vertical",
            "clearance",
            "rear_extension",
            "stable",
            "strict_score",
            "front_left_load",
            "front_right_load",
            "front_left_load_n",
            "front_right_load_n",
            "front_left_vx_m_s",
            "front_left_vy_m_s",
            "front_left_vz_m_s",
            "front_right_vx_m_s",
            "front_right_vy_m_s",
            "front_right_vz_m_s",
            "front_left_slip_speed_m_s",
            "front_right_slip_speed_m_s",
            "rear_joint_speed_rad_s",
            "rear_load",
            "hold_seconds",
            "terminated",
            "push_fraction",
            "push_delta_velocity",
        )
        self.sums = torch.zeros(len(self.names), device=device)
        self.reset()
        self.finish()

    @torch.inference_mode()
    def reset(self):
        self.sums.zero_()
        self.steps = 0

    @torch.inference_mode()
    def update(self, values):
        self.sums += torch.stack([values[name].float().mean() for name in self.names])
        self.steps += 1

    @torch.inference_mode()
    def finish(self):
        for name, value in zip(self.names, self.sums / max(self.steps, 1)):
            setattr(self, name, value)


class Go2Balance(Go2Handstand):
    def _init_buffers(self):
        super()._init_buffers()
        p = self.cfg.balance_perturbations
        c = self.cfg.handstand_curriculum
        if any(pitch != 90 for pitch in c.pitch_degrees) or c.standing_fraction != 0:
            raise ValueError(
                "balance-only training requires top resets without standing starts"
            )
        if len(p.max_delta_velocity) != len(
            self.cfg.handstand_curriculum.pitch_degrees
        ):
            raise ValueError("balance perturbation levels must match reset levels")
        if (
            any(v < 0 or not math.isfinite(v) for v in p.max_delta_velocity)
            or not math.isfinite(p.interval_s)
            or p.interval_s <= 0
            or (p.enabled and p.interval_s + c.hold_seconds >= self.max_episode_length_s)
            or not math.isfinite(p.settled_hold_s)
            or p.settled_hold_s < 0
            or not 0 <= p.undisturbed_fraction < 1
        ):
            raise ValueError("invalid balance perturbation limits or timing")
        self._push_table = torch.tensor(p.max_delta_velocity, device=self.device)
        self._episode_push_limit = torch.zeros(self.num_envs, device=self.device)
        self._push_count = torch.zeros(self.num_envs, device=self.device)
        self._last_push = torch.zeros(self.num_envs, device=self.device)
        self._body_contact_time = torch.zeros(self.num_envs, device=self.device)
        self._collapse_time = torch.zeros(self.num_envs, device=self.device)
        self.training_diagnostics = BalanceDiagnostics(self.device)
        knee = self.cfg.reward_settings.kick_knee_target
        front = [0.0, -math.pi / 2 - knee / 2, knee]
        rear = [0.0, math.pi / 2 - knee / 2, knee]
        self._balance_reference = self.dof_pos.new_tensor(front * 2 + rear * 2).expand(
            self.num_envs, -1
        )

    def _standing_posture(self):
        # Go2's torque controller uses this as its zero-action reference.
        # Zero residual must retain the top pose rather than fold to standing.
        return self._balance_reference

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._body_contact_time[reset_mask] = 0
        self._collapse_time[reset_mask] = 0
        p = self.cfg.balance_perturbations
        limits = self._push_table[self._reset_level.clamp(min=0)]
        disturbed = (
            torch.rand(self.num_envs, device=self.device) >= p.undisturbed_fraction
        )
        limits = torch.where(disturbed & p.enabled, limits, 0.0)
        self._episode_push_limit[reset_mask] = limits[reset_mask]
        # Keep undisturbed examples, but only challenged primary episodes can
        # promote a level that requires impulse recovery.
        required = p.enabled & (self._push_table[self._reset_level.clamp(min=0)] > 0)
        self._primary_reset[reset_mask] &= (~required | disturbed)[reset_mask]
        self._push_count[reset_mask] = 0
        self._last_push[reset_mask] = 0

    def _record_handstand_episodes(self, reset_mask):
        # An early hold followed by a fall must not promote long-hold training.
        self._best_hold[reset_mask & self.terminated] = 0
        if hasattr(self, "_push_count"):
            # A challenged episode ending before its first push is a failure,
            # not an excluded sample or a successful pre-push hold.
            missed_push = (
                reset_mask & (self._episode_push_limit > 0) & (self._push_count == 0)
            )
            self._best_hold[missed_push] = 0
        super()._record_handstand_episodes(reset_mask)

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._last_push.zero_()
        p = self.cfg.balance_perturbations
        if not p.enabled or p.max_delta_velocity[self.curriculum_level] == 0:
            return
        interval = max(1, math.ceil(p.interval_s / self.dt))
        eligible = (
            (self.episode_length_buf > 0)
            & (self.episode_length_buf % interval == 0)
            & (self._episode_push_limit > 0)
            & (self._push_count == 0)
            & (self._hold_time >= p.settled_hold_s)
            & ~self.terminated
            & ~self.timed_out
        )
        # Native root writes are required only on frames containing an impulse.
        if not eligible.any():
            return
        angle = torch.rand(self.num_envs, device=self.device) * (2 * math.pi)
        speed = self._episode_push_limit
        delta = torch.stack((angle.cos(), angle.sin()), dim=1) * speed[:, None]
        self.root_states[eligible, 7:9] += delta[eligible]
        self._backend.set_all_root_states()
        self._push_count[eligible] += 1
        self._last_push[eligible] = speed[eligible]
        # A hold achieved before the push is not a successful recovery.
        self._hold_time[eligible] = 0
        self._best_hold[eligible] = 0

    def _post_decimation_step(self):
        super()._post_decimation_step()
        c = self.cfg.balance_failure
        active = self.episode_length_buf * self.dt >= c.contact_grace_seconds
        collapsed = self.root_states[:, 2] < self.cfg.env.min_base_height
        self._body_contact_time[:] = torch.where(
            active & self._body_contact(), self._body_contact_time + self.dt, 0.0
        )
        self._collapse_time[:] = torch.where(
            active & collapsed, self._collapse_time + self.dt, 0.0
        )

    def _body_contact(self):
        forces = self.contact_forces[:, self.termination_contact_indices]
        return (forces.norm(dim=-1) > self.cfg.balance_failure.body_contact_force).any(dim=1)

    def _check_terminations_and_timeouts(self):
        # Replace the base task's instantaneous contact/height checks with
        # continuous failure timers. Foot unloading and slip never terminate.
        c = self.cfg.balance_failure
        self.timed_out = self.episode_length_buf >= self.max_episode_length
        self.terminated |= (self._body_contact_time + 1e-7 >= c.body_contact_seconds)
        self.terminated |= (self._collapse_time + 1e-7 >= c.collapse_seconds)

    def _valid_handstand(self):
        speed = self._rigid_body_lin_vel[:, self.feet_indices[:2], :2].norm(dim=-1)
        supported = (self._handstand_loads()[:, :2] >= self.cfg.reward_settings.min_front_load).all(dim=1)
        return supported & (speed <= self.cfg.reward_settings.valid_front_speed).all(dim=1) & ~self._body_contact()

    def _reward_handstand_balance(self):
        return super()._reward_handstand_balance() * self._valid_handstand()

    def _reward_balance_sustained(self):
        valid = self._reward_handstand_balance() >= self.cfg.handstand_curriculum.hold_score
        ramp = (self._hold_time / self.cfg.reward_settings.sustained_ramp_s).clamp(0, 1)
        return valid * ramp

    def _reward_balance_front_unloading(self):
        loads = self._handstand_loads()[:, :2]
        return -(1 - loads / self.cfg.reward_settings.min_front_load).clamp(0, 1).mean(dim=1)

    def _reward_balance_body_contact(self):
        return -self._body_contact().float()

    def _reward_termination(self):
        # The runner multiplies all rewards by dt. Falls are discrete events,
        # so keep their configured cost independent of the control frequency.
        return -self.terminated.float() / self.dt

    def _reward_balance_orientation(self):
        return self._handstand_balance_components()["vertical"]

    def _rear_extension(self):
        # Keep the original extension preference outside a small free region.
        s = self.cfg.reward_settings
        joints = self.dof_pos.index_select(1, self.actuated_dof_indices)
        error = (joints[:, [8, 11]] - s.kick_knee_target).abs()
        error = (error - s.rear_knee_deadband).clamp(min=0)
        return torch.exp(-(error / s.kick_knee_sigma).square())

    def _reward_balance_rear_motion(self):
        # Position freedom need not imply free rapid rear-leg cycling.
        velocity = self.dof_vel.index_select(1, self.actuated_dof_indices)[:, 6:]
        excess = (velocity.abs() - self.cfg.reward_settings.rear_velocity_deadband).clamp(min=0)
        return -excess.square().mean(dim=1)

    def _reward_balance_front_support(self):
        c = self._handstand_balance_components()
        return c["support"] * c["sharing"] * c["rear_off"]

    def _reward_balance_clearance(self):
        c = self._handstand_balance_components()
        return c["clearance"] * c["rear_extension"] * c["vertical"]

    def _reward_balance_stability(self):
        c = self._handstand_balance_components()
        return c["stable"] * c["vertical"] * c["support"] * c["sharing"]

    def update_training_diagnostics(self):
        """Called on collected control steps, before the runner resets episodes."""
        values = self._handstand_balance_components()
        loads = self._handstand_loads()
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[:2]]
        normal_load = self.contact_forces[:, self.feet_indices[:2], 2].clamp(min=0)
        values.update(
            strict_score=self._reward_handstand_balance(),
            front_left_load=loads[:, 0],
            front_right_load=loads[:, 1],
            rear_load=loads[:, 2:].sum(dim=1),
            hold_seconds=self._hold_time,
            terminated=self.terminated,
            push_fraction=self._last_push > 0,
            push_delta_velocity=self._last_push,
            front_left_load_n=normal_load[:, 0],
            front_right_load_n=normal_load[:, 1],
            front_left_slip_speed_m_s=velocity[:, 0, :2].norm(dim=-1),
            front_right_slip_speed_m_s=velocity[:, 1, :2].norm(dim=-1),
            rear_joint_speed_rad_s=self.dof_vel.index_select(1, self.actuated_dof_indices)[:, 6:].abs().mean(dim=1),
        )
        for foot, index in (("left", 0), ("right", 1)):
            for axis, component in enumerate("xyz"):
                values[f"front_{foot}_v{component}_m_s"] = velocity[:, index, axis]
        self.training_diagnostics.update(values)
