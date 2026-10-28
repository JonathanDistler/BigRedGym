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
            or p.interval_s <= c.hold_seconds
            or p.interval_s + c.hold_seconds >= self.max_episode_length_s
            or not math.isfinite(p.settled_hold_s)
            or p.settled_hold_s < 0
            or not 0 <= p.undisturbed_fraction < 1
        ):
            raise ValueError("invalid balance perturbation limits or timing")
        self._push_table = torch.tensor(p.max_delta_velocity, device=self.device)
        self._episode_push_limit = torch.zeros(self.num_envs, device=self.device)
        self._push_count = torch.zeros(self.num_envs, device=self.device)
        self._last_push = torch.zeros(self.num_envs, device=self.device)
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

    def _reward_balance_orientation(self):
        return self._handstand_balance_components()["vertical"]

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
        values.update(
            strict_score=self._reward_handstand_balance(),
            front_left_load=loads[:, 0],
            front_right_load=loads[:, 1],
            rear_load=loads[:, 2:].sum(dim=1),
            hold_seconds=self._hold_time,
            terminated=self.terminated,
            push_fraction=self._last_push > 0,
            push_delta_velocity=self._last_push,
        )
        self.training_diagnostics.update(values)
