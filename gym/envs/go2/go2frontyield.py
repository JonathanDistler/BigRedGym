"""Validated reference plus phase-bounded corrections in canonical joint radians."""

import torch
from gym.envs.go2.go2kickup import Go2Kickup
from gym.envs.go2.go2 import Go2
from gym.envs.go2.kickup_reference import KickReference, torch_reference


class Go2FrontYield(Go2Kickup):
    def __init__(self, cfg, device, headless, backend):
        super().__init__(cfg, device, headless, backend)
        from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
        if (
            isinstance(backend,MuJocoCPUBackend)
            and str(device) == "cpu"
            and cfg.front_yield.batched_cpu_sync
        ):
            from gym.envs.go2.frontyield_cpu import install_batched_cpu_sync

            install_batched_cpu_sync(self._backend)

    def _init_buffers(self):
        self.reference_parameters = KickReference(**self.cfg.front_yield.parameters)
        super()._init_buffers()
        self._raw_action = torch.zeros(self.num_envs, 12, device=self.device)
        self._nonfoot_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._verified_hold = torch.zeros(self.num_envs, device=self.device)
        self._best_verified_hold = torch.zeros_like(self._verified_hold)
        self._applied_residual = torch.zeros_like(self._raw_action)
        footset = set(self.feet_indices.tolist())
        self._nonfeet = [
            i for i in range(self.contact_forces.shape[1]) if i not in footset
        ]

    def _kick_reference(self, clock):
        return torch_reference(
            clock, self.reference_parameters, Go2._standing_posture(self)
        )

    def _reset_idx(self, mask):
        super()._reset_idx(mask)
        self.dof_pos_target[mask] = 0
        self._raw_action[mask] = 0
        self._nonfoot_contact[mask] = False
        self._verified_hold[mask] = 0
        self._best_verified_hold[mask] = 0
        self._applied_residual[mask] = 0

    def set_state(self, name, value):
        if name == "dof_pos_target":
            # Actor normalization/scaling is consumed exactly once. The result
            # is canonical FL/FR/RL/RR joint radians, not native actuator order.
            self._raw_action[:] = value.clamp(-1, 1)
        else:
            super().set_state(name, value)

    def residual_bounds(self, clock):
        p = self.reference_parameters
        c = self.cfg.front_yield
        level = self._reset_level.clamp(0, 6)

        def bounds(phase):
            front = torch.tensor(getattr(c, phase + "_front"), device=self.device)[
                level
            ]
            rear = torch.tensor(getattr(c, phase + "_rear"), device=self.device)[level]
            return torch.cat(
                (front[:, None].expand(-1, 6), rear[:, None].expand(-1, 6)), 1
            )

        start = p.settle + p.crouch
        end = start + p.extend * max(
            p.hip_delay + p.hip_duration, p.knee_delay + p.knee_duration
        )
        tuck = end + p.tuck_delay + p.tuck_duration
        # Start protecting the entire hip/knee push before the earliest clock.
        earliest = start + p.extend * min(p.hip_delay, p.knee_delay, 0.0)
        load, push, tuckbound, stable = [
            bounds(x) for x in ("loading", "push", "tuck", "stabilize")
        ]
        result = torch.where((clock < earliest)[:, None], load, push)
        a = ((clock - end) / c.phase_blend_s).clamp(0, 1)[:, None]
        result = result.lerp(tuckbound, a)
        b = ((clock - tuck) / c.phase_blend_s).clamp(0, 1)[:, None]
        return result.lerp(stable, b)

    def _compute_torques(self):
        ids = self.actuated_dof_indices
        clock = self._kick_clock + self._physics_offset
        q, v = self._kick_reference(clock)
        c = self.cfg.front_yield
        p = self.reference_parameters
        bounds = self.residual_bounds(clock)
        desired = (self._raw_action * self.scales["dof_pos_target"]).clamp(
            -bounds, bounds
        )
        h = self.dt / self.cfg.control.decimation
        delta = (desired - self.dof_pos_target).clamp(
            -c.slew_rad_s * h, c.slew_rad_s * h
        )
        start = p.settle + p.crouch
        end = start + p.extend * max(
            p.hip_delay + p.hip_duration, p.knee_delay + p.knee_duration
        )
        push = (clock >= start + p.extend * min(p.hip_delay, p.knee_delay, 0.0)) & (
            clock <= end
        )
        # Rate projection guarantees >=80% of directed reference extension at
        # every nonzero reference-velocity sample, independently for each knee/hip.
        for j in (7, 8, 10, 11):
            opposing = (delta[:, j] * v[:, j] < 0) & push
            cap = c.max_push_cancellation_fraction * v[:, j].abs() * h
            delta[:, j] = torch.where(
                opposing, delta[:, j].clamp(-cap, cap), delta[:, j]
            )
        self.dof_pos_target = (self.dof_pos_target + delta).clamp(-bounds, bounds)
        self._applied_residual[:] = self.dof_pos_target
        limits = self.dof_pos_limits[ids]
        target = (q + self.dof_pos_target).clamp(limits[:, 0], limits[:, 1])
        rate = v.clamp(-self.dof_vel_limits[ids], self.dof_vel_limits[ids])
        gain = torch.ones_like(target)
        gain[:, :6] = p.front_gain
        damping = torch.ones_like(target)
        damping[:, :6] = p.front_damping
        tau = self.p_gains * gain * (target - self.dof_pos[:, ids])
        tau += self.d_gains * damping * (p.feedforward * rate - self.dof_vel[:, ids])
        return tau.clamp(-self.actuated_torque_limits, self.actuated_torque_limits)

    def _post_physics_step(self):
        # This task has its own physical objective. Skip legacy impulse/reward
        # bookkeeping while retaining the base's current body observations.
        Go2._post_physics_step(self)
        h = self.dt / self.cfg.control.decimation
        self._physics_offset += h
        clock = self._kick_clock + self._physics_offset
        loads = self._handstand_loads()
        rear = loads[:, 2:]
        self._rear_was_loaded |= rear >= 0.03
        self._rear_contact[:] = torch.where(
            self._rear_contact, rear > 0.015, rear >= 0.03
        )
        clearance = self._rear_clearance()
        drift = self._front_displacement().amax(1)
        bad = self.contact_forces[:, self._nonfeet].norm(dim=-1).amax(1) > 1.0
        self._nonfoot_contact |= bad & (clock > 0.15)
        push, lift = self._phase_masks(self._physics_offset)
        upward = self._rigid_body_lin_vel[:, self.feet_indices[2:], 2] > 0
        event = (
            (rear < 0.015)
            & (clearance > 0.002)
            & upward
            & self._rear_was_loaded
            & (self._takeoff_time < 0)
            & lift[:, None]
        )
        self._takeoff_time[:] = torch.where(event, clock[:, None], self._takeoff_time)
        rates = self.dof_vel[:, self.actuated_dof_indices][:, [8, 11]]
        onset = (rates > 0.2) & push[:, None] & (self._extension_time < 0)
        self._extension_time[:] = torch.where(
            onset, clock[:, None], self._extension_time
        )
        self._push_impulse += rear * self._body_weight()[:, None] * h * push[:, None]
        valid = (
            (clearance.amin(1) >= 0.03)
            & (rear.amax(1) < 0.015)
            & (loads[:, :2].amin(1) >= 0.06)
        )
        valid &= self._rear_was_loaded.all(1) & ~self._nonfoot_contact & (drift <= 0.04)
        self._verified_hold[:] = torch.where(valid, self._verified_hold + h, 0.0)
        self._best_verified_hold[:] = torch.maximum(
            self._best_verified_hold, self._verified_hold
        )
        self._flight_time[:] = self._verified_hold
        self._best_flight[:] = self._best_verified_hold
        self._max_clearance[:] = torch.maximum(self._max_clearance, clearance.amin(1))
        self._max_drift[:] = torch.maximum(self._max_drift, drift)

    def _update_handstand_hold(self):
        physical = (
            (self._best_verified_hold >= 0.06)
            & ~self._nonfoot_contact
            & ~self.terminated
        )
        level = self._reset_level.clamp(0, 6)
        # First establish physical takeoff; later stages need forward rotation
        # with bounded speed, never unbounded angular momentum.
        angle = torch.asin(self.projected_gravity[:, 0].clamp(-1, 1))
        target = torch.deg2rad(self._goal("pitch_degrees"))
        physical &= (level < 2) | (
            (angle >= target) & (self.base_ang_vel[:, 1].abs() <= 2.0)
        )
        self._best_hold[:] = torch.where(
            physical,
            torch.maximum(self._best_hold, self._best_verified_hold),
            torch.zeros_like(self._best_hold),
        )

    def _yield_supported(self):
        return (self._handstand_loads()[:, :2].amin(1) >= 0.06) & ~self._nonfoot_contact

    def _reward_yield_takeoff(self):
        return self._yield_supported() * (self._best_verified_hold >= 0.02)

    def _reward_yield_clearance(self):
        return (
            self._yield_supported()
            * (self._rear_clearance().amin(1) / 0.12).clamp(0, 1)
            * (self._handstand_loads()[:, 2:].amax(1) < 0.015)
        )

    def _reward_yield_rotation(self):
        angle = torch.asin(self.projected_gravity[:, 0].clamp(-1, 1))
        target = torch.deg2rad(self._goal("pitch_degrees"))
        speed = self.base_ang_vel[:, 1]
        controlled = torch.exp(-((speed - 1.0) / 2.0).square())
        return (
            self._yield_supported()
            * (self._best_verified_hold >= 0.02)
            * (angle / target).clamp(0, 1)
            * controlled
        )

    def _reward_yield_terminal(self):
        angle = torch.asin(self.projected_gravity[:, 0].clamp(-1, 1))
        target = torch.deg2rad(self._goal("pitch_degrees"))
        ready = torch.exp(
            -((angle - target) / 0.35).square()
            - (self.base_ang_vel[:, 1] / 1.5).square()
        )
        return (
            self._yield_supported()
            * (self._best_verified_hold >= 0.02)
            * (self._kick_clock > 1.05)
            * ready
        )

    def _reward_yield_support(self):
        return (self._handstand_loads()[:, :2] / 0.1).clamp(0, 1).amin(1)

    def _reward_yield_drift(self):
        return -(
            (self._front_displacement().amax(1) - 0.04).clamp(min=0) / 0.04
        ).square()

    def _reward_yield_instability(self):
        return (
            -self._nonfoot_contact.float()
            - ((self.base_ang_vel.norm(dim=1) - 3.0).clamp(min=0)).square()
        )
