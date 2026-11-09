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
            "rear_left_clearance_m",
            "rear_right_clearance_m",
            "takeoff_difference_s",
            "extension_difference_s",
            "both_takeoff_fraction",
            "rear_left_push_impulse_Ns",
            "rear_right_push_impulse_Ns",
            "curriculum_success",
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
                "takeoff_tolerance",
            )
        ):
            raise ValueError("kick curriculum tables must match reset curriculum")
        m = self.cfg.kick_motion
        if m.crouch_seconds <= 0 or m.extend_seconds <= 0:
            raise ValueError("kick motion durations must be positive")
        s = self.cfg.reward_settings
        positive = (
            "motion_sigma",
            "lift_window_s",
            "rear_contact_load",
            "takeoff_clearance",
            "sync_height_sigma",
            "sync_velocity_sigma",
            "extension_velocity_sigma",
            "upward_velocity_target",
            "push_load_target",
            "max_foot_load",
            "impulse_target",
            "extension_onset_rate",
            "extension_rate_target",
        )
        if any(
            not math.isfinite(getattr(s, name)) or getattr(s, name) <= 0
            for name in positive
        ):
            raise ValueError("kick reward thresholds must be positive and finite")
        if not 0 <= s.rear_static_load < s.max_foot_load:
            raise ValueError("kick static load must be below the force limit")
        if any(not math.isfinite(x) or x <= 0 for x in c.takeoff_tolerance):
            raise ValueError("kick takeoff tolerances must be positive and finite")
        if self._body_weight().min() <= 0:
            raise ValueError("kick-up requires positive mass and downward gravity")
        self._kick_clock = torch.zeros(self.num_envs, device=self.device)
        self._front_anchor = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._front_anchor_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._support_loss_time = torch.zeros_like(self._kick_clock)
        self._support_failed = torch.zeros_like(self._front_anchor_valid)
        self._takeoff_time = torch.full((self.num_envs, 2), -1.0, device=self.device)
        self._extension_time = torch.full_like(self._takeoff_time, -1.0)
        self._rear_was_loaded = torch.zeros_like(self._takeoff_time, dtype=torch.bool)
        self._push_impulse = torch.zeros_like(self._takeoff_time)  # physical N s
        self._useful_push_impulse = torch.zeros_like(self._takeoff_time)
        self._step_push_impulse = torch.zeros_like(self._takeoff_time)
        self._step_spikes = torch.zeros_like(self._kick_clock)
        self._physics_offset = 0.0
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
        self._takeoff_time[reset_mask] = -1
        self._extension_time[reset_mask] = -1
        self._rear_was_loaded[reset_mask] = False
        self._push_impulse[reset_mask] = 0
        self._step_push_impulse[reset_mask] = 0
        self._useful_push_impulse[reset_mask] = 0
        self._step_spikes[reset_mask] = 0

    def _phase_masks(self, offset=0.0):
        m = self.cfg.kick_motion
        t = self._kick_clock + offset - m.settle_seconds
        push = (t >= m.crouch_seconds) & (t < m.crouch_seconds + m.extend_seconds)
        lift = (t >= m.crouch_seconds) & (
            t
            < m.crouch_seconds
            + m.extend_seconds
            + self.cfg.reward_settings.lift_window_s
        )
        return push, lift

    def _body_weight(self):
        gravity = getattr(getattr(self.cfg, "sim", None), "gravity", [0, 0, -9.81])
        return self._backend.link_mass.sum(dim=1) * -gravity[2]

    def _handstand_loads(self):
        return (
            self.contact_forces[:, self.feet_indices, 2].clamp(min=0)
            / self._body_weight()[:, None]
        )

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._step_push_impulse.zero_()
        self._step_spikes.zero_()
        self._physics_offset = 0.0

    def _post_physics_step(self):
        super()._post_physics_step()
        # Backend exposes MuJoCo rnePostConstraint external forces in world axes.
        # Integrate all physics samples, not just the final contact-force peak.
        s = self.cfg.reward_settings
        h = self.dt / self.cfg.control.decimation
        weight = self._body_weight()[:, None]
        loads = self.contact_forces[:, self.feet_indices[2:], 2].clamp(min=0) / weight
        self._physics_offset += h
        push, lift = self._phase_masks(self._physics_offset)
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[2:], 2]
        clearance = self._rear_clearance()
        self._rear_was_loaded |= loads > s.rear_contact_load
        airborne = (
            (loads < s.rear_contact_load)
            & (clearance > s.takeoff_clearance)
            & (velocity > 0)
        )
        first = (
            airborne & self._rear_was_loaded & (self._takeoff_time < 0) & lift[:, None]
        )
        self._takeoff_time[:] = torch.where(
            first, self._kick_clock[:, None] + self._physics_offset, self._takeoff_time
        )
        # Only the first push counts. Repeated landings cannot farm impulse.
        useful = (
            push[:, None]
            & (self._takeoff_time < 0)
            & self._valid_kick_support()[:, None]
        )
        impulse = loads * weight * h * useful
        self._push_impulse += impulse
        rates = self.dof_vel.index_select(1, self.actuated_dof_indices)[:, [8, 11]]
        first_extension = (
            (rates > s.extension_onset_rate)
            & push[:, None]
            & (self._extension_time < 0)
        )
        self._extension_time[:] = torch.where(
            first_extension,
            self._kick_clock[:, None] + self._physics_offset,
            self._extension_time,
        )
        extending = (rates / s.extension_rate_target).clamp(0, 1)
        # Static weight support is not a useful push-off. Credit excess load
        # only while the actual rear knees extend, with bounded per-foot loads.
        excess = (loads.clamp(max=s.max_foot_load) - s.rear_static_load).clamp(min=0)
        earned = excess * weight * h * useful * extending
        self._step_push_impulse += earned
        self._useful_push_impulse += earned
        all_loads = self.contact_forces[:, self.feet_indices].norm(dim=-1) / weight
        self._step_spikes += (all_loads - s.max_foot_load).clamp(min=0).square().sum(
            dim=1
        ) / self.cfg.control.decimation

    def _event_difference(self, events):
        # A missing event accumulates delay once the other leg has moved.
        times = torch.where(events >= 0, events, self._kick_clock[:, None])
        return (times[:, 0] - times[:, 1]).abs()

    def _takeoff_difference(self):
        return self._event_difference(self._takeoff_time)

    def _synchronized_takeoff(self):
        return (self._takeoff_time >= 0).all(dim=1) & (
            self._takeoff_difference() <= self._goal("takeoff_tolerance")
        )

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
            & self._synchronized_takeoff()
            & (self._extension_time >= 0).all(dim=1)
            & (
                self._event_difference(self._extension_time)
                <= self._goal("takeoff_tolerance")
            )
            & (
                (self._rear_clearance()[:, 0] - self._rear_clearance()[:, 1]).abs()
                <= self.cfg.reward_settings.sync_height_sigma
            )
            & (
                self._handstand_loads()[:, 2:]
                < self.cfg.reward_settings.rear_contact_load
            ).all(dim=1)
            & self._phase_masks()[1]
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
            .amin(dim=1)
        )

    def _reward_kick_clearance(self):
        lift = (self._rear_clearance() / self._goal("clearance")[:, None]).clamp(0, 1)
        # Only world-space lift of the lower rear foot counts.
        quality = torch.exp(
            -self._front_speed().square().sum(dim=1) / 0.04
            - self._front_displacement().square().sum(dim=1) / 0.01
        )
        return (
            lift.amin(dim=1)
            * self._phase_masks()[1]
            * quality
            * self._valid_kick_support()
            * self._reward_kick_front_support()
        )

    def _reward_kick_pitch(self):
        target = torch.sin(self._goal("pitch_degrees") * math.pi / 180)
        progress = (self.projected_gravity[:, 0] / target).clamp(0, 1)
        overshoot = (self.projected_gravity[:, 0] - target).clamp(min=0)
        return (
            progress
            * torch.exp(-overshoot.square() / 0.04)
            * self._valid_kick_support()
            * self._reward_kick_front_support()
        )

    def _reward_kick_motion(self):
        error = (
            self.dof_pos.index_select(1, self.actuated_dof_indices)
            - self._standing_posture()
        )
        m = self.cfg.kick_motion
        preparing = (self._kick_clock >= m.settle_seconds) & (
            self._kick_clock < m.settle_seconds + m.crouch_seconds + m.extend_seconds
        )
        return preparing * torch.exp(
            -error[:, 6:].square().mean(dim=1)
            / self.cfg.reward_settings.motion_sigma**2
        )

    def _reward_kick_upward_velocity(self):
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[2:], 2]
        return (
            (
                velocity.amin(dim=1) / self.cfg.reward_settings.upward_velocity_target
            ).clamp(0, 1)
            * self._phase_masks()[1]
            * self._reward_kick_front_support()
            * self._valid_kick_support()
        )

    def _reward_kick_synchronization(self):
        s = self.cfg.reward_settings
        heights = self._rear_clearance()
        velocity = self._rigid_body_lin_vel[:, self.feet_indices[2:], 2]
        score = torch.exp(
            -((heights[:, 0] - heights[:, 1]) / s.sync_height_sigma).square()
            - ((velocity[:, 0] - velocity[:, 1]) / s.sync_velocity_sigma).square()
        )
        progress = (heights.amin(dim=1) / self._goal("clearance")).clamp(0, 1)
        return (
            score
            * progress
            * self._phase_masks()[1]
            * self._reward_kick_front_support()
            * self._valid_kick_support()
        )

    def _reward_kick_extension_timing(self):
        # Compare extension rates, leaving independent joint positions available.
        rates = self.dof_vel.index_select(1, self.actuated_dof_indices)[:, [8, 11]]
        rate_error = (
            (rates[:, 0] - rates[:, 1])
            / self.cfg.reward_settings.extension_velocity_sigma
        ).square()
        timing_error = (
            self._event_difference(self._extension_time)
            / self._goal("takeoff_tolerance")
        ).square()
        return -(rate_error + timing_error).clamp(max=4) * self._phase_masks()[0]

    def _reward_kick_takeoff_timing(self):
        return (
            -(self._takeoff_difference() / self._goal("takeoff_tolerance"))
            .square()
            .clamp(max=4)
            * self._phase_masks()[1]
        )

    def _reward_kick_release(self):
        off = (
            self._handstand_loads()[:, 2:] < self.cfg.reward_settings.rear_contact_load
        ).all(dim=1)
        return off * self._synchronized_takeoff() * self._reward_kick_clearance()

    def _reward_kick_push_impulse(self):
        s = self.cfg.reward_settings
        # Average force over this control step, normalized by body weight.
        # Saturation of the episode impulse prevents prolonged-contact farming.
        weight = self._body_weight()[:, None]
        previous = self._useful_push_impulse - self._step_push_impulse
        cap = weight * s.impulse_target
        useful = (self._useful_push_impulse.minimum(cap) - previous.minimum(cap)).clamp(
            min=0
        )
        share = 2 * useful.amin(dim=1) / useful.sum(dim=1).clamp(min=1e-6)
        return (
            (
                useful.sum(dim=1) / (weight[:, 0] * self.dt * 2 * s.push_load_target)
            ).clamp(0, 1)
            * share
            * self._phase_masks()[0]
            * self._reward_kick_front_support()
        )

    def _reward_kick_force_spikes(self):
        return -self._step_spikes

    def _reward_kick_rear_slip(self):
        loaded = (
            self._handstand_loads()[:, 2:] > self.cfg.reward_settings.rear_contact_load
        )
        speed = self._rigid_body_lin_vel[:, self.feet_indices[2:], :2]
        return -(speed.square().sum(dim=-1) * loaded).sum(dim=1)

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
                rear_clearance_m=self._rear_clearance().amin(dim=1),
                rear_left_clearance_m=self._rear_clearance()[:, 0],
                rear_right_clearance_m=self._rear_clearance()[:, 1],
                takeoff_difference_s=self._takeoff_difference(),
                extension_difference_s=self._event_difference(self._extension_time),
                both_takeoff_fraction=(self._takeoff_time >= 0).all(dim=1),
                rear_left_push_impulse_Ns=self._push_impulse[:, 0],
                rear_right_push_impulse_Ns=self._push_impulse[:, 1],
                curriculum_success=self._kick_clock.new_tensor(
                    self.curriculum_success_rate
                ),
                front_slip_speed_m_s=self._front_speed(),
                front_displacement_m=self._front_displacement(),
                success=self._best_hold >= self.cfg.handstand_curriculum.hold_seconds,
            )
        )
