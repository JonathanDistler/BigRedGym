"""Opt-in contact-transition task; native dynamics, historical takeoff certificate.

A: fixed-support objective. B: loaded backward yielding. C: unloading/step/catch.
Feet are never welded, friction is never changed, and the actor owns all 12 joints.
"""

import copy
from dataclasses import dataclass
import mujoco
import numpy as np
from experiments.com_energy_phase.core import gaussian, orientation_error
from experiments.momentum_phase.core import (
    MomentumConfig,
    MomentumEnvironment,
    ManeuverPhase,
)
from experiments.morphology.dynamics_audit import ground_clearance
from experiments.morphology.learning.takeoff import TakeoffEnvironment
from experiments.front_catch.physics import sample_fast


class CatchPhase(ManeuverPhase):
    def __init__(self, times):
        super().__init__(times)
        self.caught = False
        self.current_support = False

    def update(self, time, loaded, takeoff, pitch, stable):
        previous = self.progress
        super().update(time, loaded, takeoff, pitch, False)
        if not self.current_support and previous < 0.8:
            self.progress = min(self.progress, 0.799)
        if takeoff and self.caught and self.current_support and pitch >= np.deg2rad(35):
            self.progress = max(self.progress, 0.6)
        if (
            takeoff
            and self.current_support
            and pitch >= np.deg2rad(80)
            and abs(self.omega) < 1
        ):
            self.progress = max(self.progress, 0.8)
        if takeoff and stable:
            self.progress = 1.0


@dataclass
class CatchConfig(MomentumConfig):
    adc_enabled: bool = False
    capture_enabled: bool = False
    comparison_suite: str = "original"
    anticipation_seconds: float = 0.08
    anticipation_support_sigma: float = 0.3
    early_placement_weight: float = 0.8
    rear_sweep_weight: float = 0.6
    rear_relative_momentum_scale: float = 0.4
    catch_mode: str = "A"
    condition: str = "A"
    phase_filter: bool = False
    contact_gap_seconds: float = 0.12
    minimum_unload_seconds: float = 0.02
    catch_dwell_seconds: float = 0.04
    max_impact_weight_ratio: float = 4.0
    backward_scale: float = 0.08
    loaded_bc: float = 1.0
    swing_bc: float = 0.025
    full_trace: bool = False


def backward_displacement(position, origin, axis):
    """World-frame displacement opposite initial task heading, not body-relative."""
    heading = np.array([axis[1], -axis[0], 0.0])
    return -(np.asarray(position) - np.asarray(origin)) @ heading


def support_distance(point, feet):
    a, b = feet[:, :2]
    v = b - a
    t = np.clip(np.dot(point[:2] - a, v) / max(v @ v, 1e-12), 0, 1)
    return float(max(0, np.linalg.norm(point[:2] - a - t * v) - 0.022))


def useful_swing_state(s, row, start_pitch, peak_pitch):
    """Current airborne swing, not historical progress followed by a fall/landing."""
    return bool(
        row["rear_clearance"] > 0.015
        and np.asarray(row["loads"][2:]).sum() < 0.04 * s["mass"] * 9.81
        and s["pitch"] >= max(start_pitch + 0.03, peak_pitch - np.deg2rad(5))
        and (s["pitch_velocity"] > -0.1 or s["pitch"] > np.deg2rad(75))
        and s["com_velocity_z"] > -0.25
    )


class CatchEnvironment(MomentumEnvironment):
    policy_actor_dim = 73
    policy_critic_dim = 103

    def reset(self, *args, **kwargs):
        self.start_kind = "standing"
        self.catch_origin = None
        self.contact_loaded = np.zeros(2, bool)
        self.previous_loaded = np.zeros(2, bool)
        self.unloaded_age = np.zeros(2)
        self.loaded_age = np.zeros(2)
        self.swing_air_age = np.zeros(2)
        self.last_air_time = np.zeros(2)
        self.recontacts = np.zeros(2, int)
        self.impact_failed = False
        self.max_impact_ratio = 0.0
        self.recontact_position = np.zeros((2, 3))
        self.front_gap = self.max_front_gap = 0.0
        self.ever_unloaded = False
        self.catch_hold = self.best_catch_hold = 0.0
        self.catch_credit = self.paid_catch = 0.0
        self.front_impulse = self.front_positive_impulse = (
            self.front_negative_impulse
        ) = 0.0
        self.peak_safe_pitch = 0.0
        self.peak_swing_pitch = 0.0
        self.start_pitch = 0.0
        self.paid_pitch = self.paid_placement = self.paid_recoverability = 0.0
        self.contact_sample = None
        self.head_collision = self.calf_collision = False
        self.first_collision_time = None
        self.collision_bodies = set()
        self.catch_observed = False
        self.catch_reset_time = 0.0
        super().reset(*args, **kwargs)
        self.phase = CatchPhase(self.cfg.phase_times)
        self.catch_origin = self.d.geom_xpos[self.foot_geoms[:2]].copy()
        self.contact_loaded = self.row["loads"][:2] > 0.05 * self.weight
        self.previous_loaded = self.contact_loaded.copy()
        self.start_pitch = self.row["pitch"]
        self.peak_safe_pitch = self.start_pitch
        self.head_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.body(int(self.m.geom_bodyid[g])).name.startswith("Head")
            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])
        ]
        self.robot_ids = np.flatnonzero(
            (self.m.body_rootid == self.base) & (self.m.body_mass > 0)
        )
        self.rear_indices = np.array(
            [i for i, n in enumerate(self.names) if n.startswith(("RL_", "RR_"))]
        )
        self.calf_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.body(int(self.m.geom_bodyid[g])).name.startswith(
                ("FL_calf", "FR_calf")
            )
            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])
        ]
        self.contact_sample = self.contact_measure(self.row, self.momentum_sample, 0.0)
        self.initial_capture_error = self.contact_sample["capture_error"]
        return self.observations()

    def observations(self):
        a, c = super().observations()
        s = getattr(self, "contact_sample", None)
        extra = np.zeros(12, np.float32)
        if s is not None:
            extra = np.r_[
                self.contact_loaded.astype(float),
                [s["FL_backward"] / 0.1, s["FR_backward"] / 0.1],
                [s["FL_z"] / 0.2, s["FR_z"] / 0.2],
                self.unloaded_age / 0.12,
                [s["FL_backward_velocity"], s["FR_backward_velocity"]],
                np.minimum(self.recontacts, 1),
            ].astype(np.float32)
        extra = np.clip(extra, -5, 5)
        return np.r_[a, extra].astype(np.float32), np.r_[c, extra].astype(np.float32)

    def contact_measure(self, row, s, dt):
        feet = self.d.geom_xpos[self.foot_geoms[:2]].copy()
        origin = self.catch_origin if self.catch_origin is not None else feet
        back = backward_displacement(feet, origin, self.pitch_axis)
        jp = np.zeros((3, self.m.nv))
        velocities = []
        for geom in self.foot_geoms[:2]:
            mujoco.mj_jacGeom(self.m, self.d, jp, None, int(geom))
            velocities.append(jp @ self.d.qvel)
        loaded = row["loads"][:2] > 0.05 * self.weight
        started = self.tracker.confirmed or self.start_kind in (
            "post_liftoff",
            "partial",
            "near_vertical",
        )
        if dt > 0:
            for i in range(2):
                if loaded[i]:
                    if (
                        not self.previous_loaded[i]
                        and self.swing_air_age[i] >= self.cfg.minimum_unload_seconds
                        and started
                    ):
                        self.recontacts[i] += 1
                        self.last_air_time[i] = self.swing_air_age[i]
                        self.recontact_position[i] = feet[i]
                    self.loaded_age[i] += dt
                    self.unloaded_age[i] = 0.0
                    self.swing_air_age[i] = 0.0
                else:
                    self.unloaded_age[i] += dt
                    self.loaded_age[i] = 0.0
                    if started:
                        self.swing_air_age[i] += dt
            self.previous_loaded = loaded.copy()
            self.contact_loaded = loaded.copy()
        started = self.tracker.confirmed or self.start_kind in (
            "post_liftoff",
            "partial",
            "near_vertical",
        )
        if started and dt > 0:
            self.ever_unloaded |= not loaded.all()
            self.front_gap = self.front_gap + dt if not loaded.any() else 0.0
            self.max_front_gap = max(self.max_front_gap, self.front_gap)
        force_ratio = s["front_force_z"] / self.weight
        self.max_impact_ratio = max(self.max_impact_ratio, force_ratio)
        self.impact_failed |= bool(
            started and force_ratio > self.cfg.max_impact_weight_ratio
        )
        current_impact = not self.impact_failed
        # Only count a placement after actual unloading AND stable recontact.
        caught = bool(
            started
            and loaded.all()
            and (self.loaded_age >= self.cfg.catch_dwell_seconds).all()
            and (self.recontacts > 0).any()
            and current_impact
        )
        self.catch_observed |= caught
        com = np.array([s[f"com_{a}"] for a in "xyz"])
        velocity = np.array([s[f"com_velocity_{a}"] for a in "xyz"])
        h = max(com[2] - 0.022, 0.05)
        capture = com + np.sqrt(h / np.linalg.norm(self.m.opt.gravity)) * velocity
        capture_error = support_distance(capture, feet)
        head_clear = min(
            (
                ground_clearance(self.m, self.d, g)
                for g in getattr(self, "head_geoms", [])
            ),
            default=1.0,
        )
        out = dict(
            capture_error=capture_error,
            static_capture_error=support_distance(capture, origin),
            current_com_error=support_distance(com, feet),
            front_gap=self.front_gap,
            max_front_gap=self.max_front_gap,
            front_recontacts=int(self.recontacts.sum()),
            caught=caught,
            front_force_weight_ratio=force_ratio,
            head_clearance=head_clear,
            impact_valid=current_impact,
            both_front_loaded=bool(loaded.all()),
            any_front_loaded=bool(loaded.any()),
            historical_takeoff=bool(self.tracker.confirmed),
            start_kind=self.start_kind,
            head_collision=self.head_collision,
            calf_collision=self.calf_collision,
        )
        for i, leg in enumerate(("FL", "FR")):
            out.update(
                {
                    f"{leg}_backward": float(back[i]),
                    f"{leg}_backward_velocity": float(
                        backward_displacement(
                            velocities[i], np.zeros(3), self.pitch_axis
                        )
                    ),
                    f"{leg}_contact": bool(loaded[i]),
                    f"{leg}_recontacts": int(self.recontacts[i]),
                    f"{leg}_unloaded_seconds": float(self.unloaded_age[i]),
                    f"{leg}_load": float(row["loads"][i]),
                }
            )
            for j, axis in enumerate("xyz"):
                out[f"{leg}_{axis}"] = float(feet[i, j])
        return out

    def measure(self):
        old_time = (
            getattr(self, "previous_physics", {}).get("time", None)
            if getattr(self, "previous_physics", None) is not None
            else None
        )
        row = TakeoffEnvironment.measure(self)
        if (
            not getattr(self, "record_physics", False)
            or old_time is None
            or self.d.time <= old_time + 1e-10
        ):
            return row
        previous = self.previous_physics
        s = sample_fast(self, row)
        dt = s["time"] - old_time
        safe = (
            not s["collision"]
            and not self.nonfoot_collision
            and not self.self_collision
            and np.max(abs(self.d.qvel[self.vids]) / self.speed) <= 1.001
            and (self.d.qpos[self.qids] >= self.bounds[:, 0] - 0.001).all()
            and (self.d.qpos[self.qids] <= self.bounds[:, 1] + 0.001).all()
        )
        self._momentum_prefix_safe &= safe
        self.phase.front_loaded |= bool(row["loads"][:2].sum() > row["loads"][2:].sum())
        self.phase.omega = s["pitch_velocity"]
        self.rear_work += s["rear_power"] * dt
        self.rear_positive_work += s["rear_positive_power"] * dt
        rear = 0.5 * (s["rear_moment_com"] + previous["rear_moment_com"]) * dt
        total = 0.5 * (s["moment_com"] + previous["moment_com"]) * dt
        self.rear_impulse += rear
        self.net_impulse += total
        self.balance_residual_com += s["L_com"] - previous["L_com"] - total
        self.balance_residual_fixed += (
            s["L_fixed"]
            - previous["L_fixed"]
            - 0.5 * (s["moment_fixed"] + previous["moment_fixed"]) * dt
        )
        self.momentum_sample = s
        self.previous_physics = s.copy()
        s.update(
            prefix_safe=self._momentum_prefix_safe,
            rear_work=self.rear_work,
            rear_positive_work=self.rear_positive_work,
            net_impulse=self.net_impulse,
            momentum_residual_com=self.balance_residual_com,
            momentum_residual_fixed=self.balance_residual_fixed,
            phase=self.phase.progress,
            takeoff=self.tracker.confirmed,
            flight_valid=self.tracker.flight_valid,
        )
        self.physics_trace.append(s.copy())
        self.front_impulse += (
            0.5 * (s["front_moment_com"] + previous["front_moment_com"]) * dt
        )
        self.front_positive_impulse += max(s["front_moment_com"], 0) * dt
        self.front_negative_impulse += min(s["front_moment_com"], 0) * dt
        if row["illegal_contacts"] or row["self_contacts"]:
            self.collision_bodies.update(
                row["illegal_bodies"].split("|") if row["illegal_bodies"] else []
            )
            self.head_collision |= "Head" in row["illegal_bodies"]
            self.calf_collision |= "calf" in row["illegal_bodies"]
            if self.first_collision_time is None:
                self.first_collision_time = s["time"]
        cs = self.contact_measure(row, s, dt)
        self.contact_sample = cs
        safe = self._momentum_prefix_safe and not s["collision"]
        started = self.tracker.confirmed or self.start_kind in (
            "post_liftoff",
            "partial",
            "near_vertical",
        )
        if safe:
            self.peak_safe_pitch = max(self.peak_safe_pitch, s["pitch"])
            if started and row["rear_clearance"] > 0.015:
                self.peak_swing_pitch = max(self.peak_swing_pitch, s["pitch"])
        target = np.array(self.cfg.target_rotation).reshape(3, 3)
        error = orientation_error(self.d.xmat[self.base].reshape(3, 3), target)
        energy = self.mass * np.linalg.norm(self.m.opt.gravity) * self.cfg.target_height
        stable = (
            safe
            and started
            and cs["both_front_loaded"]
            and cs["impact_valid"]
            and (row["loads"][2:] < 0.02 * self.weight).all()
            and row["rear_clearance"] > 0.1
            and error < np.deg2rad(10)
            and abs(s["potential"] - energy) < self.cfg.sigma_energy
            and s["body_angular_speed"] < 0.5
            and s["com_speed"] < 0.25
            and cs["current_com_error"] < 0.04
            and s["front_slip_speed"] < 0.05
            and s["calf_clearance"] > 0
            and cs["head_clearance"] > 0
        )
        self.catch_hold = self.catch_hold + dt if stable else 0.0
        self.best_catch_hold = max(self.best_catch_hold, self.catch_hold)
        self.phase.current_support = cs["both_front_loaded"]
        self.phase.caught = self.catch_observed
        cs.update(
            prefix_safe=safe,
            orientation_error=error,
            energy_target=energy,
            front_impulse=self.front_impulse,
            front_negative_impulse=self.front_negative_impulse,
            front_positive_impulse=self.front_positive_impulse,
            rear_impulse=self.rear_impulse,
            handstand_seconds=self.best_catch_hold,
        )
        self.physics_trace[-1].update(cs)
        return row

    def step(self, action):
        old = self.reward_sum
        prior = self.terms_sum.copy()
        self.cfg.target_filter = 1.0
        self.cfg.action_tau = 0.0
        obs, base, done, info = TakeoffEnvironment.step(self, action)
        s = self.momentum_sample
        cs = self.contact_sample
        started = self.tracker.confirmed or self.start_kind in (
            "post_liftoff",
            "partial",
            "near_vertical",
        )
        self.phase.update(
            self.d.time,
            self.tracker.loaded.all()
            or self.start_kind in ("post_liftoff", "partial", "near_vertical"),
            started,
            s["pitch"],
            self.catch_hold >= 5.0,
        )
        safe = bool(
            self._momentum_prefix_safe
            and cs["impact_valid"]
            and not self.nonfoot_collision
            and not self.self_collision
        )
        support = bool(cs["both_front_loaded"])
        if self.cfg.catch_mode == "A":
            eligible = (
                safe
                and support
                and not self.ever_unloaded
                and self.row["front_drift"] < 0.1
            )
        elif self.cfg.catch_mode == "B":
            eligible = safe and support and not self.ever_unloaded
        else:
            # Contact-loss progress is paid only after support returns.
            eligible = (
                safe and support and self.max_front_gap <= self.cfg.contact_gap_seconds
            )
        current_swing = useful_swing_state(
            s, self.row, self.start_pitch, self.peak_swing_pitch
        )
        progress = (
            float(
                np.clip(
                    (self.peak_swing_pitch - self.start_pitch)
                    / (np.pi / 2 - self.start_pitch if self.start_pitch < 1.4 else 0.2),
                    0,
                    1,
                )
            )
            if started and eligible and current_swing
            else 0.0
        )
        recovery = (
            gaussian(cs["capture_error"], 0.12)
            if started and eligible and current_swing
            else 0.0
        )
        orientation = gaussian(cs["orientation_error"], self.cfg.sigma_orientation)
        energy = gaussian(s["potential"] - cs["energy_target"], self.cfg.sigma_energy)
        terminal = (
            float(eligible and started and self.row["rear_clearance"] > 0.1)
            * energy
            * orientation
            * gaussian(s["body_angular_speed"], 0.5)
        )
        # Placement credit requires improving support and rotation together.
        placement = float(
            np.clip(
                min(cs["FL_backward"], cs["FR_backward"]) / self.cfg.backward_scale,
                0,
                1,
            )
        )
        useful = (
            started
            and eligible
            and current_swing
            and cs["capture_error"] < 0.12
            and self.peak_swing_pitch > self.start_pitch + 0.03
            and cs["capture_error"]
            < min(self.initial_capture_error, cs["static_capture_error"] - 0.005)
        )
        if self.cfg.catch_mode == "A":
            placement = 0.0
        if self.cfg.catch_mode == "C" and not cs["caught"]:
            placement = 0.0
        placement = placement * recovery if useful else 0.0
        catch_score = (
            recovery * float(cs["caught"] and useful)
            if self.cfg.catch_mode == "C"
            else 0.0
        )
        terms = dict(
            rotation_progress=2.0 * max(0, progress - self.paid_pitch),
            recoverability=0.3 * max(0, recovery - self.paid_recoverability),
            useful_placement=0.5 * max(0, placement - self.paid_placement),
            stable_recatch=0.5 * max(0, catch_score - self.paid_catch),
            terminal=4.0 * terminal * self.cfg.dt,
            stabilization=self.cfg.dt * float(self.catch_hold > 0),
            impact=-0.1
            * self.cfg.dt
            * min(
                max(
                    cs["front_force_weight_ratio"] - self.cfg.max_impact_weight_ratio, 0
                )
                ** 2,
                25,
            ),
            fixed_front=-0.1
            * self.cfg.dt
            * min(
                float(
                    np.mean(
                        np.square(
                            backward_displacement(
                                self.d.geom_xpos[self.foot_geoms[:2]],
                                self.catch_origin,
                                self.pitch_axis,
                            )
                            / 0.05
                        )
                    )
                ),
                4,
            )
            if self.cfg.catch_mode == "A" and started
            else 0.0,
        )
        self.paid_pitch = max(self.paid_pitch, progress)
        self.paid_recoverability = max(self.paid_recoverability, recovery)
        self.paid_placement = max(self.paid_placement, placement)
        self.paid_catch = max(self.paid_catch, catch_score)
        self.catch_credit = max(self.catch_credit, catch_score)
        reward = base + sum(terms.values())
        self.reward_sum = old + reward
        for k, value in terms.items():
            self.terms_sum[k] = prior.get(k, 0) + value
        out = {
            **s,
            **cs,
            **{f"reward_{k}": v for k, v in terms.items()},
            "reward": reward,
            "phase": self.phase.progress,
            "peak_swing_pitch": self.peak_swing_pitch,
            "terminal_reward": terminal,
            "catch_mode": self.cfg.catch_mode,
            "start_kind": self.start_kind,
            "backward_mean": 0.5 * (cs["FL_backward"] + cs["FR_backward"]),
            "elapsed_time": float(self.d.time) - self.catch_reset_time,
        }
        for i in range(12):
            out[f"target_{i}"] = float(self.target[i])
            out[f"joint_{i}"] = float(self.d.qpos[self.qids[i]])
        self.trace.append(out)
        success = self.best_catch_hold >= 5.0
        info.update(
            success=success and self.start_kind == "standing",
            standing_success=success and self.start_kind == "standing",
            intermediate_success=success and self.start_kind != "standing",
            start_kind=self.start_kind,
            certified_standing_takeoff=bool(
                self.tracker.confirmed and self.start_kind == "standing"
            ),
            peak_collision_free_pitch=self.peak_safe_pitch,
            peak_swing_pitch=self.peak_swing_pitch,
            catch_credit=self.catch_credit,
            front_recontacts=int(self.recontacts.sum()),
            front_impulse=self.front_impulse,
            rear_impulse=self.rear_impulse,
            front_negative_impulse=self.front_negative_impulse,
            head_collision=self.head_collision,
            calf_collision=self.calf_collision,
            collision_bodies="|".join(sorted(self.collision_bodies)),
            backward_mean=out["backward_mean"],
            FL_backward=cs["FL_backward"],
            FR_backward=cs["FR_backward"],
            current_com_error=cs["current_com_error"],
            capture_error=cs["capture_error"],
            handstand_seconds=self.best_catch_hold,
            reward=reward,
        )
        if not self.cfg.full_trace:
            self.trace = self.trace[-1:]
            self.physics_trace = self.physics_trace[-1:]
        return self.observations(), reward, done, info

    def restore_start(self, state, kind):
        self.restore(copy.deepcopy(state))
        self.start_kind = kind
        self.catch_reset_time = float(self.d.time)
        self.catch_origin = self.d.geom_xpos[self.foot_geoms[:2]].copy()
        self.recontacts = np.zeros(2, int)
        self.loaded_age = np.zeros(2)
        self.unloaded_age = np.zeros(2)
        self.swing_air_age = np.zeros(2)
        self.contact_loaded = self.row["loads"][:2] > 0.05 * self.weight
        self.previous_loaded = self.contact_loaded.copy()
        self.front_gap = self.max_front_gap = 0.0
        self.ever_unloaded = False
        self.catch_observed = False
        self.catch_hold = self.best_catch_hold = 0.0
        self.catch_credit = self.paid_catch = 0.0
        self.paid_pitch = self.paid_placement = self.paid_recoverability = 0.0
        self.start_pitch = self.row["pitch"]
        self.peak_safe_pitch = self.start_pitch
        self.peak_swing_pitch = self.start_pitch
        self.reward_sum = 0.0
        self.terms_sum = {}
        self.trace = []
        self.physics_trace = []
        self.net_impulse = self.balance_residual_com = self.balance_residual_fixed = 0.0
        self.rear_work = self.rear_positive_work = self.contact_work = 0.0
        self.front_impulse = self.front_negative_impulse = (
            self.front_positive_impulse
        ) = self.rear_impulse = 0.0
        self.contact_sample = self.contact_measure(self.row, self.momentum_sample, 0.0)
        self.initial_capture_error = self.contact_sample["capture_error"]
        return self.observations()

    def state(self):
        result = super().state()
        # Full physical/tracker/filter state, bounded logs for cheap checkpointing.
        result["values"] = dict(result["values"], trace=[], physics_trace=[])
        return result
