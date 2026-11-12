"""Takeoff-first reward, physics-rate certification, and physical-scale gates.

The old PPO task and its data are intentionally left unchanged. No runtime CPG
or reference controller is part of this environment.
"""

from collections import deque
from dataclasses import dataclass, field
import copy

import numpy as np

from experiments.morphology.learning.environment import Environment, LearningConfig


@dataclass
class TakeoffConfig(LearningConfig):
    action_scale: float = 2.0
    target_filter: float = 1.0
    terminate_on_goal: bool = True
    goal_pitch_degrees: float = 35.0
    goal_airborne_seconds: float = 0.04


class TakeoffTracker:
    """Require loaded, extending rear knees before synchronized rear release."""

    def __init__(self, dt=0.002):
        self.dt = dt
        self.window = deque(maxlen=round(0.4 / dt))
        self.loaded = np.zeros(2, dtype=bool)
        self.release = np.full(2, np.nan)
        self.previous_feet = None
        self.airborne = self.best_airborne = 0.0
        self.confirmed = False
        self.flight_certified = False
        self.event = False
        self.max_flight_pitch = 0.0
        self.positive_impulse = np.zeros(2)
        self.extension = np.zeros(2)
        self.positive_work = np.zeros(2)
        self.paid_preparation = 0.0
        self.paid_rotation = 0.0
        self.flight_valid = False
        self.synchrony = np.inf
        self.first_takeoff_time = None
        self.certified_work_min = 0.0
        self.certified_impulse_min = 0.0
        self.certified_extension_min = 0.0

    def observe(self, row, time, knee_velocity, valid=True):
        self.event = False
        loads = np.asarray(row["loads"][2:]) / row["weight"]
        feet = np.asarray(row["feet"][2:, 2])
        vertical_velocity = (
            (feet - self.previous_feet) / self.dt
            if self.previous_feet is not None
            else np.zeros(2)
        )
        self.previous_feet = feet.copy()
        self.loaded |= loads > 0.08
        # Use only load above a modest per-foot support baseline, while the
        # actual knees extend; static support is never a push-off reward.
        extending = np.maximum(knee_velocity, 0.0) * self.dt
        useful = (
            np.clip(loads - 0.15, 0, 1.5)
            * np.clip(np.asarray(knee_velocity) / 0.5, 0, 1)
            * self.dt
        )
        work = (
            np.maximum(np.asarray(row.get("rear_leg_power", [0.0, 0.0])), 0.0)
            * (loads > 0.08)
            * self.dt
        )
        self.window.append((useful, extending, work))
        self.positive_impulse = np.sum([x[0] for x in self.window], axis=0)
        self.extension = np.sum([x[1] for x in self.window], axis=0)
        self.positive_work = np.sum([x[2] for x in self.window], axis=0)
        for j in range(2):
            if loads[j] > 0.04:
                self.release[j] = np.nan
            elif (
                self.loaded[j]
                and loads[j] < 0.02
                and feet[j] - 0.022 > 0.008
                and vertical_velocity[j] > 0.02
                and np.isnan(self.release[j])
            ):
                self.release[j] = time
        self.synchrony = (
            float(abs(self.release[0] - self.release[1]))
            if np.isfinite(self.release).all()
            else np.inf
        )
        pushed = (
            self.loaded.all()
            and (self.positive_impulse >= 0.002).all()
            and (self.extension >= 0.03).all()
            and (self.positive_work >= 0.05).all()
        )
        # All certificates concern the same recent push, not an unrelated early
        # motion followed by folding the rear legs much later.
        recent = (
            np.isfinite(self.release).all() and time - float(np.min(self.release)) < 0.4
        )
        base_valid = (
            valid
            and row["front_support"]
            and (loads < 0.02).all()
            and (feet - 0.022 > 0.015).all()
            and row["pitch"] > 0.12
            and row["front_drift"] < 0.10
        )
        if not base_valid:
            self.flight_certified = False
        airborne = base_valid and (
            self.flight_certified or (pushed and recent and self.synchrony <= 0.04)
        )
        self.flight_valid = bool(airborne)
        self.airborne = self.airborne + self.dt if airborne else 0.0
        self.best_airborne = max(self.best_airborne, self.airborne)
        if self.airborne >= 0.02 - 1e-10:
            self.flight_certified = True
            self.max_flight_pitch = max(
                self.max_flight_pitch, min(row["pitch"], np.pi / 2)
            )
            if not self.confirmed:
                self.confirmed = self.event = True
                self.first_takeoff_time = time
                self.certified_work_min = float(np.min(self.positive_work))
                self.certified_impulse_min = float(np.min(self.positive_impulse))
                self.certified_extension_min = float(np.min(self.extension))
        # A bounded, one-use preparation credit cannot pay indefinitely for
        # squat cycles. It requires both actually extending, previously loaded legs.
        preparation = (
            0.5 * min(float(np.min(self.positive_impulse)) / 0.02, 1.0)
            if self.loaded.all() and valid and (self.positive_work >= 0.01).all()
            else 0.0
        )
        prep_credit = max(0.0, preparation - self.paid_preparation)
        self.paid_preparation = max(self.paid_preparation, preparation)
        rotation = np.clip((self.max_flight_pitch - 0.12) / (np.pi / 4 - 0.12), 0, 1)
        rotation_credit = max(0.0, float(rotation) - self.paid_rotation)
        self.paid_rotation = max(self.paid_rotation, float(rotation))
        flight = (
            (
                2.0
                * min(max(row["rear_clearance"], 0) / 0.08, 1)
                * np.clip(np.sin(row["pitch"]), 0, 1)
                * self.dt
            )
            if airborne
            else 0.0
        )
        return dict(
            push_preparation=prep_credit,
            simultaneous_liftoff=3.0 * float(self.event),
            airborne_rotation=rotation_credit,
            supported_flight=flight,
        )


def inverse_target_action(env, desired_target):
    """Invert the command filter; projection/error are measured, never hidden."""
    raw = env.target + (np.asarray(desired_target) - env.target) / env.cfg.target_filter
    unbounded = (raw - env.nominal) / env.cfg.action_scale
    action = np.clip(unbounded, -1, 1)
    desired = np.clip(env.nominal + env.cfg.action_scale * action, *env.bounds.T)
    filtered = env.target + env.cfg.target_filter * (desired - env.target)
    effective = env.target + np.clip(
        filtered - env.target,
        -0.7 * env.speed * env.cfg.dt,
        0.7 * env.speed * env.cfg.dt,
    )
    return action, dict(
        action_projection_fraction=float(np.mean(abs(unbounded) > 1)),
        target_error=float(np.max(abs(effective - desired_target))),
    )


class TakeoffEnvironment(Environment):
    def __init__(self, assets, config=None):
        super().__init__(assets, config or TakeoffConfig())

    def reset(self, *args, **kwargs):
        self.tracker = None
        result = super().reset(*args, **kwargs)
        self.tracker = TakeoffTracker(self.m.opt.timestep)
        self._last_tracker_time = float(self.d.time)
        self._takeoff_prefix_valid = True
        self._phase_terms = {}
        self.physics_trace = []
        self.trace_enabled = False
        self._previous_q = self.d.qpos[self.qids].copy()
        self._previous_v = self.d.qvel[self.vids].copy()
        self._previous_target = self.target.copy()
        self.knee_indices = np.array(
            [self.names.index(n) for n in ("RL_calf_joint", "RR_calf_joint")]
        )
        self.action = np.zeros(12)
        return result

    def measure(self):
        row = super().measure()
        if getattr(self, "tracker", None) is None:
            return row
        if self.d.time <= self._last_tracker_time + 1e-10:
            return row
        self._last_tracker_time = float(self.d.time)
        self._takeoff_prefix_valid &= (
            not row["illegal_contacts"] and not row["self_contacts"]
        )
        valid = (
            self._takeoff_prefix_valid
            and not row["illegal_contacts"]
            and not row["self_contacts"]
            and np.max(abs(self.d.qvel[self.vids]) / self.speed) <= 1.001
            and (self.d.qpos[self.qids] >= self.bounds[:, 0] - 0.001).all()
            and (self.d.qpos[self.qids] <= self.bounds[:, 1] + 0.001).all()
        )
        power = np.maximum(self.d.qfrc_applied[self.vids] * self._previous_v, 0.0)
        leg_power = np.array(
            [
                sum(power[j] for j, n in enumerate(self.names) if n.startswith(leg))
                for leg in ("RL", "RR")
            ]
        )
        enriched = dict(row, weight=self.weight, rear_leg_power=leg_power)
        terms = self.tracker.observe(
            enriched, self.d.time, self.d.qvel[self.vids[self.knee_indices]], valid
        )
        for key, value in terms.items():
            self._phase_terms[key] = self._phase_terms.get(key, 0.0) + value
        if self.trace_enabled:
            raw = (
                self.cfg.kp * (self.target - self._previous_q)
                - self.cfg.kd * self._previous_v
            )
            record = dict(
                time=float(self.d.time),
                scale=self.scale,
                pitch=row["pitch"],
                pitch_velocity=row["pitch_velocity"],
                rear_clearance=row["rear_clearance"],
                front_support=row["front_support"],
                front_drift=row["front_drift"],
                nonfoot_contacts=row["illegal_contacts"],
                illegal_bodies=row["illegal_bodies"],
                genuine_takeoff=self.tracker.confirmed,
                flight_valid=self.tracker.flight_valid,
                useful_impulse_left=self.tracker.positive_impulse[0],
                useful_impulse_right=self.tracker.positive_impulse[1],
                synchrony=self.tracker.synchrony
                if np.isfinite(self.tracker.synchrony)
                else -1.0,
            )
            for i in range(12):
                record[f"action_{i}"] = float(self.action[i])
                record[f"target_{i}"] = float(self.target[i])
                record[f"joint_{i}"] = float(self.d.qpos[self.qids[i]])
                record[f"velocity_{i}"] = float(self.d.qvel[self.vids[i]])
                record[f"torque_{i}"] = float(self.d.qfrc_applied[self.vids[i]])
                record[f"raw_torque_{i}"] = float(raw[i])
            for i, leg in enumerate(("FL", "FR", "RL", "RR")):
                record[f"load_{leg}"] = float(row["loads"][i])
            self.physics_trace.append(record)
        self._previous_q = self.d.qpos[self.qids].copy()
        self._previous_v = self.d.qvel[self.vids].copy()
        return row

    def step(self, action):
        before = self.reward_sum
        prior_terms = self.terms_sum.copy()
        self._phase_terms = {
            k: 0.0
            for k in (
                "push_preparation",
                "simultaneous_liftoff",
                "airborne_rotation",
                "supported_flight",
            )
        }
        self.action = np.clip(np.asarray(action), -1, 1)
        previous = self.target.copy()
        observations, _, done, old = super().step(self.action)
        failed = old["termination"] in (
            "nonfoot_collision",
            "self_collision",
            "actuator_state_limit",
        )
        # Remove all grounded pitch/support income and the old terminal-potential
        # credit. The same replacement terms apply at every physical morphology.
        terms = dict(self._phase_terms)
        terms.update(
            torque=-0.02
            * float(np.mean((self.d.qfrc_applied[self.vids] / self.effort) ** 2))
            * self.cfg.dt,
            command_change=-0.02
            * float(np.mean(((self.target - previous) / self.cfg.action_scale) ** 2)),
            failure=-0.5 * float(failed),
        )
        reward = sum(terms.values())
        self.reward_sum = before + reward
        self.terms_sum = {
            key: prior_terms.get(key, 0.0) + value for key, value in terms.items()
        }
        goal = (
            self.tracker.confirmed
            and self.tracker.max_flight_pitch >= np.deg2rad(self.cfg.goal_pitch_degrees)
            and self.tracker.best_airborne >= self.cfg.goal_airborne_seconds
            and not failed
        )
        if goal and self.cfg.terminate_on_goal:
            done = True
        info = self.metrics()
        info.update(
            termination="takeoff_goal"
            if goal and not failed and self.cfg.terminate_on_goal
            else old["termination"],
            timeout=old["timeout"],
            reward=reward,
            saturation=old["saturation"],
        )
        info.update({f"reward_{k}": value for k, value in terms.items()})
        return observations, reward, done, info

    def metrics(self):
        result = super().metrics()
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            result.update(
                legacy_takeoff=result["takeoff"],
                takeoff=tracker.confirmed,
                genuine_flight_pitch=tracker.max_flight_pitch,
                genuine_airborne_seconds=tracker.best_airborne,
                takeoff_time=tracker.first_takeoff_time
                if tracker.first_takeoff_time is not None
                else -1.0,
                useful_impulse_min=float(np.min(tracker.positive_impulse)),
                loaded_positive_work_min=float(np.min(tracker.positive_work)),
                certified_work_min=tracker.certified_work_min,
                certified_impulse_min=tracker.certified_impulse_min,
                certified_extension_min=tracker.certified_extension_min,
                takeoff_synchrony=tracker.synchrony
                if np.isfinite(tracker.synchrony)
                else -1.0,
            )
        return result


@dataclass
class TakeoffCurriculum:
    method: str
    protocol: dict
    stage: int = 0
    stage_entry_step: int = 0
    outcomes: list = field(default_factory=list)
    passing_windows: int = 0
    events: list = field(default_factory=list)
    windows: list = field(default_factory=list)

    @property
    def scale(self):
        return 1.0 if self.method == "A" else self.protocol["scales"][self.stage]

    def completed(self, episode, actual_steps):
        p = self.protocol
        if (
            self.method == "A"
            or self.stage == len(p["scales"]) - 1
            or episode["scale"] != self.scale
        ):
            return
        # Quality is local to the certified takeoff; no eight-second whole-episode
        # front-load criterion that disqualifies valid short kicks under exploration.
        passed = (
            episode["takeoff"]
            and episode["genuine_flight_pitch"]
            >= np.deg2rad(p["progress_pitch_degrees"][self.stage])
            and episode["genuine_airborne_seconds"]
            >= p["progress_airborne_seconds"][self.stage]
            and episode["max_speed_ratio"] <= 1.001
            and episode["joint_limit_violation"] < 0.001
        )
        self.outcomes.append(bool(passed))
        if len(self.outcomes) < p["stage_window_episodes"]:
            return
        rate = float(np.mean(self.outcomes))
        self.outcomes.clear()
        self.passing_windows = (
            self.passing_windows + 1 if rate >= p["stage_required_fraction"] else 0
        )
        self.windows.append(
            dict(actual_steps=actual_steps, scale=self.scale, qualifying_fraction=rate)
        )
        if (
            self.passing_windows >= p["stage_passing_windows"]
            and actual_steps - self.stage_entry_step >= p["minimum_stage_steps"]
        ):
            previous = self.scale
            self.stage += 1
            self.stage_entry_step = actual_steps
            self.passing_windows = 0
            self.events.append(
                dict(
                    actual_steps=actual_steps,
                    previous_scale=previous,
                    scale=self.scale,
                    evidence=copy.deepcopy(self.windows[-1]),
                )
            )
