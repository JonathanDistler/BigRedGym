"""One state-conditioned actor, native contact physics, no within-episode resets."""

from dataclasses import dataclass
import numpy as np
from experiments.front_catch.core import CatchConfig
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.multi_impulse.sensors import Corrections, joint_sample

PHASES = ("PRELOAD", "KICK", "ROTATE", "CATCH", "REACCELERATE", "RETRY", "HOLD")


@dataclass
class MultiConfig(CatchConfig):
    corrective: bool = False
    balance_training: bool = False
    context_enabled: bool = True
    balance_stage: int = 0
    target_base_height: float = 0.57
    seconds: float = 12.0


class Phase:
    def __init__(self):
        self.index = 0
        self.age = 0.0
        self.pending = 0
        self.pending_age = 0.0
        self.weights = np.eye(7)[0].copy()

    def update(
        self, pitch, omega, rear_air, rear_loaded, front, hold, pushes, extension, dt
    ):
        predicted = pitch + 0.12 * omega
        if hold or (self.index == 6 and pitch > np.deg2rad(75) and abs(omega) < 1.2):
            candidate = 6
        elif rear_loaded and pushes > 0 and pitch < np.deg2rad(70):
            candidate = 5
        elif rear_loaded and extension > 0.25 and front:
            candidate = 1
        elif predicted > np.deg2rad(70) and front:
            candidate = 3
        elif rear_air and omega < 0.25 and pitch < np.deg2rad(75) and front:
            candidate = 4
        elif rear_air and omega > 0.25:
            candidate = 2
        else:
            candidate = 0
        self.age += dt
        if candidate != self.pending:
            self.pending = candidate
            self.pending_age = 0.0
        self.pending_age += dt
        if candidate != self.index and self.pending_age >= 0.06:
            self.index = candidate
            self.age = 0.0
        alpha = 1 - np.exp(-dt / 0.06)
        self.weights += (np.eye(7)[self.index] - self.weights) * alpha
        return self.weights


class MultiEnvironment(CaptureEnvironment):
    def reset(self, *args, **kwargs):
        self.redistribution_previous = None
        self.corrections = Corrections()
        self.movement = Phase()
        self.multi_hold = self.multi_best_hold = 0.0
        self.success_paid = False
        self.ever_hold_success = False
        self.progress_credit = self.brake_credit = 0.0
        self.multi_sample = {}
        self._multi_time = -1.0
        self.mechanisms = []
        self.last_mechanism_time = -1.0
        self.native_rows = []
        return super().reset(*args, **kwargs)

    def restore_start(self, state, kind):
        super().restore_start(state, kind)
        self.redistribution_previous = None
        self.corrections = Corrections()
        self.movement = Phase()
        self.multi_hold = self.multi_best_hold = 0.0
        self.success_paid = False
        self.ever_hold_success = False
        self.progress_credit = self.brake_credit = 0.0
        self.multi_sample = {}
        self._multi_time = float(self.d.time)
        self.mechanisms = []
        self.last_mechanism_time = -1.0
        self.native_rows = []
        return self.observations()

    def observations(self):
        a, c = super().observations()
        # Preserve the B actor's original eight-second clock convention even
        # when the physical episode horizon is extended.
        a[42] = self.steps * self.cfg.dt / 8.0
        c[42] = a[42]
        s = getattr(self, "momentum_sample", None)
        extra = np.zeros(20, np.float32)
        if s is not None and getattr(self.cfg, "context_enabled", True):
            extra = np.r_[
                self.movement.weights,
                s["pitch_velocity"] / 3,
                s["L_com"] / 2,
                s["com_support_x"] / 0.3,
                s["com_support_y"] / 0.3,
                [s["com_velocity_" + x] for x in "xyz"],
                self.row["loads"] / self.weight,
                min(self.corrections.loaded_age, 0.4) / 0.4,
                min(self.multi_hold, 5) / 5,
            ].astype(np.float32)
        assert extra.shape == (20,)
        return np.r_[a, np.clip(extra, -5, 5)].astype(np.float32), np.r_[
            c, np.clip(extra, -5, 5)
        ].astype(np.float32)

    def measure(self):
        row = super().measure()
        if (
            not getattr(self, "record_physics", False)
            or self.d.time <= self._multi_time + 1e-9
            or not self.physics_trace
        ):
            return row
        self._multi_time = float(self.d.time)
        s = self.physics_trace[-1]
        dt = self.m.opt.timestep
        self.row = row
        self.multi_sample = self.corrections.observe(self, row, s)
        safe = bool(
            s["prefix_safe"]
            and not row["illegal_contacts"]
            and not row["self_contacts"]
        )
        rear_air = bool(
            row["rear_clearance"] > 0.10
            and (row["loads"][2:] < 0.02 * self.weight).all()
        )
        feet = self.d.geom_xpos[self.foot_geoms[:2], :2]
        delta = feet[1] - feet[0]
        com = self.d.subtree_com[self.base]
        u = np.clip((com[:2] - feet[0]) @ delta / max(delta @ delta, 1e-10), 0, 1)
        support_error = float(np.linalg.norm(com[:2] - feet[0] - u * delta))
        stable = bool(
            safe
            and rear_air
            and row["front_support"]
            and s["orientation_error"] < np.deg2rad(10)
            and abs(self.d.qpos[2] - self.cfg.target_base_height) < 0.06
            and abs(s["com_height"] - self.cfg.target_height) < 0.06
            and s["com_speed"] < 0.25
            and s["body_angular_speed"] < 0.5
            and support_error < 0.04
            and s["front_slip_speed"] < 0.05
            and s["calf_clearance"] > 0
            and s["head_clearance"] > 0
            and s["front_force_weight_ratio"] < 4.0
        )
        self.multi_hold = self.multi_hold + dt if stable else 0.0
        self.multi_best_hold = max(self.multi_best_hold, self.multi_hold)
        if self.multi_hold >= 5.0:
            self.ever_hold_success = True
        knee = [
            i
            for i, n in enumerate(self.names)
            if n.startswith(("RL_", "RR_")) and "calf" in n
        ]
        extension = float(np.mean(self.d.qvel[self.vids[knee]]))
        self.movement.update(
            s["pitch"],
            s["pitch_velocity"],
            row["rear_clearance"] > 0.015 and not s["rear_loaded"],
            self.corrections.loaded_age > 0.04,
            s["any_front_loaded"],
            stable,
            len(self.corrections.pushes),
            extension,
            dt,
        )
        # Mechanisms are measured separately from reward. Internal swing never
        # counts as newly generated total angular momentum or a rear push-off.
        if self.d.time - self.last_mechanism_time > 0.12:
            if (
                s["front_moment_com"] > 0.3
                and abs(s["rear_moment_com"]) < 0.2
                and s["any_front_loaded"]
            ):
                self.mechanisms.append(
                    dict(
                        time=float(self.d.time),
                        kind="front_ground_torque",
                        pitch=s["pitch"],
                    )
                )
                self.last_mechanism_time = float(self.d.time)
        s.update(
            self.multi_sample,
            multi_handstand_seconds=self.multi_hold,
            support_error_actual=support_error,
            movement_phase=PHASES[self.movement.index],
        )
        if self.cfg.full_trace:
            s.update(joint_sample(self))
            self.native_rows.append(s.copy())
        return row

    def step(self, action):
        before = self.reward_sum
        prior = self.terms_sum.copy()
        (a, c), reward, done, info = super().step(action)
        s = self.momentum_sample
        cs = self.contact_sample
        safe = bool(
            self._momentum_prefix_safe
            and not self.nonfoot_collision
            and not self.self_collision
        )
        active = self.cfg.corrective or (
            self.cfg.balance_training and self.start_kind != "standing"
        )
        terms = {}
        if active:
            # Progress is paid once per episode, never once per kick or oscillation.
            progress = float(
                np.clip((s["pitch"] - max(0, self.start_pitch)) / (np.pi / 2), 0, 1)
            )
            eligible = (
                safe
                and cs["any_front_loaded"]
                and (self.tracker.confirmed or self.start_kind != "standing")
            )
            orientation = float(np.exp(-0.5 * (cs["orientation_error"] / 0.23) ** 2))
            height = float(
                np.exp(-0.5 * ((s["com_height"] - self.cfg.target_height) / 0.06) ** 2)
            )
            support = float(
                np.exp(
                    -0.5
                    * (self.physics_trace[-1].get("support_error_actual", 1.0) / 0.06)
                    ** 2
                )
            )
            airborne = float(
                self.row["rear_clearance"] > 0.1
                and (self.row["loads"][2:] < 0.02 * self.weight).all()
            )
            pose = (
                float(eligible)
                * orientation
                * height
                * support
                * airborne
                * float(self.row["front_support"])
            )
            calm = float(
                np.exp(
                    -0.5 * (s["com_speed"] / 0.3) ** 2
                    - 0.5 * (s["body_angular_speed"] / 0.7) ** 2
                )
            )
            brake = orientation * float(eligible and s["pitch"] > np.deg2rad(75)) * calm
            torque = float(np.mean((self.d.qfrc_applied[self.vids] / self.effort) ** 2))
            terms = dict(
                progress=3
                * max(0, progress - self.progress_credit)
                * float(eligible and self.cfg.corrective),
                brake=0.5 * max(0, brake - self.brake_credit),
                pose=2 * self.cfg.dt * pose,
                hold=8 * self.cfg.dt * pose * calm,
                effort=-0.01 * self.cfg.dt * torque,
                collision=-5 * float(done and not safe),
                clearance=-0.05
                * self.cfg.dt
                * float(self.movement.index in (1, 2, 3, 4))
                * float(np.clip((0.02 - s["calf_clearance"]) / 0.02, 0, 1)) ** 2,
                success=20
                * float(
                    self.ever_hold_success and not getattr(self, "success_paid", False)
                ),
            )
            self.success_paid = self.ever_hold_success
            if eligible:
                self.progress_credit = max(self.progress_credit, progress)
            self.brake_credit = max(self.brake_credit, brake)
            reward = float(sum(terms.values()))
            self.reward_sum = before + reward
            self.terms_sum = prior
            for k, v in terms.items():
                self.terms_sum[k] = prior.get(k, 0) + v
            for key in list(self.trace[-1]):
                if key.startswith("reward_"):
                    del self.trace[-1][key]
            self.trace[-1].update({"reward_" + k: v for k, v in terms.items()})
        rel = float(self.trace[-1].get("rear_relative_momentum", 0.0))
        previous = getattr(self, "redistribution_previous", None)
        current = (
            rel,
            float(s["L_com"]),
            self.corrections.front_J + self.corrections.rear_J,
        )
        if previous is not None and safe and abs(rel - previous[0]) > 0.03:
            external = current[2] - previous[2]
            if abs(external) < 0.02 and abs(current[1] - previous[1] - external) < 0.03:
                self.mechanisms.append(
                    dict(
                        time=float(self.d.time),
                        kind="internal_redistribution",
                        pitch=s["pitch"],
                        external_impulse=external,
                    )
                )
        self.redistribution_previous = current
        extra = dict(
            **self.multi_sample,
            movement_phase=PHASES[self.movement.index],
            handstand_seconds=self.multi_best_hold,
            success=self.ever_hold_success and self.start_kind == "standing",
            standing_success=self.ever_hold_success and self.start_kind == "standing",
            intermediate_success=self.ever_hold_success
            and self.start_kind != "standing",
            reward=reward,
            episode_reward=self.reward_sum,
            correction_events=self.corrections.pushes,
            mechanism_events=self.mechanisms,
            attained90=bool(
                self.corrections.peak >= np.deg2rad(85) and self.multi_best_hold > 0
            ),
        )
        self.trace[-1].update(
            {
                k: v
                for k, v in extra.items()
                if k not in ("correction_events", "mechanism_events")
            }
        )
        info.update(extra)
        return self.observations(), reward, done, info

    def state(self):
        result = super().state()
        result["values"] = dict(result["values"], native_rows=[])
        return result
