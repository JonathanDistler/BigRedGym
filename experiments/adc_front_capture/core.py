"""Single-actor support learning, retaining native collisions and motor limits."""

import numpy as np
from experiments.front_catch.anticipatory import AnticipatoryEnvironment
from experiments.front_catch.data import validate_state
from experiments.adc_front_capture.mechanics import measurements
from experiments.adc_front_capture.curriculum import damping


class CaptureEnvironment(AnticipatoryEnvironment):
    def reset(self, *args, **kwargs):
        self.recontact_impulses = np.zeros(2)
        self.first_recontact_times = np.full(2, np.nan)
        self.capture_paid = self.clearance_paid = self.rotation_paid = 0.0
        self.first_useful_front_time = None
        self.partial_seconds = 0.0
        self.capture_phase = 0
        self.last_front_q = None
        result = super().reset(*args, **kwargs)
        self.last_front_q = self.d.qpos[self.qids].copy()
        return result

    def restore_start(self, state, kind):
        super().restore_start(state, kind)
        self.steps = 0
        valid, details = validate_state(self)
        if not valid:
            raise ValueError("Rejected intermediate reset: " + str(details))
        self.recontact_impulses = np.zeros(2)
        self.first_recontact_times = np.full(2, np.nan)
        self.capture_paid = self.clearance_paid = self.rotation_paid = 0.0
        self.first_useful_front_time = None
        self.partial_seconds = 0.0
        self.capture_phase = 0
        self.last_front_q = self.d.qpos[self.qids].copy()
        return self.observations()

    def contact_measure(self, row, s, dt):
        before = self.recontacts.copy()
        out = super().contact_measure(row, s, dt)
        if dt > 0:
            for i, leg in enumerate(("FL", "FR")):
                if self.recontacts[i] > before[i] and np.isnan(
                    self.first_recontact_times[i]
                ):
                    self.first_recontact_times[i] = float(self.d.time)
                if self.recontacts[i] > 0 and self.loaded_age[i] <= 0.04:
                    self.recontact_impulses[i] += max(0, float(row["loads"][i])) * dt
                out[leg + "_recontact_impulse_z"] = float(self.recontact_impulses[i])
                out[leg + "_first_recontact_time"] = float(
                    self.first_recontact_times[i]
                )
        return out

    def set_dynamics(self, stiffness, nominal_k, nominal_b):
        self.cfg.kp = float(stiffness)
        # Approximate critical damping away from nominal, smoothly returning to
        # original damping at the endpoint. No hidden change to final dynamics.
        fraction = np.clip((stiffness - nominal_k) / (0.01 * nominal_k), 0, 1)
        self.cfg.kd = (1 - fraction) * nominal_b + fraction * damping(self, stiffness)
        return dict(kp=self.cfg.kp, kd=np.asarray(self.cfg.kd).tolist())

    def step(self, action):
        old = self.reward_sum
        prior = self.terms_sum.copy()
        old_q = self.d.qpos[self.qids].copy()
        obs, reward, done, info = super().step(action)
        s, cs = self.momentum_sample, self.contact_sample
        mech = measurements(self, with_ik=self.cfg.capture_enabled)
        safe = bool(
            self._momentum_prefix_safe
            and not self.nonfoot_collision
            and not self.self_collision
        )
        started = self.tracker.confirmed or self.start_kind in (
            "post_liftoff",
            "partial",
            "near_vertical",
        )
        predicted_pitch = s["pitch"] + 0.10 * s["pitch_velocity"]
        preparing = bool(
            safe
            and cs["any_front_loaded"]
            and (
                s["pitch_velocity"] > 0.1
                or self.row["loads"][:2].sum() > 0.55 * self.weight
            )
        )
        # State/contact events only. These phases can move backwards after a fall.
        self.capture_phase = (
            4
            if self.catch_hold > 0
            else 3
            if started and predicted_pitch > np.deg2rad(65) and cs["both_front_loaded"]
            else 2
            if started
            else 1
            if preparing
            else 0
        )
        progression = float(
            np.clip((s["pitch"] - self.start_pitch) / (np.pi / 2), 0, 1)
        )
        rising = bool(s["pitch_velocity"] > 0.1 and predicted_pitch > s["pitch"])
        distance = (
            mech["capture_support_distance"]
            if mech["lip_valid"]
            else mech["actual_support_distance"]
        )
        support_quality = (
            0.0
            if not np.isfinite(distance)
            else float(np.exp(-0.5 * (distance / 0.10) ** 2))
        )
        # A planted leg, loaded slide, staggered step and simultaneous reach may
        # all earn this credit; no foot displacement is independently rewarded.
        load_ok = (
            all(
                mech[f"{leg}_single_support_static_ratio"] <= 1
                for i, leg in enumerate(("FL", "FR"))
                if self.contact_loaded[i]
            )
            if self.contact_loaded.sum() == 1
            else True
        )
        viable = bool(
            safe and cs["any_front_loaded"] and cs["impact_valid"] and load_ok
        )
        collision_quality = float(np.clip(mech["illegal_clearance"] / 0.05, 0, 1))
        readiness = bool(preparing and predicted_pitch > np.deg2rad(10) and rising)
        geometry_score = (
            support_quality
            * collision_quality
            * max(progression, 0.1 if readiness else 0.0)
        )
        geometry_score *= float(viable and (rising or self.catch_hold > 0))
        clearance_score = collision_quality * progression * float(viable and rising)
        front_ids = [
            i for i, n in enumerate(self.names) if n.startswith(("FL_", "FR_"))
        ]
        moved = (
            float(np.max(abs(self.d.qpos[self.qids[front_ids]] - old_q[front_ids])))
            > 0.005
        )
        if (
            self.first_useful_front_time is None
            and moved
            and geometry_score > self.capture_paid + 1e-5
        ):
            self.first_useful_front_time = float(self.d.time)
        if (
            safe
            and started
            and np.deg2rad(45) <= s["pitch"] <= np.deg2rad(100)
            and self.row["rear_clearance"] > 0.015
        ):
            self.partial_seconds += self.cfg.dt
        if self.cfg.capture_enabled:
            self.phase.progress = (0.0, 0.2, 0.4, 0.6, 0.8)[self.capture_phase]
            # Replace legacy displacement and always-on leaning/lift rewards.
            terminal = (
                float(viable and started and self.row["rear_clearance"] > 0.10)
                * np.exp(-0.5 * (cs["orientation_error"] / 0.20) ** 2)
                * np.exp(
                    -0.5 * ((s["com_height"] - self.cfg.target_height) / 0.06) ** 2
                )
                * support_quality
            )
            torque = np.mean((self.d.qfrc_applied[self.vids] / self.effort) ** 2)
            terms = dict(
                capture=1.0 * max(0, geometry_score - self.capture_paid),
                clearance=0.5 * max(0, clearance_score - self.clearance_paid),
                safe_rotation=2.0
                * max(0, progression - self.rotation_paid)
                * float(viable and rising),
                pose=4.0 * self.cfg.dt * terminal,
                balance=4.0
                * self.cfg.dt
                * terminal
                * np.exp(-0.5 * (s["com_speed"] / 0.25) ** 2)
                * np.exp(-0.5 * (s["body_angular_speed"] / 0.5) ** 2),
                effort=-0.02 * self.cfg.dt * min(float(torque), 1),
                collision=-5.0 * float(not safe and done),
                clearance_barrier=-0.1
                * self.cfg.dt
                * float(preparing or started)
                * (1 - collision_quality) ** 2,
                success=20.0
                * float(self.catch_hold >= 5.0 and self.catch_hold - self.cfg.dt < 5.0),
            )
            # Bounded credit accumulated once; no stationary lean income.
            self.capture_paid = max(self.capture_paid, geometry_score)
            self.clearance_paid = max(self.clearance_paid, clearance_score)
            if viable and rising:
                self.rotation_paid = max(self.rotation_paid, progression)
            for key in list(info):
                if key.startswith("reward_"):
                    del info[key]
            info.update({"reward_" + key: value for key, value in terms.items()})
            reward = float(sum(terms.values()))
            self.reward_sum = old + reward
            self.terms_sum = prior
            for k, v in terms.items():
                self.terms_sum[k] = prior.get(k, 0) + v
            for k in list(self.trace[-1]):
                if k.startswith("reward_"):
                    del self.trace[-1][k]
            self.trace[-1].update({f"reward_{k}": v for k, v in terms.items()})
        transition = bool(viable and (self.catch_observed or self.catch_hold > 0.1))
        extra = dict(
            **mech,
            capture_phase=self.capture_phase,
            predicted_pitch=predicted_pitch,
            first_useful_front_time=self.first_useful_front_time,
            partial_inversion_seconds=self.partial_seconds,
            valid_support_transition=transition,
            actual_kp=float(self.cfg.kp),
            actual_kd_mean=float(np.mean(self.cfg.kd)),
        )
        self.trace[-1].update(extra, reward=reward)
        info.update(
            extra,
            reward=reward,
            episode_reward=self.reward_sum,
            elapsed_episode_seconds=float(self.d.time) - self.catch_reset_time,
        )
        for key, value in cs.items():
            if "recontact_impulse" in key or "first_recontact_time" in key:
                info[key] = value
        return self.observations(), reward, done, info
