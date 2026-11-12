import math
import mujoco
import numpy as np
from experiments.morphology.learning.rotation import RotationEnvironment
from experiments.handstand_v2.measurement import Measurement, STAGES, shaping


class V2Environment(RotationEnvironment):
    def reset(self, *args, **kwargs):
        self.measurement = Measurement()
        self.stage = getattr(self, "stage", 1)
        self.trace = []
        self.last_terms = {}
        self._v2_time = -1.0
        return super().reset(*args, **kwargs)

    def set_stage(self, stage):
        self.stage = stage

    def measure(self):
        row = super().measure()
        if getattr(self, "tracker", None) is None or self.d.time <= self._v2_time:
            return row
        self._v2_time = float(self.d.time)
        speed = float(np.max(abs(self.d.qvel[self.vids]) / self.speed))
        joint = max(
            0.0,
            float(np.max(self.bounds[:, 0] - self.d.qpos[self.qids])),
            float(np.max(self.d.qpos[self.qids] - self.bounds[:, 1])),
        )
        torque = float(np.max(abs(self.d.qfrc_applied[self.vids]) / self.effort))
        safe = (
            not row["illegal_contacts"]
            and not row["self_contacts"]
            and speed <= 1.001
            and joint < 0.001
            and torque <= 1.00001
            and np.isfinite(self.d.qpos).all()
        )
        # Measure material velocity at the ground contact point.
        # Foot-center travel can include rolling or flight.
        slip_speeds = np.zeros(2)
        self_pairs = []
        for contact in self.d.contact:
            if contact.dist > 0:
                continue
            b1, b2 = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 and b2:
                self_pairs.append(
                    self.m.body(int(b1)).name + "/" + self.m.body(int(b2)).name
                )
            if bool(b1) == bool(b2):
                continue
            body = int(b2 if b1 == 0 else b1)
            foot = self.foot_lookup.get(body, 4)
            if foot >= 2 or row["loads"][foot] <= 0.05 * self.weight:
                continue
            jac = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jac, None, contact.pos, body)
            velocity = jac @ self.d.qvel
            normal = contact.frame.reshape(3, 3)[0]
            tangent = velocity - normal * np.dot(normal, velocity)
            slip_speeds[foot] = max(slip_speeds[foot], np.linalg.norm(tangent))
        enriched = dict(
            row,
            weight=self.weight,
            ground_slip_speed=float(slip_speeds.max()),
            self_bodies="|".join(self_pairs),
            speed_ratio=speed,
            joint_violation=joint,
            torque_ratio=torque,
        )
        before_slip = self.measurement.slip
        certified = self.measurement.observe(
            enriched,
            float(self.d.time),
            self.m.opt.timestep,
            safe,
            self.tracker.confirmed,
            STAGES[self.stage][1],
        )
        model = dict(self.rotation_sample or {})
        terms = shaping(
            enriched,
            safe and not self.measurement.failed,
            self.tracker.confirmed,
            STAGES[self.stage][1],
            self.m.opt.timestep,
            (self.measurement.slip - before_slip) / self.m.opt.timestep,
            model.get("front_moment_y", 0.0),
        )
        for k, v in terms.items():
            self.last_terms[k] = self.last_terms.get(k, 0.0) + v
        support = np.r_[row["feet"][:2, :2].mean(0), 0.0]
        angular = self.d.subtree_angmom[self.base] + np.cross(
            row["com"] - support, self.mass * row["com_velocity"]
        )
        if model.get("model_eligible", False):
            model["desired_com_omega"] = float(
                np.sign(model["pendulum_theta"])
                * min(
                    2.0,
                    np.sqrt(
                        2
                        * model["energy_deficit"]
                        / max(model["inertia_support_y"], 1e-8)
                    ),
                )
            )
            model["predicted_com_omega_next"] = (
                model["pendulum_omega"]
                + self.m.opt.timestep * model["pendulum_acceleration"]
            )
        model.update(
            time=float(self.d.time),
            pitch=row["pitch"],
            pitch_velocity=row["pitch_velocity"],
            rear_clearance=row["rear_clearance"],
            load_FL=row["loads"][0],
            load_FR=row["loads"][1],
            loaded_slip=self.measurement.slip,
            front_drift=row["front_drift"],
            gap=self.measurement.gap,
            max_gap=self.measurement.max_gap,
            certified=certified,
            safe=safe,
            ground_slip_speed=enriched["ground_slip_speed"],
            illegal_bodies=row["illegal_bodies"],
            self_contacts=row["self_contacts"],
            self_bodies="|".join(self_pairs),
            support_momentum_y=float(angular[1]),
            speed_ratio=speed,
            joint_violation=joint,
            **{f"v2_{k}": v for k, v in terms.items()},
        )
        self.trace.append(model)
        return row

    def step(self, action):
        self.last_terms = {}
        obs, reward, done, info = super().step(action)
        if self.measurement.failed and not done:
            done = True
            info["termination"] = "v2_strict_safety"
            info["timeout"] = False
        extra = sum(self.last_terms.values())
        reward += extra
        self.reward_sum += extra
        info.update({f"reward_v2_{k}": v for k, v in self.last_terms.items()})
        info["reward"] = reward
        info.update(self.metrics())
        return obs, reward, done, info

    def metrics(self):
        info = super().metrics()
        m = self.measurement
        target, hold = STAGES[self.stage][1:]
        success = bool(
            info.get("takeoff", False)
            if self.stage == 0
            else target in m.crossings and (hold == 0 or m.best_catch >= hold)
        )
        if self.stage == 5:
            success = bool(success and m.best_handstand >= 5.0)
        info.update(
            stable_handstand_seconds=m.best_handstand,
            front_recovery_seconds=m.best_front_recovery,
            curriculum_success=success,
            supported_pitch_degrees=math.degrees(m.certified_peak),
            peak_body_pitch_degrees=math.degrees(m.peak),
            loaded_pitch_degrees=math.degrees(m.loaded_peak),
            front_contact_gap_seconds=m.max_gap,
            front_contact_slip_m=m.slip,
            catch_seconds=m.best_catch,
            recovered=m.recovered,
            touchdown_speed_mps=m.touchdown_speed,
        )
        return info
