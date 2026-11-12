"""Opt-in rotation objective and approximate moving-support pendulum diagnostics.

World +y momentum has the same sign as positive body pitch. COM angle theta
is atan2(-dx, -dz), from downward vertical: upright COM is +/- pi, NOT zero.
The pendulum equation excludes rear push, support acceleration and shape torque.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment


def potential_increment(previous, current, gamma, terminal=False):
    return gamma * (0.0 if terminal else current) - previous


def pendulum_state(relative, relative_velocity, mass, inertia_com, momentum_y):
    dx, dz = relative[[0, 2]]
    vx, vz = relative_velocity[[0, 2]]
    length = max(float(np.hypot(dx, dz)), 1e-8)
    rate = (dx * vx + dz * vz) / length
    omega = (dz * vx - dx * vz) / length**2
    theta = float(np.arctan2(-dx, -dz))
    inertia = inertia_com + mass * length**2
    barrier = mass * 9.81 * max(0.0, length - dz)
    kinetic = 0.5 * inertia * omega**2
    # Only kinetic motion toward COM-over-support counts against the barrier.
    toward = omega * np.sign(theta) > 0
    useful = kinetic if toward else 0.0
    return dict(
        pendulum_theta=theta,
        pendulum_omega=float(omega),
        effective_length=length,
        length_rate=float(rate),
        inertia_com_y=float(inertia_com),
        inertia_support_y=float(inertia),
        pendulum_acceleration=float(
            -9.81 / length * np.sin(theta) - 2 * rate / length * omega
        ),
        potential_barrier=float(barrier),
        rotational_energy=float(kinetic),
        centroidal_rotation_energy=float(0.5 * momentum_y**2 / max(inertia_com, 1e-8)),
        energy_deficit=float(max(0.0, barrier - useful)),
    )


def contact_wrenches(env, com):
    """Actual MuJoCo ground-on-robot wrench; frame rows are contact axes."""
    force = np.zeros((3, 3))
    moment = np.zeros((3, 3))
    points, weights = [], []
    local = np.empty(6)
    for index in range(env.d.ncon):
        contact = env.d.contact[index]
        if contact.dist > 0:
            continue
        b1, b2 = env.m.geom_bodyid[[contact.geom1, contact.geom2]]
        if bool(b1) == bool(b2):
            continue
        body = int(b2 if b1 == 0 else b1)
        foot = env.foot_lookup.get(body, 4)
        group = 0 if foot < 2 else 1 if foot < 4 else 2
        mujoco.mj_contactForce(env.m, env.d, index, local)
        sign = 1.0 if b1 == 0 else -1.0
        rotation = contact.frame.reshape(3, 3).T
        world_force = sign * rotation @ local[:3]
        couple = sign * rotation @ local[3:]
        force[group] += world_force
        moment[group] += np.cross(contact.pos - com, world_force) + couple
        if group == 0 and world_force[2] > 0:
            points.append(contact.pos.copy())
            weights.append(world_force[2])
    support = np.average(points, weights=weights, axis=0) if points else None
    return force, moment, support


@dataclass
class RotationConfig(TakeoffConfig):
    terminate_on_goal: bool = False
    rotation_enabled: bool = True
    pendulum_velocity_guidance: bool = True
    rotation_potential_weight: float = 2.0
    milestone_weight: float = 0.5
    rotation_command_weight: float = 0.01
    rotation_collision_weight: float = 1.0


class RotationEnvironment(TakeoffEnvironment):
    def reset(self, *args, **kwargs):
        self.rotation_sample = None
        self.rotation_trace = []
        self.record_rotation_trace = False
        self._model_previous = None
        self._rotation_phi = 0.0
        self._milestones_paid = set()
        self._last_model_time = -1.0
        result = super().reset(*args, **kwargs)
        return result

    def measure(self):
        row = super().measure()
        if (
            getattr(self, "tracker", None) is None
            or self.d.time <= self._last_model_time + 1e-10
        ):
            return row
        self._last_model_time = float(self.d.time)
        com = row["com"]
        force, moment, support = contact_wrenches(self, com)
        supported = support is not None and row["front_support"]
        if support is None:
            support = row["feet"][:2].mean(0).copy()
            support[2] = 0.0
        previous = self._model_previous
        dt = self.m.opt.timestep
        support_v = (
            np.zeros(3) if previous is None else (support - previous["support"]) / dt
        )
        stable_support = (
            supported
            and previous is not None
            and previous["supported"]
            and np.linalg.norm(support_v) < 0.15
        )
        relative = com - support
        relative_v = row["com_velocity"] - support_v
        rotations = self.d.ximat.reshape(-1, 3, 3)
        # Full articulated instantaneous rigid-axis inertia, not trunk inertia.
        inertia_yy = np.sum(
            np.sum(rotations[:, 1, :] ** 2 * self.m.body_inertia, axis=1)
            + self.m.body_mass
            * ((self.d.xipos[:, 0] - com[0]) ** 2 + (self.d.xipos[:, 2] - com[2]) ** 2)
        )
        momentum = self.d.subtree_angmom[self.base].copy()
        model = pendulum_state(relative, relative_v, self.mass, inertia_yy, momentum[1])
        rear_unloaded = force[1, 2] < 0.04 * self.weight
        eligible = bool(
            stable_support
            and rear_unloaded
            and not row["illegal_contacts"]
            and not row["self_contacts"]
            and row["front_drift"] < 0.10
        )
        age = (
            previous["eligible_age"] + dt if previous is not None and eligible else 0.0
        )
        measured_acceleration = (
            (model["pendulum_omega"] - previous["omega"]) / dt
            if previous is not None
            else 0.0
        )
        valid_prediction = eligible and age >= 0.01
        raw = (
            self.cfg.kp * (self.target - self.d.qpos[self.qids])
            - self.cfg.kd * self.d.qvel[self.vids]
        )
        sample = dict(
            time=float(self.d.time),
            scale=self.scale,
            pitch=row["pitch"],
            pitch_velocity=row["pitch_velocity"],
            momentum_x=float(momentum[0]),
            momentum_y=float(momentum[1]),
            momentum_z=float(momentum[2]),
            support_valid=bool(supported),
            support_speed=float(np.linalg.norm(support_v)),
            model_eligible=valid_prediction,
            measured_pendulum_acceleration=float(measured_acceleration),
            pendulum_residual=float(
                measured_acceleration - model["pendulum_acceleration"]
            )
            if valid_prediction
            else float("nan"),
            com_support_x=float(relative[0]),
            com_support_z=float(relative[2]),
            gravitational_potential=float(
                np.sum(self.m.body_mass * self.d.xipos[:, 2]) * 9.81
            ),
            takeoff=bool(self.tracker.confirmed),
            flight_valid=bool(self.tracker.flight_valid),
            rear_unloaded=bool(rear_unloaded),
            nonfoot_collision=bool(row["illegal_contacts"]),
            self_collision=bool(row["self_contacts"]),
            front_drift=row["front_drift"],
            tracking_rms=float(
                np.sqrt(np.mean((self.target - self.d.qpos[self.qids]) ** 2))
            ),
            saturation=float(np.mean(abs(raw) >= self.effort)),
            actuator_power=float(
                self.d.qfrc_applied[self.vids] @ self.d.qvel[self.vids]
            ),
            **model,
        )
        for i, axis in enumerate("xyz"):
            sample[f"com_{axis}"] = float(com[i])
            sample[f"com_velocity_{axis}"] = float(row["com_velocity"][i])
            sample[f"support_{axis}"] = float(support[i])
        for group, label in enumerate(("front", "rear", "nonfoot")):
            for i, axis in enumerate("xyz"):
                sample[f"{label}_force_{axis}"] = float(force[group, i])
                sample[f"{label}_moment_{axis}"] = float(moment[group, i])
        for i in range(12):
            sample[f"joint_{i}"] = float(self.d.qpos[self.qids[i]])
            sample[f"joint_velocity_{i}"] = float(self.d.qvel[self.vids[i]])
            sample[f"target_{i}"] = float(self.target[i])
            sample[f"action_{i}"] = float(self.action[i])
            sample[f"torque_{i}"] = float(self.d.qfrc_applied[self.vids[i]])
        self.rotation_sample = sample
        self._model_previous = dict(
            support=support.copy(),
            supported=supported,
            omega=model["pendulum_omega"],
            eligible_age=age,
        )
        if self.record_rotation_trace:
            self.rotation_trace.append(sample.copy())
        return row

    def rotation_potential(self):
        s = self.rotation_sample
        if (
            s is None
            or not self.tracker.confirmed
            or not s["support_valid"]
            or not s["rear_unloaded"]
            or s["nonfoot_collision"]
            or s["self_collision"]
            or self.max_speed_ratio > 1.001
            or self.max_joint_violation >= 0.001
        ):
            return 0.0
        progress = np.clip((s["pitch"] - 0.12) / (np.pi / 2 - 0.12), 0, 1)
        desired = min(2.0, max(0.0, 3 * (np.pi / 2 - s["pitch"])))
        if self.cfg.pendulum_velocity_guidance and s["model_eligible"]:
            desired = min(
                2.0,
                np.sqrt(2 * s["energy_deficit"] / max(s["inertia_support_y"], 1e-8)),
            ) * np.clip((np.pi / 2 - s["pitch"]) / np.deg2rad(20), 0, 1)
        speed_match = np.exp(-(((s["pitch_velocity"] - desired) / 1.5) ** 2))
        return float(
            progress
            + 0.15 * speed_match * progress
            + 0.1 * progress * float(s["support_valid"])
        )

    def step(self, action):
        previous_extra = {key: self.terms_sum.get(key, 0.0) for key in ROTATION_TERMS}
        observations, reward, done, info = super().step(action)
        if not self.cfg.rotation_enabled:
            return observations, reward, done, info
        s = self.rotation_sample
        phi = 0.0 if done else self.rotation_potential()
        milestone = 0.0
        if (
            self.tracker.confirmed
            and s["support_valid"]
            and s["rear_unloaded"]
            and not self.nonfoot_collision
            and not self.self_collision
            and abs(s["pitch_velocity"]) <= 3
            and self.max_speed_ratio <= 1.001
            and self.max_joint_violation < 0.001
        ):
            for angle in (45, 60, 75, 85):
                if (
                    s["pitch"] >= np.deg2rad(angle)
                    and angle not in self._milestones_paid
                ):
                    self._milestones_paid.add(angle)
                    milestone += self.cfg.milestone_weight
        terms = dict(
            rotation_progress=self.cfg.rotation_potential_weight
            * potential_increment(self._rotation_phi, phi, self.cfg.gamma, done),
            rotation_milestone=milestone,
            rotation_arrival=0.25
            * self.cfg.dt
            * float(
                self.tracker.confirmed
                and s["support_valid"]
                and abs(s["pitch"] - np.pi / 2) < np.deg2rad(10)
                and abs(s["pitch_velocity"]) < 0.5
                and abs(s["com_support_x"]) < 0.10
                and not self.nonfoot_collision
            ),
            rotation_overrun=-self.cfg.dt
            * float(
                max(0.0, s["pitch"] - np.deg2rad(100)) ** 2
                + max(0.0, abs(s["pitch_velocity"]) - 5) ** 2
            ),
            rotation_command=-self.cfg.rotation_command_weight
            * self.cfg.dt
            * float(np.mean(np.asarray(action) ** 2)),
            rotation_collision=-self.cfg.rotation_collision_weight
            * float(
                info["termination"]
                in ("nonfoot_collision", "self_collision", "actuator_state_limit")
            ),
        )
        self._rotation_phi = phi
        extra = sum(terms.values())
        self.reward_sum += extra
        for key, value in terms.items():
            self.terms_sum[key] = previous_extra.get(key, 0.0) + value
        info.update(self.metrics())
        info.update({f"reward_{key}": value for key, value in terms.items()})
        info["reward"] = reward + extra
        return observations, reward + extra, done, info

    def metrics(self):
        info = super().metrics()
        sample = getattr(self, "rotation_sample", None)
        if sample is not None:
            for key in (
                "momentum_y",
                "rotational_energy",
                "energy_deficit",
                "effective_length",
                "pendulum_theta",
                "pendulum_omega",
                "front_moment_y",
                "rear_moment_y",
                "com_support_x",
                "support_valid",
                "tracking_rms",
            ):
                info[key] = sample[key]
        return info


ROTATION_TERMS = (
    "rotation_progress",
    "rotation_milestone",
    "rotation_arrival",
    "rotation_overrun",
    "rotation_command",
    "rotation_collision",
)
