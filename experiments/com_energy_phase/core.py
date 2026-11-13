"""Opt-in articulated COM/energy instrumentation and one-shot kick-up policy task.

SI units throughout. Pendulum diagnostics are approximations, never a simulator.
"""

from dataclasses import dataclass
import math
import mujoco
import numpy as np
from experiments.morphology.learning.environment import Environment
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment


def coefficient(dt, tau):
    if dt <= 0 or tau < 0:
        raise ValueError("dt must be positive and tau nonnegative")
    return 1.0 if tau == 0 else -math.expm1(-dt / tau)


def gaussian(error, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return float(np.exp(-0.5 * (error / sigma) ** 2))


def orientation_error(rotation, target):
    return float(np.arccos(np.clip((np.trace(target.T @ rotation) - 1) / 2, -1, 1)))


def measurements(env):
    """Independent mass/Jacobian sums, checked against MuJoCo subtree quantities."""
    m, d = env.m, env.d
    ids = np.flatnonzero((m.body_rootid == env.base) & (m.body_mass > 0))
    mass = float(m.body_mass[ids].sum())
    com = np.average(d.xipos[ids], axis=0, weights=m.body_mass[ids])
    velocity = np.zeros(3)
    momentum = np.zeros(3)
    inertia = np.zeros((3, 3))
    kinetic = 0.0
    spin_kinetic = 0.0
    jp = np.zeros((3, m.nv))
    jr = np.zeros_like(jp)
    for b in ids:
        mujoco.mj_jacBodyCom(m, d, jp, jr, int(b))
        v, omega = jp @ d.qvel, jr @ d.qvel
        r = d.xipos[b] - com
        rot = d.ximat[b].reshape(3, 3)
        ib = (rot * m.body_inertia[b]) @ rot.T
        velocity += m.body_mass[b] * v / mass
        momentum += ib @ omega + np.cross(r, m.body_mass[b] * v)
        inertia += ib + m.body_mass[b] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        spin_kinetic += 0.5 * omega @ ib @ omega
        kinetic += 0.5 * m.body_mass[b] * np.dot(v, v) + 0.5 * omega @ ib @ omega
    mujoco.mj_subtreeVel(m, d)
    force = np.zeros(6)
    positions = []
    weights = []
    normals = []
    for contact_index, c in enumerate(d.contact):
        if c.dist > 0:
            continue
        b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
        if bool(b1) == bool(b2):
            continue
        if int(max(b1, b2)) in env.feet[:2]:
            mujoco.mj_contactForce(m, d, contact_index, force)
            if force[0] > 0:
                positions.append(c.pos.copy())
                weights.append(force[0])
                normal = c.frame[:3].copy()
                normals.append(normal if normal[2] >= 0 else -normal)
    support = (
        np.average(positions, axis=0, weights=weights) if positions else np.zeros(3)
    )
    normal = (
        np.average(normals, axis=0, weights=weights)
        if normals
        else np.array([0.0, 0.0, 1.0])
    )
    normal /= np.linalg.norm(normal)
    foot_speeds = []
    for geom in env.foot_geoms[:2]:
        mujoco.mj_jacGeom(m, d, jp, None, int(geom))
        foot_speeds.append(np.linalg.norm((jp @ d.qvel)[:2]))
    valid = bool(positions and env.row["front_support"])
    pivot_valid = valid and max(foot_speeds) < 0.15
    relative = com - support
    length = float(np.linalg.norm(relative))
    support_inertia = inertia[1, 1] + mass * (relative[0] ** 2 + relative[2] ** 2)
    support_momentum = momentum + np.cross(relative, mass * velocity)
    # theta=0 is COM vertically ABOVE support; positive x is positive theta.
    theta = float(np.arctan2(relative[0], relative[2]))
    gravity = float(np.linalg.norm(m.opt.gravity))
    rotor_kinetic = float(0.5 * np.dot(m.dof_armature, d.qvel**2))
    out = dict(
        rotor_kinetic=rotor_kinetic,
        mujoco_kinetic=float(kinetic + rotor_kinetic),
        mass=mass,
        com_height=float(com[2]),
        trunk_height=float(d.xipos[env.base, 2]),
        potential=mass * gravity * com[2],
        kinetic=float(kinetic),
        translational_kinetic=float(0.5 * mass * velocity @ velocity),
        link_spin_kinetic=float(spin_kinetic),
        total_energy=float(kinetic + mass * gravity * com[2]),
        internal_kinetic=float(max(0, kinetic - 0.5 * mass * velocity @ velocity)),
        inertia_com_y=float(inertia[1, 1]),
        inertia_support_y=float(support_inertia),
        momentum_y=float(momentum[1]),
        support_momentum_y=float(support_momentum[1]),
        support_valid=valid,
        pivot_valid=pivot_valid,
        front_slip_speed=float(max(foot_speeds)),
        support_height=float(support[2]),
        height_over_plane=float(relative @ normal),
        support_potential=float(mass * gravity * (com[2] - support[2])),
        pendulum_length=length,
        pendulum_angle=theta,
        pendulum_gravity_angle=float(
            np.arccos(np.clip(relative[2] / max(length, 1e-12), -1, 1))
        ),
        com_support_x=float(relative[0]),
        com_support_y=float(relative[1]),
        point_mass_barrier=float(mass * gravity * (length - relative[2])),
        rigid_rotation_estimate=float(
            momentum[1] ** 2 / (2 * max(inertia[1, 1], 1e-9))
        ),
        com_check=float(np.linalg.norm(com - d.subtree_com[env.base])),
        velocity_check=float(np.linalg.norm(velocity - d.subtree_linvel[env.base])),
        momentum_check=float(np.linalg.norm(momentum - d.subtree_angmom[env.base])),
    )
    # Horizontal distance to the actual two-foot line segment, including foot radius.
    feet = d.geom_xpos[env.foot_geoms[:2], :2]
    segment = feet[1] - feet[0]
    t = np.clip(
        np.dot(com[:2] - feet[0], segment) / max(segment @ segment, 1e-12), 0, 1
    )
    out["support_region_distance"] = float(
        max(0, np.linalg.norm(com[:2] - feet[0] - t * segment) - 0.022)
    )
    for label, vector in [
        ("com", com),
        ("trunk_com", d.xipos[env.base]),
        ("com_velocity", velocity),
        ("momentum", momentum),
        ("support", support),
        ("support_momentum", support_momentum),
    ]:
        out.update(
            {f"{label}_{axis}": float(vector[i]) for i, axis in enumerate("xyz")}
        )
    return out


class Phase:
    def __init__(self, times=(0.2, 0.6, 0.74, 0.95, 1.4), event_conditioned=True):
        if len(times) != 5 or np.any(np.diff(times) <= 0) or times[0] <= 0:
            raise ValueError("five increasing positive phase times required")
        self.times = tuple(times)
        self.event_conditioned = event_conditioned
        self.progress = 0.0
        self.takeoff_time = None

    def update(self, time, loaded, takeoff, pitch, stable):
        if takeoff and self.takeoff_time is None:
            self.takeoff_time = time
        p = float(np.interp(time, (0,) + self.times, (0, 0.2, 0.4, 0.6, 0.8, 1)))
        if self.event_conditioned:
            if not loaded:
                p = min(p, 0.199)
            elif not takeoff:
                p = min(p, 0.399)
            else:
                p = 0.4 + 0.399 * np.clip(
                    (time - self.takeoff_time) / (self.times[3] - self.times[2]), 0, 1
                )
                if pitch >= np.deg2rad(75):
                    p = max(p, 0.8)
                if stable:
                    p = 1.0
        self.progress = max(self.progress, p)

    def encode(self):
        # Half-circle avoids identical start/end encodings for a one-shot maneuver.
        phi = np.pi * self.progress
        return np.array([np.sin(phi), np.cos(phi), self.progress], dtype=np.float32)


@dataclass
class EnergyConfig(TakeoffConfig):
    terminate_on_goal: bool = False
    energy_enabled: bool = True
    phase_enabled: bool = True
    event_phase: bool = True
    phase_times: tuple = (0.2, 0.6, 0.74, 0.95, 1.4)
    target_height: float = 0.5  # Runner MUST replace with measured reference.
    target_rotation: tuple = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    sigma_energy: float = 12.0
    sigma_orientation: float = 0.35
    energy_tau: float = 0.0
    orientation_tau: float = 0.0
    action_tau: float = 0.0
    target_weight: float = 4.0
    pumping_weight: float = 0.2
    stage: int = 0


class EnergyEnvironment(TakeoffEnvironment):
    def observations(self):
        # Base method asserts the historical dimensions; append phase consistently.
        a, c = Environment.observations(self)
        if self.cfg.phase_enabled:
            encoding = (
                self.phase.encode()
                if hasattr(self, "phase")
                else np.array([0, 1, 0], np.float32)
            )
            a = np.r_[a, encoding].astype(np.float32)
            c = np.r_[c, encoding].astype(np.float32)
        return a, c

    def reset(self, *args, **kwargs):
        self.intermediate_start = False
        self.phase = Phase(self.cfg.phase_times, self.cfg.event_phase)
        self.energy_ema = self.orientation_ema = 0.0
        self.paid_energy = self.paid_pump = self.paid_orientation = 0.0
        self.valid_hold = self.best_valid_hold = 0.0
        self.trace = []
        self.actuator_work = 0.0
        self.positive_actuator_work = 0.0
        self._work_time = -1.0
        self._interval_support = True
        result = super().reset(*args, **kwargs)
        self.previous_delta = np.zeros(12)
        self.sample = measurements(self)
        self.initial_energy = self.sample["potential"]
        self.fixed_length = None
        self.previous_support = None
        return result

    def reset_intermediate(self, integration, seed, provenance):
        """Explicit episode reset to a checked balance snapshot, never a handoff.

        This trains terminal control without claiming that standing takeoff occurred.
        Only previously validated same-geometry snapshots should be supplied.
        """
        self.reset(1.0, seed)
        mujoco.mj_setState(
            self.m, self.d, integration, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(self.m, self.d)
        self.row = Environment.measure(self)
        q = self.d.qpos[self.qids]
        sample = measurements(self)
        safe = (
            sample["support_valid"]
            and self.row["rear_clearance"] > 0.015
            and self.row["loads"][2:].sum() < 0.04 * self.weight
            and not self.row["illegal_contacts"]
            and not self.row["self_contacts"]
            and np.all(q >= self.bounds[:, 0] - 0.001)
            and np.all(q <= self.bounds[:, 1] + 0.001)
            and np.max(abs(self.d.qvel[self.vids]) / self.speed) <= 1.001
            and abs(self.row["pitch"] - np.pi / 2) < np.deg2rad(10)
        )
        if not safe:
            raise ValueError("Intermediate reset failed support/geometry/safety checks")
        self.intermediate_start = True
        self.intermediate_provenance = provenance
        self.target = q.copy()
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()
        self._previous_q = q.copy()
        self._previous_v = self.d.qvel[self.vids].copy()
        self._last_tracker_time = self._work_time = float(self.d.time)
        self.initial_energy = sample["potential"]
        self.sample = sample
        self.phase.progress = 0.8
        self.phase.takeoff_time = float(self.d.time)
        self.paid_orientation = float(
            np.clip(
                1
                - orientation_error(
                    self.d.xmat[self.base].reshape(3, 3),
                    np.array(self.cfg.target_rotation).reshape(3, 3),
                )
                / (np.pi / 2),
                0,
                1,
            )
        )
        return self.observations()

    def measure(self):
        row = super().measure()
        if (
            hasattr(self, "actuator_work")
            and getattr(self, "tracker", None) is not None
            and self.d.time > self._work_time + 1e-10
        ):
            self._interval_support &= bool(
                row["front_support"]
                and not row["illegal_contacts"]
                and not row["self_contacts"]
            )
            self._work_time = float(self.d.time)
            self.positive_actuator_work += (
                max(0.0, float(self.d.qfrc_applied[self.vids] @ self.d.qvel[self.vids]))
                * self.m.opt.timestep
            )
            self.actuator_work += (
                float(self.d.qfrc_applied[self.vids] @ self.d.qvel[self.vids])
                * self.m.opt.timestep
            )
        return row

    def step(self, action):
        before = self.target.copy()
        self._interval_support = True
        self.cfg.target_filter = coefficient(self.cfg.dt, self.cfg.action_tau)
        obs, reward, done, info = super().step(action)
        s = measurements(self)
        support = np.array([s[f"support_{a}"] for a in "xyz"])
        s["support_reference_speed"] = (
            float(np.linalg.norm(support - self.previous_support) / self.cfg.dt)
            if self.previous_support is not None and s["support_valid"]
            else 0.0
        )
        s["pivot_valid"] = bool(
            s["pivot_valid"]
            and self.previous_support is not None
            and s["support_reference_speed"] < 0.15
        )
        self.previous_support = support if s["support_valid"] else None
        if self.fixed_length is None and s["support_valid"]:
            self.fixed_length = s["pendulum_length"]
        self.sample = s
        target = s["mass"] * np.linalg.norm(self.m.opt.gravity) * self.cfg.target_height
        error = orientation_error(
            self.d.xmat[self.base].reshape(3, 3),
            np.array(self.cfg.target_rotation).reshape(3, 3),
        )
        safe = (
            not (self.nonfoot_collision or self.self_collision)
            and self.max_speed_ratio <= 1.001
            and self.max_joint_violation < 0.001
        )
        valid = bool(
            safe
            and s["support_valid"]
            and (
                self.tracker.flight_valid
                or (
                    self.intermediate_start
                    and self.row["rear_clearance"] > 0.015
                    and self.row["loads"][2:].sum() < 0.04 * self.weight
                )
            )
            and self.row["front_drift"] < 0.1
        )
        target_valid = bool(
            valid
            and self._interval_support
            and s["support_region_distance"] < 0.06
            and abs(self.row["pitch_velocity"]) < 2.0
            and np.linalg.norm(self.row["com_velocity"]) < 0.75
        )
        terminal = bool(
            target_valid
            and error < np.deg2rad(10)
            and abs(self.row["pitch_velocity"]) < 0.5
            and np.linalg.norm(self.row["com_velocity"]) < 0.25
            and s["support_region_distance"] < 0.04
        )
        self.valid_hold = self.valid_hold + self.cfg.dt if terminal else 0.0
        self.best_valid_hold = max(self.best_valid_hold, self.valid_hold)
        self.phase.update(
            self.d.time,
            self.tracker.loaded.all() or self.intermediate_start,
            self.tracker.confirmed or self.intermediate_start,
            self.row["pitch"],
            terminal,
        )
        re = gaussian(s["potential"] - target, self.cfg.sigma_energy)
        ro = gaussian(error, self.cfg.sigma_orientation)
        if target_valid:
            self.energy_ema += coefficient(self.cfg.dt, self.cfg.energy_tau) * (
                re - self.energy_ema
            )
            self.orientation_ema += coefficient(
                self.cfg.dt, self.cfg.orientation_tau
            ) * (ro - self.orientation_ema)
        else:
            self.energy_ema = self.orientation_ema = 0.0
        # One-use best-so-far increments cannot pay for stationary leaning or cycles.
        useful = (
            np.clip(
                (s["potential"] - self.initial_energy)
                / max(target - self.initial_energy, 1.0),
                0,
                1,
            )
            if valid
            else 0.0
        )
        rotation = np.clip(1 - error / (np.pi / 2), 0, 1) if valid else 0.0
        deficit = max(target - s["potential"], 0.0)
        desired_ke = min(deficit, 15.0)
        pump = (
            (
                np.clip(s["rigid_rotation_estimate"] / max(desired_ke, 1.0), 0, 1)
                * np.clip(self.row["pitch_velocity"] / 2, 0, 1)
                * np.clip(deficit / 15, 0, 1)
            )
            if valid and deficit > 5
            else 0.0
        )
        delta = self.target - before
        terms = dict(
            energy_progress=float(max(0, useful - self.paid_energy)),
            orientation_progress=float(max(0, rotation - self.paid_orientation)),
            pumping=self.cfg.pumping_weight * float(max(0, pump - self.paid_pump)),
            target=self.cfg.target_weight
            * self.energy_ema
            * self.orientation_ema
            * target_valid
            * self.cfg.dt,
            braking=-0.1
            * self.cfg.dt
            * valid
            * float(error < 0.35)
            * min(self.row["pitch_velocity"] ** 2, 25),
            stable=self.cfg.dt * float(terminal),
            five_seconds=float(
                self.valid_hold >= 5 and self.valid_hold - self.cfg.dt < 5
            ),
            target_acceleration=-0.002
            * self.cfg.dt
            * float(
                np.mean(((delta - self.previous_delta) / (self.cfg.dt**2 * 500.0)) ** 2)
            ),
            power=-0.0001
            * self.cfg.dt
            * max(0, float(self.d.qfrc_applied[self.vids] @ self.d.qvel[self.vids]))
            * float(error < 0.35),
        )
        self.paid_energy = max(self.paid_energy, useful)
        self.paid_orientation = max(self.paid_orientation, rotation)
        self.paid_pump = max(self.paid_pump, pump)
        stage_gain = (0.25, 0.5, 1.0, 1.0, 1.0)[self.cfg.stage]
        for key in ("energy_progress", "orientation_progress", "pumping"):
            terms[key] *= stage_gain
        self.previous_delta = delta
        extra = sum(terms.values()) if self.cfg.energy_enabled else 0.0
        self.reward_sum += extra
        if self.cfg.energy_enabled:
            for k, v in terms.items():
                self.terms_sum[k] = self.terms_sum.get(k, 0) + v
        reward += extra
        s.update(
            time=float(self.d.time),
            pitch=self.row["pitch"],
            pitch_velocity=self.row["pitch_velocity"],
            orientation_error=error,
            energy_target=float(target),
            energy_deficit=float(target - s["potential"]),
            required_support_momentum_estimate=float(
                np.sqrt(2 * s["inertia_support_y"] * max(target - s["potential"], 0))
            ),
            r_energy=re,
            r_orientation=ro,
            r_target=terms["target"] if self.cfg.energy_enabled else 0.0,
            r_energy_smooth=self.energy_ema,
            r_orientation_smooth=self.orientation_ema,
            valid=valid,
            target_valid=target_valid,
            phase=self.phase.progress,
            work=self.actuator_work,
            positive_actuator_work=self.positive_actuator_work,
            reward=reward,
            fixed_length_potential=float(
                s["mass"]
                * np.linalg.norm(self.m.opt.gravity)
                * (self.fixed_length or 0.0)
                * np.cos(s["pendulum_gravity_angle"])
            ),
            rear_clearance=self.row["rear_clearance"],
            front_support=self.row["front_support"],
            collision=bool(self.nonfoot_collision or self.self_collision),
        )
        for i in range(12):
            s.update(
                {
                    f"raw_target_{i}": float(
                        np.clip(
                            self.nominal + self.cfg.action_scale * np.asarray(action),
                            *self.bounds.T,
                        )[i]
                    ),
                    f"target_{i}": float(self.target[i]),
                    f"joint_{i}": float(self.d.qpos[self.qids[i]]),
                    f"torque_{i}": float(self.d.qfrc_applied[self.vids[i]]),
                    f"joint_velocity_{i}": float(self.d.qvel[self.vids[i]]),
                    f"requested_torque_{i}": float(
                        self.cfg.kp * (self.target[i] - self.d.qpos[self.qids[i]])
                        - self.cfg.kd * self.d.qvel[self.vids[i]]
                    ),
                }
            )
        self.trace.append(s.copy())
        info.update(
            success=self.best_valid_hold >= 5,
            transition_success=self.best_valid_hold >= 5
            and not self.intermediate_start,
            intermediate_start=self.intermediate_start,
            handstand_seconds=self.best_valid_hold,
            reward=reward,
        )
        return self.observations(), reward, done, info


class Stages:
    """A-E progression requires repeated measured episodes, never iteration count."""

    def __init__(self, window=16, rate=0.75):
        self.stage = 0
        self.window = window
        self.rate = rate
        self.outcomes = []
        self.events = []

    def observe(self, info):
        pitch = info["genuine_flight_pitch"]
        passed = (
            info["takeoff"]
            if self.stage == 0
            else pitch >= np.deg2rad((35, 45, 60, 80)[self.stage])
            if self.stage < 4
            else info["success"]
        )
        # Early milestones certify the safe prefix; a later fall is logged separately.
        if self.stage >= 3:
            passed = (
                passed and not info["nonfoot_collision"] and not info["self_collision"]
            )
        if self.stage == 3:
            passed = passed and info["recovery_seconds"] >= 0.1
        self.outcomes.append(bool(passed))
        if len(self.outcomes) >= self.window:
            rate = float(np.mean(self.outcomes))
            self.outcomes = []
            if rate >= self.rate and self.stage < 4:
                self.stage += 1
                self.events.append(dict(stage=self.stage, rate=rate))
