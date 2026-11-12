"""Full MuJoCo rollout, feasible state proposals, and independent safety metrics."""

import math
import mujoco
import numpy as np
from experiments.morphology.learning.environment import Environment
from experiments.morphology.learning.rotation import RotationEnvironment


class World(RotationEnvironment):
    def measure(self):
        row = super().measure()
        monitor = getattr(self, "monitor", None)
        if (
            monitor is not None
            and self.d.time > getattr(self, "_monitor_time", -1.0) + 1e-10
        ):
            self._monitor_time = float(self.d.time)
            self.last_monitor_sample = monitor.observe(row, self.m.opt.timestep)
        return row

    def physical_row(self):
        return Environment.measure(self)

    def tick(self, target):
        self.target = np.asarray(target).copy()
        raw = (
            self.cfg.kp * (self.target - self.d.qpos[self.qids])
            - self.cfg.kd * self.d.qvel[self.vids]
        )
        torque = np.clip(raw, -self.effort, self.effort)
        velocity = self.d.qvel[self.vids]
        torque[(abs(velocity) >= self.speed) & (torque * velocity > 0)] = 0
        self.d.qfrc_applied[:] = 0.0
        self.d.qfrc_applied[self.vids] = torque
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        return self.physical_row()

    def screen(self):
        row = self.physical_row()
        q = self.d.qpos[self.qids]
        violation = max(
            0.0,
            float(np.max(self.bounds[:, 0] - q)),
            float(np.max(q - self.bounds[:, 1])),
        )
        speed = float(np.max(abs(self.d.qvel[self.vids]) / self.speed))
        torque = float(np.max(abs(self.d.qfrc_applied[self.vids]) / self.effort))
        valid = (
            not row["illegal_contacts"]
            and not row["self_contacts"]
            and violation < 0.001
            and speed <= 1.001
            and torque <= 1.00001
            and np.isfinite(self.d.qpos).all()
            and np.isfinite(self.d.qvel).all()
        )
        return bool(valid), dict(
            joint_violation=violation,
            speed_ratio=speed,
            torque_ratio=torque,
            bodies=row["illegal_bodies"],
            self_contacts=row["self_contacts"],
        )


def rotation(pitch, roll=0.0):
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cp, 0, sp], [sr * sp, cr, -sr * cp], [-cr * sp, sr, cr * cp]])


def feasible_state(
    w,
    reference,
    seed,
    pitch=90.0,
    omega=0.0,
    roll=0.0,
    roll_rate=0.0,
    front_delta=0.0,
    rear_delta=0.0,
    com_shift=0.0,
    foot_height=0.0,
    scale=1.0,
):
    """Forward kinematic proposal plus planted-front IK; reject, never fallback.

    These are reset proposals, not online handoff operations. Contact forces are
    computed by MuJoCo, never assigned. Dynamic contact forces need settling.
    """
    w.reset(scale, seed)
    rng = np.random.default_rng(seed)
    angle = math.radians(pitch)
    w.d.qpos[w.qids] = reference
    front = np.array([0, 1, 2, 3, 4, 5])
    rear = np.arange(6, 12)
    # Interpolate front-leg kinematics with pitch.
    # A rotated standing configuration would not define valid upright support.
    w.d.qpos[w.qids[[1, 4]]] += math.pi / 2 - angle
    w.d.qpos[w.qids[[2, 5]]] += front_delta
    w.d.qpos[w.qids[rear]] += rng.normal(0, 0.001, 6)
    w.d.qpos[w.qids[[8, 11]]] += rear_delta
    matrix = rotation(angle, math.radians(roll))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.flatten())
    w.d.qpos[3:7] = quat
    w.d.qpos[:3] = [0, 0, 0.5]
    w.d.qvel[:] = 0.0
    mujoco.mj_forward(w.m, w.d)
    feet = w.d.geom_xpos[w.foot_geoms[:2]].copy()
    w.d.qpos[2] += 0.022 + 0.0002 - feet[:, 2].mean()
    mujoco.mj_forward(w.m, w.d)
    desired = w.d.geom_xpos[w.foot_geoms[:2]].copy()
    desired[:, 2] = 0.022 + 0.0002
    desired[0, 2] += foot_height
    w.d.qpos[0] += com_shift
    w.d.qpos[w.qids[front]] += rng.normal(0, 0.0005, 6)
    for _ in range(60):
        mujoco.mj_forward(w.m, w.d)
        actual = w.d.geom_xpos[w.foot_geoms[:2]]
        error = (
            (desired - actual)[:, 2] if abs(roll) > 0 else (desired - actual).flatten()
        )
        if np.linalg.norm(error) < 1e-7:
            break
        jac = []
        for geom in w.foot_geoms[:2]:
            jp = np.zeros((3, w.m.nv))
            mujoco.mj_jacGeom(w.m, w.d, jp, None, int(geom))
            jac.append(
                jp[2:3, w.vids[front]] if abs(roll) > 0 else jp[:, w.vids[front]]
            )
        jac = np.vstack(jac)
        increment = np.linalg.lstsq(jac, error, rcond=1e-7)[0]
        w.d.qpos[w.qids[front]] += np.clip(increment, -0.08, 0.08)
    mujoco.mj_forward(w.m, w.d)
    residual = desired - w.d.geom_xpos[w.foot_geoms[:2]]
    error = float(np.linalg.norm(residual[:, 2] if abs(roll) > 0 else residual))
    w.d.qvel[3:6] = matrix.T @ np.array([roll_rate, omega, 0.0])
    jac = []
    for geom in w.foot_geoms[:2]:
        jp = np.zeros((3, w.m.nv))
        mujoco.mj_jacGeom(w.m, w.d, jp, None, int(geom))
        jac.append(jp)
    jac = np.vstack(jac)
    w.d.qvel[w.vids[front]] = np.linalg.lstsq(
        jac[:, w.vids[front]], -jac @ w.d.qvel, rcond=1e-7
    )[0]
    mujoco.mj_forward(w.m, w.d)
    safe, details = w.screen()
    clearance = w.physical_row()["rear_clearance"]
    accepted = safe and error < 1e-5 and clearance > 0.015
    w.target = w.d.qpos[w.qids].copy()
    w.anchor = w.d.geom_xpos[w.foot_geoms[:2], :2].copy()
    return accepted, dict(ik_error=error, rear_clearance=clearance, **details)


class Metrics:
    def __init__(self, w):
        self.w = w
        self.anchor = w.d.geom_xpos[w.foot_geoms[:2], :2].copy()
        self.hold = self.best_hold = self.slip = self.drift = self.gap = (
            self.max_gap
        ) = 0.0
        self.peak = self.supported_peak = 0.0
        self.failed = False
        self.first_failure = None
        self.max_torque = self.max_speed = self.max_joint = 0.0
        self.had_gap = False
        self.ever_supported = False
        self.require_takeoff = False
        self.recovery = 0.0
        self.recovered = False

    def observe(self, row, dt):
        w = self.w
        safe, limits = w.screen()
        self.max_torque = max(self.max_torque, limits["torque_ratio"])
        self.max_speed = max(self.max_speed, limits["speed_ratio"])
        self.max_joint = max(self.max_joint, limits["joint_violation"])
        if not safe and self.first_failure is None:
            self.first_failure = dict(time=float(w.d.time), **limits)
        self.failed |= not safe
        self.peak = max(self.peak, row["pitch"])
        loaded = row["front_support"]
        if loaded and safe:
            self.supported_peak = max(self.supported_peak, row["pitch"])
        active = not self.require_takeoff or bool(w.tracker.confirmed)
        if active:
            self.ever_supported |= loaded
            self.gap = (
                0.0 if loaded else (self.gap + dt if self.ever_supported else 0.0)
            )
            self.max_gap = max(self.max_gap, self.gap)
            self.had_gap |= self.ever_supported and not loaded
        slip_speed = 0.0
        for contact in w.d.contact:
            if contact.dist > 0:
                continue
            b1, b2 = w.m.geom_bodyid[[contact.geom1, contact.geom2]]
            if bool(b1) == bool(b2):
                continue
            body = int(b2 if b1 == 0 else b1)
            index = w.foot_lookup.get(body, 4)
            if index >= 2 or row["loads"][index] < 0.05 * w.weight:
                continue
            jp = np.zeros((3, w.m.nv))
            mujoco.mj_jac(w.m, w.d, jp, None, contact.pos, body)
            v = jp @ w.d.qvel
            n = contact.frame.reshape(3, 3)[0]
            slip_speed = max(slip_speed, float(np.linalg.norm(v - n * (n @ v))))
        self.slip += slip_speed * dt
        self.drift = max(
            self.drift,
            float(np.max(np.linalg.norm(row["feet"][:2, :2] - self.anchor, axis=1))),
        )
        matrix = w.d.xmat[w.base].reshape(3, 3)
        gravity = matrix.T @ np.array([0, 0, -1.0])
        tilt = math.degrees(math.acos(float(np.clip(gravity[0], -1, 1))))
        stable = (
            loaded
            and not self.failed
            and tilt <= 10
            and np.linalg.norm(w.d.qvel[3:6]) <= 1.5
            and np.linalg.norm(row["com_velocity"]) <= 0.35
            and row["rear_clearance"] > 0.10
            and row["loads"][2:].sum() < 0.03 * w.weight
            and slip_speed <= 0.08
            and self.drift <= 0.04
        )
        self.hold = self.hold + dt if stable else 0.0
        self.best_hold = max(self.best_hold, self.hold)
        self.recovery = (
            self.recovery + dt
            if loaded and safe and not self.failed and self.drift <= 0.04
            else 0.0
        )
        self.recovered |= self.had_gap and self.recovery >= 0.02
        rowout = dict(
            time=float(w.d.time),
            pitch=math.degrees(row["pitch"]),
            omega=float(w.d.qvel[4]),
            roll_rate=float(w.d.qvel[3]),
            tilt=tilt,
            com_x=float(row["com"][0]),
            com_z=float(row["com"][2]),
            com_vx=float(row["com_velocity"][0]),
            com_vz=float(row["com_velocity"][2]),
            com_support_x=float(row["com"][0] - row["feet"][:2, 0].mean()),
            force_FL=float(row["loads"][0]),
            force_FR=float(row["loads"][1]),
            slip_speed=slip_speed,
            slip=self.slip,
            drift=self.drift,
            gap=self.gap,
            hold=self.hold,
            safe=safe,
            **limits,
        )
        ground = np.flatnonzero(w.m.geom_bodyid == 0)
        for leg in ('FL', 'FR'):
            calf = w.m.body(leg + '_calf').id
            geoms = np.flatnonzero(w.m.geom_bodyid == calf)
            rowout[leg + '_calf_clearance'] = min(
                (mujoco.mj_geomDistance(w.m, w.d, int(g), int(f), 1.0, None)
                 for g in geoms for f in ground), default=1.0
            )
        for j in range(12):
            rowout[f"q{j}"] = float(w.d.qpos[w.qids[j]])
            rowout[f"v{j}"] = float(w.d.qvel[w.vids[j]])
        return rowout

    def result(self):
        return dict(
            success=bool(self.best_hold >= 5.0 - 1e-8),
            hold=self.best_hold,
            supported_peak=math.degrees(self.supported_peak),
            peak=math.degrees(self.peak),
            loaded_slip=self.slip,
            loaded_drift=self.drift,
            recovered=bool(self.recovered),
            max_gap=self.max_gap,
            collision=bool(
                self.first_failure
                and (
                    self.first_failure["bodies"] or self.first_failure["self_contacts"]
                )
            ),
            first_failure=self.first_failure,
            max_speed=self.max_speed,
            max_torque=self.max_torque,
            max_joint=self.max_joint,
        )
