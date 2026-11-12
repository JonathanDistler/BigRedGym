"""Isolated CPU Go2 standing-to-handstand task, shared by feasibility and PPO.

One policy commands position offsets about standing. No reference is used by
training; the scripted push below is only a matched feasibility probe.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from gym.envs.go2.leg_morphology import (
    SCALES,
    actuator_parameters,
    build_model,
    generate,
)


@dataclass
class Config:
    seconds: float = 8.0
    dt: float = 0.01
    kp: float = 60.0
    kd: float = 3.0
    action_scale: float = 1.5
    success_hold: float = 5.0
    pitch_tolerance: float = 0.174533
    max_omega: float = 0.5
    max_com_speed: float = 0.25
    min_rear_clearance: float = 0.10
    min_front_load: float = 0.05
    max_front_drift: float = 0.10


class Task:
    def __init__(self, assets, config=None):
        self.cfg = config or Config()
        self.models = {}
        assets = Path(assets)
        assets.mkdir(parents=True, exist_ok=True)
        for scale in SCALES:
            path = assets / f"go2_{scale:.2f}.urdf"
            # Assets belong to a newly created experiment directory.
            if not path.exists():
                generate(scale, path)
            self.models[scale] = build_model(path)
        self.reset(1.0, 0)

    def reset(self, scale, seed, pitch=0.0, omega=0.0):
        self.scale = scale
        self.m = self.models[scale]
        self.d = mujoco.MjData(self.m)
        self.names, self.effort, self.speed = actuator_parameters(self.m)
        self.jids = np.array([self.m.joint(n).id for n in self.names])
        self.qids = self.m.jnt_qposadr[self.jids]
        self.vids = self.m.jnt_dofadr[self.jids]
        self.feet = np.array(
            [self.m.body(f"{leg}_foot").id for leg in ("FL", "FR", "RL", "RR")]
        )
        self.foot_geoms = np.array(
            [np.flatnonzero(self.m.geom_bodyid == body)[0] for body in self.feet]
        )
        self.base = self.m.body("base").id
        self.nominal = np.array(
            [0 if "hip" in n else 0.58 if "thigh" in n else -1.16 for n in self.names]
        )
        self.d.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
        self.d.qpos[self.qids] = self.nominal
        if pitch:
            # Front legs remain vertical in world coordinates.
            for i, name in enumerate(self.names):
                if name[:2] in ("FL", "FR") and "thigh" in name:
                    self.d.qpos[self.qids[i]] -= pitch
        rng = np.random.default_rng(seed)
        self.d.qpos[self.qids] += rng.uniform(-0.005, 0.005, 12)
        mujoco.mj_forward(self.m, self.d)
        # Same joint pose, foot clearance and normalized perturbation across scales.
        relevant = self.foot_geoms[:2] if pitch else self.foot_geoms
        self.d.qpos[2] += 0.025 - self.d.geom_xpos[relevant, 2].min()
        self.d.qvel[4] = omega
        mujoco.mj_forward(self.m, self.d)
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()
        self.previous_action = np.zeros(12)
        self.hold = self.best_hold = self.t = 0.0
        self.rear_impulse = 0.0
        self.reward_sum = 0.0
        self.last = self.measure()
        self.done = False
        return self.observation()

    def observation(self):
        r = self.d.xmat[self.base].reshape(3, 3)
        return np.concatenate(
            (
                r.T @ [0, 0, -1],
                self.d.qvel[:6],
                self.d.qpos[self.qids] - self.nominal,
                self.d.qvel[self.vids],
                self.previous_action,
            )
        ).astype(np.float32)

    def measure(self):
        m, d = self.m, self.d
        mass = m.body_mass
        com = (mass[:, None] * d.xipos).sum(0) / mass.sum()
        pivot = d.geom_xpos[self.foot_geoms[:2]].mean(0).copy()
        pivot[2] = 0
        inertia = np.zeros((3, 3))
        for i in range(1, m.nbody):
            rotation = d.ximat[i].reshape(3, 3)
            delta = d.xipos[i] - com
            inertia += rotation @ np.diag(m.body_inertia[i]) @ rotation.T
            inertia += mass[i] * ((delta @ delta) * np.eye(3) - np.outer(delta, delta))
        loads = np.zeros(4)
        illegal = self_collisions = 0
        illegal_names = []
        for j in range(d.ncon):
            contact = d.contact[j]
            b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 != 0 and b2 != 0:
                self_collisions += 1
                continue
            body = b2 if b1 == 0 else b1
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, j, force)
            vertical = abs((contact.frame.reshape(3, 3).T @ force[:3])[2])
            idx = np.flatnonzero(self.feet == body)
            if len(idx):
                loads[idx[0]] += vertical
            elif contact.dist <= 0:
                illegal += 1
                illegal_names.append(m.body(body).name)
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            m, d, mujoco.mjtObj.mjOBJ_BODY, self.base, body_velocity, 0
        )
        r = d.xmat[self.base].reshape(3, 3)
        axis_rate = np.cross(body_velocity[:3], r[:, 0])
        pitch_rate = (-r[0, 0] * axis_rate[2] + r[2, 0] * axis_rate[0]) / max(
            r[0, 0] ** 2 + r[2, 0] ** 2, 1e-12
        )
        pitch = np.arctan2(-r[2, 0], r[0, 0])
        velocity = np.zeros(6)
        mujoco.mj_subtreeVel(m, d)
        velocity[:] = np.r_[d.subtree_angmom[self.base], d.subtree_linvel[self.base]]
        row = dict(
            scale=self.scale,
            time=self.t,
            pitch=pitch,
            pitch_velocity=float(pitch_rate),
            thigh_length=0.213 * self.scale,
            calf_length=0.213 * self.scale,
            pendulum_length=float(np.linalg.norm(com - pivot)),
            pitch_inertia_com=float(inertia[1, 1]),
            pitch_inertia_support=float(
                inertia[1, 1] + mass.sum() * ((com[0] - pivot[0]) ** 2 + com[2] ** 2)
            ),
            rear_clearance=float(d.geom_xpos[self.foot_geoms[2:], 2].min() - 0.022),
            front_drift=float(
                np.linalg.norm(
                    d.geom_xpos[self.foot_geoms[:2], :2] - self.anchor, axis=1
                ).max()
            ),
            illegal_contacts=illegal,
            self_collisions=self_collisions,
            illegal_contact_bodies="|".join(sorted(set(illegal_names))),
            com_speed=float(np.linalg.norm(velocity[3:])),
            mass=float(mass.sum()),
        )
        for i, axis in enumerate("xyz"):
            row[f"com_{axis}"] = float(com[i])
            row[f"support_{axis}"] = float(pivot[i])
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            row[f"load_{leg}"] = float(loads[i])
        row["front_support"] = bool(
            (loads[:2] > self.cfg.min_front_load * mass.sum() * 9.81).all()
        )
        return row

    def step(self, action, target=None):
        if self.done:
            raise RuntimeError("reset required after episode termination")
        c = self.cfg
        action = np.clip(np.asarray(action), -1, 1)
        target = self.nominal + c.action_scale * action if target is None else target
        target = np.clip(target, *self.m.jnt_range[self.jids].T)
        saturation = velocity_limit = rear_torque = impulse = excess = 0.0
        collision = False
        repeats = round(c.dt / self.m.opt.timestep)
        for _ in range(repeats):
            raw = (
                c.kp * (target - self.d.qpos[self.qids]) - c.kd * self.d.qvel[self.vids]
            )
            torque = np.clip(raw, -self.effort, self.effort)
            # No positive mechanical power beyond the URDF no-load speed.
            overspeed = (np.abs(self.d.qvel[self.vids]) >= self.speed) & (
                torque * self.d.qvel[self.vids] > 0
            )
            torque[overspeed] = 0
            saturation += np.mean(np.abs(raw) >= self.effort) / repeats
            velocity_limit += np.mean(overspeed) / repeats
            rear = np.array([n.startswith(("RL", "RR")) for n in self.names])
            rear_torque = max(rear_torque, float(np.max(np.abs(torque[rear]))))
            self.d.qfrc_applied[:] = 0
            self.d.qfrc_applied[self.vids] = torque
            mujoco.mj_step(self.m, self.d)
            self.t = self.d.time
            row = self.measure()
            rear_load = row["load_RL"] + row["load_RR"]
            impulse += rear_load * self.m.opt.timestep
            excess += (
                max(0.0, rear_load - 0.5 * row["mass"] * 9.81) * self.m.opt.timestep
            )
            collision |= row["illegal_contacts"] > 0
        self.rear_impulse += impulse
        upright = abs(row["pitch"] - np.pi / 2) < c.pitch_tolerance
        valid = (
            upright
            and abs(row["pitch_velocity"]) < c.max_omega
            and row["com_speed"] < c.max_com_speed
            and row["front_support"]
            and row["rear_clearance"] > c.min_rear_clearance
            and row["front_drift"] < c.max_front_drift
            and not collision
        )
        self.hold = self.hold + c.dt if valid else 0.0
        self.best_hold = max(self.best_hold, self.hold)
        terms = dict(
            pitch=float(np.exp(-(((row["pitch"] - np.pi / 2) / 0.5) ** 2))),
            lift=float(np.clip(row["rear_clearance"] / 0.2, 0, 1)),
            support=float(row["front_support"]),
            stability=float(valid),
            torque=-0.001 * float(np.mean((torque / self.effort) ** 2)),
            action_rate=-0.01 * float(np.mean((action - self.previous_action) ** 2)),
        )
        reward = c.dt * (
            4 * terms["pitch"]
            + terms["lift"]
            + terms["support"]
            + 4 * terms["stability"]
            + terms["torque"]
            + terms["action_rate"]
        )
        cause = (
            "illegal_contact"
            if collision
            else "timeout"
            if self.t >= c.seconds - 1e-9
            else "running"
        )
        self.done = cause != "running"
        if collision:
            reward -= 2
        self.previous_action = action.copy()
        self.reward_sum += reward
        row.update(
            saturation=saturation,
            velocity_limit_fraction=velocity_limit,
            rear_peak_torque=rear_torque,
            rear_impulse=self.rear_impulse,
            rear_step_impulse=impulse,
            rear_excess_step_impulse=excess,
            success=self.best_hold >= c.success_hold,
            best_hold=self.best_hold,
            termination=cause,
            reward=reward,
        )
        row.update({f"reward_{k}": v for k, v in terms.items()})
        self.last = row
        return self.observation(), reward, self.done, row


def reference(task, amplitude=1.0, duration=0.18):
    """Matched open-loop crouch/push probe; never part of PPO's controller."""
    target = task.nominal.copy()
    crouch = np.clip((task.t - 0.20) / 0.25, 0, 1)
    extend = np.clip((task.t - 0.45) / duration, 0, 1)
    crouch = crouch**2 * (3 - 2 * crouch)
    extend = extend**2 * (3 - 2 * extend)
    for i, name in enumerate(task.names):
        if name.startswith(("RL", "RR")):
            folded = 1.2 if "thigh" in name else -2.4 if "calf" in name else 0
            thrust = -0.2 if "thigh" in name else -0.85 if "calf" in name else 0
            goal = (1 - crouch) * target[i] + crouch * folded
            target[i] += amplitude * ((1 - extend) * goal + extend * thrust - target[i])
    return target


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def metadata(config):
    import hashlib
    import subprocess
    from gym.envs.go2.leg_morphology import SOURCE

    return dict(
        config=asdict(config),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_diff=subprocess.check_output(["git", "diff"], text=True),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
    )
