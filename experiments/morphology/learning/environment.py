"""CPG-free physical-morphology PPO environment with privileged critic state.

All scales share one reward and unmodified native collision/motor parameters.
Initial poses are a fixed relative-extension IK family, not a posture curriculum.
"""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from experiments.morphology.dynamics_audit import Audit
from scripts.audit_morphology_dynamics import FEET
from gym.envs.go2.leg_morphology import actuator_parameters


@dataclass
class LearningConfig:
    dt: float = 0.02
    seconds: float = 8.0
    kp: float = 60.0
    kd: float = 3.0
    action_scale: float = 1.2
    target_filter: float = 0.25
    gamma: float = 0.995
    takeoff_seconds: float = 0.02
    initial_relative_height: float = 0.34
    reset_clearance: float = 0.001


class Assets:
    def __init__(self, directory):
        audit = Audit(Path(directory))
        self.models = audit.models
        self.poses = {}
        self.nominals = {}
        self.metadata = {}
        for scale, model in self.models.items():
            audit.reset(scale, 0)
            guess = audit.d.qpos.copy()
            guess[audit.qids] = audit.nominal
            q, error = audit.pose(0.34 * scale, feet=FEET, guess=guess)
            if error > 1e-6:
                raise RuntimeError("standing IK failed")
            row = audit.static(q)
            if row["nonfoot_clearance"] < 0.015 or row["static_torque_ratio"] >= 1:
                raise RuntimeError("invalid morphology standing pose")
            self.poses[scale] = q.copy()
            self.nominals[scale] = q[audit.qids].copy()
            names, effort, speed = actuator_parameters(model)
            joints = np.array([model.joint(n).id for n in names])
            feet = np.array(
                [model.body(f"{leg}_foot").id for leg in ("FL", "FR", "RL", "RR")]
            )
            foot_geoms = np.array(
                [np.flatnonzero(model.geom_bodyid == b)[0] for b in feet]
            )
            self.metadata[scale] = dict(
                names=names,
                effort=effort,
                speed=speed,
                qids=model.jnt_qposadr[joints],
                vids=model.jnt_dofadr[joints],
                bounds=model.jnt_range[joints].copy(),
                feet=feet,
                foot_geoms=foot_geoms,
                base=model.body("base").id,
                foot_lookup={int(body): i for i, body in enumerate(feet)},
            )


class Environment:
    actor_dim = 44
    critic_dim = 74

    def __init__(self, assets, config=None):
        self.assets = assets
        self.cfg = config or LearningConfig()
        self.reset(1.0, 0)

    def reset(self, scale, seed):
        self.scale, self.seed = scale, int(seed)
        self.m = self.assets.models[scale]
        for key, value in self.assets.metadata[scale].items():
            setattr(self, key, value)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = self.assets.poses[scale]
        rng = np.random.default_rng(seed)
        # Same relative pose and paired perturbation draws at every morphology.
        self.d.qpos[self.qids] += rng.uniform(-0.003, 0.003, 12)
        self.d.qvel[4] = rng.uniform(-0.02, 0.02)
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[2] += (
            0.022
            + self.cfg.reset_clearance
            - self.d.geom_xpos[self.foot_geoms, 2].min()
        )
        mujoco.mj_forward(self.m, self.d)
        self.nominal = self.assets.nominals[scale]
        self.target = self.nominal.copy()
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()
        self.mass = self.m.body_mass.sum()
        self.weight = self.mass * 9.81
        self.steps = 0
        self.hold = self.best_hold = self.recovery = self.best_recovery = 0.0
        self.airborne = self.best_airborne = 0.0
        self.takeoff = False
        self.takeoff_drift = 1e6
        self.reward_sum = 0.0
        self.terms_sum = {}
        self.max_pitch = self.max_supported_pitch = self.max_omega = 0.0
        self.max_clearance = 0.0
        self.front_support_samples = self.samples = 0
        self.saturation_sum = self.speed_sum = 0.0
        self.max_speed_ratio = self.max_torque_ratio = self.max_joint_violation = 0.0
        self.nonfoot_collision = self.self_collision = False
        self.row = self.measure()
        self.initial_phi = self.phi = self.potential(self.row)
        return self.observations()

    def measure(self):
        m, d = self.m, self.d
        rotation = d.xmat[self.base].reshape(3, 3)
        feet = d.geom_xpos[self.foot_geoms]
        loads = np.zeros(4)
        illegal = []
        self_contacts = 0
        force = np.empty(6)
        for j in range(d.ncon):
            contact = d.contact[j]
            if contact.dist > 0:
                continue
            b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 and b2:
                self_contacts += 1
                continue
            body = int(b2 if b1 == 0 else b1)
            if body in self.foot_lookup:
                mujoco.mj_contactForce(m, d, j, force)
                loads[self.foot_lookup[body]] += abs(
                    contact.frame.reshape(3, 3)[:, 2] @ force[:3]
                )
            else:
                illegal.append(m.body(body).name)
        mujoco.mj_subtreeVel(m, d)
        com = d.subtree_com[self.base].copy()
        com_velocity = d.subtree_linvel[self.base].copy()
        world_omega = rotation @ d.qvel[3:6]
        axis_rate = np.cross(world_omega, rotation[:, 0])
        omega = (-rotation[0, 0] * axis_rate[2] + rotation[2, 0] * axis_rate[0]) / max(
            rotation[0, 0] ** 2 + rotation[2, 0] ** 2, 1e-9
        )
        return dict(
            pitch=float(np.arctan2(-rotation[2, 0], rotation[0, 0])),
            pitch_velocity=float(omega),
            com=com,
            com_velocity=com_velocity,
            feet=feet.copy(),
            loads=loads,
            rear_clearance=float(feet[2:, 2].min() - 0.022),
            front_support=bool((loads[:2] > 0.05 * self.weight).all()),
            front_drift=float(np.linalg.norm(feet[:2, :2] - self.anchor, axis=1).max()),
            illegal_contacts=len(illegal),
            illegal_bodies="|".join(sorted(set(illegal))),
            self_contacts=self_contacts,
        )

    def observations(self):
        r = self.d.xmat[self.base].reshape(3, 3)
        actor = np.r_[
            r.T @ [0, 0, -1],
            self.d.qvel[3:6],
            self.d.qpos[self.qids] - self.nominal,
            self.d.qvel[self.vids] / self.speed,
            (self.target - self.nominal) / self.cfg.action_scale,
            self.steps * self.cfg.dt / self.cfg.seconds,
            self.scale,
        ].astype(np.float32)
        row = self.row
        support = np.r_[row["feet"][:2, :2].mean(0), 0.0]
        critic = np.r_[
            actor,
            r.T @ self.d.qvel[:3],
            self.d.qpos[2],
            row["com"] - support,
            row["com_velocity"],
            (row["feet"] - row["com"]).ravel(),
            row["loads"] / self.weight,
            (row["feet"][:2, :2] - self.anchor).ravel(),
        ].astype(np.float32)
        assert actor.shape == (self.actor_dim,) and critic.shape == (self.critic_dim,)
        return actor, critic

    @staticmethod
    def potential(row):
        return 2.0 * np.clip(np.sin(row["pitch"]), 0, 1) * float(row["front_support"])

    def step(self, action):
        c = self.cfg
        action = np.clip(action, -1, 1)
        desired = np.clip(self.nominal + c.action_scale * action, *self.bounds.T)
        filtered = self.target + c.target_filter * (desired - self.target)
        new_target = self.target + np.clip(
            filtered - self.target, -0.7 * self.speed * c.dt, 0.7 * self.speed * c.dt
        )
        action_change = np.mean(((new_target - self.target) / c.action_scale) ** 2)
        self.target = new_target
        collision = False
        self_hit = False
        saturation = speed_fraction = torque_cost = 0.0
        event = False
        repeats = round(c.dt / self.m.opt.timestep)
        for _ in range(repeats):
            raw = (
                c.kp * (self.target - self.d.qpos[self.qids])
                - c.kd * self.d.qvel[self.vids]
            )
            torque = np.clip(raw, -self.effort, self.effort)
            overspeed = (abs(self.d.qvel[self.vids]) >= self.speed) & (
                torque * self.d.qvel[self.vids] > 0
            )
            torque[overspeed] = 0
            self.d.qfrc_applied[:] = 0
            self.d.qfrc_applied[self.vids] = torque
            mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            row = self.measure()
            collision |= row["illegal_contacts"] > 0
            self_hit |= row["self_contacts"] > 0
            saturation += np.mean(abs(raw) >= self.effort) / repeats
            speed_fraction += np.mean(overspeed) / repeats
            torque_cost += np.mean((torque / self.effort) ** 2) / repeats
            self.max_speed_ratio = max(
                self.max_speed_ratio,
                float(np.max(abs(self.d.qvel[self.vids]) / self.speed)),
            )
            self.max_torque_ratio = max(
                self.max_torque_ratio, float(np.max(abs(torque) / self.effort))
            )
            violation = max(
                0.0,
                np.max(self.bounds[:, 0] - self.d.qpos[self.qids]),
                np.max(self.d.qpos[self.qids] - self.bounds[:, 1]),
            )
            self.max_joint_violation = max(self.max_joint_violation, float(violation))
            valid = (
                not collision
                and not self_hit
                and violation < 0.001
                and self.max_speed_ratio <= 1.001
            )
            lifted = (
                valid
                and row["front_support"]
                and row["rear_clearance"] > 0.015
                and row["loads"][2:].sum() < 0.04 * self.weight
                and row["pitch"] > 0.12
                and row["front_drift"] < 0.10
            )
            self.airborne = self.airborne + self.m.opt.timestep if lifted else 0.0
            self.best_airborne = max(self.best_airborne, self.airborne)
            if self.airborne >= c.takeoff_seconds and not self.takeoff:
                self.takeoff = event = True
                self.takeoff_drift = row["front_drift"]
            recover = (
                valid
                and row["front_support"]
                and row["rear_clearance"] > 0.10
                and abs(row["pitch"] - np.pi / 2) < np.deg2rad(15)
                and abs(row["pitch_velocity"]) < 2.0
                and np.linalg.norm(row["com_velocity"]) < 0.75
                and row["front_drift"] < 0.10
            )
            self.recovery = self.recovery + self.m.opt.timestep if recover else 0.0
            self.best_recovery = max(self.best_recovery, self.recovery)
            stable = (
                recover
                and abs(row["pitch"] - np.pi / 2) < np.deg2rad(10)
                and abs(row["pitch_velocity"]) < 0.5
                and np.linalg.norm(row["com_velocity"]) < 0.25
            )
            self.hold = self.hold + self.m.opt.timestep if stable else 0.0
            self.best_hold = max(self.best_hold, self.hold)
            if valid:
                self.max_pitch = max(self.max_pitch, row["pitch"])
                self.max_omega = max(self.max_omega, row["pitch_velocity"])
                self.max_clearance = max(self.max_clearance, row["rear_clearance"])
                if row["front_support"]:
                    self.max_supported_pitch = max(
                        self.max_supported_pitch, row["pitch"]
                    )
            self.front_support_samples += int(row["front_support"])
            self.samples += 1
        self.steps += 1
        self.nonfoot_collision |= collision
        self.self_collision |= self_hit
        timeout = self.steps * c.dt >= c.seconds - 1e-9
        invalid_state = self.max_joint_violation > 0.01 or self.max_speed_ratio > 1.05
        done = collision or self_hit or timeout or invalid_state
        cause = (
            "nonfoot_collision"
            if collision
            else "self_collision"
            if self_hit
            else "actuator_state_limit"
            if invalid_state
            else "timeout"
            if timeout
            else "running"
        )
        support = float(row["front_support"] and not collision and not self_hit)
        pitch = np.clip(np.sin(row["pitch"]), 0, 1)
        lift = np.clip(row["rear_clearance"] / 0.15, 0, 1) * support
        next_phi = 0.0 if done and not timeout else self.potential(row)
        terms = dict(
            pitch=2.0 * pitch * support * c.dt,
            lift=2.0 * lift * pitch * c.dt,
            front_support=0.05 * support * c.dt,
            stability=8.0 * float(self.hold > 0) * c.dt,
            recovery=2.0 * float(self.recovery > 0) * c.dt,
            potential=c.gamma * next_phi - self.phi,
            takeoff_bonus=float(event),
            torque=-0.02 * torque_cost * c.dt,
            action_rate=-0.02 * action_change,
            fall=-0.5 * float(done and not timeout),
        )
        reward = sum(terms.values())
        self.phi = next_phi
        self.reward_sum += reward
        for key, value in terms.items():
            self.terms_sum[key] = self.terms_sum.get(key, 0.0) + value
        self.saturation_sum += saturation
        self.speed_sum += speed_fraction
        self.row = row
        info = self.metrics()
        info.update(
            termination=cause,
            timeout=timeout,
            reward=reward,
            saturation=saturation,
            speed_limit_fraction=speed_fraction,
        )
        info.update({f"reward_{k}": v for k, v in terms.items()})
        return self.observations(), reward, done, info

    def metrics(self):
        r = self.row
        result = dict(
            scale=self.scale,
            episode_steps=self.steps,
            episode_seconds=self.steps * self.cfg.dt,
            episode_reward=self.reward_sum,
            max_pitch=self.max_pitch,
            max_supported_pitch=self.max_supported_pitch,
            max_pitch_velocity=self.max_omega,
            rear_clearance=self.max_clearance,
            takeoff=self.takeoff,
            supported_airborne_seconds=self.best_airborne,
            front_support_fraction=self.front_support_samples / max(self.samples, 1),
            nonfoot_collision=self.nonfoot_collision,
            self_collision=self.self_collision,
            recovery_seconds=self.best_recovery,
            handstand_seconds=self.best_hold,
            success=self.best_hold >= 5.0,
            torque_saturation=self.saturation_sum / max(self.steps, 1),
            speed_limit_fraction=self.speed_sum / max(self.steps, 1),
            max_speed_ratio=self.max_speed_ratio,
            max_torque_ratio=self.max_torque_ratio,
            joint_limit_violation=self.max_joint_violation,
            pitch=r["pitch"],
            pitch_velocity=r["pitch_velocity"],
            front_support=r["front_support"],
            front_drift=r["front_drift"],
            current_rear_clearance=r["rear_clearance"],
            com_speed=float(np.linalg.norm(r["com_velocity"])),
            illegal_bodies=r["illegal_bodies"],
        )
        for i, a in enumerate("xyz"):
            result[f"com_{a}"] = float(r["com"][i])
            result[f"com_velocity_{a}"] = float(r["com_velocity"][i])
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            result[f"load_{leg}"] = float(r["loads"][i])
        return result

    def state(self):
        # Models/compiled metadata are immutable and shared. Everything else is
        # serialized, including MuJoCo integration/warmstart state for resumption.
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.m, spec))
        mujoco.mj_getState(self.m, self.d, state, spec)
        keys = [
            k
            for k in self.__dict__
            if k
            not in (
                "assets",
                "cfg",
                "m",
                "d",
                "names",
                "effort",
                "speed",
                "qids",
                "vids",
                "bounds",
                "feet",
                "foot_geoms",
                "base",
                "foot_lookup",
            )
        ]
        return dict(integration=state, values={k: self.__dict__[k] for k in keys})

    def restore(self, state):
        self.reset(state["values"]["scale"], state["values"]["seed"])
        self.__dict__.update(state["values"])
        mujoco.mj_setState(
            self.m, self.d, state["integration"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(self.m, self.d)
