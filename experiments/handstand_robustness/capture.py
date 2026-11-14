"""Capture instrumentation and explicit controller comparisons; simulation only."""

import mujoco
import numpy as np
import torch
from torch import nn
from experiments.handstand_robustness.takeoff import ResidualPolicy

from experiments.handstand_robustness.core import Environment
from gym.envs.go2.handstand_replay import PHYSICS_DT, STATE_SPEC, clearance
from scripts.handstand_robustness import legacy_teacher


class CaptureEnvironment(Environment):
    def __init__(self, model, trajectory, seed=0, condition=None, trace=True):
        self.trace_enabled = trace
        self.physics = []
        self.contacts = []
        self.controls = []
        self.first_illegal = None
        self.first_geometric_illegal = None
        self.first_safety_violation = None
        self.max_observer_velocity_error = 0.0
        self.max_observer_height_error = 0.0
        self.max_rear_clearance = -float("inf")
        self.front_support_transitions = []
        self._last_front_support = None
        self.foot_impulse_total = np.zeros((4, 3))
        self.capture_foot_impulse = np.zeros((4, 3))
        self.max_front_force = 0.0
        self.max_torque_ratio = 0.0
        self.command = np.zeros(12)
        super().__init__(model, trajectory, seed, condition)
        self.calves = [self.geoms((leg + "_calf",)) for leg in ("FL", "FR", "RL", "RR")]

    def after_physics_step(self):
        # mj_step solver contacts/forces BEFORE mj_forward recomputes constraints.
        m, d = self.m, self.d
        if hasattr(self, "imu_observer"):
            rotation = d.xmat[self.base].reshape(3, 3)
            specific_force = rotation.T @ (d.qacc[:3] - m.opt.gravity)
            self.imu_observer.update(specific_force, rotation)
        forces = np.zeros((4, 3))
        touch = np.zeros(4, dtype=bool)
        for i, c in enumerate(d.contact):
            if c.dist > 0:
                continue
            b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
            if b1 == b2 == 0:
                continue
            wrench = np.empty(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            world = c.frame.reshape(3, 3).T @ wrench[:3]
            illegal = bool(
                (b1 and b2)
                or (not b1 and b2 not in self.feet)
                or (not b2 and b1 not in self.feet)
            )
            if (b1 == 0) != (b2 == 0):
                b = b2 if b1 == 0 else b1
                if b in self.feet:
                    j = self.feet.index(b)
                    touch[j] = True
                    forces[j] += world * (1 if b1 == 0 else -1)
            event = dict(
                interval_start_s=float(d.time - PHYSICS_DT),
                time_s=float(d.time),
                geom_ids=[int(c.geom1), int(c.geom2)],
                geom_names=[
                    m.geom(int(g)).name or f"geom_{g}" for g in [c.geom1, c.geom2]
                ],
                bodies=[m.body(b1).name, m.body(b2).name],
                position_world_m=c.pos.copy().tolist(),
                normal_geom1_to_geom2=c.frame[:3].copy().tolist(),
                force_on_geom2_world_N=world.tolist(),
                impulse_on_geom2_world_Ns=(world * PHYSICS_DT).tolist(),
                contact_wrench=wrench.tolist(),
                distance_m=float(c.dist),
                illegal=illegal,
            )
            if illegal and self.first_illegal is None:
                self.first_illegal = event
            if self.trace_enabled and (0.8 <= d.time <= 2.3 or illegal):
                self.contacts.append(event)
        supported = bool((forces[:2, 2] > 0.05 * self.weight).all())
        if supported != self._last_front_support:
            self.front_support_transitions.append(
                dict(time_s=float(d.time), supported=supported)
            )
            self._last_front_support = supported
        self.foot_impulse_total += forces * PHYSICS_DT
        if d.time - PHYSICS_DT >= 0.8 - 1e-12:
            self.capture_foot_impulse += forces * PHYSICS_DT
        self.max_front_force = max(self.max_front_force, float(forces[:2, 2].sum()))
        self.solver_forces = forces
        self.solver_touch = touch

    def record_metrics(self):
        super().record_metrics()
        self.max_torque_ratio = max(
            self.max_torque_ratio,
            float(abs(self.d.qfrc_applied[self.vids] / self.effort).max()),
        )
        if self.first_geometric_illegal is None:
            for c in self.d.contact:
                if c.dist > 0:
                    continue
                b1, b2 = map(int, self.m.geom_bodyid[[c.geom1, c.geom2]])
                if b1 == b2 == 0:
                    continue
                illegal = bool(
                    (b1 and b2)
                    or (not b1 and b2 not in self.feet)
                    or (not b2 and b1 not in self.feet)
                )
                if illegal:
                    self.first_geometric_illegal = dict(
                        time_s=float(self.d.time),
                        geom_ids=[int(c.geom1), int(c.geom2)],
                        bodies=[self.m.body(b1).name, self.m.body(b2).name],
                        position_world_m=c.pos.tolist(),
                        normal_geom1_to_geom2=c.frame[:3].tolist(),
                        scope=(
                            "post-integration geometry; force impulse is reported "
                            "for following solver interval"
                        ),
                    )
                    break
        if not self.safe and self.first_safety_violation is None:
            speed_ratio = abs(self.d.qvel[self.vids] / self.speed)
            range_bad = (self.d.qpos[self.qids] < self.bounds[:, 0] - 0.001) | (
                self.d.qpos[self.qids] > self.bounds[:, 1] + 0.001
            )
            reason = (
                "illegal_contact"
                if self.first_geometric_illegal is not None
                else "joint_speed"
                if (speed_ratio > 1.001).any()
                else "joint_range"
                if range_bad.any()
                else "safety_gate"
            )
            self.first_safety_violation = dict(
                time_s=float(self.d.time),
                reason=reason,
                max_speed_ratio=float(speed_ratio.max()),
            )
        if hasattr(self, "imu_observer"):
            self.max_observer_velocity_error = max(
                self.max_observer_velocity_error,
                float(np.linalg.norm(self.imu_observer.velocity - self.d.qvel[:3])),
            )
            self.max_observer_height_error = max(
                self.max_observer_height_error,
                abs(float(self.imu_observer.height - self.d.qpos[2])),
            )
        self.max_rear_clearance = max(
            self.max_rear_clearance,
            float((self.d.geom_xpos[self.foot_geoms[2:], 2] - 0.022).min()),
        )
        if not self.trace_enabled:
            return
        r = self.d.xmat[self.base].reshape(3, 3)
        axis = self.trajectory["pitch_axis"]
        pitch = np.arctan2(-r[2, 0], r[:, 0] @ [axis[1], -axis[0], 0])
        row = dict(
            time_s=float(self.d.time),
            pitch_deg=float(np.rad2deg(pitch)),
            pitch_rate_rad_s=float(axis @ r @ self.d.qvel[3:6]),
            com_position_m=self.d.subtree_com[self.base].copy().tolist(),
            com_velocity_m_s=self.d.subtree_linvel[self.base].copy().tolist(),
            foot_positions_world_m=self.d.geom_xpos[self.foot_geoms].copy().tolist(),
            com_minus_front_support_m=(
                self.d.subtree_com[self.base]
                - self.d.geom_xpos[self.foot_geoms[:2]].mean(0)
            ).tolist(),
            foot_contacts=self.solver_touch.tolist(),
            foot_forces_N=self.solver_forces.tolist(),
            calf_clearance_m=[
                min(clearance(self.m, self.d, g) for g in geoms)
                for geoms in self.calves
            ],
            rear_foot_clearance_m=float(
                (self.d.geom_xpos[self.foot_geoms[2:], 2] - 0.022).min()
            ),
            q=self.d.qpos[self.qids].copy().tolist(),
            dq=self.d.qvel[self.vids].copy().tolist(),
            q_target=self.target.copy().tolist(),
            torque_Nm=self.d.qfrc_applied[self.vids].copy().tolist(),
            effort_limit_Nm=self.effort.tolist(),
            safe=self.safe,
            action=self.command.copy().tolist(),
            strict_hold_s=self.hold,
        )
        if hasattr(self, "imu_observer"):
            row["observer_root_velocity_error_m_s"] = (
                self.imu_observer.velocity - self.d.qvel[:3]
            ).tolist()
            row["observer_root_height_error_m"] = float(
                self.imu_observer.height - self.d.qpos[2]
            )
        self.physics.append(row)

    def step(self, action):
        self.command = np.asarray(action).copy()
        obs, reward, done = super().step(action)
        if self.trace_enabled:
            k = self.index - 1
            ref = mujoco.MjData(self.m)
            mujoco.mj_setState(
                self.m,
                ref,
                self.trajectory["reference_states"][
                    min(self.index, len(self.trajectory["actions"]))
                ],
                STATE_SPEC,
            )
            self.controls.append(
                dict(
                    time_s=k * 0.02,
                    reference_index=k,
                    action=self.command.tolist(),
                    reference_action=self.trajectory["actions"][k].tolist(),
                    action_error_max=float(
                        abs(self.command - self.trajectory["actions"][k]).max()
                    ),
                    effective_phase_index=int(
                        getattr(self, "effective_phase_index", k)
                    ),
                    phase_error_s=(int(getattr(self, "effective_phase_index", k)) - k)
                    * 0.02,
                    joint_error_max_rad=float(
                        abs(self.d.qpos[self.qids] - ref.qpos[self.qids]).max()
                    ),
                    quaternion_error_rad=float(
                        2
                        * np.arccos(
                            np.clip(abs(self.d.qpos[3:7] @ ref.qpos[3:7]), 0, 1)
                        )
                    ),
                )
            )
        return obs, reward, done

    def report(self):
        r = self.result()
        r.update(
            first_illegal=self.first_illegal,
            first_geometric_illegal=self.first_geometric_illegal,
            maximum_rear_clearance_m=self.max_rear_clearance,
            front_support_transitions=self.front_support_transitions,
            first_safety_violation=self.first_safety_violation,
            max_observer_velocity_error_m_s=self.max_observer_velocity_error,
            max_observer_height_error_m=self.max_observer_height_error,
            foot_impulse_total_Ns=self.foot_impulse_total.tolist(),
            capture_window_foot_impulse_Ns=self.capture_foot_impulse.tolist(),
            max_front_force_N=self.max_front_force,
            max_torque_ratio=self.max_torque_ratio,
        )
        if self.physics:
            r["maximum_rear_clearance_m"] = max(
                x["rear_foot_clearance_m"] for x in self.physics
            )
            r["minimum_calf_clearance_m"] = np.min(
                [x["calf_clearance_m"] for x in self.physics], axis=0
            ).tolist()
        return r


class Controller:
    def __init__(self, name, trajectory, checkpoint=None, balance=None):
        self.name, self.t = name, trajectory
        from scripts.handstand_takeoff import load

        self.actor = load(checkpoint)[0] if checkpoint else None
        self.balance = legacy_teacher(balance) if balance else None

    def __call__(self, e):
        k = e.index
        reference = self.t["actions"][min(k, len(self.t["actions"]) - 1)]
        e.effective_phase_index = min(k, 47) if self.name in ("prior", "student") else k
        if self.name == "reference":
            return reference.copy()
        if self.name in ("prior", "student"):
            with torch.no_grad():
                return self.actor(
                    torch.tensor(e.observation())[None], feedback=self.name != "prior"
                )[0].numpy()
        if self.name == "hybrid":
            if k < 69:
                return reference.copy()
            b = self.balance
            x = e.balance_features()
            u = np.clip(
                b["weight"] @ x[b["channels"]] + b["bias"], b["lower"], b["upper"]
            )
            t = k * 0.02 - 1.32
            duration = 0.64517335
            amp = max(0, min(1, t / 0.1, (duration - t) / 0.15))
            u[[1, 4]] += -0.0204364 * amp
            u[[2, 5]] += 0.10256115 * amp
            return u.clip(-1, 1)
        raise ValueError(self.name)


class KinematicBalance:
    """IMU+encoder-only reconstruction; bilateral stationary front stance assumed.

    Separate FK data is never integrated. No world base position/velocity or
    simulator force is accepted. The IMU quaternion is wxyz in the model frame.
    Hardware estimator accuracy and the stance assumption are NOT validated.
    """

    def __init__(self, trajectory, balance):
        from gym.envs.go2.handstand_replay import Replay, build_model

        self.robot = Replay(build_model(), trajectory)
        self.b = legacy_teacher(balance) if balance else None
        self.nominal = trajectory["nominal"]

    def features(self, obs, quaternion):
        r = self.robot
        m, d = r.m, r.d
        d.qpos[:3] = [0, 0, 1]
        d.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
        d.qpos[r.qids] = obs[6:18] + self.nominal
        d.qvel[:3] = 0
        d.qvel[3:6] = obs[3:6]
        d.qvel[r.vids] = obs[18:30] * r.speed
        mujoco.mj_forward(m, d)
        jac = np.zeros((3, m.nv))
        footjac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, r.base)
        for g in r.foot_geoms[:2]:
            partial = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, partial, None, int(g))
            footjac += partial / 2
        height = (
            d.subtree_com[r.base, 2] - d.geom_xpos[r.foot_geoms[:2], 2].mean() + 0.022
        )
        velocity = (jac - footjac) @ d.qvel
        x = np.zeros(93, np.float32)
        x[:30] = obs[:30]
        x[55] = height / 0.6
        x[56:59] = velocity
        return x

    def __call__(self, obs, quaternion):
        b = self.b
        x = self.features(obs, quaternion)
        return np.clip(
            b["weight"] @ x[b["channels"]] + b["bias"], b["lower"], b["upper"]
        )


class InertialObserver:
    """Ideal 500 Hz IMU integration from calibrated standing, not hardware proof."""

    def __init__(self, initial_height):
        self.velocity = np.zeros(3)
        self.height = float(initial_height)

    def update(self, specific_force_body, rotation):
        acceleration = rotation @ np.asarray(specific_force_body) + [0, 0, -9.81]
        self.velocity += PHYSICS_DT * acceleration
        self.height += PHYSICS_DT * self.velocity[2]


class InertialBalance(KinematicBalance):
    def __init__(self, trajectory, balance):
        super().__init__(trajectory, balance)
        # Known ordinary standing calibration, never a mid-trajectory state read.
        d = self.robot.d
        self.observer = InertialObserver(float(d.qpos[2]))

    def features(self, obs, quaternion):
        # Populate only IMU/encoder FK state through the parent implementation.
        super().features(obs, quaternion)
        r = self.robot
        m, d = r.m, r.d
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, r.base)
        height = self.observer.height + d.subtree_com[r.base, 2] - d.qpos[2]
        velocity = self.observer.velocity + jac @ d.qvel
        x = np.zeros(93, np.float32)
        x[:30] = obs[:30]
        x[55] = height / 0.6
        x[56:59] = velocity
        return x


def sensor_observation(e):
    if not hasattr(e, "capture_kinematics"):
        raise ValueError(
            "Capture IMU/encoder observer must be attached before stepping"
        )
    obs = Environment.observation(e)
    x = e.capture_kinematics.features(obs, e.d.qpos[3:7].copy())
    return np.r_[obs, x[55:59]].astype(np.float32)


class CapturePolicy(ResidualPolicy):
    checkpoint_kind = "capture_residual_v1"

    def __init__(self, actions, centers, balance, active_steps=110, bound=0.05):
        super().__init__(actions, centers, bound)
        self.scale = torch.tensor(
            [0.02] * 3 + [0.1] * 3 + [0.02] * 36 + [0.02, 0.1, 0.1, 0.1]
        )
        self.gains = nn.Parameter(torch.zeros(len(actions), 12, 46))
        self.critic = nn.Sequential(nn.Linear(47, 64), nn.Tanh(), nn.Linear(64, 1))
        self.register_buffer("active_steps", torch.tensor(active_steps))
        self.register_buffer(
            "balance_weight", torch.tensor(balance["weight"], dtype=torch.float32)
        )
        self.register_buffer(
            "balance_bias", torch.tensor(balance["bias"], dtype=torch.float32)
        )
        self.register_buffer(
            "balance_lower", torch.tensor(balance["lower"], dtype=torch.float32)
        )
        self.register_buffer(
            "balance_upper", torch.tensor(balance["upper"], dtype=torch.float32)
        )

    def inputs(self, obs):
        if obs.shape[-1] != 47:
            raise ValueError("Capture actor requires 47 sensor/estimate channels")
        k = (obs[..., 42] * 400).round().long().clamp(0, int(self.active_steps) - 1)
        delta = torch.cat(
            [
                obs[..., :42] - self.centers[k, :42],
                obs[..., 43:47] - self.centers[k, 43:47],
            ],
            -1,
        )
        return k, (delta / self.scale).clamp(-20, 20)

    def forward(self, obs, feedback=True):
        prefix = super().forward(obs, feedback)
        k = (obs[..., 42] * 400).round().long().clamp(0, len(self.actions) - 1)
        features = torch.cat([obs[..., :30], obs[..., 43:47]], -1)
        centers = torch.cat([self.centers[k, :30], self.centers[k, 43:47]], -1)
        correction = ((features - centers) @ self.balance_weight.T).clamp(
            -self.bound, self.bound
        )
        balance = (self.actions[k] + correction).clamp(-1, 1)
        late = (obs[..., 42] * 400).round() >= self.active_steps
        if not feedback:
            balance = self.actions[k]
        return torch.where(late[..., None], balance, prefix)


class CaptureTrainingEnvironment(CaptureEnvironment):
    def __init__(self, model, trajectory, seed=0, condition=None, horizon=110):
        super().__init__(model, trajectory, seed, condition, trace=False)
        self.horizon = horizon
        self.capture_kinematics = InertialBalance(trajectory, None)
        self.imu_observer = self.capture_kinematics.observer
        self.finite = True
        self.reward_totals = {}
        self._capture_paid = False

    def observation(self):
        if not hasattr(self, "capture_kinematics"):
            return Environment.observation(self)
        return sensor_observation(self)

    def step(self, action):
        obs, _, done = super().step(action)
        ref = mujoco.MjData(self.m)
        mujoco.mj_setState(
            self.m,
            ref,
            self.trajectory["reference_states"][
                min(self.index, len(self.trajectory["actions"]))
            ],
            STATE_SPEC,
        )
        tracking = float(
            np.exp(-20 * np.mean((self.d.qpos[self.qids] - ref.qpos[self.qids]) ** 2))
        )
        captured = self.safe and self.takeoff and self.hold >= 0.1
        terms = dict(
            tracking=0.02 * tracking,
            capture_event=2.0 * int(captured and not self._capture_paid),
            terminal_capture=2.0 * int(self.index >= self.horizon and captured),
            stable_hold=0.02 * int(self.hold > 0),
            collision=-2.0 * int(not self.safe),
            residual_cost=-0.001
            * float(
                np.mean(
                    (
                        np.asarray(action)
                        - self.trajectory["actions"][
                            min(self.index - 1, len(self.trajectory["actions"]) - 1)
                        ]
                    )
                    ** 2
                )
            ),
        )
        self._capture_paid |= captured
        self.last_terms = terms
        for k, v in terms.items():
            self.reward_totals[k] = self.reward_totals.get(k, 0.0) + v
        reward = sum(terms.values())
        self.finite &= bool(np.isfinite(obs).all() and np.isfinite(reward))
        return (
            obs,
            reward,
            (not self.safe)
            or self.index >= self.horizon
            or (self.horizon >= 320 and self.best_hold >= 5 and self.takeoff),
        )

    def result(self):
        row = super().result()
        success = bool(self.safe and self.takeoff and self.hold >= 0.1)
        rotation = self.d.xmat[self.base].reshape(3, 3)
        axis = self.trajectory["pitch_axis"]
        ref = mujoco.MjData(self.m)
        mujoco.mj_setState(
            self.m,
            ref,
            self.trajectory["reference_states"][
                min(self.index, len(self.trajectory["actions"]))
            ],
            STATE_SPEC,
        )
        quaterror = 2 * np.arccos(np.clip(abs(self.d.qpos[3:7] @ ref.qpos[3:7]), 0, 1))
        support = bool((self.solver_forces[:2, 2] > 0.05 * self.weight).all())
        clear = float((self.d.geom_xpos[self.foot_geoms[2:], 2] - 0.022).min())
        prefix = bool(
            self.safe
            and self.takeoff
            and support
            and clear > 0.1
            and quaterror < np.deg2rad(5)
        )
        row.update(
            standing_start=True,
            capture_success=success,
            objective_success=success,
            prefix_success=prefix,
            current_hold_seconds=self.hold,
            endpoint_pitch_rate_rad_s=float(axis @ rotation @ self.d.qvel[3:6]),
            foot_forces_N=self.solver_forces.tolist(),
            first_geometric_illegal=self.first_geometric_illegal,
            first_illegal=self.first_illegal,
            maximum_rear_clearance_m=self.max_rear_clearance,
            takeoff_success=bool(self.takeoff and self.safe),
            finite=self.finite,
            reward_components=self.reward_totals,
            control_steps=self.index,
        )
        return row


class SensorTeacher:
    """Validated feedforward prior plus live sensor-error feedback at every step."""

    def __init__(self, trajectory, feedback, balance, centers, bound=0.05):
        self.t, self.f, self.centers = trajectory, feedback, centers
        self.b = legacy_teacher(balance)
        self.bound = bound

    def __call__(self, e):
        k = min(e.index, len(self.t["actions"]) - 1)
        obs = e.observation()
        if obs.shape != (47,):
            raise ValueError("Expected 47 sensor/estimated channels")
        if k < 69:
            correction = -self.f["gains"][k] @ (obs[:42] - self.centers[k, :42])
        else:
            features = np.r_[obs[:30], obs[43:47]]
            center = np.r_[self.centers[k, :30], self.centers[k, 43:47]]
            correction = self.b["weight"] @ (features - center)
        return np.clip(
            self.t["actions"][k] + np.clip(correction, -self.bound, self.bound), -1, 1
        )
