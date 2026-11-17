"""Timestamped body-origin IMU integration and sensor-only COM kinematics.

Quaternion wxyz maps calibrated body to world; gyro is body rad/s; specific
force is body m/s² at the free-joint origin, including positive-up gravity.
No contact constraint or world root measurement is used. Mounting correction
must occur upstream: raw uncalibrated IMU measurements are not accepted.
"""

import mujoco
import numpy as np
from .policy import JOINTS, SPEED


def finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite sensor shape {shape}")
    return value


def rotation(quaternion):
    q = finite(quaternion, (4,))
    if abs(np.linalg.norm(q) - 1) > 0.001:
        raise ValueError("Uncalibrated quaternion")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q / np.linalg.norm(q))
    return matrix.reshape(3, 3)


class SensorEstimator:
    def __init__(
        self, model, initial_height, epoch, *, calibrated=False, joint_order=JOINTS
    ):
        if not calibrated or list(joint_order) != JOINTS:
            raise ValueError(
                "Explicit frame/height calibration and canonical joint order required"
            )
        if not np.isfinite([initial_height, epoch]).all():
            raise ValueError("Invalid initialization")
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(name).id for name in JOINTS]
        self.qids = model.jnt_qposadr[joints]
        self.vids = model.jnt_dofadr[joints]
        self.base = model.body("base").id
        self.height = float(initial_height)
        self.velocity = np.zeros(3)
        self.timestamp = float(epoch)
        self.samples = 0

    def update(self, timestamp, force, quaternion):
        force = finite(force, (3,))
        r = rotation(quaternion)
        dt = float(timestamp) - self.timestamp
        if not np.isfinite(dt) or dt <= 0 or dt > 0.003:
            raise ValueError("Duplicate, out-of-order or missing IMU measurement")
        # Semi-implicit Euler, exactly as research InertialObserver at dt=.002.
        self.velocity += dt * (r @ force + [0, 0, -9.81])
        self.height += dt * self.velocity[2]
        self.timestamp = float(timestamp)
        self.samples += 1

    def com(self, quaternion, gyro, q, dq, now):
        if (
            not np.isfinite(now)
            or now - self.timestamp < -1e-9
            or now - self.timestamp > 0.006
        ):
            raise ValueError("Stale or future IMU estimate")
        rotation(quaternion)
        d, m = self.data, self.model
        d.qpos[:3] = [0, 0, 1]
        d.qpos[3:7] = quaternion
        d.qpos[self.qids] = finite(q, (12,))
        d.qvel[:3] = 0
        d.qvel[3:6] = finite(gyro, (3,))
        d.qvel[self.vids] = finite(dq, (12,))
        mujoco.mj_forward(m, d)  # FK/Jacobian only: never mj_step.
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, self.base)
        return np.r_[
            (self.height + d.subtree_com[self.base, 2] - d.qpos[2]) / 0.6,
            self.velocity + jac @ d.qvel,
        ].astype(np.float32)

    def observation(self, quaternion, gyro, q, dq, target, nominal, step, now):
        r = rotation(quaternion)
        q, dq = finite(q, (12,)), finite(dq, (12,))
        target, nominal = finite(target, (12,)), finite(nominal, (12,))
        base = np.r_[
            r.T @ [0, 0, -1],
            finite(gyro, (3,)),
            q - nominal,
            dq / SPEED,
            (target - nominal) / 2,
            step * 0.02 / 8,
        ].astype(np.float32)
        # Research FK receives rounded float32 observation channels.
        com = self.com(
            quaternion, base[3:6], base[6:18] + nominal, base[18:30] * SPEED, now
        )
        result = np.r_[base, com].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite observation after sensor normalization")
        return result
