"""Explicit IMU mounting/bias calibration. Constants require staff measurement."""

import mujoco
import numpy as np
from .estimator import finite, rotation


class IMUCalibration:
    def __init__(
        self,
        body_from_imu,
        world_alignment,
        lever_arm_body,
        accel_bias_imu,
        gyro_bias_imu,
    ):
        self.mount = finite(body_from_imu, (3, 3))
        self.world = finite(world_alignment, (3, 3))
        for matrix in (self.mount, self.world):
            if not np.allclose(
                matrix.T @ matrix, np.eye(3), atol=1e-8
            ) or not np.isclose(np.linalg.det(matrix), 1):
                raise ValueError("Calibration requires proper rotations")
        self.lever = finite(lever_arm_body, (3,))
        self.accel_bias = finite(accel_bias_imu, (3,))
        self.gyro_bias = finite(gyro_bias_imu, (3,))
        self.previous_time = None
        self.previous_gyro = None

    def correct(self, timestamp, quaternion, gyro, specific_force):
        if not np.isfinite(timestamp):
            raise ValueError("Invalid calibration timestamp")
        omega = self.mount @ (finite(gyro, (3,)) - self.gyro_bias)
        alpha = np.zeros(3)
        if self.previous_time is not None:
            dt = timestamp - self.previous_time
            if dt <= 0 or dt > 0.003:
                raise ValueError("Calibration sample gap")
            alpha = (omega - self.previous_gyro) / dt
        # First sample must be stationary; derivative noise is not filtered.
        elif np.linalg.norm(omega) > 0.02:
            raise ValueError("Calibration initialization must be stationary")
        r = self.world @ rotation(quaternion) @ self.mount.T
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, r.ravel())
        force = self.mount @ (finite(specific_force, (3,)) - self.accel_bias)
        force -= np.cross(alpha, self.lever) + np.cross(
            omega, np.cross(omega, self.lever)
        )
        self.previous_time = float(timestamp)
        self.previous_gyro = omega.copy()
        return q, omega, force
