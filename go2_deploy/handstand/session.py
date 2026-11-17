"""Software-only lifecycle and command acknowledgement contract.

This class never publishes commands. Hardware transport and failure response
require course staff review. A cancelled/faulted session cannot emit targets.
"""

import numpy as np
import torch
from .policy import TargetPath, JOINTS, SPEED
from .estimator import SensorEstimator, finite, rotation


class Session:
    def __init__(self, policy, trajectory, model):
        self.policy, self.trajectory, self.model = policy, trajectory, model
        ids = [model.joint(name).id for name in JOINTS]
        self.path = TargetPath(trajectory, model.jnt_range[ids])
        self.active = False
        self.pending = None

    def activate(self, epoch, initial_height, *, calibrated=False):
        self.cancel()
        self.estimator = SensorEstimator(
            self.model, initial_height, epoch, calibrated=calibrated
        )
        self.path.reset()
        self.telemetry = None
        self.active = True
        self.last_proposal_time = None

    def cancel(self):
        self.active = False
        self.pending = None

    def ingest(self, timestamp, force, quaternion, gyro, q, dq):
        if not self.active:
            raise RuntimeError("Explicit activation required")
        try:
            if self.telemetry is None and timestamp == self.estimator.timestamp:
                # Initial standing sample defines epoch; integrate no interval twice.
                finite(force, (3,))
                rotation(quaternion)
            else:
                self.estimator.update(timestamp, force, quaternion)
            finite(gyro, (3,))
            finite(q, (12,))
            finite(dq, (12,))
            self.telemetry = (
                np.array(quaternion),
                np.array(gyro),
                np.array(q),
                np.array(dq),
            )
        except (ValueError, TypeError):
            self.cancel()
            raise

    def snapshot(self, now):
        if not self.active or self.telemetry is None or self.pending is not None:
            raise RuntimeError("Inactive, missing telemetry, or unacknowledged command")
        try:
            return self.estimator.observation(
                *self.telemetry,
                self.path.target,
                self.path.nominal,
                self.path.step,
                now,
            )
        except ValueError:
            self.cancel()
            raise

    def propose(self, now):
        obs = self.snapshot(now)
        if (
            self.last_proposal_time is not None
            and abs(now - self.last_proposal_time - 0.02) > 0.002
        ):
            self.cancel()
            raise ValueError("Control cadence differs from 50 Hz")
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs)).numpy()
        finite(action, (12,))
        # Apply only to a temporary path: state advances on acknowledgement.
        desired = np.clip(
            self.path.nominal + 2 * action.clip(-1, 1), *self.path.bounds.T
        )
        self.last_proposal_time = now
        self.pending = self.path.target + np.clip(
            desired - self.path.target, -0.7 * SPEED * 0.02, 0.7 * SPEED * 0.02
        )
        return action.copy(), self.pending.copy()

    def acknowledge(self, target):
        if (
            not self.active
            or self.pending is None
            or not np.array_equal(target, self.pending)
        ):
            self.cancel()
            raise ValueError("Applied target differs from proposed bounded target")
        self.path.target = self.pending.copy()
        self.path.step += 1
        self.pending = None
