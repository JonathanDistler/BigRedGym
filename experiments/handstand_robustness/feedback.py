"""Finite-horizon trajectory LQR projected into sensor observations.

Offline finite differences use simulator state. Online feedback only consumes
43 actor inputs. This is a model-based controller, not a trained neural actor.
Projection removes unobservable state directions and can lose stabilization.
"""

import mujoco
import numpy as np

from gym.envs.go2.handstand_replay import (
    ACTION_SCALE,
    CONTROL_DT,
    KD,
    KP,
    STATE_SPEC,
    Replay,
)


def observation(robot, data, target, index):
    r = data.xmat[robot.base].reshape(3, 3)
    return np.r_[
        r.T @ [0, 0, -1],
        data.qvel[3:6],
        data.qpos[robot.qids] - robot.trajectory["nominal"],
        data.qvel[robot.vids] / robot.speed,
        (target - robot.trajectory["nominal"]) / 2,
        index * CONTROL_DT / 8,
    ]


def build(model, trajectory):
    robot = Replay(model, trajectory)
    n = len(trajectory["actions"])
    data = mujoco.MjData(model)
    refdata = mujoco.MjData(model)
    refs, targets = [], [trajectory["initial_target"].copy()]
    for action in trajectory["actions"]:
        desired = np.clip(
            trajectory["nominal"] + ACTION_SCALE * action, *robot.bounds.T
        )
        targets.append(
            targets[-1]
            + np.clip(
                desired - targets[-1],
                -0.7 * robot.speed * CONTROL_DT,
                0.7 * robot.speed * CONTROL_DT,
            )
        )
    A, B, C = [], [], []
    eps = 1e-5
    for k in range(n):
        mujoco.mj_setState(
            model, refdata, trajectory["reference_states"][k], STATE_SPEC
        )
        mujoco.mj_forward(model, refdata)
        refs.append(observation(robot, refdata, targets[k], k))

        def integrate(dx, du):
            mujoco.mj_setState(
                model, data, trajectory["reference_states"][k], STATE_SPEC
            )
            mujoco.mj_integratePos(model, data.qpos, dx[:18], 1.0)
            data.qvel[:] += dx[18:36]
            target = targets[k] + dx[36:]
            mujoco.mj_forward(model, data)
            observed = observation(robot, data, target, k)
            action = np.clip(trajectory["actions"][k] + du, -1, 1)
            desired = np.clip(trajectory["nominal"] + 2 * action, *robot.bounds.T)
            target += np.clip(
                desired - target,
                -0.7 * robot.speed * CONTROL_DT,
                0.7 * robot.speed * CONTROL_DT,
            )
            for _ in range(10):
                vel = data.qvel[robot.vids].copy()
                torque = np.clip(
                    KP * (target - data.qpos[robot.qids]) - KD * vel,
                    -robot.effort,
                    robot.effort,
                )
                torque[(abs(vel) >= robot.speed) & (torque * vel > 0)] = 0
                data.qfrc_applied[:] = 0
                data.qfrc_applied[robot.vids] = torque
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
            return data.qpos.copy(), data.qvel.copy(), target, observed

        def difference(a, b):
            dq = np.zeros(18)
            mujoco.mj_differentiatePos(model, dq, 2 * eps, b[0], a[0])
            return np.r_[dq, (a[1] - b[1]) / (2 * eps), (a[2] - b[2]) / (2 * eps)]

        aa, bb, cc = np.zeros((48, 48)), np.zeros((48, 12)), np.zeros((42, 48))
        for j in range(48):
            dx = np.zeros(48)
            dx[j] = eps
            plus, minus = integrate(dx, np.zeros(12)), integrate(-dx, np.zeros(12))
            aa[:, j] = difference(plus, minus)
            cc[:, j] = (plus[3][:42] - minus[3][:42]) / (2 * eps)
        for j in range(12):
            du = np.zeros(12)
            du[j] = eps
            bb[:, j] = difference(
                integrate(np.zeros(48), du), integrate(np.zeros(48), -du)
            )
        A.append(aa)
        B.append(bb)
        C.append(cc)
        if k % 50 == 0:
            print(f"linearized {k}/{n}", flush=True)
    # SI-unit costs: rotation and joint tracking dominate; translation is weak
    # because the projected actor cannot observe absolute horizontal position.
    Q = np.diag(
        np.r_[
            [1, 1, 20],
            [80, 80, 5],
            np.full(12, 4),
            np.full(3, 1),
            np.full(3, 2),
            np.full(12, 0.05),
            np.full(12, 0.1),
        ]
    )
    R = np.eye(12) * 0.5
    P = Q * 10
    gains = [None] * n
    residuals = []
    for k in reversed(range(n)):
        gain = np.linalg.solve(R + B[k].T @ P @ B[k], B[k].T @ P @ A[k])
        gains[k] = gain @ np.linalg.pinv(C[k], rcond=1e-5)
        residuals.append(float(np.linalg.norm(gains[k] @ C[k] - gain)))
        P = Q + A[k].T @ P @ (A[k] - B[k] @ gain)
        P = 0.5 * (P + P.T)
    return dict(
        gains=np.array(gains),
        observations=np.array(refs),
        actions=trajectory["actions"],
        projection_residual=np.array(residuals),
    )


class FeedbackPolicy:
    def __init__(self, archive):
        self.gains = archive["gains"]
        self.observations = archive["observations"]
        self.actions = archive["actions"]

    def __call__(self, observations):
        import torch

        values = observations.detach().cpu().numpy()
        result = []
        for obs in values:
            k = int(
                np.clip(
                    round(float(obs[42]) * 8 / CONTROL_DT), 0, len(self.actions) - 1
                )
            )
            correction = self.gains[k] @ (obs[:42] - self.observations[k, :42])
            result.append(np.clip(self.actions[k] - correction, -1, 1))
        return torch.tensor(np.array(result), dtype=torch.float64)
