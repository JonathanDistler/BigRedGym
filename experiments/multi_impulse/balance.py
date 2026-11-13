"""Native equilibrium/LQR reset validation and optional offline supervision.

The LQR is a probe/teacher only. Deployment and PPO always use one learned actor.
"""

import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.linalg import solve_discrete_are
from experiments.recoverability.configuration_planning.geometry import (
    ConfigurationProblem,
)


def native_step(e, q, v, target):
    d = mujoco.MjData(e.m)
    d.qpos[:] = q
    d.qvel[:] = v
    for _ in range(round(e.cfg.dt / e.m.opt.timestep)):
        mujoco.mj_forward(e.m, d)
        raw = (
            e.cfg.kp * (target - d.qpos[e.qids]) - np.asarray(e.cfg.kd) * d.qvel[e.vids]
        )
        torque = np.clip(raw, -e.effort, e.effort)
        torque[(abs(d.qvel[e.vids]) >= e.speed) & (torque * d.qvel[e.vids] > 0)] = 0
        d.qfrc_applied[:] = 0
        d.qfrc_applied[e.vids] = torque
        mujoco.mj_step(e.m, d)
    mujoco.mj_forward(e.m, d)
    return d


def equilibrium(assets, cfg, height=0.57):
    problem = ConfigurationProblem(assets, cfg)
    e = problem.w
    seed = e.nominal.copy()
    for i, n in enumerate(e.names):
        if n.startswith(("FL_", "FR_")):
            seed[i] = 0 if "hip" in n else -0.9 if "thigh" in n else -0.6
    r = problem.solve(90, height, 0, [0.28, 0, 0, 0], seed, max_nfev=100)
    if not r["geometric_cell"]:
        raise RuntimeError("No geometrically valid equilibrium proposal")
    q = np.array(r["qpos"])
    problem.set(q)
    J = []
    for i in range(2):
        jp = np.zeros((3, e.m.nv))
        point = problem.d.geom_xpos[e.foot_geoms[i]].copy()
        point[2] -= 0.022
        mujoco.mj_jac(e.m, problem.d, jp, None, point, int(e.feet[i]))
        J.append(jp)
    J = np.vstack(J)
    force = np.linalg.lstsq(J[:, :6].T, problem.d.qfrc_bias[:6], rcond=1e-9)[0]
    tau = (problem.d.qfrc_bias - J.T @ force)[e.vids]
    lo = np.maximum(e.bounds[:, 0], e.nominal - cfg.action_scale)
    hi = np.minimum(e.bounds[:, 1], e.nominal + cfg.action_scale)
    guess = np.r_[
        q[2] - 0.0005, np.clip(q[e.qids] + tau / cfg.kp, lo + 1e-6, hi - 1e-6)
    ]
    d = mujoco.MjData(e.m)

    def residual(x):
        d.qpos[:] = q
        d.qpos[2] = x[0]
        d.qvel[:] = 0
        d.qfrc_applied[:] = 0
        d.qfrc_applied[e.vids] = cfg.kp * (x[1:] - d.qpos[e.qids])
        mujoco.mj_forward(e.m, d)
        return d.qacc.copy()

    fit = least_squares(
        residual,
        guess,
        bounds=(np.r_[q[2] - 0.0024, lo], np.r_[q[2] + 0.0001, hi]),
        max_nfev=200,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        diff_step=1e-5,
    )
    error = float(np.linalg.norm(residual(fit.x)))
    q[2] = fit.x[0]
    if error > 0.1:
        raise RuntimeError(
            "Native equilibrium acceleration residual "
            + str(error)
            + " qacc="
            + str(d.qacc)
            + " dz="
            + str(fit.x[0] - r["qpos"][2])
        )
    return (
        q,
        fit.x[1:],
        dict(
            acceleration_residual=error,
            geometry=r,
            max_target_torque_ratio=float(
                np.max(abs(cfg.kp * (fit.x[1:] - q[e.qids])) / e.effort)
            ),
        ),
    )


class BalanceTeacher:
    def __init__(self, e, q, target):
        self.q = q.copy()
        self.target = target.copy()
        self.nv = e.m.nv
        # Ground translation and absolute yaw do not belong to the balancing task.
        self.positions = np.array([i for i in range(e.m.nv) if i not in (0, 1, 5)])
        self.indices = np.r_[self.positions, e.m.nv + np.arange(e.m.nv)]
        size = len(self.indices)
        A = np.zeros((size, size))
        B = np.zeros((size, 12))
        eps = 1e-5

        def result(q0, v, u):
            d = native_step(e, q0, v, u)
            delta = np.zeros(e.m.nv)
            mujoco.mj_differentiatePos(e.m, delta, 1.0, self.q, d.qpos)
            return np.r_[delta, d.qvel][self.indices]

        for j, index in enumerate(self.indices):
            values = []
            for sign in (-1, 1):
                q0 = self.q.copy()
                v = np.zeros(e.m.nv)
                if index < e.m.nv:
                    delta = np.zeros(e.m.nv)
                    delta[index] = sign * eps
                    mujoco.mj_integratePos(e.m, q0, delta, 1.0)
                else:
                    v[index - e.m.nv] = sign * eps
                values.append(result(q0, v, target))
            A[:, j] = (values[1] - values[0]) / (2 * eps)
        for j in range(12):
            delta = np.zeros(12)
            delta[j] = eps
            B[:, j] = (
                result(q, np.zeros(e.m.nv), target + delta)
                - result(q, np.zeros(e.m.nv), target - delta)
            ) / (2 * eps)
        weights = np.ones(2 * e.m.nv)
        weights[2] = 30
        weights[3:6] = [120, 400, 0]
        weights[e.m.nv + 3 : e.m.nv + 6] = [8, 15, 8]
        Q = np.diag(weights[self.indices])
        R = np.eye(12) * 0.5
        P = solve_discrete_are(0.999 * A, 0.999 * B, Q, R)
        self.K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        self.max_eigenvalue = float(max(abs(np.linalg.eigvals(A - B @ self.K))))

    def command(self, e):
        delta = np.zeros(e.m.nv)
        mujoco.mj_differentiatePos(e.m, delta, 1.0, self.q, e.d.qpos)
        error = np.r_[delta, e.d.qvel][self.indices]
        target = np.clip(
            self.target - self.K @ error,
            np.maximum(e.bounds[:, 0], e.nominal - e.cfg.action_scale),
            np.minimum(e.bounds[:, 1], e.nominal + e.cfg.action_scale),
        )
        return np.clip((target - e.nominal) / e.cfg.action_scale, -1, 1)
