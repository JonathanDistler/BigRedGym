"""Trajectory LQR from native dynamics, reduced to encoder/IMU output feedback.

No learned CapturePolicy parameters, COM truth, or accelerometer integration
are used by the inference actor. The successful reference is a warm start.
"""

import time
from pathlib import Path
import mujoco
import numpy as np
import torch
from torch import nn
from gym.envs.go2.handstand_replay import (
    build_model,
    load_trajectory,
    STATE_SPEC,
    CONTROL_DT,
    KP,
    KD,
)
from .model import REFERENCE_SHA, JOINTS
from gym.envs.base.urdf_limits import parse_urdf_limits
from gym.envs.go2.go2_config import Go2Cfg
from gym import GYM_ROOT_DIR

SPEED = np.tile([30.1, 30.1, 15.7], 4)


def features(model, data, qids, vids, nominal):
    rotation = data.xmat[model.body("base").id].reshape(3, 3)
    return np.r_[
        rotation.T @ [0, 0, -1],
        data.qvel[3:6],
        data.qpos[qids] - nominal,
        data.qvel[vids] / SPEED,
    ]


class TrajectoryLQR(nn.Module):
    """43-channel fixed trajectory + model-derived correction in target radians."""

    def __init__(self, actions, centers, gains, limit=0.1):
        super().__init__()
        if not np.isfinite(limit) or not 0 < limit <= 0.2:
            raise ValueError("Invalid feedback target limit")
        for name, value in [
            ("actions", actions),
            ("centers", centers),
            ("gains", gains),
        ]:
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite LQR asset")
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.limit = float(limit)
        if (
            self.actions.shape != (len(actions), 12)
            or self.centers.shape != (len(actions), 30)
            or self.gains.shape != (len(actions), 12, 30)
        ):
            raise ValueError("Invalid trajectory LQR dimensions")

    def forward(self, obs: torch.Tensor):
        if obs.size(-1) != 43:
            raise ValueError("Trajectory tracker requires 43 observations")
        k = (obs[..., 42] * 400).round().long().clamp(0, self.actions.size(0) - 1)
        delta = obs[..., :30] - self.centers[k]
        correction = -torch.einsum("...ij,...j->...i", self.gains[k], delta)
        correction = correction.clamp(-self.limit, self.limit)
        return (self.actions[k] + correction / 2).clamp(-1, 1)


def fit(reference, start=110, spacing=5, reduced=False):
    if start < 0 or spacing < 1:
        raise ValueError("Invalid feedback schedule")
    started = time.monotonic()
    m = build_model()
    d = mujoco.MjData(m)
    ref = load_trajectory(Path(reference), REFERENCE_SHA, m)
    qids = m.jnt_qposadr[[m.joint(n).id for n in JOINTS]]
    vids = m.jnt_dofadr[[m.joint(n).id for n in JOINTS]]
    bounds = m.jnt_range[[m.joint(n).id for n in JOINTS]]
    effort, speed = np.array(
        [
            parse_urdf_limits(Go2Cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR))[n]
            for n in JOINTS
        ]
    ).T
    # Linearize the float64 warm start; evaluation/adapter enforce float32 wire targets.
    target = ref["initial_target"].copy()
    targets = []
    for action in ref["actions"]:
        target += np.clip(
            np.clip(ref["nominal"] + 2 * action, *bounds.T) - target,
            -0.7 * speed * CONTROL_DT,
            0.7 * speed * CONTROL_DT,
        )
        targets.append(target.copy())
    targets = np.asarray(targets)
    n = len(targets)
    feet = [m.body(leg + "_foot").id for leg in ("FL", "FR")]
    geoms = [int(np.flatnonzero(m.geom_bodyid == b)[0]) for b in feet]
    centers = []
    base = []
    for state in ref["reference_states"][:-1]:
        mujoco.mj_setState(m, d, state, STATE_SPEC)
        mujoco.mj_forward(m, d)
        centers.append(features(m, d, qids, vids, ref["nominal"]))
        base.append((d.qpos.copy(), d.qvel.copy()))
    centers = np.asarray(centers)

    def step(k, perturb=None, du=None):
        mujoco.mj_setState(m, d, ref["reference_states"][k], STATE_SPEC)
        if perturb is not None:
            mujoco.mj_integratePos(m, d.qpos, perturb[:18], 1.0)
            d.qvel[:] += perturb[18:]
        command = np.clip(targets[k] + (np.zeros(12) if du is None else du), *bounds.T)
        mujoco.mj_forward(m, d)
        for _ in range(10):
            v = d.qvel[vids].copy()
            tau = np.clip(KP * (command - d.qpos[qids]) - KD * v, -effort, effort)
            tau[(abs(v) >= speed) & (tau * v > 0)] = 0
            d.qfrc_applied[:] = 0
            d.qfrc_applied[vids] = tau
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
        return d.qpos.copy(), d.qvel.copy()

    def difference(q0, v0, q, v):
        tangent = np.empty(18)
        mujoco.mj_differentiatePos(m, tangent, 1, q0, q)
        return np.r_[tangent, v - v0]

    def reconstruction(k):
        q0, v0 = base[k]
        mujoco.mj_setState(m, d, ref["reference_states"][k], STATE_SPEC)
        mujoco.mj_forward(m, d)
        local = []
        anchor = []
        for b, g in zip(feet, geoms):
            point = d.geom_xpos[g].copy()
            point[2] -= m.geom_size[g, 0]
            anchor.append(point)
            local.append(d.xmat[b].reshape(3, 3).T @ (point - d.xpos[b]))

        def reconstruct(obs):
            g0 = centers[k, :3]
            gravity = obs[:3] / np.linalg.norm(obs[:3])
            theta = -np.cross(g0, gravity - g0)
            d.qpos[:] = q0
            mujoco.mj_integratePos(
                m, d.qpos, np.r_[np.zeros(3), theta, np.zeros(12)], 1.0
            )
            d.qpos[qids] = obs[6:18] + ref["nominal"]
            d.qvel[:] = 0
            d.qvel[3:6] = obs[3:6]
            d.qvel[vids] = obs[18:30] * speed
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            points = [
                d.xpos[b] + d.xmat[b].reshape(3, 3) @ p for b, p in zip(feet, local)
            ]
            q = d.qpos.copy()
            q[:3] += np.mean(np.asarray(anchor) - points, axis=0)
            velocities = []
            for b, point in zip(feet, points):
                jac = np.zeros((3, 18))
                mujoco.mj_jac(m, d, jac, None, point, b)
                velocities.append(-jac @ d.qvel)
            return q, np.r_[np.mean(velocities, axis=0), d.qvel[3:]]

        qcenter, vcenter = reconstruct(centers[k])
        cols = []
        for j in range(30):
            hi = centers[k].copy()
            lo = hi.copy()
            hi[j] += 1e-5
            lo[j] -= 1e-5
            qh, vh = reconstruct(hi)
            ql, vl = reconstruct(lo)
            cols.append(
                (
                    difference(qcenter, vcenter, qh, vh)
                    - difference(qcenter, vcenter, ql, vl)
                )
                / 2e-5
            )
        return np.asarray(cols).T

    knots = np.unique(np.r_[np.arange(start, n, spacing), n - 1]).astype(int)
    matrices = {}
    for k in knots:
        qbase, vbase = step(k)
        a = []
        b = []
        for j in range(36):
            dx = np.zeros(36)
            dx[j] = 1e-4
            qp, vp = step(k, dx)
            qm, vm = step(k, -dx)
            a.append(
                (difference(qbase, vbase, qp, vp) - difference(qbase, vbase, qm, vm))
                / 2e-4
            )
        for j in range(12):
            du = np.zeros(12)
            du[j] = 1e-3
            qp, vp = step(k, du=du)
            qm, vm = step(k, du=-du)
            b.append(
                (difference(qbase, vbase, qp, vp) - difference(qbase, vbase, qm, vm))
                / 2e-3
            )
        matrices[k] = (np.array(a).T, np.array(b).T, reconstruction(k))
    # Riccati recursion on the sensor-observable contact manifold avoids
    # designing gains for unobservable global translation/COM drift states.
    qfull = np.diag(
        np.r_[[400] * 3, [400] * 3, [10] * 12, [25] * 3, [25] * 3, [1] * 12]
    )
    rcost = np.eye(12) * 100

    def sensor_projection(k):
        g = centers[min(k, n - 1), :3]
        skew = np.array([[0, -g[2], g[1]], [g[2], 0, -g[0]], [-g[1], g[0], 0]])
        c = np.zeros((30, 36))
        c[:3, 3:6] = skew
        c[3:6, 21:24] = np.eye(3)
        c[6:18, 6:18] = np.eye(12)
        c[18:30, 24:36] = np.diag(1 / speed)
        return c

    models = {}
    for k, (a, b, e) in matrices.items():
        if reduced:
            c = sensor_projection(k + 1)
            models[k] = (c @ a @ e, c @ b, e.T @ qfull @ e + 1e-8 * np.eye(30))
        else:
            models[k] = (a, b, qfull)

    def riccati(a, b, p, q):
        gain = np.linalg.solve(rcost + b.T @ p @ b, b.T @ p @ a)
        value = q + a.T @ p @ a - a.T @ p @ b @ gain
        return (value + value.T) / 2, gain

    a, b, q = models[n - 1]
    p = q.copy()
    for _ in range(200):
        p, _ = riccati(a, b, p, q)
    gains = np.zeros((n, 12, 30))
    spectra = []
    for k in range(n - 1, start - 1, -1):
        j = int(knots[max(0, np.searchsorted(knots, k, side="right") - 1)])
        a, b, q = models[j]
        p, gain = riccati(a, b, p, q)
        gains[k] = gain if reduced else gain @ matrices[j][2]
        afull, bfull, _ = matrices[j]
        spectra.append(
            float(
                max(
                    abs(
                        np.linalg.eigvals(
                            afull - bfull @ gains[k] @ sensor_projection(j)
                        )
                    )
                )
            )
        )
    return TrajectoryLQR(ref["actions"], centers, gains), dict(
        kind="native_trajectory_lqr_v1",
        reduced_observer=reduced,
        max_local_closed_loop_eigenvalue=max(spectra),
        start_step=start,
        model_spacing=spacing,
        model_knots=knots.tolist(),
        fit_seconds=time.monotonic() - started,
        max_gain=float(abs(gains).max()),
        reference_sha256=REFERENCE_SHA,
    )
