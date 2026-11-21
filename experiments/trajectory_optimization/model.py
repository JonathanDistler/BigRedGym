"""Nominal Go2 model. Native contact-solver forces are not used by the NLP."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import mujoco
import numpy as np
from scipy.optimize import least_squares
from gym.envs.go2.handstand_replay import build_model, clearance, STATE_SPEC
from gym.envs.base.urdf_limits import parse_urdf_limits
from gym.envs.go2.go2_config import Go2Cfg
from gym import GYM_ROOT_DIR

JOINTS = [
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
REFERENCE_SHA = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"


@dataclass(frozen=True)
class Settings:
    intervals: int = 12
    duration: float = 2.4
    mesh_times: tuple = ()
    contact_acceleration: bool = True
    release_fraction: float = 0.30
    friction: float = 0.6
    crouch_height: float = 0.27
    foot_radius: float = 0.022
    min_front_rear_distance: float = 0.10
    min_foot_hip_distance: float = 0.08
    body_clearance: float = 0.003
    calf_clearance: float = 0.003
    calf_buffer: float = 0.03
    calf_penalty_weight: float = 0.0
    sweep_substeps: int = 0
    foot_velocity_tolerance: float = 0.002
    max_foot_drift: float = 0.04
    max_force_bodyweights: float = 3.0
    joint_margin: float = 0.001
    # A conservative RESEARCH assumption, not measured motor characteristics.
    torque_curve_calibrated: bool = False
    equality_tolerance: float = 1e-5
    inequality_tolerance: float = 1e-6

    def __post_init__(self):
        values = [
            v for v in self.__dict__.values() if not isinstance(v, (bool, tuple, list))
        ]
        if not np.isfinite(values).all() or self.intervals < 2 or self.duration <= 0:
            raise ValueError("Invalid finite mesh/duration")
        if not (0 < self.release_fraction < 1) or self.friction <= 0:
            raise ValueError("Invalid contact schedule/friction")
        if (
            min(
                self.crouch_height,
                self.foot_radius,
                self.min_front_rear_distance,
                self.min_foot_hip_distance,
                self.body_clearance,
                self.calf_clearance,
                self.calf_buffer,
                self.foot_velocity_tolerance,
                self.max_foot_drift,
                self.max_force_bodyweights,
                self.joint_margin,
                self.equality_tolerance,
                self.inequality_tolerance,
            )
            <= 0
        ):
            raise ValueError("Constraint margins must be positive")

        if (
            self.calf_penalty_weight < 0
            or not isinstance(self.sweep_substeps, int)
            or self.sweep_substeps < 0
            or self.calf_buffer < self.calf_clearance
        ):
            raise ValueError("Invalid calf clearance/penalty/sweep settings")

        if self.mesh_times:
            t = np.asarray(self.mesh_times, dtype=float)
            if (
                t.shape != (self.intervals + 1,)
                or not np.isfinite(t).all()
                or t[0] != 0
                or abs(t[-1] - self.duration) > 1e-12
                or np.any(np.diff(t) <= 0)
            ):
                raise ValueError("Mesh must increase strictly from zero to duration")

    @property
    def times(self):
        return (
            np.asarray(self.mesh_times)
            if self.mesh_times
            else np.linspace(0, self.duration, self.intervals + 1)
        )

    def step(self, k):
        return float(self.times[k + 1] - self.times[k])

    @property
    def dt(self):
        return self.duration / self.intervals

    def contacts(self, k):
        return np.array(
            [
                True,
                True,
                self.times[k] < self.duration * self.release_fraction,
                self.times[k] < self.duration * self.release_fraction,
            ]
        )


class Robot:
    def __init__(self, settings=None):
        self.cfg = settings or Settings()
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        if self.m.nq != 19 or self.m.nv != 18:
            raise ValueError("Expected floating-base Go2")
        names = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        if names != JOINTS:
            raise ValueError("Incorrect joint ordering")
        self.qids = self.m.jnt_qposadr[1:].copy()
        self.vids = self.m.jnt_dofadr[1:].copy()
        self.bounds = self.m.jnt_range[1:].copy()
        limits = parse_urdf_limits(Go2Cfg.asset.file.format(GYM_ROOT_DIR=GYM_ROOT_DIR))
        self.effort, self.speed = np.array([limits[n] for n in JOINTS]).T
        self.base = self.m.body("base").id
        self.feet = [self.m.body(leg + "_foot").id for leg in ("FL", "FR", "RL", "RR")]
        self.hips = [self.m.body(leg + "_hip").id for leg in ("FL", "FR", "RL", "RR")]
        self.foot_geoms = [
            int(np.flatnonzero(self.m.geom_bodyid == b)[0]) for b in self.feet
        ]
        self.robot_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] and g not in self.foot_geoms
        ]
        self.calf_geoms = [
            [
                g
                for g in self.robot_geoms
                if self.m.body(int(self.m.geom_bodyid[g])).name == leg + "_calf"
            ]
            for leg in ("FL", "FR", "RL", "RR")
        ]
        if any(not geoms for geoms in self.calf_geoms):
            raise ValueError("Missing calf collision geometry")
        self.mass = float(self.m.body_subtreemass[self.base])
        if self.cfg.friction > self.m.geom_friction[0, 0]:
            raise ValueError("Requested friction exceeds nominal ground friction")
        self.anchor = np.zeros(19)
        self.anchor[3] = 1
        self.M = np.empty((18, 18))
        self.geometry_pairs = []
        for g in range(1, self.m.ngeom):
            for h in range(g + 1, self.m.ngeom):
                b, c = int(self.m.geom_bodyid[g]), int(self.m.geom_bodyid[h])
                wb, wc = int(self.m.body_weldid[b]), int(self.m.body_weldid[c])
                if (
                    wb == wc
                    or self.m.body_weldid[self.m.body_parentid[wb]] == wc
                    or self.m.body_weldid[self.m.body_parentid[wc]] == wb
                ):
                    continue
                signature = (min(b, c) << 16) + max(b, c)
                if signature in self.m.exclude_signature:
                    continue
                if (self.m.geom_contype[g] & self.m.geom_conaffinity[h]) or (
                    self.m.geom_contype[h] & self.m.geom_conaffinity[g]
                ):
                    self.geometry_pairs.append((g, h))

    def chart(self, q):
        y = np.empty(18)
        mujoco.mj_differentiatePos(self.m, y, 1.0, self.anchor, q)
        return y

    def pose(self, y):
        q = self.anchor.copy()
        mujoco.mj_integratePos(self.m, q, np.asarray(y, dtype=float), 1.0)
        return q

    def set(self, q, v=None):
        self.d.qpos[:] = q
        self.d.qvel[:] = np.zeros(18) if v is None else v
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def foot_kinematics(self):
        position = self.d.geom_xpos[self.foot_geoms].copy()
        position[:, 2] -= self.cfg.foot_radius
        jac = []
        for g in self.foot_geoms:
            j = np.zeros((3, 18))
            mujoco.mj_jacGeom(self.m, self.d, j, None, g)
            jac.append(j)
        return position, np.array(jac)

    def calf_heights(self):
        """Lowest primitive surface, not the joint origin or geometry center."""
        return np.array(
            [min(clearance(self.m, self.d, g) for g in gs) for gs in self.calf_geoms]
        )

    def terms(self, q, v):
        self.set(q, v)
        mujoco.mj_fullM(self.m, self.d, self.M)
        M = self.M.copy()
        h = (self.d.qfrc_bias - self.d.qfrc_passive).copy()
        p, J = self.foot_kinematics()
        hip = self.d.xpos[self.hips].copy()
        clear = np.array([clearance(self.m, self.d, g) for g in self.robot_geoms])
        sep = np.array(
            [
                mujoco.mj_geomDistance(self.m, self.d, g, h, 0.2, None)
                for g, h in self.geometry_pairs
            ]
        )
        return M, h, p, J, hip, clear, sep

    def jdot_v(self, q, v, J):
        if np.linalg.norm(v) == 0:
            return np.zeros((4, 3))
        eps = 1e-6 / max(1.0, np.linalg.norm(v))
        qp = q.copy()
        qm = q.copy()
        mujoco.mj_integratePos(self.m, qp, v, eps)
        mujoco.mj_integratePos(self.m, qm, v, -eps)
        self.set(qp, v)
        _, Jp = self.foot_kinematics()
        self.set(qm, v)
        _, Jm = self.foot_kinematics()
        return np.einsum("fij,j->fi", (Jp - Jm) / (2 * eps), v)

    def torque_envelope(self, dq):
        return self.effort * np.maximum(0.0, 1.0 - abs(np.asarray(dq)) / self.speed)

    def endpoints(self, reference):
        reference = Path(reference)
        if hashlib.sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA:
            raise ValueError("Reference hash mismatch")
        with np.load(reference, allow_pickle=False) as f:
            if str(f["mujoco_version"]) != mujoco.__version__:
                raise ValueError("Pinned reference requires matching MuJoCo")
            mujoco.mj_setState(self.m, self.d, f["initial_state"], STATE_SPEC)
            nominal = f["nominal"].copy()
            mujoco.mj_forward(self.m, self.d)
            anchors, _ = self.foot_kinematics()
            anchors[:, 2] = 0
            crouch = self.d.qpos.copy()
            crouch[2] = self.cfg.crouch_height

            def crouch_error(joints):
                q = crouch.copy()
                q[self.qids] = joints
                self.set(q)
                p, _ = self.foot_kinematics()
                return (p - anchors).ravel()

            fit = least_squares(
                crouch_error,
                crouch[self.qids],
                bounds=self.bounds.T,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=500,
            )
            crouch[self.qids] = fit.x
            if np.max(abs(crouch_error(fit.x))) > 1e-7:
                raise ValueError("Crouch IK failed; no state fabricated")
            mujoco.mj_setState(self.m, self.d, f["reference_states"][-1], STATE_SPEC)
            goal = self.d.qpos.copy()
            self.set(goal)
            p, _ = self.foot_kinematics()
            goal[:2] += anchors[:2, :2].mean(0) - p[:2, :2].mean(0)
            goal[2] = 0.57
            # Six front joint coordinates + body XY: exact support and static COM.
            variables = np.r_[goal[:2], goal[self.qids[:6]]]

            def terminal_error(x):
                q = goal.copy()
                q[:2] = x[:2]
                q[self.qids[:6]] = x[2:]
                self.set(q)
                feet, _ = self.foot_kinematics()
                segment = anchors[1, :2] - anchors[0, :2]
                offset = self.d.subtree_com[self.base, :2] - anchors[0, :2]
                perp = np.cross(np.r_[segment, 0], np.r_[offset, 0])[
                    2
                ] / np.linalg.norm(segment)
                return np.r_[(feet[:2] - anchors[:2]).ravel(), perp]

            fit2 = least_squares(
                terminal_error,
                variables,
                bounds=(
                    np.r_[[-2, -2], self.bounds[:6, 0]],
                    np.r_[[2, 2], self.bounds[:6, 1]],
                ),
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            goal[:2] = fit2.x[:2]
            goal[self.qids[:6]] = fit2.x[2:]
            if np.max(abs(terminal_error(fit2.x))) > 1e-7:
                raise ValueError("Terminal support IK failed")
        return crouch, goal, anchors, nominal
