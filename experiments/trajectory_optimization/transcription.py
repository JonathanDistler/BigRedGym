"""Direct transcription: minimize zero, subject to manipulator/contact constraints.

Variables at each node: configuration chart18, velocity18, acceleration18,
joint torque12, world point-foot force12. No policy/reward or reference tracking
objective is used. Reference is used only to specify the terminal boundary pose.
"""

import mujoco
import numpy as np
from scipy.sparse import lil_matrix

WIDTH = 78
Y = slice(0, 18)
V = slice(18, 36)
A = slice(36, 54)
TAU = slice(54, 66)
FORCE = slice(66, 78)


class Problem:
    def __init__(self, robot, start, goal, anchors, nominal):
        self.robot = robot
        self.cfg = robot.cfg
        self.start = np.asarray(start).copy()
        self.goal = np.asarray(goal).copy()
        self.anchors = np.asarray(anchors).copy()
        self.nominal = np.asarray(nominal).copy()
        self.nodes = self.cfg.intervals + 1
        self.groups = {}

    def unpack(self, x):
        x = np.asarray(x, dtype=float)
        if x.size != self.nodes * WIDTH or not np.isfinite(x).all():
            raise ValueError("Expected finite trajectory decision vector")
        return x.reshape(self.nodes, WIDTH)

    @staticmethod
    def objective(x):
        return 0.0

    @staticmethod
    def objective_gradient(x):
        return np.zeros_like(x)

    def target(self, row):
        return (
            self.robot.pose(row[Y])[self.robot.qids]
            + (row[TAU] + 3 * row[V][self.robot.vids]) / 60
        )

    def bounds(self):
        r = self.robot
        lo = np.full((self.nodes, WIDTH), -np.inf)
        hi = -lo.copy()
        lo[:, 0:3] = [-2, -2, 0.12]
        hi[:, 0:3] = [2, 2, 1.1]
        lo[:, 3:6] = -np.pi
        hi[:, 3:6] = np.pi
        lo[:, 6:18] = r.bounds[:, 0] + self.cfg.joint_margin
        hi[:, 6:18] = r.bounds[:, 1] - self.cfg.joint_margin
        lo[:, V] = -np.r_[[5] * 3, [10] * 3, r.speed]
        hi[:, V] = -lo[:, V]
        lo[:, A] = -200
        hi[:, A] = 200
        lo[:, TAU] = -r.effort
        hi[:, TAU] = r.effort
        lo[:, FORCE] = -self.cfg.max_force_bodyweights * r.mass * 9.81
        hi[:, FORCE] = -lo[:, FORCE]
        return lo.ravel(), hi.ravel()

    def seed(self, static=False):
        r = self.robot
        times = self.cfg.times
        rows = np.zeros((self.nodes, WIDTH))
        start = r.chart(self.start)
        goal = start if static else r.chart(self.goal)
        t = times / self.cfg.duration
        blend = 3 * t * t - 2 * t * t * t
        blend[1] = 0
        blend[-2] = 1
        rows[:, Y] = start + (goal - start) * blend[:, None]
        for k in range(self.nodes - 1):
            mujoco.mj_differentiatePos(
                r.m,
                rows[k, V],
                self.cfg.step(k),
                r.pose(rows[k, Y]),
                r.pose(rows[k + 1, Y]),
            )
        rows[:-1, A] = np.diff(rows[:, V], axis=0) / np.diff(times)[:, None]
        for k, row in enumerate(rows):
            q = r.pose(row[Y])
            M, h, p, J, *_ = r.terms(q, row[V])
            contacts = self.cfg.contacts(k)
            needed = M @ row[A] + h
            columns = np.concatenate([J[i].T for i in range(4) if contacts[i]], axis=1)
            active = np.linalg.lstsq(columns[:6], needed[:6], rcond=None)[0].reshape(
                -1, 3
            )
            f = np.zeros((4, 3))
            f[contacts] = active
            f[:, 2] = np.maximum(0, f[:, 2])
            row[FORCE] = f.ravel()
            row[TAU] = np.clip(
                (needed - np.einsum("fij,fi->j", J, f))[r.vids],
                -0.95 * r.effort,
                0.95 * r.effort,
            )
        return rows.ravel()

    def kinematic_seed(self):
        """IK/tangent initialization only; does not certify feasibility."""
        from scipy.optimize import least_squares

        r = self.robot
        times = self.cfg.times
        rows = self.seed().reshape(self.nodes, WIDTH)
        r.set(self.goal)
        goal_feet, _ = r.foot_kinematics()
        last_joints = self.start[r.qids].copy()
        release_time = self.cfg.duration * self.cfg.release_fraction
        for k, row in enumerate(rows):
            if k in (0, 1):
                row[Y] = r.chart(self.start)
            if k == self.nodes - 1:
                row[Y] = r.chart(self.goal)
            q = r.pose(row[Y])
            desired = self.anchors.copy()
            if times[k] >= release_time:
                phase = (times[k] - release_time) / (self.cfg.duration - release_time)
                phase = phase * phase * (3 - 2 * phase)
                desired[2:] = (1 - phase) * self.anchors[2:] + phase * goal_feet[2:]
                desired[2:, 2] += 0.08 * np.sin(np.pi * phase)

            def error(joints):
                trial = q.copy()
                trial[r.qids] = joints
                r.set(trial)
                feet, _ = r.foot_kinematics()
                return (feet - desired).ravel()

            fit = least_squares(
                error,
                last_joints,
                bounds=(
                    r.bounds[:, 0] + self.cfg.joint_margin,
                    r.bounds[:, 1] - self.cfg.joint_margin,
                ),
                max_nfev=250,
            )
            q[r.qids] = fit.x
            last_joints = fit.x.copy()
            row[Y] = r.chart(q)
        rows[0, Y] = r.chart(self.start)
        rows[1, Y] = rows[0, Y]
        rows[-1, Y] = r.chart(self.goal)
        rows[:, V] = 0
        rows[:, A] = 0
        for k in range(self.nodes - 1):
            q = r.pose(rows[k, Y])
            qn = r.pose(rows[k + 1, Y])
            mujoco.mj_differentiatePos(r.m, rows[k, V], self.cfg.step(k), q, qn)
        rows[0, V] = 0
        for k, row in enumerate(rows):
            q = r.pose(row[Y])
            r.set(q, row[V])
            _, J = r.foot_kinematics()
            active = J[self.cfg.contacts(k)].reshape(-1, 18)
            row[V][6:] += np.linalg.lstsq(active[:, 6:], -active @ row[V], rcond=None)[
                0
            ]
        rows[:-1, A] = np.diff(rows[:, V], axis=0) / np.diff(times)[:, None]
        for k, row in enumerate(rows):
            q = r.pose(row[Y])
            M, h, p, J, *_ = r.terms(q, row[V])
            mask = self.cfg.contacts(k)
            active = J[mask].reshape(-1, 18)
            jd = r.jdot_v(q, row[V], J)[mask].ravel()
            row[A][6:] += np.linalg.lstsq(
                active[:, 6:], -active @ row[A] - jd, rcond=None
            )[0]
            if k == self.nodes - 1:
                row[A] = 0
            need = M @ row[A] + h
            matrix = active.T
            f = np.zeros((4, 3))
            f[mask] = np.linalg.lstsq(matrix[:6], need[:6], rcond=None)[0].reshape(
                -1, 3
            )
            f[:, 2] = np.maximum(0, f[:, 2])
            row[FORCE] = f.ravel()
            row[TAU] = np.clip(
                (need - np.einsum("fij,fi->j", J, f))[6:],
                -0.95 * r.effort,
                0.95 * r.effort,
            )
        return rows.ravel()

    def blocks(self, x, solver=False):
        rows = self.unpack(x)
        r = self.robot
        c = self.cfg
        bw = r.mass * 9.81
        blocks = []

        def add(kind, name, value, scale, depends):
            blocks.append(
                (
                    kind,
                    name,
                    np.asarray(value, dtype=float).ravel(),
                    float(scale),
                    tuple(depends),
                )
            )

        if not solver:
            add(
                "eq",
                "initial_configuration",
                rows[0, Y] - r.chart(self.start),
                0.1,
                [0],
            )
            add("eq", "initial_velocity", rows[0, V], 1, [0])
            add(
                "eq",
                "terminal_configuration",
                rows[-1, Y] - r.chart(self.goal),
                0.1,
                [self.nodes - 1],
            )
            add("eq", "terminal_velocity", rows[-1, V], 1, [self.nodes - 1])
            add("eq", "terminal_acceleration", rows[-1, A], 9.81, [self.nodes - 1])
        for k, row in enumerate(rows):
            q = r.pose(row[Y])
            v = row[V]
            a = row[A]
            f = row[FORCE].reshape(4, 3)
            M, h, p, J, hip, clear, sep = r.terms(q, v)
            contact = c.contacts(k)
            generalized = np.einsum("fij,fi->j", J, f)
            generalized[r.vids] += row[TAU]
            add("eq", "manipulator_dynamics", M @ a + h - generalized, bw, [k])
            jdv = r.jdot_v(q, v, J)
            fv = np.einsum("fij,j->fi", J, v)
            fa = np.einsum("fij,j->fi", J, a) + jdv
            for i in range(4):
                if contact[i]:
                    if not solver or k not in (0, 1, self.nodes - 1):
                        add("eq", "stance_ground_gap", p[i, 2], 0.1, [k])
                    if c.contact_acceleration and (not solver or k != self.nodes - 1):
                        add("eq", "stance_zero_acceleration", fa[i], 9.81, [k])
                    # Numerical drift guard in addition to zero acceleration.
                    add(
                        "ineq",
                        "stance_velocity",
                        np.r_[
                            c.foot_velocity_tolerance - fv[i],
                            c.foot_velocity_tolerance + fv[i],
                        ],
                        c.foot_velocity_tolerance,
                        [k],
                    )
                    add(
                        "ineq",
                        "stance_drift",
                        c.max_foot_drift
                        - np.linalg.norm(p[i, :2] - self.anchors[i, :2]),
                        c.max_foot_drift,
                        [k],
                    )
                elif not solver:
                    add("eq", "swing_zero_force", f[i], bw, [k])
            add("ineq", "foot_ground_nonpenetration", p[:, 2], 0.1, [k])
            add("ineq", "body_calf_head_clearance", clear - c.body_clearance, 0.1, [k])
            add("ineq", "native_self_clearance", sep - c.body_clearance, 0.1, [k])
            distances = [
                np.linalg.norm(p[i] - p[j]) for i in range(2) for j in range(2, 4)
            ]
            add(
                "ineq",
                "front_rear_foot_separation",
                np.array(distances) - c.min_front_rear_distance,
                0.1,
                [k],
            )
            foot_centers = p + [0, 0, c.foot_radius]
            add(
                "ineq",
                "foot_hip_separation",
                np.linalg.norm(foot_centers - hip, axis=1) - c.min_foot_hip_distance,
                0.1,
                [k],
            )
            add("ineq", "unilateral_force", f[:, 2], bw, [k])
            add(
                "ineq",
                "friction_cone",
                c.friction * f[:, 2] - np.linalg.norm(f[:, :2], axis=1),
                bw,
                [k],
            )
            add(
                "ineq",
                "normal_load_cap",
                c.max_force_bodyweights * bw - f[:, 2],
                bw,
                [k],
            )
            limit = r.torque_envelope(v[r.vids])
            add(
                "ineq",
                "velocity_dependent_torque",
                np.r_[limit - row[TAU], limit + row[TAU]],
                r.effort.max(),
                [k],
            )
            target = self.target(row)
            add(
                "ineq",
                "command_joint_limits",
                np.r_[target - r.bounds[:, 0], r.bounds[:, 1] - target],
                1,
                [k],
            )
            add(
                "ineq",
                "normalized_action_bounds",
                np.r_[2 - (target - self.nominal), 2 + (target - self.nominal)],
                1,
                [k],
            )
            if k < self.nodes - 1:
                nxt = rows[k + 1]
                predicted = q.copy()
                mujoco.mj_integratePos(r.m, predicted, v, c.step(k))
                position_error = np.empty(18)
                mujoco.mj_differentiatePos(
                    r.m, position_error, 1, predicted, r.pose(nxt[Y])
                )
                if not solver or k != 0:
                    add(
                        "eq", "explicit_euler_position", position_error, 0.1, [k, k + 1]
                    )
                add(
                    "eq",
                    "explicit_euler_velocity",
                    nxt[V] - v - c.step(k) * a,
                    1,
                    [k, k + 1],
                )
                slew = 0.7 * r.speed * c.step(k)
                delta = self.target(nxt) - target
                add(
                    "ineq",
                    "target_slew",
                    np.r_[slew - delta, slew + delta],
                    1,
                    [k, k + 1],
                )
        return blocks

    def values(self, x, kind):
        return np.concatenate(
            [v / s for k, n, v, s, d in self.blocks(x, solver=True) if k == kind]
        )

    def sparsity(self, x, kind):
        blocks = [b for b in self.blocks(x, solver=True) if b[0] == kind]
        mat = lil_matrix((sum(len(b[2]) for b in blocks), len(x)), dtype=int)
        i = 0
        for _, _, v, _, depends in blocks:
            for node in depends:
                mat[i : i + len(v), node * WIDTH : (node + 1) * WIDTH] = 1
            i += len(v)
        return mat.tocsr()

    def audit(self, x):
        groups = {}
        for kind, name, v, scale, _ in self.blocks(x):
            error = abs(v) if kind == "eq" else np.maximum(0, -v)
            g = groups.setdefault(
                name,
                {
                    "kind": kind,
                    "maximum_raw_violation": 0.0,
                    "maximum_scaled_violation": 0.0,
                },
            )
            g["maximum_raw_violation"] = max(
                g["maximum_raw_violation"], float(error.max(initial=0))
            )
            g["maximum_scaled_violation"] = max(
                g["maximum_scaled_violation"], float((error / scale).max(initial=0))
            )
        lo, hi = self.bounds()
        bound_error = float(np.maximum(np.maximum(lo - x, x - hi), 0).max(initial=0))
        eq = max(
            (
                g["maximum_scaled_violation"]
                for g in groups.values()
                if g["kind"] == "eq"
            ),
            default=0,
        )
        iq = max(
            (
                g["maximum_scaled_violation"]
                for g in groups.values()
                if g["kind"] == "ineq"
            ),
            default=0,
        )
        return {
            "objective": 0.0,
            "maximum_scaled_equality_violation": eq,
            "maximum_scaled_inequality_violation": iq,
            "maximum_bound_violation": bound_error,
            "groups": groups,
            "mesh_feasible": bool(
                eq <= self.cfg.equality_tolerance
                and iq <= self.cfg.inequality_tolerance
                and bound_error <= 1e-9
            ),
            "hardware_ready": False,
        }
