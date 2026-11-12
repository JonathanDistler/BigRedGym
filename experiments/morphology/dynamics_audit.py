"""IK, collision clearance, support wrench and instrumented morphology rollouts.

No learned policy or model edits. Forces and events are sampled at physics rate.
"""

import mujoco
import numpy as np

from experiments.morphology.task import Task


def ground_clearance(model, data, geom):
    """Exact primitive/compiled-mesh lower support point against the z=0 plane."""
    kind = model.geom_type[geom]
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    z = data.geom_xpos[geom, 2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return z - np.abs(rotation[2]) @ size
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (
            z
            - abs(rotation[2, 2]) * size[1]
            - size[0] * np.linalg.norm(rotation[2, :2])
        )
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return z + np.min(model.mesh_vert[start : start + count] @ rotation[2])
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.inf
    raise ValueError(f"unsupported collision geometry {kind}")


class Audit(Task):
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.collision_geoms = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] and self.m.geom_contype[g]
        ]
        self.rear_geoms = [
            g
            for g in self.collision_geoms
            if self.m.body(self.m.geom_bodyid[g]).name.startswith(
                ("RL_calf", "RR_calf")
            )
        ]
        self.nonfoot_geoms = [
            g for g in self.collision_geoms if g not in self.foot_geoms
        ]
        self.rear_geoms += [
            g
            for g in self.collision_geoms
            if self.m.body(self.m.geom_bodyid[g]).name.startswith(
                ("RL_calflower", "RR_calflower")
            )
        ]
        self.rear_geoms = sorted(set(self.rear_geoms))
        self.pivot = self.d.geom_xpos[self.foot_geoms[:2]].mean(0).copy()
        self.pivot[2] = 0
        self.external_angular_impulse = self.rear_angular_impulse = 0.0
        self.positive_work = self.net_work = self.push_impulse = 0.0
        self.first_event = None
        return obs

    def set_pose(self, qpos, omega=0.0):
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = qpos
        self.d.qvel[4] = omega
        mujoco.mj_forward(self.m, self.d)
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()
        self.pivot = self.d.geom_xpos[self.foot_geoms[:2]].mean(0).copy()
        self.pivot[2] = 0
        self.t = self.hold = self.best_hold = 0.0
        self.done = False
        self.external_angular_impulse = self.rear_angular_impulse = 0.0
        self.positive_work = self.net_work = self.push_impulse = 0.0
        self.first_event = None

    def solve_ik(self, feet, qpos, active=(0, 1, 2, 3), iterations=50):
        """Damped joint-space IK with native MuJoCo translational Jacobians."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = qpos
        error = np.inf
        for _ in range(iterations):
            mujoco.mj_forward(self.m, d)
            error = max(
                np.linalg.norm(feet[i] - d.geom_xpos[self.foot_geoms[i]])
                for i in active
            )
            if error < 1e-7:
                break
            for i in active:
                leg = ("FL", "FR", "RL", "RR")[i]
                indices = np.array(
                    [j for j, n in enumerate(self.names) if n.startswith(leg)]
                )
                jac = np.zeros((3, self.m.nv))
                mujoco.mj_jacGeom(self.m, d, jac, None, self.foot_geoms[i])
                j = jac[:, self.vids[indices]]
                e = feet[i] - d.geom_xpos[self.foot_geoms[i]]
                change = j.T @ np.linalg.solve(j @ j.T + 1e-6 * np.eye(3), e)
                d.qpos[self.qids[indices]] += np.clip(change, -0.15, 0.15)
                bounds = self.m.jnt_range[self.jids[indices]]
                d.qpos[self.qids[indices]] = np.clip(
                    d.qpos[self.qids[indices]], *bounds.T
                )
        mujoco.mj_forward(self.m, d)
        error = max(
            np.linalg.norm(feet[i] - d.geom_xpos[self.foot_geoms[i]]) for i in active
        )
        return d.qpos.copy(), float(error)

    def pose(
        self, height, pitch=0.0, x=0.0, feet=None, active=(0, 1, 2, 3), guess=None
    ):
        q = self.d.qpos.copy() if guess is None else guess.copy()
        q[:3] = [x, 0, height]
        q[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
        return self.solve_ik(feet, q, active)

    def clearances(self):
        return dict(
            rear_calf_clearance=min(
                ground_clearance(self.m, self.d, g) for g in self.rear_geoms
            ),
            nonfoot_clearance=min(
                ground_clearance(self.m, self.d, g) for g in self.nonfoot_geoms
            ),
        )

    def static(self, q, active=(0, 1, 2, 3)):
        self.d.qpos[:] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        jacobians = []
        for i in active:
            jac = np.zeros((3, self.m.nv))
            point = self.d.geom_xpos[self.foot_geoms[i]].copy()
            point[2] -= 0.022
            mujoco.mj_jac(self.m, self.d, jac, None, point, self.feet[i])
            jacobians.append(jac)
        jac = np.concatenate(jacobians, axis=0)
        force = np.linalg.lstsq(jac[:, :6].T, self.d.qfrc_bias[:6], rcond=None)[0]
        torque = self.d.qfrc_bias - jac.T @ force
        self.static_torque = torque[self.vids].copy()
        forces = force.reshape(-1, 3)
        return dict(
            static_torque_ratio=float(np.max(abs(torque[self.vids]) / self.effort)),
            static_base_residual=float(np.linalg.norm(torque[:6])),
            min_normal_force=float(forces[:, 2].min()),
            static_friction_ratio=float(
                np.max(
                    np.linalg.norm(forces[:, :2], axis=1)
                    / np.maximum(forces[:, 2], 1e-9)
                )
            ),
            **self.measure(),
            **self.clearances(),
        )

    def sample(self, target, phase, torque_ff=None):
        """One 2ms physics step. Never changes or disables collision settings."""
        dt = self.m.opt.timestep
        raw = (
            self.cfg.kp * (target - self.d.qpos[self.qids])
            - self.cfg.kd * self.d.qvel[self.vids]
        )
        if torque_ff is not None:
            raw += torque_ff
        torque = np.clip(raw, -self.effort, self.effort)
        overspeed = (abs(self.d.qvel[self.vids]) >= self.speed) & (
            torque * self.d.qvel[self.vids] > 0
        )
        torque[overspeed] = 0
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_applied[self.vids] = torque
        work = torque * self.d.qvel[self.vids] * dt
        self.positive_work += np.maximum(work, 0).sum()
        self.net_work += work.sum()
        mujoco.mj_step(self.m, self.d)
        # mj_step leaves some derived quantities at the pre-integration state.
        mujoco.mj_forward(self.m, self.d)
        self.t = self.d.time
        row = self.measure()
        gravity_moment = np.cross(
            np.array([row[f"com_{a}"] for a in "xyz"]) - self.pivot,
            [0, 0, -row["mass"] * 9.81],
        )[1]
        moment, rear_moment = gravity_moment, 0.0
        events = []
        rear_impulse_step = 0.0
        for j in range(self.d.ncon):
            contact = self.d.contact[j]
            b1, b2 = self.m.geom_bodyid[[contact.geom1, contact.geom2]]
            if b1 and b2:
                continue
            body = b2 if b1 == 0 else b1
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, j, force)
            world = contact.frame.reshape(3, 3).T @ force[:3]
            if b2 == 0:
                world = -world
            moment += np.cross(contact.pos - self.pivot, world)[1]
            if self.m.body(body).name.startswith(("RL_foot", "RR_foot")):
                rear_moment += np.cross(contact.pos - self.pivot, world)[1]
                if phase == "push":
                    rear_impulse_step += max(world[2], 0) * dt
            if body not in self.feet and contact.dist <= 0:
                events.append(
                    dict(
                        time=self.t,
                        phase=phase,
                        body=self.m.body(body).name,
                        geom=int(contact.geom2 if b1 == 0 else contact.geom1),
                        distance=float(contact.dist),
                        force=float(np.linalg.norm(world)),
                        pitch=row["pitch"],
                        joint_positions=self.d.qpos[self.qids].tolist(),
                        root_position=self.d.qpos[:3].tolist(),
                        contact_position=contact.pos.tolist(),
                    )
                )
        # Exclude the first illegal-contact sample from useful impulse metrics.
        if not events:
            self.external_angular_impulse += moment * dt
            self.rear_angular_impulse += rear_moment * dt
            self.push_impulse += rear_impulse_step
        if events and self.first_event is None:
            self.first_event = events[0]
        valid = (
            abs(row["pitch"] - np.pi / 2) < self.cfg.pitch_tolerance
            and abs(row["pitch_velocity"]) < self.cfg.max_omega
            and row["com_speed"] < self.cfg.max_com_speed
            and row["front_support"]
            and row["rear_clearance"] > 0.10
            and row["front_drift"] < 0.10
            and not events
        )
        self.hold = self.hold + dt if valid else 0.0
        self.best_hold = max(self.best_hold, self.hold)
        row.update(self.clearances())
        row.update(
            phase=phase,
            external_angular_impulse=self.external_angular_impulse,
            rear_angular_impulse=self.rear_angular_impulse,
            push_impulse=self.push_impulse,
            positive_work=self.positive_work,
            net_work=self.net_work,
            saturation=float(np.mean(abs(raw) >= self.effort)),
            speed_limit_fraction=float(np.mean(overspeed)),
            measured_speed_ratio=float(
                np.max(abs(self.d.qvel[self.vids]) / self.speed)
            ),
            torque_ratio=float(np.max(abs(torque) / self.effort)),
            joint_limit_violation=float(
                max(
                    0.0,
                    np.max(self.m.jnt_range[self.jids, 0] - self.d.qpos[self.qids]),
                    np.max(self.d.qpos[self.qids] - self.m.jnt_range[self.jids, 1]),
                )
            ),
            gravitational_potential_energy=row["mass"] * 9.81 * row["com_z"],
            target_clipped=bool(
                np.any(
                    (target < self.m.jnt_range[self.jids, 0])
                    | (target > self.m.jnt_range[self.jids, 1])
                )
            ),
            best_hold=self.best_hold,
            success=self.best_hold >= 5.0,
            angular_momentum_support=float(
                self.d.subtree_angmom[self.base, 1]
                + np.cross(
                    np.array([row[f"com_{a}"] for a in "xyz"]) - self.pivot,
                    self.m.body_mass.sum() * self.d.subtree_linvel[self.base],
                )[1]
            ),
        )
        return row, events


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def pendulum_acceleration(theta, omega, length, length_rate, gravity=9.81):
    """Point-mass model, theta from downward vertical; no support torque."""
    if np.any(np.asarray(length) <= 0):
        raise ValueError("pendulum length must be positive")
    return -gravity / length * np.sin(theta) - 2 * length_rate / length * omega


def frozen_energy_barrier(mass, horizontal_offset, height, gravity=9.81):
    """Peak gravitational rise to COM-over-pivot at fixed sagittal radius."""
    return mass * gravity * (np.hypot(horizontal_offset, height) - height)
