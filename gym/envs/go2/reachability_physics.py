"""Local, conditional torque-work bounds and unilateral-contact screening.

This is a forward-dynamics curriculum, never reverse integration of friction.
"""

import mujoco
import numpy as np


def work_speed_bound(inertia, torque, travel, speed_limit):
    """From rest: 1/2 I_eff omega² <= sum(tau_max * available travel).

    I_eff is the contact-compatible kinetic metric at the sampled pose. Frozen
    geometry and no gravity/initial energy make this a sampling envelope, not
    a global upper bound on the coupled robot's speed.
    """
    if inertia <= 0 or np.any(np.asarray(travel) < 0):
        raise ValueError("positive inertia and nonnegative travel required")
    return min(float(speed_limit), float(np.sqrt(2 * np.dot(torque, travel) / inertia)))


class ContactSampler:
    """CPU MuJoCo scratch simulation; never mutates the live environment."""

    def __init__(self, env):
        backend = env._backend
        if not hasattr(backend, "_datas"):
            raise ValueError("go2reachability currently requires MuJoCo CPU")
        self.env = env
        self.model = backend._mjm
        self.data = mujoco.MjData(self.model)
        self.order = backend._canonical_to_native_dof_np
        self.q = 7 + self.order
        self.v = 6 + self.order
        self.rear = np.arange(6, 12)
        self.feet = backend._canonical_to_native_body_np[env.feet_indices.cpu().numpy()]
        self.limits = env.dof_pos_limits.cpu().numpy()
        self.torque = env.torque_limits.cpu().numpy()
        self.speed = env.dof_vel_limits.cpu().numpy()
        self.kp = env.p_gains.cpu().numpy().reshape(-1, 12)[0]
        self.kd = env.d_gains.cpu().numpy().reshape(-1, 12)[0]

    def load(self, root, joints):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = root[:3]
        d.qpos[3:7] = root[[6, 3, 4, 5]]
        d.qpos[self.q] = joints
        mujoco.mj_forward(self.model, d)

    def direction(self, rear_direction):
        """Keep both planted foot centers stationary, including tangential motion."""
        m, d = self.model, self.data
        jac = []
        for body in self.feet[:2]:
            jp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, None, int(body))
            jac.append(jp)
        jac = np.vstack(jac)
        velocity = np.zeros(m.nv)
        velocity[self.v[self.rear]] = rear_direction
        free = np.r_[np.arange(6), self.v[:6]]
        velocity[free] = np.linalg.lstsq(jac[:, free], -jac @ velocity, rcond=None)[0]
        if np.linalg.norm(jac @ velocity) > 1e-7:
            raise ValueError("contact velocity projection failed")
        mass = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, mass)
        return velocity, float(velocity @ mass @ velocity)

    def bound(self, joints, direction, inertia):
        # Total rear-motor work across remaining travel in the chosen direction.
        travel = np.where(
            direction >= 0,
            self.limits[6:, 1] - joints[6:],
            joints[6:] - self.limits[6:, 0],
        )
        nonzero = np.abs(direction) > 1e-10
        path_length = np.min(travel[nonzero] / np.abs(direction[nonzero]))
        travel = np.abs(direction) * path_length
        motor_limit = np.min(self.speed[6:][nonzero] / np.abs(direction[nonzero]))
        return work_speed_bound(inertia, self.torque[6:], travel, motor_limit)

    def contact_ok(self):
        m, d = self.model, self.data
        loads = np.zeros(2)
        for i in range(d.ncon):
            c = d.contact[i]
            bodies = m.geom_bodyid[[c.geom1, c.geom2]]
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force)
            for j, body in enumerate(self.feet[:2]):
                if body in bodies and 0 in bodies:
                    loads[j] += force[0]
        speeds = []
        for body in self.feet[:2]:
            jp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, None, int(body))
            speeds.append(np.linalg.norm(jp @ d.qvel))
        c = self.env.cfg.reachability
        return bool(
            np.all(loads >= c.min_normal_force) and max(speeds) <= c.max_slip_speed
        )

    def probe(self, velocity, joints):
        m, d = self.model, self.data
        d.qvel[:] = velocity
        mujoco.mj_forward(m, d)
        for _ in range(
            max(
                1,
                int(np.ceil(self.env.cfg.reachability.probe_seconds / m.opt.timestep)),
            )
        ):
            tau = np.clip(
                self.kp * (joints - d.qpos[self.q]) - self.kd * d.qvel[self.v],
                -self.torque,
                self.torque,
            )
            d.qfrc_applied[self.v] = tau
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            if not self.contact_ok():
                return False
        return True
