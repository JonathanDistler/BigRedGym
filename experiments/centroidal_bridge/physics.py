"""Independent world-frame articulated centroidal audit; no dynamics surrogate."""

import mujoco
import numpy as np


def audit(e):
    m, d = e.m, e.d
    ids = np.flatnonzero((m.body_rootid == e.base) & (m.body_mass > 0))
    assert e.base != 0 and m.body_parentid[e.base] == 0
    mujoco.mj_subtreeVel(m, d)
    mass = m.body_mass[ids].sum()
    com = np.average(d.xipos[ids], axis=0, weights=m.body_mass[ids])
    ag = np.zeros((3, m.nv))
    inertia = np.zeros((3, 3))
    links = {}
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    velocity = np.zeros(3)
    for b in ids:
        mujoco.mj_jacBodyCom(m, d, jp, jr, int(b))
        r = d.xipos[b] - com
        rot = d.ximat[b].reshape(3, 3)
        ib = (rot * m.body_inertia[b]) @ rot.T
        cross = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        matrix = ib @ jr + m.body_mass[b] * cross @ jp
        ag += matrix
        links[m.body(int(b)).name] = (matrix @ d.qvel).tolist()
        velocity += m.body_mass[b] * (jp @ d.qvel) / mass
        inertia += ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    contacts = []
    moment = np.zeros(3)
    force_sum = np.zeros(3)
    for i, c in enumerate(d.contact):
        bodies = m.geom_bodyid[[c.geom1, c.geom2]]
        robot = [int(b) in ids for b in bodies]
        if robot[0] == robot[1]:
            continue
        b = int(bodies[1] if robot[1] else bodies[0])
        wrench = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, wrench)
        sign = 1 if robot[1] else -1
        f, tau = (sign * c.frame.reshape(3, 3).T @ wrench.reshape(2, 3).T).T
        torque = np.cross(c.pos - com, f) + tau
        mujoco.mj_jac(m, d, jp, jr, c.pos, b)
        contacts.append(
            dict(
                body=m.body(b).name,
                position=c.pos.tolist(),
                velocity=(jp @ d.qvel).tolist(),
                force=f.tolist(),
                torque=tau.tolist(),
                moment=torque.tolist(),
            )
        )
        moment += torque
        force_sum += f
    h = ag @ d.qvel
    out = dict(
        time=float(d.time),
        com=com.tolist(),
        velocity=velocity.tolist(),
        H=h.tolist(),
        inertia=inertia.tolist(),
        links=links,
        contacts=contacts,
        moment=moment.tolist(),
        force=force_sum.tolist(),
        AG=ag.tolist(),
        subtree_error=float(np.linalg.norm(h - d.subtree_angmom[e.base])),
        com_error=float(np.linalg.norm(com - d.subtree_com[e.base])),
        velocity_error=float(np.linalg.norm(velocity - d.subtree_linvel[e.base])),
    )
    return out
