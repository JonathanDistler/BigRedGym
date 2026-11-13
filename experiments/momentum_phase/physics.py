"""SI articulated momentum and measured ground wrench diagnostics.

The pitch axis is fixed by the initial heading. No body-pitch/pendulum-angle
identification is used. Moving support-point transport is explicitly measured.
"""

from types import SimpleNamespace
import mujoco
import numpy as np
from experiments.com_energy_phase.core import measurements
from experiments.morphology.dynamics_audit import ground_clearance


def heading_axis(rotation):
    forward = rotation[:2, 0]
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("Initial heading must be measured before vertical rotation")
    forward = forward / np.linalg.norm(forward)
    return np.array([-forward[1], forward[0], 0.0])


def sample(env, row, axis=None):
    m, d = env.m, env.d
    view = SimpleNamespace(
        m=m, d=d, base=env.base, feet=env.feet, foot_geoms=env.foot_geoms, row=row
    )
    s = measurements(view)
    axis = env.pitch_axis if axis is None else axis
    com = np.array([s[f"com_{a}"] for a in "xyz"])
    vel = np.array([s[f"com_velocity_{a}"] for a in "xyz"])
    support = np.array([s[f"support_{a}"] for a in "xyz"])
    h = np.array([s[f"momentum_{a}"] for a in "xyz"])
    hs = h + np.cross(com - support, s["mass"] * vel)
    # Full tensor projected onto a fixed task pitch axis, including link offsets.
    inertia = 0.0
    for body in np.flatnonzero((m.body_rootid == env.base) & (m.body_mass > 0)):
        r = d.xipos[body] - com
        rot = d.ximat[body].reshape(3, 3)
        ib = (rot * m.body_inertia[body]) @ rot.T
        inertia += axis @ ib @ axis + m.body_mass[body] * (r @ r - (r @ axis) ** 2)
    r = com - support
    inertia_support = inertia + s["mass"] * (r @ r - (r @ axis) ** 2)
    forces = np.zeros((3, 3))
    moments = np.zeros((3, 3))
    contact_power = np.zeros(3)
    wrench = np.zeros(6)
    jp = np.zeros((3, m.nv))
    for i, contact in enumerate(d.contact):
        b1, b2 = m.geom_bodyid[[contact.geom1, contact.geom2]]
        if bool(b1) == bool(b2):
            continue
        body = int(b2 if b1 == 0 else b1)
        if m.body_rootid[body] != env.base:
            continue
        mujoco.mj_contactForce(m, d, i, wrench)
        frame = contact.frame.reshape(3, 3)
        sign = 1.0 if b1 == 0 else -1.0
        force = sign * (frame.T @ wrench[:3])
        torque = sign * (frame.T @ wrench[3:])
        group = 0 if body in env.feet[:2] else 1 if body in env.feet[2:] else 2
        forces[group] += force
        moments[group] += np.cross(contact.pos - com, force) + torque
        mujoco.mj_jac(m, d, jp, None, contact.pos, body)
        contact_power[group] += float(force @ (jp @ d.qvel))
    external_force = np.zeros(3)
    external_moment = np.zeros(3)
    for b in np.flatnonzero(m.body_rootid == env.base):
        external_force += d.xfrc_applied[b, :3]
        external_moment += d.xfrc_applied[b, 3:] + np.cross(
            d.xipos[b] - com, d.xfrc_applied[b, :3]
        )
    net_force = forces.sum(0) + s["mass"] * m.opt.gravity + external_force
    moment_com = moments.sum(0) + external_moment
    moment_support = moment_com + np.cross(com - support, net_force)
    # A fixed initial support reference is also logged for an unambiguous balance law.
    fixed = getattr(env, "fixed_support", support)
    h_fixed = h + np.cross(com - fixed, s["mass"] * vel)
    moment_fixed = moment_com + np.cross(com - fixed, net_force)
    calf = []
    for g in range(m.ngeom):
        name = m.body(int(m.geom_bodyid[g])).name
        if name.startswith(("FL_calf", "FR_calf")) and (
            m.geom_contype[g] or m.geom_conaffinity[g]
        ):
            calf.append(ground_clearance(m, d, g))
    rear = np.array(
        [i for i, n in enumerate(env.names) if n.startswith(("RL_", "RR_"))]
    )
    power = d.qfrc_applied[env.vids] * d.qvel[env.vids]
    rotation = d.xmat[env.base].reshape(3, 3)
    heading = np.array([axis[1], -axis[0], 0.0])
    forward = rotation[:, 0]
    forward_rate = np.cross(rotation @ d.qvel[3:6], forward)
    parallel = float(forward @ heading)
    parallel_rate = float(forward_rate @ heading)
    pitch = float(np.arctan2(-forward[2], parallel))
    pitch_rate = float(
        (-parallel * forward_rate[2] + forward[2] * parallel_rate)
        / max(parallel**2 + forward[2] ** 2, 1e-9)
    )
    s.update(
        time=float(d.time),
        pitch=pitch,
        pitch_velocity=pitch_rate,
        body_angular_speed=float(np.linalg.norm(d.qvel[3:6])),
        L_com=float(h @ axis),
        L_support=float(hs @ axis),
        L_fixed=float(h_fixed @ axis),
        I_com=float(inertia),
        I_support=float(inertia_support),
        K_system=s["mujoco_kinetic"],
        mechanical_energy=s["potential"] + s["mujoco_kinetic"],
        rotational_proxy=float((h @ axis) ** 2 / (2 * max(inertia, 1e-9))),
        moment_com=float(moment_com @ axis),
        moment_support=float(moment_support @ axis),
        moment_fixed=float(moment_fixed @ axis),
        rear_power=float(power[rear].sum()),
        rear_positive_power=float(np.maximum(power[rear], 0).sum()),
        actuator_power=float(power.sum()),
        calf_clearance=float(min(calf)),
        front_support=bool(row["front_support"]),
        rear_clearance=row["rear_clearance"],
        rear_loaded=bool((row["loads"][2:] > 0.04 * env.weight).any()),
        collision=bool(row["illegal_contacts"] or row["self_contacts"]),
        foot_drift=row["front_drift"],
        com_speed=float(np.linalg.norm(vel)),
    )
    for group, name in enumerate(("front", "rear", "nonfoot")):
        s[f"{name}_moment_com"] = float(moments[group] @ axis)
        s[f"{name}_contact_power"] = float(contact_power[group])
        for j, a in enumerate("xyz"):
            s[f"{name}_force_{a}"] = float(forces[group, j])
    for j, a in enumerate("xyz"):
        s[f"pitch_axis_{a}"] = float(axis[j])
        s[f"linear_momentum_{a}"] = float(s["mass"] * vel[j])
    return s
