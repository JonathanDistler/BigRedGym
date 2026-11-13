"""Fast native MuJoCo quantities, cross-checked against independent link Jacobians.

Native subtree momentum/COM and full-system energy avoid repeating 43 body
Jacobians at every physics sample. No approximate dynamics replace MuJoCo.
"""

import mujoco
import numpy as np
from experiments.morphology.dynamics_audit import ground_clearance


def sample_fast(e, row):
    m, d = e.m, e.d
    ids = e.robot_ids
    mass = e.mass
    axis = e.pitch_axis
    com = d.subtree_com[e.base].copy()
    velocity = d.subtree_linvel[e.base].copy()
    h = d.subtree_angmom[e.base].copy()
    rotations = d.ximat[ids].reshape(-1, 3, 3)
    body_axes = np.einsum(
        "bij,bj->bi", rotations.transpose(0, 2, 1), np.broadcast_to(axis, (len(ids), 3))
    )
    r = d.xipos[ids] - com
    inertia = float(
        np.sum(m.body_inertia[ids] * body_axes**2)
        + np.sum(m.body_mass[ids] * (np.sum(r * r, axis=1) - (r @ axis) ** 2))
    )
    forces = np.zeros((3, 3))
    moments = np.zeros((3, 3))
    positions = []
    weights = []
    wrench = np.empty(6)
    for i, c in enumerate(d.contact):
        b1, b2 = m.geom_bodyid[[c.geom1, c.geom2]]
        if bool(b1) == bool(b2):
            continue
        body = int(b2 if b1 == 0 else b1)
        if m.body_rootid[body] != e.base:
            continue
        mujoco.mj_contactForce(m, d, i, wrench)
        frame = c.frame.reshape(3, 3)
        sign = 1.0 if b1 == 0 else -1.0
        force = sign * frame.T @ wrench[:3]
        torque = sign * frame.T @ wrench[3:]
        group = 0 if body in e.feet[:2] else 1 if body in e.feet[2:] else 2
        forces[group] += force
        moments[group] += np.cross(c.pos - com, force) + torque
        if group == 0 and wrench[0] > 0:
            positions.append(c.pos.copy())
            weights.append(wrench[0])
    support = (
        np.average(positions, axis=0, weights=weights) if positions else np.zeros(3)
    )
    relative = com - support
    hs = h + np.cross(relative, mass * velocity)
    fixed = e.fixed_support
    hf = h + np.cross(com - fixed, mass * velocity)
    external_force = d.xfrc_applied[ids, :3].sum(0)
    external_moment = d.xfrc_applied[ids, 3:].sum(0) + np.cross(
        d.xipos[ids] - com, d.xfrc_applied[ids, :3]
    ).sum(0)
    force = forces.sum(0) + mass * m.opt.gravity + external_force
    moment = moments.sum(0) + external_moment
    support_moment = moment + np.cross(relative, force)
    fixed_moment = moment + np.cross(com - fixed, force)
    mujoco.mj_energyPos(m, d)
    mujoco.mj_energyVel(m, d)
    potential = float(d.energy[0])
    kinetic = float(d.energy[1])
    rotor = float(0.5 * np.dot(m.dof_armature, d.qvel**2))
    rotation = d.xmat[e.base].reshape(3, 3)
    heading = np.array([axis[1], -axis[0], 0.0])
    forward = rotation[:, 0]
    rate = np.cross(rotation @ d.qvel[3:6], forward)
    parallel = forward @ heading
    omega = float(
        (-parallel * rate[2] + forward[2] * (rate @ heading))
        / max(parallel**2 + forward[2] ** 2, 1e-9)
    )
    speeds = []
    jp = np.zeros((3, m.nv))
    for g in e.foot_geoms[:2]:
        mujoco.mj_jacGeom(m, d, jp, None, int(g))
        speeds.append(np.linalg.norm((jp @ d.qvel)[:2]))
    calf = min(ground_clearance(m, d, g) for g in e.calf_geoms)
    power = d.qfrc_applied[e.vids] * d.qvel[e.vids]
    support_valid = bool(positions and row["front_support"])
    out = dict(
        time=float(d.time),
        mass=mass,
        com_height=float(com[2]),
        potential=potential,
        K_system=kinetic,
        kinetic=kinetic - rotor,
        rotor_kinetic=rotor,
        mujoco_kinetic=kinetic,
        mechanical_energy=potential + kinetic,
        L_com=float(h @ axis),
        L_support=float(hs @ axis),
        L_fixed=float(hf @ axis),
        I_com=inertia,
        I_support=float(
            inertia + mass * (relative @ relative - (relative @ axis) ** 2)
        ),
        moment_com=float(moment @ axis),
        moment_support=float(support_moment @ axis),
        moment_fixed=float(fixed_moment @ axis),
        rear_power=float(power[e.rear_indices].sum()),
        rear_positive_power=float(np.maximum(power[e.rear_indices], 0).sum()),
        actuator_power=float(power.sum()),
        calf_clearance=float(calf),
        front_support=bool(row["front_support"]),
        support_valid=support_valid,
        front_slip_speed=float(max(speeds)),
        rear_clearance=row["rear_clearance"],
        rear_loaded=bool((row["loads"][2:] > 0.04 * e.weight).any()),
        collision=bool(row["illegal_contacts"] or row["self_contacts"]),
        foot_drift=row["front_drift"],
        com_speed=float(np.linalg.norm(velocity)),
        body_angular_speed=float(np.linalg.norm(d.qvel[3:6])),
        pitch=float(np.arctan2(-forward[2], parallel)),
        pitch_velocity=omega,
        com_support_x=float(relative[0]),
        com_support_y=float(relative[1]),
        pendulum_length=float(np.linalg.norm(relative)),
        rotational_proxy=float((h @ axis) ** 2 / (2 * max(inertia, 1e-9))),
    )
    feet = d.geom_xpos[e.foot_geoms[:2], :2]
    segment = feet[1] - feet[0]
    t = np.clip(
        np.dot(com[:2] - feet[0], segment) / max(segment @ segment, 1e-12), 0, 1
    )
    out["support_region_distance"] = float(
        max(0, np.linalg.norm(com[:2] - feet[0] - t * segment) - 0.022)
    )
    for label, vector in [
        ("com", com),
        ("com_velocity", velocity),
        ("support", support),
        ("momentum", h),
        ("support_momentum", hs),
    ]:
        for j, a in enumerate("xyz"):
            out[f"{label}_{a}"] = float(vector[j])
    for group, label in enumerate(("front", "rear", "nonfoot")):
        out[f"{label}_moment_com"] = float(moments[group] @ axis)
        for j, a in enumerate("xyz"):
            out[f"{label}_force_{a}"] = float(forces[group, j])
    return out
