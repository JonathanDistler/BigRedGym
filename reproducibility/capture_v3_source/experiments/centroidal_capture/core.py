"""Opt-in capture metrics and smooth corrections to saved physical rollouts."""

import numpy as np
import mujoco
from experiments.morphology.dynamics_audit import ground_clearance


def metrics(e, detailed=False):
    m, d = e.m, e.d
    mujoco.mj_subtreeVel(m, d)
    feet = d.geom_xpos[e.foot_geoms].copy()
    vel = []
    jp = np.zeros((3, m.nv))
    for g in e.foot_geoms:
        mujoco.mj_jacGeom(m, d, jp, None, int(g))
        vel.append(jp @ d.qvel)
    vel = np.array(vel)
    com = d.subtree_com[e.base].copy()
    cv = d.subtree_linvel[e.base].copy()
    r = d.xmat[e.base].reshape(3, 3)
    loads = np.array(e.row["loads"])
    # Foot midpoint is defined even in flight, but is not called actual support.
    relative = com - feet[:2].mean(0)
    s = e.momentum_sample
    out = dict(
        time=float(d.time),
        orientation_error=float(e.contact_sample["orientation_error"]),
        pose_error=float(
            np.sqrt(
                np.mean(
                    (
                        d.qpos[e.qids]
                        - np.array(
                            [
                                -0.1543,
                                -0.9677,
                                -1.1838,
                                0.1543,
                                -0.9677,
                                -1.1838,
                                -0.0546,
                                0.7147,
                                -1.4534,
                                0.0546,
                                0.7147,
                                -1.4534,
                            ]
                        )
                    )
                    ** 2
                )
            )
        ),
        pitch=float(s["pitch"]),
        pitch_velocity=float(s["pitch_velocity"]),
        roll=float(np.arctan2(r[2, 1], r[2, 2])),
        yaw=float(np.arctan2(r[1, 0], r[0, 0])),
        omega=(r @ d.qvel[3:6]).tolist(),
        H=d.subtree_angmom[e.base].tolist(),
        com=com.tolist(),
        com_velocity=cv.tolist(),
        feet=feet.tolist(),
        foot_velocity=vel.tolist(),
        gaps=(feet[:, 2] - 0.022).tolist(),
        loads=loads.tolist(),
        com_foot_xy=relative[:2].tolist(),
        safe=bool(e._momentum_prefix_safe),
        collision=bool(e.nonfoot_collision or e.self_collision),
        rear_clearance=float(s["rear_clearance"]),
        capture_distance=float(e.capture_distance),
        energy=float(s["mechanical_energy"]),
        front_moment=float(s["front_moment_com"]),
        rear_moment=float(s["rear_moment_com"]),
        front_loaded=bool(loads[:2].sum() > 0.15 * e.weight),
        both_front=bool(np.all(loads[:2] > 0.04 * e.weight)),
        torque_ratio=float(np.max(abs(d.qfrc_applied[e.vids]) / e.effort)),
        speed_ratio=float(np.max(abs(d.qvel[e.vids]) / e.speed)),
    )
    # No claim that Euler angles are nonsingular near vertical: matrix also saved.
    if detailed:
        out.update(
            rotation=r.tolist(),
            q=d.qpos[e.qids].tolist(),
            qd=d.qvel[e.vids].tolist(),
            target=e.target.tolist(),
            tau=d.qfrc_applied[e.vids].tolist(),
            torque_margin=(e.effort - abs(d.qfrc_applied[e.vids])).tolist(),
            joint_lower_margin=(d.qpos[e.qids] - e.bounds[:, 0]).tolist(),
            joint_upper_margin=(e.bounds[:, 1] - d.qpos[e.qids]).tolist(),
            illegal_contacts=list(e.row["illegal_contacts"])
            if isinstance(e.row["illegal_contacts"], list)
            else int(e.row["illegal_contacts"]),
        )
        clear = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = m.body(b).name
            if b and (m.geom_contype[g] or m.geom_conaffinity[g]):
                group = (
                    "calf"
                    if "calf" in name
                    else "thigh"
                    if "thigh" in name
                    else "foot"
                    if "foot" in name
                    else "body"
                )
                clear[group] = min(clear.get(group, 1e9), ground_clearance(m, d, g))
        out["clearances"] = clear
        out["hip_positions"] = [
            d.xpos[m.body(f"{leg}_thigh").id].tolist()
            for leg in ("FL", "FR", "RL", "RR")
        ]
    return out


def capture_quality(x):
    """Near-vertical supported arrival, low speed and small geometric error."""
    p = x["pitch"]
    v = np.array(x["com_velocity"])
    w = np.array(x["omega"])
    gap = np.array(x["gaps"][:2])
    fv = np.array(x["foot_velocity"][:2])
    angle = abs(p - np.pi / 2)
    cost = (
        2.5 * angle
        + 2 * np.linalg.norm(x["com_foot_xy"])
        + 0.45 * np.linalg.norm(v)
        + 0.20 * np.linalg.norm(w)
        + 0.3 * np.linalg.norm(x["H"])
    )
    cost += 2 * np.maximum(gap, 0).mean() + 0.08 * np.linalg.norm(fv[:, :2])
    # Impact shaping on downward near-ground approach, never rewards impacts.
    cost += 0.12 * np.mean(
        np.maximum(-fv[:, 2] - 0.5, 0) * np.exp(-np.maximum(gap, 0) / 0.08)
    )
    near = p > np.deg2rad(60)
    cost += 1.5 * x.get("orientation_error", abs(p - np.pi / 2))
    cost += 0.5 * x.get("pose_error", 0)
    cost += 0.4 * min(x.get("capture_distance", 0), 5)
    quality = 10 - 3 * cost
    quality += near * (2 * x["front_loaded"] + x["both_front"])
    quality -= 12 * max(0, np.deg2rad(60) - p)
    if x["rear_clearance"] < 0.05:
        quality -= 3
    return float(quality)


def correction(params, t):
    return np.array(
        [np.interp(t, [0, 0.16, 0.34, 0.56, 0.8], params[:, i]) for i in range(6)]
    )


def modify(u, names, offset):
    u = u.copy()
    for i, name in enumerate(names):
        if "thigh" not in name and "calf" not in name:
            continue
        group = (
            0 if name.startswith("FL_") else 2 if name.startswith("FR_") else 4
        ) + int("calf" in name)
        u[i] += offset[group]
    return u.clip(-1, 1)
