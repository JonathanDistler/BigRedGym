"""Online geometry diagnostics. No direction or stepping strategy is prescribed."""

import mujoco
import numpy as np
from experiments.morphology.dynamics_audit import ground_clearance


def region_distance(point, feet, loaded):
    points = feet[np.asarray(loaded, bool), :2]
    if not len(points):
        return float("nan")
    if len(points) == 1:
        return float(max(0, np.linalg.norm(point[:2] - points[0]) - 0.022))
    delta = points[1] - points[0]
    t = np.clip((point[:2] - points[0]) @ delta / max(delta @ delta, 1e-12), 0, 1)
    return float(max(0, np.linalg.norm(point[:2] - points[0] - t * delta) - 0.022))


def reach(env, leg, target, iterations=8):
    """Native Jacobian IK at frozen current base; candidate only, never a reset.

    Check torque authority for a vertical load and all illegal ground geometry.
    Does not certify a dynamic capture basin or full contact wrench feasibility.
    """
    d = mujoco.MjData(env.m)
    d.qpos[:] = env.d.qpos
    ids = np.array([i for i, n in enumerate(env.names) if n.startswith(leg + "_")])
    qids, vids = env.qids[ids], env.vids[ids]
    geom = int(env.foot_geoms[("FL", "FR").index(leg)])
    jp = np.zeros((3, env.m.nv))
    for _ in range(iterations):
        mujoco.mj_forward(env.m, d)
        error = target - d.geom_xpos[geom]
        if np.linalg.norm(error) < 0.004:
            break
        mujoco.mj_jacGeom(env.m, d, jp, None, geom)
        j = jp[:, vids]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), error)
        d.qpos[qids] = np.clip(
            d.qpos[qids] + np.clip(dq, -0.15, 0.15), *env.bounds[ids].T
        )
    mujoco.mj_forward(env.m, d)
    mujoco.mj_jacGeom(env.m, d, jp, None, geom)
    error = float(np.linalg.norm(target - d.geom_xpos[geom]))
    # Conservative full weight on one front leg, with actual Jacobian leverage.
    static_ratio = float(
        np.max(abs(jp[:, vids].T @ [0, 0, env.weight]) / env.effort[ids])
    )
    illegal = any(
        c.dist < 0
        and (
            int(env.m.geom_bodyid[c.geom1]) not in (0, *env.feet)
            or int(env.m.geom_bodyid[c.geom2]) not in (0, *env.feet)
        )
        for c in d.contact
    )
    return error < 0.01 and not illegal and static_ratio <= 1, error, static_ratio


def measurements(env, with_ik=True):
    m, d = env.m, env.d
    s = env.momentum_sample
    com = d.subtree_com[env.base].copy()
    vel = d.subtree_linvel[env.base].copy()
    rot = d.xmat[env.base].reshape(3, 3)
    feet = d.geom_xpos[env.foot_geoms[:2]].copy()
    loaded = env.row["loads"][:2] > 0.05 * env.weight
    height = com[2] - 0.022
    # LIP validity is intentionally restrictive; ballistic forecast remains diagnostic.
    valid_lip = bool(
        height > 0.1
        and loaded.all()
        and abs(vel[2]) < 0.15
        and abs(s["pitch_velocity"]) < 0.5
        and abs(s["L_com"]) < 0.4
    )
    capture = com + vel * np.sqrt(max(height, 0.05) / 9.81)
    future = com + 0.10 * vel
    future[2] -= 0.5 * 9.81 * 0.10**2
    target = capture.copy() if valid_lip else future.copy()
    target[2] = 0.022
    out = dict(
        lip_valid=valid_lip,
        actual_support_count=int(loaded.sum()),
        actual_support_distance=region_distance(com, feet, loaded),
        capture_support_distance=region_distance(capture, feet, loaded),
        support_velocity=float(np.linalg.norm(vel[:2])),
        capture_x=float(capture[0]),
        capture_y=float(capture[1]),
        forecast_x=float(target[0]),
        forecast_y=float(target[1]),
    )
    for leg, i in [("FL", 0), ("FR", 1)]:
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacGeom(m, d, jp, None, int(env.foot_geoms[i]))
        footvel = jp @ d.qvel
        bodyfoot = rot.T @ (feet[i] - d.xpos[env.base])
        # Bound forecast request in the horizontal plane around the current foot.
        candidate = target.copy()
        shift = target[:2] - feet[i, :2]
        candidate[:2] = feet[i, :2] + shift * min(
            1, 0.10 / max(np.linalg.norm(shift), 1e-9)
        )
        reachable, error, loadratio = (
            reach(env, leg, candidate) if with_ik else (False, 1.0, 1.0)
        )
        out.update(
            {
                f"{leg}_target_reachable": bool(reachable),
                f"{leg}_ik_error": error,
                f"{leg}_single_support_static_ratio": loadratio,
                f"{leg}_predicted_recontact_seconds": float(
                    np.clip((feet[i, 2] - 0.022) / max(-footvel[2], 0.01), 0, 0.5)
                ),
            }
        )
        for label, vec in [
            ("world", feet[i]),
            ("world_velocity", footvel),
            ("body", bodyfoot),
            ("com_relative", feet[i] - com),
            ("target", candidate),
        ]:
            for axis, value in zip("xyz", vec):
                out[f"{leg}_{label}_{axis}"] = float(value)
        for part in ("hip", "thigh", "calf", "foot"):
            b = m.body(f"{leg}_{part}").id
            for axis, value in zip("xyz", d.xpos[b]):
                out[f"{leg}_{part}_{axis}"] = float(value)
        # Knee location is the calf joint anchor; calf body origin is that anchor.
        for axis, value in zip("xyz", d.xpos[m.body(f"{leg}_calf").id]):
            out[f"{leg}_knee_{axis}"] = float(value)
        for j, n in enumerate(env.names):
            if n.startswith(leg + "_"):
                out[n + "_angle"] = float(d.qpos[env.qids[j]])
                out[n + "_velocity"] = float(d.qvel[env.vids[j]])
    minimum = 1.0
    for prefix in ("FL_calf", "FR_calf", "FL_thigh", "FR_thigh", "Head", "base"):
        geoms = [
            g
            for g in range(m.ngeom)
            if m.body(int(m.geom_bodyid[g])).name.startswith(prefix)
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        clearance = min((ground_clearance(m, d, g) for g in geoms), default=1.0)
        out[prefix + "_ground_clearance"] = float(clearance)
        minimum = min(minimum, clearance)
    out["illegal_clearance"] = minimum
    return out
