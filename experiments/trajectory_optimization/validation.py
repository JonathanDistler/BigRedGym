"""Independent physical replay and fail-closed export of optimization candidates."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from gym.envs.go2.handstand_replay import PHYSICS_DT, CONTROL_DT, KP, KD
from .model import Robot, Settings, JOINTS
from .transcription import Problem, Y, V


def audit_saved(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    path = directory / "candidate.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["candidate_sha256"]:
        raise ValueError("Candidate hash mismatch")
    settings = Settings(**report["settings"])
    robot = Robot(settings)
    with np.load(path, allow_pickle=False) as data:
        saved = {key: data[key].copy() for key in data.files}
    problem = Problem(
        robot, saved["start"], saved["goal"], saved["anchors"], saved["nominal"]
    )
    rows = problem.unpack(saved["decision"])
    # Do not trust duplicated arrays in an archive independently of decisions.
    for name, expected in [
        ("time", settings.times),
        ("qpos", np.array([robot.pose(row[Y]) for row in rows])),
        ("qvel", rows[:, V]),
        ("targets", np.array([problem.target(row) for row in rows])),
    ]:
        if saved[name].shape != expected.shape or not np.array_equal(
            saved[name], expected
        ):
            raise ValueError("Archive/decision mismatch: " + name)
    actual = problem.audit(saved["decision"])
    return robot, problem, saved, actual


def strict_contact_audit(problem, decision):
    """Recheck coarse-mesh experiments against the original contact criteria."""
    cfg = replace(problem.cfg, contact_acceleration=True, foot_velocity_tolerance=0.002)
    robot = Robot(cfg)
    strict = Problem(
        robot, problem.start, problem.goal, problem.anchors, problem.nominal
    )
    return strict.audit(decision)


def position_commands(saved, speed, hold_seconds):
    """The exact 50 Hz bounded position sequence used by replay AND export."""
    end = float(saved["time"][-1]) + hold_seconds
    grid = np.arange(int(np.ceil(end / CONTROL_DT))) * CONTROL_DT
    target = saved["targets"][0].copy()
    commands = []
    for t in grid:
        desired = np.array(
            [np.interp(t, saved["time"], saved["targets"][:, j]) for j in range(12)]
        )
        target += np.clip(
            desired - target, -0.7 * speed * CONTROL_DT, 0.7 * speed * CONTROL_DT
        )
        commands.append(target.copy())
    return grid, np.array(commands)


def replay(directory, hold_seconds=5.0):
    robot, problem, saved, mesh = audit_saved(directory)
    if hold_seconds < 5:
        raise ValueError("Five-second hold cannot be weakened")
    m = robot.m
    d = mujoco.MjData(m)
    d.qpos[:] = saved["start"]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    initial_height = float(d.qpos[2])
    target = saved["targets"][0].copy()
    errors = {
        "joint_tracking_rad": 0.0,
        "root_position_tracking_m": 0.0,
        "root_orientation_tracking_rad": 0.0,
        "minimum_body_clearance_m": float("inf"),
        "max_speed_ratio": 0.0,
        "max_torque_speed_violation_Nm": 0.0,
    }
    illegal = None
    state_safe = True
    hold = best_hold = 0.0
    from gym.envs.go2.handstand_replay import clearance, strict_balance, TARGET_ROTATION

    grid, commands = position_commands(saved, robot.speed, hold_seconds)
    # Native compliant contacts; never inject planned forces or reset the state.
    for time, target in zip(grid, commands):
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            velocity = d.qvel[robot.vids].copy()
            torque = np.clip(
                KP * (target - d.qpos[robot.qids]) - KD * velocity,
                -robot.effort,
                robot.effort,
            )
            torque[(abs(velocity) >= robot.speed) & (torque * velocity > 0)] = 0
            errors["max_torque_speed_violation_Nm"] = max(
                errors["max_torque_speed_violation_Nm"],
                float(
                    np.maximum(0, abs(torque) - robot.torque_envelope(velocity)).max()
                ),
            )
            d.qfrc_applied[:] = 0
            d.qfrc_applied[robot.vids] = torque
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            errors["max_speed_ratio"] = max(
                errors["max_speed_ratio"],
                float(abs(d.qvel[robot.vids] / robot.speed).max()),
            )
            lowest = min(clearance(m, d, g) for g in robot.robot_geoms)
            errors["minimum_body_clearance_m"] = min(
                errors["minimum_body_clearance_m"], lowest
            )
            for contact in d.contact:
                if contact.dist > 0:
                    continue
                b1, b2 = map(int, m.geom_bodyid[[contact.geom1, contact.geom2]])
                if (b1 and b2) or any(b and b not in robot.feet for b in (b1, b2)):
                    illegal = {
                        "time_s": float(d.time),
                        "bodies": [m.body(b1).name, m.body(b2).name],
                    }
                    break
            i = min(
                int(np.searchsorted(saved["time"], d.time, side="right") - 1),
                len(saved["time"]) - 1,
            )
            expected = saved["qpos"][max(0, i)]
            error = np.empty(18)
            mujoco.mj_differentiatePos(m, error, 1, expected, d.qpos)
            errors["joint_tracking_rad"] = max(
                errors["joint_tracking_rad"], float(abs(error[6:]).max())
            )
            errors["root_position_tracking_m"] = max(
                errors["root_position_tracking_m"], float(np.linalg.norm(error[:3]))
            )
            errors["root_orientation_tracking_rad"] = max(
                errors["root_orientation_tracking_rad"],
                float(np.linalg.norm(error[3:6])),
            )
            # New crouch-start trial: require final specified pose/velocity and
            # continuous bilateral support/clearance. This is a new crouch task;
            # it is not the 67/100 CapturePolicy evaluation.
            foot = d.geom_xpos[robot.foot_geoms].copy()
            foot[:, 2] -= robot.cfg.foot_radius
            loads = np.zeros(4)
            for ci, contact in enumerate(d.contact):
                if contact.dist > 0:
                    continue
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if (m.geom_bodyid[g1] == 0) == (m.geom_bodyid[g2] == 0):
                    continue
                g = g2 if m.geom_bodyid[g1] == 0 else g1
                if g in robot.foot_geoms:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, wrench)
                    force = contact.frame.reshape(3, 3).T @ wrench[:3]
                    loads[robot.foot_geoms.index(g)] += abs(force[2])
            egoal = np.empty(18)
            mujoco.mj_differentiatePos(m, egoal, 1, saved["goal"], d.qpos)
            mujoco.mj_subtreeVel(m, d)
            com = d.subtree_com[robot.base]
            segment = foot[1, :2] - foot[0, :2]
            u = np.clip(
                (com[:2] - foot[0, :2]) @ segment / max(segment @ segment, 1e-10), 0, 1
            )
            support_error = np.linalg.norm(com[:2] - foot[0, :2] - u * segment)
            slip = 0.0
            for g in robot.foot_geoms[:2]:
                jac = np.zeros((3, 18))
                mujoco.mj_jacGeom(m, d, jac, None, g)
                slip = max(slip, float(np.linalg.norm((jac @ d.qvel)[:2])))
            rotation = d.xmat[robot.base].reshape(3, 3)
            legal_joints = bool(
                (d.qpos[robot.qids] >= robot.bounds[:, 0] - 0.001).all()
                and (d.qpos[robot.qids] <= robot.bounds[:, 1] + 0.001).all()
            )
            state_safe &= legal_joints
            strict = strict_balance(
                safe=state_safe
                and illegal is None
                and legal_joints
                and errors["max_speed_ratio"] <= 1.001,
                rear_clearance=float(foot[2:, 2].min()),
                rear_loads=loads[2:] / (robot.mass * 9.81),
                front_support=bool((loads[:2] > 0.05 * robot.mass * 9.81).all()),
                orientation=float(
                    np.arccos(
                        np.clip((np.trace(TARGET_ROTATION.T @ rotation) - 1) / 2, -1, 1)
                    )
                ),
                base_height=float(d.qpos[2]),
                com_height=float(com[2]),
                com_speed=float(np.linalg.norm(d.subtree_linvel[robot.base])),
                angular_speed=float(np.linalg.norm(d.qvel[3:6])),
                support_error=float(support_error),
                slip=slip,
                calf_clearance=lowest,
                head_clearance=lowest,
                force_ratio=float(loads[:2].sum() / (robot.mass * 9.81)),
            )
            # The original strict gate is retained; endpoint leg/body matching is
            # an ADDITIONAL requirement for this new trajectory boundary problem.
            stable = bool(
                strict
                and d.time >= saved["time"][-1]
                and np.max(abs(egoal[6:])) < 0.15
                and np.linalg.norm(egoal[:2]) < 0.05
                and lowest >= robot.cfg.body_clearance
            )
            hold = hold + PHYSICS_DT if stable else 0.0
            best_hold = max(best_hold, hold)
            if illegal is not None:
                break
        if illegal is not None:
            break
    result = {
        "schema": 1,
        "scope": "Independent native position-PD replay with original strict hold gate",
        "mesh_audit": mesh,
        "replay_seconds": float(d.time),
        "initial_crouch_height_m": initial_height,
        "first_illegal_contact": illegal,
        "continuous_terminal_hold_seconds": best_hold,
        "terminal_goal_error": egoal.tolist(),
        "errors": errors,
        "candidate_sha256": hashlib.sha256(
            (Path(directory) / "candidate.npz").read_bytes()
        ).hexdigest(),
        "native_replay_pass": bool(
            illegal is None
            and best_hold >= 5
            and errors["max_torque_speed_violation_Nm"] <= 1e-6
            and mesh["mesh_feasible"]
            and state_safe
            and errors["minimum_body_clearance_m"] >= robot.cfg.body_clearance
            and errors["joint_tracking_rad"] <= 0.15
            and errors["root_position_tracking_m"] <= 0.10
            and errors["root_orientation_tracking_rad"] <= 0.2
        ),
        "hardware_ready": False,
        "prefix_joint_legal": bool(state_safe),
    }
    (Path(directory) / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export(directory, destination):
    directory = Path(directory)
    destination = Path(destination)
    robot, problem, saved, mesh = audit_saved(directory)
    strict = strict_contact_audit(problem, saved["decision"])
    if not strict["mesh_feasible"]:
        raise ValueError("NOT READY: original strict contact audit must pass")
    proof = replay(directory)  # Recompute; never trust a success flag in JSON.
    digest = hashlib.sha256((directory / "candidate.npz").read_bytes()).hexdigest()
    if (
        proof.get("candidate_sha256") != digest
        or not mesh["mesh_feasible"]
        or not proof.get("native_replay_pass")
    ):
        raise ValueError(
            "NOT READY: independent mesh and five-second native replay must pass"
        )
    if not robot.cfg.torque_curve_calibrated:
        raise ValueError(
            "NOT READY: staff-calibrated speed/torque envelope unavailable"
        )
    # Staff-review archive remains explicitly software-only even after numerical passes.
    if destination.exists():
        raise FileExistsError("Export destination exists")
    destination.mkdir(parents=True)
    grid, targets = position_commands(saved, robot.speed, 5.0)
    if (abs((targets - saved["nominal"]) / 2) > 1 + 1e-9).any():
        raise ValueError("Normalized action bounds violated")
    np.savez_compressed(
        destination / "position_reference.npz",
        time=grid,
        targets=targets,
        nominal=saved["nominal"],
        initial_target=targets[0],
        joint_order=np.array(JOINTS),
    )
    meta = {
        "schema": 1,
        "kind": "constrained_position_trajectory_v1",
        "control_hz": 50,
        "kp": 60,
        "kd": 3,
        "ema": False,
        "action_scale": 2.0,
        "hardware_ready": False,
        "sha256": hashlib.sha256(
            (destination / "position_reference.npz").read_bytes()
        ).hexdigest(),
        "candidate_sha256": digest,
    }
    (destination / "deploy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
