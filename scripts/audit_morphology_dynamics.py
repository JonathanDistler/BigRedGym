"""IK-based morphology dynamics audit; does not train PPO or touch old outputs."""

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from experiments.morphology.dynamics_audit import Audit, smooth
from experiments.morphology.task import Config, metadata, reference, write_json
from gym.envs.go2.leg_morphology import SCALES
from scripts.morphology_feasibility import save_csv

FEET = np.array(
    [
        [0.1917, 0.142, 0.022],
        [0.1917, -0.142, 0.022],
        [-0.1951, 0.142, 0.022],
        [-0.1951, -0.142, 0.022],
    ]
)


def target_pose(task, standing, pitch=np.pi / 2):
    """Front IK plus COM-over-support iteration; no external stabilizing force."""
    x = 0.22
    q = standing.copy()
    for _ in range(20):
        q, error = task.pose(
            0.1934 * np.sin(pitch) + 0.022 + 0.26 * task.scale,
            pitch=pitch,
            x=x,
            feet=FEET,
            active=(0, 1),
            guess=q,
        )
        row = task.static(q, (0, 1))
        x += FEET[:2, 0].mean() - row["com_x"]
    return q, error


def waypoints(task, family):
    height = 0.34 * task.scale if family == "relative_extension" else 0.30
    crouch_height = 0.27 * task.scale if family == "relative_extension" else 0.24
    guess = task.d.qpos.copy()
    guess[task.qids] = task.nominal
    standing, es = task.pose(height, feet=FEET, guess=guess)
    crouch, ec = task.pose(crouch_height, feet=FEET, guess=standing)
    target, et = target_pose(task, standing)
    if max(es, ec, et) > 1e-5:
        raise RuntimeError(f"IK residual {es, ec, et}")
    return standing, crouch, target


def command(task, standing, crouch, duration, knee, feedback, previous):
    t = task.t
    q0, qc = standing[task.qids], crouch[task.qids]
    if t < 0.2:
        desired, phase, pitch = q0.copy(), "settle", 0.0
    elif t < 0.45:
        desired, phase, pitch = q0 + (qc - q0) * smooth((t - 0.2) / 0.25), "crouch", 0.0
    else:
        thrust = qc.copy()
        pitch = np.deg2rad(25)
        for i, name in enumerate(task.names):
            if name.startswith(("FL", "FR")):
                thrust[i] = (
                    0
                    if "hip" in name
                    else -pitch - knee / 2
                    if "thigh" in name
                    else knee
                )
            else:
                thrust[i] = 0 if "hip" in name else 0.30 if "thigh" in name else -0.85
        if t < 0.45 + duration:
            amount = smooth((t - 0.45) / duration)
            desired = qc + (thrust - qc) * amount
            pitch *= amount
            phase = "push"
        else:
            amount = smooth((t - 0.45 - duration) / 0.5)
            pitch = np.deg2rad(25) + amount * np.deg2rad(55)
            final = thrust.copy()
            for i, name in enumerate(task.names):
                if name.startswith(("FL", "FR")):
                    final[i] = (
                        0
                        if "hip" in name
                        else -np.deg2rad(80) + 0.8
                        if "thigh" in name
                        else -1.6
                    )
                else:
                    final[i] = 0 if "hip" in name else 1.0 if "thigh" in name else -1.8
            desired = thrust + (final - thrust) * amount
            phase = "rotation"
        if feedback:
            measured = task.measure()
            correction = (
                -0.6 * (pitch - measured["pitch"]) + 0.07 * measured["pitch_velocity"]
            )
            for i, name in enumerate(task.names):
                if name.startswith(("FL", "FR")) and "thigh" in name:
                    desired[i] += correction
    desired = np.clip(desired, *task.m.jnt_range[task.jids].T)
    # Position command speeds conservatively bounded, shared by all variants.
    desired = previous + np.clip(
        desired - previous,
        -0.7 * task.speed * task.m.opt.timestep,
        0.7 * task.speed * task.m.opt.timestep,
    )
    return desired, phase


def summarize(rows, task, **identity):
    valid_lift = [
        r["rear_clearance"] > 0.015
        and r["front_support"]
        and r["pitch"] > 0
        and r["illegal_contacts"] == 0
        and r["joint_limit_violation"] < 0.001
        and r["measured_speed_ratio"] <= 1.001
        and r["load_RL"] + r["load_RR"] < 0.04 * r["mass"] * 9.81
        for r in rows
    ]
    longest = current = 0
    for lifted in valid_lift:
        current = current + 1 if lifted else 0
        longest = max(longest, current)
    event = task.first_event
    return dict(
        **identity,
        scale=task.scale,
        seconds=task.t,
        max_pitch=max(r["pitch"] for r in rows),
        min_pitch=min(r["pitch"] for r in rows),
        max_supported_pitch=max(
            [r["pitch"] for r in rows if r["front_support"]] + [0.0]
        ),
        max_pitch_velocity=max(r["pitch_velocity"] for r in rows),
        max_rear_clearance=max(r["rear_clearance"] for r in rows),
        takeoff=longest * task.m.opt.timestep >= 0.02,
        supported_airborne_seconds=longest * task.m.opt.timestep,
        front_support_fraction=np.mean([r["front_support"] for r in rows]),
        push_impulse=task.push_impulse,
        rear_angular_impulse=task.rear_angular_impulse,
        external_angular_impulse=task.external_angular_impulse,
        positive_work=task.positive_work,
        net_work=task.net_work,
        saturation=np.mean([r["saturation"] for r in rows]),
        speed_limit_fraction=np.mean([r["speed_limit_fraction"] for r in rows]),
        measured_speed_ratio=max(r["measured_speed_ratio"] for r in rows),
        torque_ratio=max(r["torque_ratio"] for r in rows),
        joint_limit_violation=max(r["joint_limit_violation"] for r in rows),
        peak_potential_rise=max(r["gravitational_potential_energy"] for r in rows)
        - rows[0]["gravitational_potential_energy"],
        min_nonfoot_clearance=min(r["nonfoot_clearance"] for r in rows),
        min_rear_calf_clearance=min(r["rear_calf_clearance"] for r in rows),
        first_contact_time=event["time"] if event else "",
        first_contact_phase=event["phase"] if event else "none",
        first_contact_body=event["body"] if event else "none",
        success=task.best_hold >= 5.0,
        best_hold=task.best_hold,
        max_com_speed=max(r["com_speed"] for r in rows),
    )


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.output / "source"
    source.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/dynamics_audit.py"),
        Path("experiments/morphology/task.py"),
        Path("gym/envs/go2/leg_morphology.py"),
    ):
        shutil.copyfile(path, source / path.name)
    cfg = Config()
    task = Audit(a.output / "assets", cfg)
    write_json(
        a.output / "configuration.json",
        dict(
            **metadata(cfg),
            scales=a.scales,
            seeds=list(range(a.seeds)),
            physics_dt=0.002,
            analytical_model="fixed front pivot, U=mgz; frozen-posture peak mg(L-z)",
            feet=FEET.tolist(),
            families=["relative_extension", "world_geometry"],
            contact_termination="first nonfoot ground penetration",
            takeoff_criterion=(
                ">=20ms rear clearance >15mm; rear load <4% BW; "
                "both front loads >5% BW; positive pitch"
            ),
        ),
    )
    diagnosis = []
    events = []
    poses = []
    geometry = []
    checks = []
    traces = []
    attempts = []
    balance = []
    # Reproduce old reference for every scale, recording actual first 2ms event.
    for scale in (1.0, 0.95, 0.90, 0.85, 0.80):
        for amp in (0.6, 1.0):
            for duration in (0.12, 0.24):
                task.reset(scale, 0)
                reset = task.clearances()
                old = None
                rows = []
                for step in range(1000):
                    if step % 5 == 0:
                        old = reference(task, amp, duration)
                    phase = (
                        "settle"
                        if task.t < 0.2
                        else "crouch"
                        if task.t < 0.45
                        else "push"
                        if task.t < 0.45 + duration
                        else "rotation"
                    )
                    row, contacts = task.sample(old, phase)
                    rows.append(row)
                    if contacts:
                        break
                event = task.first_event
                identity = dict(
                    evaluation="legacy",
                    family="legacy",
                    mode="open_loop",
                    seed=0,
                    duration=duration,
                    knee=amp,
                )
                diagnosis.append(
                    dict(
                        **summarize(rows, task, **identity),
                        reset_rear_clearance=reset["rear_calf_clearance"],
                        reset_nonfoot_clearance=reset["nonfoot_clearance"],
                        rearward_tipping_time=next(
                            (r["time"] for r in rows if r["pitch"] < -np.deg2rad(10)),
                            "",
                        ),
                    )
                )
                if event:
                    events.append(dict(**event, scale=scale, **identity))
        print(f"legacy contact audit {scale:.2f}", flush=True)
    save_csv(a.output / "legacy_contact_diagnosis.csv", diagnosis)
    for scale in a.scales:
        task.reset(scale, 0)
        for family in ("relative_extension", "world_geometry"):
            standing, crouch, target = waypoints(task, family)
            static_stand = task.static(standing)
            static_target = task.static(target, (0, 1))
            mass = static_stand["mass"]
            g = 9.81
            length = static_stand["pendulum_length"]
            z = static_stand["com_z"]
            geometry.append(
                dict(
                    scale=scale,
                    family=family,
                    segment_length=0.213 * scale,
                    mass=mass,
                    com_x=static_stand["com_x"],
                    com_y=static_stand["com_y"],
                    com_z=z,
                    pendulum_length=length,
                    pitch_inertia=static_stand["pitch_inertia_com"],
                    support_pitch_inertia=static_stand["pitch_inertia_support"],
                    measured_target_com_z=static_target["com_z"],
                    measured_target_delta_U=mass * g * (static_target["com_z"] - z),
                    analytical_frozen_barrier=mass * g * (length - z),
                    analytical_frozen_barrier_per_kg=g * (length - z),
                )
            )
            for label, q, active in (
                ("standing", standing, (0, 1, 2, 3)),
                ("crouch", crouch, (0, 1, 2, 3)),
                ("target", target, (0, 1)),
            ):
                row = task.static(q, active)
                poses.append(dict(**row, family=family, pose=label, qpos=q.tolist()))
            # Audit the crouch path with all feet pinned.
            previous = standing[task.qids]
            for k in range(51):
                fraction = k / 50
                height = (1 - smooth(fraction)) * standing[2] + smooth(
                    fraction
                ) * crouch[2]
                q, error = task.pose(height, feet=FEET, guess=standing)
                row = task.static(q)
                speed = (
                    np.max(abs(q[task.qids] - previous) / (0.25 / 50) / task.speed)
                    if k
                    else 0.0
                )
                previous = q[task.qids]
                checks.append(
                    dict(
                        family=family,
                        fraction=fraction,
                        ik_error=error,
                        command_speed_ratio=speed,
                        **row,
                    )
                )
            # Front-pinned geometric path; this is not a dynamic rollout.
            path_energy = []
            q = standing.copy()
            for k in range(51):
                f = k / 50
                pitch = f * np.pi / 2
                x = (1 - f) * standing[0] + f * target[0]
                h = (1 - f) * standing[2] + f * target[2]
                q, error = task.pose(h, pitch, x, FEET, (0, 1), q)
                q[task.qids[6:]] = standing[task.qids[6:]]
                row = task.static(q, (0, 1))
                checks.append(
                    dict(
                        family=family,
                        fraction=1 + f,
                        ik_error=error,
                        command_speed_ratio=0.0,
                        **row,
                    )
                )
                path_energy.append(mass * g * (row["com_z"] - z))
            geometry[-1]["measured_ik_path_barrier"] = max(path_energy)
            for duration in (0.12, 0.24):
                for knee in (-1.8, -2.2):
                    for mode in ("open_loop", "pitch_feedback"):
                        for seed in range(a.seeds):
                            task.set_pose(
                                standing,
                                np.random.default_rng(seed).uniform(-0.02, 0.02),
                            )
                            previous = standing[task.qids].copy()
                            rows = []
                            identity = dict(
                                evaluation="takeoff",
                                family=family,
                                mode=mode,
                                seed=seed,
                                duration=duration,
                                knee=knee,
                            )
                            for k in range(round(1.8 / task.m.opt.timestep)):
                                desired, phase = command(
                                    task,
                                    standing,
                                    crouch,
                                    duration,
                                    knee,
                                    mode == "pitch_feedback",
                                    previous,
                                )
                                previous = desired.copy()
                                row, contacts = task.sample(desired, phase)
                                row.update(identity)
                                rows.append(row.copy())
                                if contacts:
                                    events.append(
                                        dict(**contacts[0], scale=scale, **identity)
                                    )
                                    break
                            attempts.append(summarize(rows, task, **identity))
                            traces.extend(rows)
            print(f"takeoff {scale:.2f} {family}", flush=True)
        # Independent pitch/omega balance grid, with COM aligned using IK.
        standing, _, _ = waypoints(task, "relative_extension")
        for angle in (80, 85, 90):
            q, error = target_pose(task, standing, np.deg2rad(angle))
            for omega in (-0.3, 0.0, 0.3):
                for mode in ("joint_hold", "gravity_pitch_feedback"):
                    task.static(q, (0, 1))
                    feedforward = task.static_torque.copy()
                    task.set_pose(q, omega)
                    rows = []
                    identity = dict(
                        evaluation="balance",
                        family="com_aligned",
                        mode=mode,
                        seed=0,
                        duration=omega,
                        knee=angle,
                    )
                    for k in range(round(6.0 / task.m.opt.timestep)):
                        desired = q[task.qids].copy()
                        if mode == "gravity_pitch_feedback":
                            r = task.measure()
                            correction = (
                                -0.6 * (np.pi / 2 - r["pitch"])
                                + 0.07 * r["pitch_velocity"]
                            )
                            for i, name in enumerate(task.names):
                                if name.startswith(("FL", "FR")) and "thigh" in name:
                                    desired[i] += correction
                        desired = np.clip(desired, *task.m.jnt_range[task.jids].T)
                        row, contacts = task.sample(
                            desired,
                            "balance",
                            feedforward if mode == "gravity_pitch_feedback" else None,
                        )
                        row.update(identity)
                        rows.append(row.copy())
                        if contacts:
                            events.append(dict(**contacts[0], scale=scale, **identity))
                            break
                    balance.append(summarize(rows, task, **identity))
                    traces.extend(rows)
        print(f"balance {scale:.2f}", flush=True)
        # Write progress so long audits remain inspectable.
        save_csv(a.output / "takeoff.csv", attempts)
        save_csv(a.output / "balance.csv", balance)
        save_csv(a.output / "geometry.csv", geometry)
        save_csv(a.output / "path_checks.csv", checks)
        save_csv(a.output / "trajectories.csv", traces)
        write_json(a.output / "poses.json", poses)
        write_json(a.output / "first_contacts.json", events)
    write_json(
        a.output / "completion.json",
        dict(
            legacy_attempts=len(diagnosis),
            takeoff_attempts=len(attempts),
            balance_attempts=len(balance),
            extended_training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[1.0, 0.95, 0.90], choices=SCALES
    )
    parser.add_argument("--seeds", type=int, default=3)
    run(parser.parse_args())
