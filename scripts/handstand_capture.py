"""Matched capture diagnostics, local viability probes and prefix validation."""

import argparse
import json
from pathlib import Path
import sys
import time
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from experiments.handstand_robustness.capture import CaptureEnvironment, Controller
from gym.envs.go2.handstand_replay import build_model, load_trajectory, STATE_SPEC


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rollout(
    t,
    controller,
    seed=91000,
    condition=None,
    steps=115,
    trace=True,
    diagnostic_tail=False,
):
    e = CaptureEnvironment(build_model(), t, seed, condition, trace)
    for k in range(steps):
        _, _, done = e.step(controller(e))
        if done and not diagnostic_tail:
            break
    return e


def compare(args, t):
    rows = {}
    for name in ["reference", "prior", "hybrid", "student"]:
        c = Controller(
            name,
            t,
            args.checkpoint if name in ["prior", "student"] else None,
            args.balance,
        )
        rows[name] = []
        for i in range(args.trials + 1):
            condition = (
                {} if i == 0 else {"joint_position": 0.003, "joint_velocity": 0.03}
            )
            e = rollout(t, c, 91000 + i, condition, diagnostic_tail=True)
            row = e.report() | dict(
                seed=91000 + i,
                condition=condition,
                diagnostic_tail=True,
                scope=(
                    "native physics logged through 2.3s; failed at first "
                    "safety violation"
                ),
            )
            rows[name].append(row)
            if i == 0:
                write(
                    args.output / f"{name}_trace.json",
                    dict(
                        result=row,
                        physics=e.physics,
                        contacts=e.contacts,
                        controls=e.controls,
                    ),
                )
            print(
                name,
                i,
                e.safe,
                e.first_illegal and e.first_illegal["time_s"],
                flush=True,
            )
    return rows


def plots(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    for name in ["reference", "prior", "hybrid", "student"]:
        d = json.loads((args.output / f"{name}_trace.json").read_text())
        rows = d["physics"]
        ts = np.array([r["time_s"] for r in rows])
        p = np.array([r["com_position_m"] for r in rows])
        v = np.array([r["com_velocity_m_s"] for r in rows])
        curves = [
            ([r["pitch_deg"] for r in rows], "Pitch (deg)"),
            ([r["pitch_rate_rad_s"] for r in rows], "Pitch rate (rad/s)"),
            (p[:, 2], "COM height (m)"),
            (v[:, 0], "COM vx (m/s)"),
            (
                [sum(np.array(r["foot_forces_N"])[:2, 2]) for r in rows],
                "Front force z (N)",
            ),
            ([min(r["calf_clearance_m"][2:]) for r in rows], "Rear calf clearance (m)"),
            (
                [
                    max(abs(np.array(r["torque_Nm"])) / r["effort_limit_Nm"])
                    for r in rows
                ],
                "Peak torque / cap",
            ),
            ([r["action_error_max"] for r in d["controls"]], "Action max deviation"),
        ]
        for ax, (values, label) in zip(axes.flat, curves):
            times = (
                ts if len(values) == len(ts) else [r["time_s"] for r in d["controls"]]
            )
            ax.plot(times, values, label=name)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.axvline(0.96, color="k", linestyle=":", alpha=0.3)
            ax.set_xlim(0.8, 2.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Continuous simulation time (s)")
    axes[-1, 1].set_xlabel("Continuous simulation time (s)")
    fig.tight_layout()
    fig.savefig(args.output / "capture_comparison.png", dpi=160)
    plt.close(fig)


def viability(args, t):
    cases = [("nominal", 0.0, 0)]
    cases += [("pitch", v, 0) for v in [-0.003, -0.001, 0.001, 0.003]]
    cases += [("roll", v, 0) for v in [-0.001, 0.001]]
    cases += [("angular_velocity", v, 0) for v in [-0.05, -0.01, 0.01, 0.05]]
    cases += [("joint_position", v, 0) for v in [-0.002, -0.0005, 0.0005, 0.002]]
    cases += [("contact_gap", v, 0) for v in [-0.0005, -0.0002, 0.0002, 0.0005]]
    cases += [("phase_offset", 0.0, v) for v in [-2, -1, 1, 2]]
    rows = []
    for kind, value, offset in cases:
        e = CaptureEnvironment(build_model(), t, trace=False)
        for k in range(64):
            e.step(t["actions"][k])
        if kind in ["pitch", "roll"]:
            delta = np.zeros(e.m.nv)
            delta[4 if kind == "pitch" else 3] = value
            mujoco.mj_integratePos(e.m, e.d.qpos, delta, 1.0)
        elif kind == "angular_velocity":
            e.d.qvel[4] += value
        elif kind == "joint_position":
            e.d.qpos[e.qids] += np.random.default_rng(11864).uniform(
                -abs(value), abs(value), 12
            ) * np.sign(value)
        elif kind == "contact_gap":
            e.d.qpos[2] += value
        mujoco.mj_forward(e.m, e.d)
        invalid = any(
            c.dist < -0.003
            or (
                c.dist < 0
                and any(
                    b and b not in e.feet for b in e.m.geom_bodyid[[c.geom1, c.geom2]]
                )
            )
            for c in e.d.contact
        )
        if invalid:
            rows.append(
                dict(kind=kind, value=value, offset_steps=offset, invalid_reset=True)
            )
            continue
        safe22 = None
        while e.index < len(t["actions"]):
            k = int(np.clip(e.index + offset, 0, len(t["actions"]) - 1))
            _, _, done = e.step(t["actions"][k])
            if e.index == 110:
                safe22 = e.safe
            if done:
                break
        row = e.report()
        row.update(
            kind=kind,
            value=value,
            offset_steps=offset,
            local_five_second_strict=row["complete_success"],
            complete_success=False,
            collision_free_through_2_2=bool(safe22),
            starting_condition=(
                "explicit 1.28s snapshot perturbation; NOT full standing-start success"
            ),
            reference_deadline_s=6.4,
            invalid_reset=False,
        )
        rows.append(row)
        print(kind, value, offset, e.safe, e.best_hold, flush=True)
    return rows


def rescue(args, t):
    actor = Controller("student", t, args.checkpoint, args.balance)
    rows = []
    for switch in [48, 50, 52, 55]:
        e = CaptureEnvironment(build_model(), t, trace=False)
        for k in range(320):
            u = actor(e) if k < switch else t["actions"][k]
            _, _, done = e.step(u)
            if done:
                break
        rows.append(
            e.report()
            | dict(
                switch_time_s=switch * 0.02,
                protocol=(
                    "continuous scheduled diagnostic switch to reference; "
                    "not autonomous success"
                ),
            )
        )
        print("switch", switch * 0.02, e.safe, e.best_hold, flush=True)
    return rows


def centers(t):
    from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

    e = CaptureTrainingEnvironment(build_model(), t, horizon=320)
    observations = []
    error = 0.0
    state = np.empty(mujoco.mj_stateSize(e.m, STATE_SPEC))
    for k, u in enumerate(t["actions"]):
        observations.append(e.observation())
        e.step(u)
        mujoco.mj_getState(e.m, e.d, state, STATE_SPEC)
        error = max(error, float(abs(state - t["reference_states"][k + 1]).max()))
    if not e.result()["complete_success"] or error > 1e-10:
        raise ValueError("Sensor observer must not change exact reference dynamics")
    return np.array(observations)


def capture_bc(args, t):
    from experiments.handstand_robustness.capture import (
        CaptureTrainingEnvironment,
        CapturePolicy,
        SensorTeacher,
    )
    from scripts.handstand_robustness import legacy_teacher
    from scripts.handstand_takeoff import save, evaluate

    reference = centers(t)
    f = np.load(args.feedback, allow_pickle=False)
    b = legacy_teacher(args.balance)
    teacher = SensorTeacher(t, f, args.balance, reference)
    actor = CapturePolicy(t["actions"], reference, b, active_steps=110)
    # Analytical derivative transfer from the validated feedback law, not fake
    # labels on impossible states. BC uses only actual successful rollouts below.
    with torch.no_grad():
        for k in range(110):
            jac = np.zeros((12, 46))
            if k < 69:
                jac[:, :42] = -f["gains"][k]
            else:
                jac[:, :30] = b["weight"][:, :30]
                jac[:, 42:] = b["weight"][:, 30:]
            actor.gains[k].copy_(
                torch.tensor(
                    jac * actor.scale.numpy()[None, :] / float(actor.bound),
                    dtype=torch.float32,
                )
            )
    xs = []
    ys = []
    episodes = []
    for i in range(args.demonstrations):
        cond = {} if i == 0 else {"joint_position": 0.002, "joint_velocity": 0.02}
        e = CaptureTrainingEnvironment(build_model(), t, 121000 + i, cond, horizon=110)
        samples = []
        for k in range(110):
            x = e.observation()
            u = teacher(e)
            samples.append((x, u.copy()))
            _, _, done = e.step(u)
            if done:
                break
        row = e.result()
        episodes.append(row | dict(seed=121000 + i, condition=cond))
        if row["capture_success"]:
            xs.extend(x for x, u in samples)
            ys.extend(u for x, u in samples)
        print("teacher", i, row["capture_success"], e.index, flush=True)
    if len(xs) < 220:
        raise ValueError("No adequately validated capture demonstrations")
    x = torch.tensor(np.array(xs))
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    baseline = evaluate(t, actor, args.trials)
    optimizer = torch.optim.Adam([actor.gains], lr=1e-4)
    curve = []
    for epoch in range(args.epochs):
        for ids in torch.randperm(len(x)).split(256):
            loss = ((actor(x[ids]) - y[ids]) / actor.bound).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            curve.append(dict(epoch=epoch, loss=float(loss.detach())))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    args.trajectory_sha256 = (
        "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
    )
    save(args.output / "bc.pt", actor, optimizer, 0, args, [], None)
    np.savez(
        args.output / "demonstrations.npz",
        observations=np.array(xs),
        actions=np.array(ys),
    )
    return dict(
        teacher_trials=len(episodes),
        teacher_successes=sum(r["capture_success"] for r in episodes),
        episodes=episodes,
        initial_transfer=baseline,
        curve=curve,
        student=evaluate(t, actor, args.trials),
    )


def prefix_eval(args, t):
    from experiments.handstand_robustness.capture import CaptureTrainingEnvironment
    from scripts.handstand_takeoff import load

    actor = load(args.checkpoint)[0]
    rows = []
    for name in args.controllers:
        controller = (
            Controller(
                name,
                t,
                args.baseline_checkpoint if name == "prior" else None,
                args.balance,
            )
            if name != "student"
            else None
        )
        for steps in [63, 75, 88, 100, 110]:
            for i in range(args.trials + 1):
                cond = (
                    {} if i == 0 else {"joint_position": 0.003, "joint_velocity": 0.03}
                )
                e = CaptureTrainingEnvironment(
                    build_model(), t, 141000 + i, cond, horizon=steps
                )
                for k in range(steps):
                    if name == "student":
                        with torch.no_grad():
                            u = actor(torch.tensor(e.observation())[None])[0].numpy()
                    else:
                        u = controller(e)
                    _, _, done = e.step(u)
                    if done:
                        break
                row = e.report()
                rows.append(
                    row
                    | dict(
                        controller=name,
                        seed=141000 + i,
                        nominal=i == 0,
                        prefix_seconds=steps * 0.02,
                    )
                )
                print(
                    "prefix",
                    name,
                    steps,
                    i,
                    row["prefix_success"],
                    row["capture_success"],
                    flush=True,
                )
    return rows


def full_eval(args, t):
    from experiments.handstand_robustness.capture import (
        CaptureTrainingEnvironment,
        SensorTeacher,
    )
    from scripts.handstand_takeoff import load

    selected = load(args.checkpoint)[0]
    result = {}
    reference = centers(t) if "teacher" in args.controllers else None
    teacher = (
        SensorTeacher(t, np.load(args.feedback), args.balance, reference)
        if reference is not None
        else None
    )
    for name in args.controllers:
        c = (
            Controller(
                name,
                t,
                args.baseline_checkpoint if name in ["prior"] else None,
                args.balance,
            )
            if name not in ["student", "teacher"]
            else None
        )
        rows = []
        for i in range(args.trials + 1):
            cond = {} if i == 0 else {"joint_position": 0.003, "joint_velocity": 0.03}
            e = CaptureTrainingEnvironment(
                build_model(), t, args.evaluation_seed + i, cond, horizon=400
            )
            for k in range(400):
                if name == "student":
                    with torch.no_grad():
                        u = selected(torch.tensor(e.observation())[None])[0].numpy()
                elif name == "teacher":
                    u = teacher(e)
                else:
                    u = c(e)
                _, _, done = e.step(u)
                if done:
                    break
            row = e.report() | dict(
                seed=args.evaluation_seed + i,
                condition=cond,
                horizon_seconds=8.0,
                standing_start=True,
                controller=name,
                privileged_actor=name == "hybrid",
            )
            rows.append(row)
            write(
                args.output / ("partial_" + name + ".json"),
                dict(controller=name, rows=rows, partial=True),
            )
            print(
                "full",
                name,
                i,
                row["complete_success"],
                row["strict_hold_seconds"],
                flush=True,
            )
        result[name] = dict(
            nominal=rows[0],
            successes=sum(r["complete_success"] for r in rows),
            trials=len(rows),
            heldout_successes=sum(r["complete_success"] for r in rows[1:]),
            heldout_trials=len(rows) - 1,
            rows=rows,
        )
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "mode",
        choices=["compare", "plot", "viability", "rescue", "bc", "prefix", "full"],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path("artifacts/takeoff_latest.pt")
            if Path("artifacts/takeoff_latest.pt").exists()
            else Path("output/takeoff_portable/latest.pt")
        ),
    )
    p.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=(
            Path("artifacts/takeoff_latest.pt")
            if Path("artifacts/takeoff_latest.pt").exists()
            else Path("output/takeoff_portable/latest.pt")
        ),
    )
    p.add_argument(
        "--balance",
        type=Path,
        default=(
            Path("artifacts/balance.pt")
            if Path("artifacts/balance.pt").exists()
            else Path(
                "/home/jonathandistler/Distler_Gym/BigRedGym/output/retained_go2_preparation_v3/balance.pt"
            )
        ),
    )
    p.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    p.add_argument("--trials", type=int, default=2)
    p.add_argument(
        "--feedback",
        type=Path,
        default=(
            Path("artifacts/takeoff_teacher.npz")
            if Path("artifacts/takeoff_teacher.npz").exists()
            else Path("output/feedback_corrected/controller.npz")
        ),
    )
    p.add_argument("--demonstrations", type=int, default=12)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=4110)
    p.add_argument("--evaluation-seed", type=int, default=201000)
    p.add_argument(
        "--controllers", nargs="+", default=["reference", "prior", "hybrid", "student"]
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.mode == "plot":
        plots(args)
        return
    t = load_trajectory(
        args.trajectory,
        "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92",
        build_model(),
    )
    from experiments.handstand_robustness.core import sha256

    provenance = dict(
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        source_sha256={
            str(path): sha256(path)
            for path in [
                Path(__file__),
                Path("experiments/handstand_robustness/capture.py"),
                Path("experiments/handstand_robustness/core.py"),
                Path("scripts/handstand_takeoff.py"),
            ]
        },
        reference_sha256=sha256(args.trajectory),
        checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint.exists() else None,
        balance_sha256=sha256(args.balance),
        feedback_sha256=sha256(args.feedback),
    )
    for filename in provenance["source_sha256"]:
        source_path = Path(filename)
        relative = (
            source_path.relative_to(Path.cwd())
            if source_path.is_absolute()
            else source_path
        )
        target = args.output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_path.read_bytes())
    started = time.perf_counter()
    result = (
        {
            "compare": compare,
            "viability": viability,
            "rescue": rescue,
            "bc": capture_bc,
            "prefix": prefix_eval,
            "full": full_eval,
        }[args.mode]
    )(args, t)
    write(
        args.output / (args.mode + ".json"),
        dict(
            result=result, seconds=time.perf_counter() - started, provenance=provenance
        ),
    )
    if args.mode == "compare":
        plots(args)


if __name__ == "__main__":
    main()
