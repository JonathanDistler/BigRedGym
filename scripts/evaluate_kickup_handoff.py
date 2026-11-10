"""Frozen-policy recovery atlas, kick diagnostics and evaluation-only handoff.

No optimizer or checkpoint writes. All outputs go to a new experiment directory.
"""

import argparse
import csv
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
import gym.envs  # noqa: F401
from gym.envs.go2.frontyield_cpu import install_batched_cpu_sync
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.envs.go2.kickup_reference import KickReference
from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.original_cfg import (
    checkpoint_configs,
    _saved_env_imports,
    original_cfg_source_dir,
)
from gym.utils.task_registry import task_registry, select_backend
from gym.utils.robot_metrics import configure_contact_cpg
from gym.utils.torch_quat import quat_from_euler_xyz
from scripts.evaluate_frontyield import matched_reset
from scripts.search_kickup_reference import Experiment


def wilson(k, n):
    z = 1.96
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def blend_target(initial, desired, elapsed, duration=0.20):
    u = min(max(elapsed / duration, 0.0), 1.0)
    return initial.lerp(desired, u * u * (3 - 2 * u))


class Policy:
    def __init__(self, checkpoint, task, trials, legacy_cpg=False):
        self.path = Path(checkpoint)
        cfg, train = checkpoint_configs(checkpoint, task)
        if legacy_cpg:
            configure_contact_cpg(
                argparse.Namespace(
                    cpg=True, contact_modes=["stance", "stance", "swing", "swing"]
                ),
                cfg,
                train,
            )
        cfg.env.num_envs = trials
        cfg.env.episode_length_s = 12
        cfg.seed = 7
        cfg.push_robots.toggle = False
        if task == "go2balance":
            cfg.balance_perturbations.enabled = False
        train.runner.device = "cpu"
        train.runner.resume = False
        train.logging.enable_local_saving = False
        task_registry.convert_frequencies_to_params(cfg, train)
        task_registry.set_log_dir_name(train)
        set_seed(7)
        with _saved_env_imports(original_cfg_source_dir(self.path.parent)):
            module = importlib.import_module(f"gym.envs.go2.{task}")
            task_class = getattr(
                module, "Go2Balance" if task == "go2balance" else "Go2FrontYield"
            )
        self.env = task_class(
            cfg=cfg,
            device="cpu",
            headless=True,
            backend=select_backend(cfg, "cpu", "mujoco"),
        )
        install_batched_cpu_sync(self.env._backend)
        self.runner = task_registry.make_alg_runner(self.env, train)
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        self.runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        self.runner.switch_to_eval()
        self.runner.alg.actor.requires_grad_(False)
        self.level = (
            saved.get("environment_training_state", {})
            .get("handstand_curriculum", {})
            .get("level", 0)
        )
        self.env.curriculum_level = self.level
        self.meta = dict(
            checkpoint=str(self.path),
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            env=class_to_dict(cfg),
            actor=self.runner.actor_cfg,
            level=self.level,
        )
        self.reset()

    def reset(self):
        e = self.env
        e._episode_active.zero_()
        e._reset_idx(torch.ones(e.num_envs, dtype=torch.bool))
        e._reset_level[:] = self.level
        e.terminated.zero_()
        e.timed_out.zero_()

    def action(self, zero=False):
        r = self.runner
        r.set_actions(
            r.actor_cfg["actions"],
            torch.zeros(self.env.num_envs, 12) if zero else r.get_inference_actions(),
            False,
        )


def sync(e):
    e._backend._sync_state_from_mujoco()
    e.base_quat[:] = e.root_states[:, 3:7]
    e._refresh_base_observations(torch.ones(e.num_envs, dtype=torch.bool))
    e.dof_pos_obs[:] = e.dof_pos - e.default_dof_pos


def states(e):
    return [(d.qpos.copy(), d.qvel.copy()) for d in e._backend._datas]


def restore(e, samples):
    for i, (q, v) in enumerate(samples):
        d, m = e._backend._datas[i], e._backend._model_for_env(i)
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = q, v
        mujoco.mj_forward(m, d)
    sync(e)


def measurements(e):
    b = e._backend
    feet = b._canonical_to_native_body_np[e.feet_indices.numpy()]
    rows = []
    for i, d in enumerate(b._datas):
        m = b._model_for_env(i)
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        rot = d.xmat[base].reshape(3, 3)
        # Heading-invariant signed pitch. Positive is nose down.
        pitch = np.arctan2(-rot[2, 0], np.linalg.norm(rot[:2, 0]))
        # Use projected gravity to extend the pitch continuously past vertical.
        pitch = np.arctan2(-rot[2, 0], rot[2, 2])
        force = np.zeros((4, 3))
        moments = np.zeros(2)
        bodyhit = False
        wrench = np.zeros(6)
        for contact_id, c in enumerate(d.contact):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 != 0 and b2 != 0:
                continue
            body = b2 if b1 == 0 else b1
            if body in feet:
                mujoco.mj_contactForce(m, d, contact_id, wrench)
                world = c.frame.reshape(3, 3).T @ wrench[:3]
                applied = world if b1 == 0 else -world
                j = np.where(feet == body)[0][0]
                force[j] += applied
                moments[int(j >= 2)] += (
                    np.cross(c.pos - d.subtree_com[base], applied) @ rot[:, 1]
                )
            elif c.dist < -1e-5:
                bodyhit = True
        mujoco.mj_subtreeVel(m, d)
        com = d.subtree_com[base]
        support = d.xpos[feet[:2]].mean(axis=0)
        # Body y remains well-defined even at vertical.
        lateral = rot[:, 1]
        forward = np.cross(lateral, np.array([0.0, 0.0, 1.0]))
        forward /= max(np.linalg.norm(forward), 1e-8)
        row = dict(
            pitch_rad=float(pitch),
            pitch_rate=float(d.cvel[base, :3] @ lateral),
            angular_momentum=float(d.subtree_angmom[base] @ lateral),
            com_x=float(com @ forward),
            com_z=float(com[2]),
            com_support_x=float((com - support) @ forward),
            base_z=float(d.qpos[2]),
            body_contact=bool(bodyhit),
            front_pitch_moment_Nm=float(moments[0]),
            rear_pitch_moment_Nm=float(moments[1]),
            com_speed=float(np.linalg.norm(d.subtree_linvel[base])),
            front_drift=float(e._front_displacement()[i].max())
            if hasattr(e, "_front_anchor")
            else 0.0,
        )
        for j in range(4):
            row[f"foot_x_{j}"] = float(d.xpos[feet[j], 0])
            row[f"foot_y_{j}"] = float(d.xpos[feet[j], 1])
            row[f"force_{j}"] = float(max(force[j, 2], 0))
            row[f"clearance_{j}"] = float(d.xpos[feet[j], 2] - FOOT_RADIUS)
        for j, native in enumerate(b._canonical_to_native_dof_np):
            row[f"joint_{j}"] = float(d.qpos[7 + native])
            row[f"joint_velocity_{j}"] = float(d.qvel[6 + native])
        rows.append(row)
    return rows


def feature(row):
    # Dimensionless, explicitly scaled metric; not a probability of recovery.
    return np.array(
        [
            row["pitch_rad"] / 0.2,
            row["pitch_rate"],
            row["com_support_x"] / 0.05,
            row["base_z"] / 0.05,
        ]
        + [row[f"joint_{j}"] / 0.3 for j in range(12)]
    )


def dump(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        )
    )


def recovery(
    balance, samples, dest, label, start_targets=None, models=None, blend_seconds=0.20
):
    """Replay full native states, never project a trajectory sample into a pose."""
    balance.reset()
    e = balance.env
    if models is not None:
        b = e._backend
        b._models = [copy.copy(m) for m in models]
        b._mjm = b._models[0]
        b._datas = [mujoco.MjData(m) for m in b._models]
        for i, m in enumerate(b._models):
            b._link_mass_t[i] = torch.tensor(
                m.body_mass[b._canonical_to_native_body_np], dtype=torch.float32
            )
            b._contact_friction_t[i] = float(m.geom_friction[0, 0])
    restore(e, samples)
    e.handstand_stage[:] = 2
    ref = e._standing_posture()
    target = (
        e.dof_pos[:, e.actuated_dof_indices].clone()
        if start_targets is None
        else start_targets.clone()
    )
    e.dof_pos_target[:] = target - ref
    e.dof_pos_history[:] = e.dof_pos_target.repeat(1, 3)
    original = e._compute_torques
    last = target.clone()
    desired_target = target.clone()
    max_delta = 0.0

    def blended(this):
        nonlocal max_delta
        # Blend the switch only; persistent slew limits alter balance dynamics.
        applied = blend_target(
            target,
            desired_target,
            float(this.episode_length_buf[0]) * this.dt,
            blend_seconds,
        )
        delta = applied - last
        last.copy_(applied)
        max_delta = max(max_delta, float(delta.abs().max()))
        this.dof_pos_target[:] = last - this._standing_posture()
        return original()

    e._compute_torques = MethodType(blended, e)
    initial = measurements(e)
    longest = np.zeros(e.num_envs)
    hold = np.zeros(e.num_envs)
    failed = np.zeros(e.num_envs, dtype=bool)
    trace = []
    try:
        for step in range(round(6.5 / e.dt)):
            balance.action()
            desired_target.copy_(e._standing_posture() + e.dof_pos_target)
            e.step()
            rows = measurements(e)
            for i, row in enumerate(rows):
                failed[i] |= bool(e.terminated[i]) or row["body_contact"]
                weight = float(e._backend.link_mass[i].sum()) * 9.81
                valid = (
                    abs(row["pitch_rad"] - np.pi / 2) <= np.deg2rad(15)
                    and abs(row["pitch_rate"]) <= 1.5
                    and min(row["force_0"], row["force_1"]) >= 0.06 * weight
                    and max(row["force_2"], row["force_3"]) <= 0.015 * weight
                    and min(row["clearance_2"], row["clearance_3"]) >= 0.20
                    and not failed[i]
                )
                hold[i] = hold[i] + e.dt if valid else 0
                longest[i] = max(longest[i], hold[i])
                if step % 10 == 0:
                    trace.append(dict(trial=i, time=(step + 1) * e.dt, **row))
            if failed.all():
                break
    finally:
        e._compute_torques = original
    results = [
        dict(
            trial=i,
            success=bool(longest[i] >= 5 - 1e-6),
            hold_s=float(longest[i]),
            failed=bool(failed[i]),
            initial=initial[i],
        )
        for i in range(e.num_envs)
    ]
    k = sum(r["success"] for r in results)
    report = dict(
        label=label,
        trials=results,
        successes=k,
        count=e.num_envs,
        wilson95=wilson(k, e.num_envs),
        max_target_step_rad=max_delta,
        blend_seconds=blend_seconds,
    )
    dump(dest / f"{label}.json", report)
    with (dest / f"{label}.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=trace[0])
        w.writeheader()
        w.writerows(trace)
    print(
        f"recovery {label}: {k}/{e.num_envs}, best hold {max(longest):.2f}s", flush=True
    )
    return report


def kick(policy, ex, dest, name, zero=False, parameters=None):
    policy.reset()
    e = policy.env
    old = e.reference_parameters
    if parameters:
        e.reference_parameters = KickReference(**parameters)
    matched_reset(e, ex, list(range(3000, 3000 + e.num_envs)))
    traces = [[] for _ in range(e.num_envs)]
    snapshots = {}
    targets = {}
    control_samples = []
    original = e._post_physics_step

    def capture():
        original()
        t = float(e._kick_clock[0] + e._physics_offset)
        q, v = e._kick_reference(e._kick_clock + e._physics_offset)
        for i, row in enumerate(measurements(e)):
            for j in range(12):
                row[f"reference_{j}"] = float(q[i, j])
                row[f"reference_velocity_{j}"] = float(v[i, j])
                row[f"residual_{j}"] = float(e.dof_pos_target[i, j])
            traces[i].append(dict(time=t, **row))

    e._post_physics_step = capture
    try:
        for step in range(round(1.6 / e.dt)):
            policy.action(zero)
            e.step()
            t = round((step + 1) * e.dt, 5)
            q, _ = e._kick_reference(e._kick_clock + e._physics_offset)
            control_samples.append(
                (t, states(e), (q + e.dof_pos_target).clone(), measurements(e))
            )
            if any(abs(t - v) < e.dt / 2 for v in (0.90, 1.0, 1.1, 1.2, 1.35, 1.6)):
                snapshots[t] = states(e)
                q, _ = e._kick_reference(e._kick_clock + e._physics_offset)
                targets[t] = q + e.dof_pos_target
    finally:
        e._post_physics_step = original
        e.reference_parameters = old
    results = []
    for i, rows in enumerate(traces):
        push = [r for r in rows if 0.7 <= r["time"] <= 1.05]
        results.append(
            dict(
                trial=i,
                maximum_pitch_deg=float(np.rad2deg(max(r["pitch_rad"] for r in rows))),
                terminal_pitch_deg=float(np.rad2deg(rows[-1]["pitch_rad"])),
                terminal_pitch_velocity=rows[-1]["pitch_rate"],
                rear_clearance_m=max(
                    min(r["clearance_2"], r["clearance_3"]) for r in rows
                ),
                peak_angular_momentum=max(r["angular_momentum"] for r in rows),
                rear_impulse_Ns=sum(r["force_2"] + r["force_3"] for r in push)
                * e.cfg.sim_dt,
                front_impulse_Ns=sum(r["force_0"] + r["force_1"] for r in push)
                * e.cfg.sim_dt,
                body_contact=any(r["body_contact"] for r in rows),
                maximum_front_drift_m=max(r["front_drift"] for r in rows),
                front_support_fraction=sum(
                    min(r["force_0"], r["force_1"]) >= 0.06 * float(e._body_weight()[i])
                    for r in rows
                    if r["time"] >= 0.7
                )
                / sum(r["time"] >= 0.7 for r in rows),
            )
        )
        with (dest / f"{name}_{i}.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    dump(dest / f"{name}.json", results)
    np.savez_compressed(
        dest / f"{name}_states.npz",
        **{
            f"{t}_{i}_{kind}": value
            for t, samples in snapshots.items()
            for i, sample in enumerate(samples)
            for kind, value in zip(("qpos", "qvel"), sample)
        },
    )
    print(
        name,
        "mean peak pitch",
        np.mean([r["maximum_pitch_deg"] for r in results]),
        flush=True,
    )
    return dict(
        results=results,
        traces=traces,
        snapshots=snapshots,
        targets=targets,
        control_samples=control_samples,
        models=[copy.copy(e._backend._model_for_env(i)) for i in range(e.num_envs)],
    )


def supervise(balance, run, atlas, dest, name):
    """Causal gate evaluated at every control step, then replay exact switch states.

    Independent MuJoCo worlds make replay equivalent to pausing each selected
    world at its first gate crossing. No state interpolation or pose projection.
    Unknown regions deliberately produce no switch.
    """
    centers = [
        feature(r["initial"])
        for a in atlas
        if a["successes"] / a["count"] >= 0.75
        for r in a["trials"]
        if r["success"]
    ]
    n = balance.env.num_envs
    chosen = [None] * n
    supported = np.zeros(n, dtype=int)
    for t, samples, targets, rows in run["control_samples"]:
        for i, row in enumerate(rows):
            supported[i] = (
                supported[i] + 1 if min(row["force_0"], row["force_1"]) >= 8.0 else 0
            )
            near = (
                bool(centers)
                and min(np.linalg.norm(feature(row) - c) for c in centers) <= 0.75
            )
            gate = (
                near
                and supported[i] >= 3
                and not row["body_contact"]
                and np.deg2rad(70) <= row["pitch_rad"] <= np.deg2rad(110)
                and abs(row["pitch_rate"]) <= 1.5
            )
            if chosen[i] is None and gate:
                chosen[i] = (t, samples[i], targets[i])
    holds = [0.0] * n
    if any(x is not None for x in chosen):
        fallback = run["control_samples"][-1]
        samples = [x[1] if x else fallback[1][i] for i, x in enumerate(chosen)]
        targets = torch.stack(
            [x[2] if x else fallback[2][i] for i, x in enumerate(chosen)]
        )
        r = recovery(
            balance,
            samples,
            dest,
            f"{name}_supervised_recovery",
            targets,
            run["models"],
        )
        holds = [r["trials"][i]["hold_s"] if x else 0.0 for i, x in enumerate(chosen)]
    report = dict(
        controller=name,
        switches=sum(x is not None for x in chosen),
        full_transition_successes=sum(h >= 5 - 1e-6 for h in holds),
        trials=n,
        hold_s=holds,
        switch_times=[x[0] if x else None for x in chosen],
        gate=(
            "75% empirical region, scaled distance <=.75, 30ms front support, "
            "pitch 70–110deg, rate <=1.5rad/s"
        ),
    )
    dump(dest / f"{name}_supervisor.json", report)
    return report


def probes(balance, dest, specs=None):
    reports = []
    b = balance.env._backend
    order = b._canonical_to_native_dof_np
    feet = b._canonical_to_native_body_np[balance.env.feet_indices.numpy()]
    default_specs = [
        (p, v, 0.0, 0.0)
        for p in (30, 45, 60, 70, 80, 90, 100)
        for v in (-1.0, 0.0, 1.0)
    ]
    default_specs += [
        (90, 0.0, hip, height)
        for hip, height in ((-0.15, 0), (0.15, 0), (0, 0.02), (0, 0.05))
    ]
    for pitch, rate, hip, height in default_specs if specs is None else specs:
        balance.reset()
        for i, d in enumerate(b._datas):
            m = b._model_for_env(i)
            # Yaw-invariant sagittal probe, small independent joint perturbations.
            rng = np.random.default_rng(4000 + i)
            angles = torch.tensor(
                [
                    rng.normal(0, 0.004),
                    np.deg2rad(pitch) + rng.normal(0, 0.004),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
            quat = quat_from_euler_xyz(*angles).numpy()
            d.qpos[3:7] = quat[[3, 0, 1, 2]]
            d.qpos[7 + order] += rng.normal(0, 0.004, 12)
            d.qpos[7 + order[[1, 4]]] += hip
            d.qvel[:] = 0
            d.qvel[4] = rate + rng.normal(0, 0.02)
            mujoco.mj_forward(m, d)
            d.qpos[2] += FOOT_RADIUS + 0.001 + height - d.xpos[feet, 2].min()
            mujoco.mj_forward(m, d)
        label = f"pose_p{pitch}_v{rate}_hip{hip}_h{height}"
        report = recovery(balance, states(balance.env), dest, label)
        report["spec"] = dict(
            pitch=pitch, rate=rate, front_hip_offset=hip, height=height
        )
        reports.append(report)
        dump(dest / "atlas.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--kick-checkpoint", required=True)
    p.add_argument("--balance-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--skip-probes", action="store_true")
    p.add_argument("--atlas-path", type=Path)
    p.add_argument("--candidate-reference", type=Path)
    p.add_argument("--bridge-durations", type=float, nargs="*", default=[])
    p.add_argument(
        "--balance-cpg",
        action="store_true",
        help="Restore documented CLI overrides for older balance runs",
    )
    args = p.parse_args()
    if args.trials < 16:
        p.error("At least 16 trials are required")
    if args.bridge_durations and (
        not args.candidate_reference or min(args.bridge_durations) <= 0
    ):
        p.error("Positive bridge durations require --candidate-reference")
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    with torch.inference_mode():
        balance = Policy(
            args.balance_checkpoint, "go2balance", args.trials, args.balance_cpg
        )
        policy = Policy(args.kick_checkpoint, "go2frontyield", args.trials)
        bm, km = balance.env._backend._mjm, policy.env._backend._mjm
        for kind, count in (
            (mujoco.mjtObj.mjOBJ_JOINT, bm.njnt),
            (mujoco.mjtObj.mjOBJ_BODY, bm.nbody),
        ):
            assert [mujoco.mj_id2name(bm, kind, i) for i in range(count)] == [
                mujoco.mj_id2name(km, kind, i) for i in range(count)
            ], "Native state layouts differ"
        assert (bm.nq, bm.nv) == (km.nq, km.nv)
        dump(
            dest / "provenance.json",
            dict(
                balance=balance.meta,
                kick=policy.meta,
                hold_definition=(
                    "continuous 5s, pitch 75–105deg, rate <=1.5rad/s, "
                    "both front loads >=6%BW, rear loads <=1.5%BW, "
                    "clearance >=20cm, no fall/body contact; 6.5s horizon"
                ),
                recovery_physics=(
                    "pose probes use saved nominal balance model; "
                    "trajectory replays preserve kick native models and qpos/qvel"
                ),
                target_blend_seconds=0.20,
                training_launched=False,
            ),
        )
        ex = Experiment()
        data = {
            name: kick(policy, ex, dest, name, zero=zero)
            for name, zero in (("reference", True), ("existing", False))
        }
        if args.candidate_reference:
            parameters = json.loads(args.candidate_reference.read_text())
            dump(dest / "candidate_reference.json", parameters)
            data["improved_reference"] = kick(
                policy, ex, dest, "improved_reference", zero=True, parameters=parameters
            )
            data["improved_residual"] = kick(
                policy, ex, dest, "improved_residual", parameters=parameters
            )
        atlas = (
            json.loads(args.atlas_path.read_text())
            if args.atlas_path
            else ([] if args.skip_probes else probes(balance, dest))
        )
        if args.atlas_path:
            source = json.loads(
                (args.atlas_path.parent / "provenance.json").read_text()
            )
            source_balance = source.get("balance", source)
            if source_balance["sha256"] != balance.meta["sha256"]:
                raise ValueError(
                    "Recovery atlas belongs to a different balance checkpoint"
                )
            if any(a["count"] < 16 or len(a["trials"]) != a["count"] for a in atlas):
                raise ValueError("Recovery regions require at least 16 recorded trials")
            dump(dest / "atlas.json", atlas)
        recoveries = {}
        for name, run in data.items():
            recoveries[name] = []
            for t, samples in run["snapshots"].items():
                report = recovery(
                    balance,
                    samples,
                    dest,
                    f"{name}_switch_{t}",
                    run["targets"][t],
                    run["models"],
                )
                report["switch_time"] = t
                recoveries[name].append(report)
        successful = [
            feature(r["initial"]) for a in atlas for r in a["trials"] if r["success"]
        ]
        gaps = {}
        for name, run in data.items():
            gaps[name] = (
                [
                    min(
                        float(
                            np.linalg.norm(
                                np.asarray(successful) - feature(row), axis=1
                            ).min()
                        )
                        for row in rows
                    )
                    for rows in run["traces"]
                ]
                if successful
                else None
            )
        supervisors = {
            name: supervise(balance, run, atlas, dest, name)
            for name, run in data.items()
        }
        bridges = []
        if args.bridge_durations:
            run = data["improved_residual"]
            for t in (1.1, 1.2):
                for duration in args.bridge_durations:
                    bridges.append(
                        recovery(
                            balance,
                            run["snapshots"][t],
                            dest,
                            f"bridge_{t}_{duration}",
                            run["targets"][t],
                            run["models"],
                            duration,
                        )
                    )
        dump(
            dest / "summary.json",
            dict(
                comparison={k: v["results"] for k, v in data.items()},
                recovery=recoveries,
                supervisors=supervisors,
                bridges=bridges,
                distance_to_successful_samples=gaps,
                atlas_successes=[
                    dict(
                        label=a["label"],
                        successes=a["successes"],
                        count=a["count"],
                        wilson95=a["wilson95"],
                    )
                    for a in atlas
                ],
            ),
        )
        for obj in (balance, policy):
            assert (
                hashlib.sha256(obj.path.read_bytes()).hexdigest() == obj.meta["sha256"]
            )
            obj.env._backend.close()


if __name__ == "__main__":
    main()
