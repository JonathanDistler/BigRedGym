"""Build sparse forward-simulated rollouts, then independently label each reset."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.envs.go2.recovery_state import capture, sync
from gym.envs.go2.recovery_frontier import native_contact_flags, replay_batch
from gym.envs.go2.posture_transition import PostureTransition
from gym.utils.torch_quat import quat_from_euler_xyz
from scripts.evaluate_kickup_handoff import Policy, Experiment, dump
from scripts.optimize_posture_transition import bank, Evaluator, install, digest
from scripts.reproduce_transition_hold import BALANCE, KICK, REFERENCE

TIMES = (0.08, 0.45, 0.95, 1.65, 2.6, 3.4, 4.4)
PROTECTED = [
    BALANCE,
    KICK,
    REFERENCE,
    "output/handoff_cpg_comparison/atlas.json",
    "gym/envs/go2/references/posture_transition_adaptive.json",
    "gym/envs/go2/references/kickup_front_yield.json",
    "datasets/go2kickup_recovery_validation_v2/train.npz",
]


def save_bank(path, samples, labels):
    arrays = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    success = np.array([r["frozen"]["success"] for r in labels])
    near = (
        np.array(
            [
                r["frozen"]["hold_s"] >= 0.1
                or (
                    abs(r["frozen"]["initial"]["pitch_rad"] - np.pi / 2) < 0.3
                    and r["frozen"]["initial"]["com_speed"] < 0.35
                )
                for r in labels
            ]
        )
        & ~success
    )
    arrays.update(
        version=np.array(2),
        seed=np.array([r["source_seed"] for r in labels]),
        source_id=np.array([r["source_id"] for r in labels]),
        family=np.array([r["family"] for r in labels]),
        stratum=np.where(success, 0, np.where(near, 1, 2)),
        frozen_success=success,
        frozen_hold_s=np.array([r["frozen"]["hold_s"] for r in labels]),
    )
    # Stage 0 has demonstrated nominal recoveries; all later stages retain
    # established recoverable states and gradually admit frontier/handoff states.
    nominal = np.array([r["kind"] == "nominal" for r in labels])
    for stage in range(6):
        arrays[f"stage_{stage}"] = (
            success & nominal
            if stage == 0
            else success
            if stage == 1
            else success | near
            if stage == 2
            else np.ones(len(labels), dtype=bool)
        )
    if not arrays["stage_0"].any():
        raise RuntimeError("No demonstrated nominal recovery; refuse training bank")
    np.savez_compressed(path, **arrays)
    return arrays


@torch.inference_mode()
def collect_pose_rollouts(split, count, start, heading, velocity_angle=0.0):
    samples, labels, trajectories = [], [], []

    def add(e, i, source, time, kind):
        if native_contact_flags(e)[i] or e.root_states[i, 2] < 0.3:
            return
        samples.append(capture(e, i))
        labels.append(dict(**source, time=float(time), kind=kind))

    policy = Policy(BALANCE, "go2balance", count, True)
    e, b = policy.env, policy.env._backend
    policy.reset()
    sources = []
    for i, d in enumerate(b._datas):
        group = i % 4
        seed = start + i
        rng = np.random.default_rng(seed)
        sign = (-1) ** (i // 4)
        yaw = heading + rng.uniform(-0.15, 0.15)
        pitch_error = np.deg2rad([0.15, 0.8, 2.5, 6.0][group]) * sign
        rate = [0.0, 0.03, 0.1, 0.3][group] * (-sign)
        velocity = [0.0, 0.01, 0.035, 0.1][group] * sign
        angles = torch.tensor([rng.normal(0, 0.002), np.pi / 2 + pitch_error, yaw])
        d.qpos[3:7] = quat_from_euler_xyz(*angles).numpy()[[3, 0, 1, 2]]
        order = b._canonical_to_native_dof_np
        d.qpos[7 + order] += rng.normal(0, 0.002, 12)
        # Controlled front support asymmetry, never invented foot forces.
        d.qpos[7 + order[[1, 4]]] += (
            np.array([1, -1]) * [0.0, 0.002, 0.008, 0.025][group]
        )
        d.qvel[:] = 0
        d.qvel[:2] = velocity * np.array(
            [np.cos(yaw + velocity_angle), np.sin(yaw + velocity_angle)]
        )
        d.qvel[4] = rate
        m = b._model_for_env(i)
        mujoco.mj_forward(m, d)
        feet = b._canonical_to_native_body_np[e.feet_indices.numpy()]
        d.qpos[2] += (
            FOOT_RADIUS
            + 0.001
            + [0.0, 0.0, 0.005, 0.015][group]
            - d.xpos[feet, 2].min()
        )
        mujoco.mj_forward(m, d)
        source = dict(
            split=split,
            source_id=f"{split}_pose_{seed}",
            source_seed=seed,
            family=(
                f"{split}_pose_heading_{heading}"
                + (f"_velocity_angle_{velocity_angle}" if velocity_angle else "")
            ),
            perturbation=dict(
                pitch_error=pitch_error,
                pitch_rate=rate,
                velocity=velocity,
                heading=yaw,
                group=group,
            ),
        )
        sources.append(source)
        trajectories.append(source)
    sync(e)
    e.dof_pos_target[:] = e.dof_pos - e._standing_posture()
    e.dof_pos_history[:] = e.dof_pos_target.repeat(1, 3)
    collision = np.zeros(count, dtype=bool)
    old = e._post_physics_step

    def audit():
        old()
        collision[:] |= native_contact_flags(e)

    e._post_physics_step = audit
    try:
        for step in range(round(4.5 / e.dt)):
            t = round(step * e.dt, 5)
            if any(abs(t - v) < e.dt / 2 for v in (0.6, 1.4, 2.3)):
                for i, d in enumerate(b._datas):
                    group = sources[i]["perturbation"]["group"]
                    yaw = sources[i]["perturbation"]["heading"]
                    amplitude = [0.0, 0.01, 0.035, 0.075][group]
                    direction = 1 if t != 1.4 else -1
                    d.qvel[:2] += (
                        direction * amplitude * np.array([np.cos(yaw), np.sin(yaw)])
                    )
                    mujoco.mj_forward(b._model_for_env(i), d)
                sync(e)
            policy.action()
            e.step()
            time = round((step + 1) * e.dt, 5)
            if any(abs(time - v) < e.dt / 2 for v in TIMES):
                for i in range(count):
                    if not collision[i]:
                        group = sources[i]["perturbation"]["group"]
                        add(
                            e,
                            i,
                            sources[i],
                            time,
                            "nominal"
                            if group == 0
                            else "atlas_boundary"
                            if group >= 2
                            else "disturbed",
                        )
    finally:
        e._post_physics_step = old
        b.close()
    print(f"{split} pose rollouts done; {len(samples)} samples", flush=True)

    return samples, labels, trajectories


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    dest = a.output_dir
    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = {p: digest(p) for p in PROTECTED}
    dump(dest / "protected.json", hashes)
    samples, labels, trajectories = [], [], []

    def add(e, i, source, time, kind):
        if native_contact_flags(e)[i] or e.root_states[i, 2] < 0.3:
            return
        samples.append(capture(e, i))
        labels.append(dict(**source, time=float(time), kind=kind))

    for spec in (
        ("train", 16, 18000, 0.0),
        ("validation", 8, 19000, 0.6),
        ("heldout", 8, 20000, -1.1),
    ):
        extra_samples, extra_labels, extra_trajectories = collect_pose_rollouts(*spec)
        samples.extend(extra_samples)
        labels.extend(extra_labels)
        trajectories.extend(extra_trajectories)

    # Preserve the original 16-world layout for exact seed-6003 reproduction.
    atlas = json.loads(Path(PROTECTED[3]).read_text())
    for split, start, timing in (
        ("train", 6000, 0.0),
        ("validation", 24000, 0.025),
        ("heldout", 25000, -0.025),
    ):
        balance = Policy(BALANCE, "go2balance", 16, True)
        kick = Policy(KICK, "go2frontyield", 16)
        seeds = list(range(start, start + 16))
        data = bank(kick, Experiment(), seeds, "aggressive")
        sources = [
            dict(
                split=split,
                source_id=f"{split}_kick_{s}",
                source_seed=s,
                family=f"{split}_standing_timing_{timing}",
                perturbation=dict(bridge_timing_delta_s=timing),
            )
            for s in seeds
        ]
        trajectories.extend(sources)
        # Actual raw kick states are included in the difficult stratum.
        install(balance, data, list(range(16)), 1.1)
        for i in range(16):
            if not data["samples"][1.1]["collision"][i]:
                add(balance.env, i, sources[i], 1.1, "standing_kick")
        parameters = PostureTransition(**json.loads(Path(PROTECTED[4]).read_text()))
        parameters = replace(
            parameters,
            takeover_s=parameters.takeover_s + timing,
            switch_s=parameters.switch_s + timing,
        )

        def collect(e, time, handoff, failed):
            elapsed = time - handoff
            # Sparse continuation samples, at least 0.35 s apart.
            if any(
                abs(elapsed - v) <= e.dt / 2
                for v in (0.05, 0.45, 0.95, 1.65, 2.6, 3.4, 4.4)
            ):
                for i in range(16):
                    if not failed[i]:
                        add(e, i, sources[i], time, "transition")

        report = Evaluator(balance, atlas, True, True).evaluate(
            parameters,
            data,
            list(range(16)),
            dest / f"{split}_transition.csv",
            state_callback=collect,
        )
        dump(dest / f"{split}_transition.json", report)
        balance.env._backend.close()
        kick.env._backend.close()
        print(f"{split} transition done; {len(samples)} total samples", flush=True)

    # Label every snapshot by an independent complete frozen-controller replay.
    classifier = Policy(BALANCE, "go2balance", 16, True)
    try:
        for start in range(0, len(samples), 16):
            outcomes, _ = replay_batch(classifier, samples[start : start + 16])
            for i, outcome in enumerate(outcomes):
                labels[start + i]["frozen"] = outcome
            dump(
                dest / "label_progress.json",
                dict(completed=min(start + 16, len(samples)), total=len(samples)),
            )
            print(
                f"labelled {min(start + 16, len(samples))}/{len(samples)}; "
                f"{sum(r['success'] for r in outcomes)} recoveries in batch",
                flush=True,
            )
    finally:
        classifier.env._backend.close()
    valid = [i for i, r in enumerate(labels) if r["frozen"]["validity"] == "valid"]
    dump(
        dest / "excluded.json",
        [r for r in labels if r["frozen"]["validity"] != "valid"],
    )
    for split in ("train", "validation", "heldout"):
        ids = [i for i in valid if labels[i]["split"] == split]
        rows = [labels[i] for i in ids]
        arrays = save_bank(dest / f"{split}.npz", [samples[i] for i in ids], rows)
        dump(dest / f"{split}_labels.json", rows)
        print(
            split,
            len(ids),
            "states",
            len(set(arrays["source_id"])),
            "sources",
            int(arrays["frozen_success"].sum()),
            "recoverable",
            flush=True,
        )
    dump(dest / "trajectories.json", trajectories)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f"Protected input changed: {path}")
    dump(
        dest / "provenance.json",
        dict(
            protected_sha256=hashes,
            rollouts=len(trajectories),
            split_rule="Whole rollout and family; seed6003 family is train only.",
            saved_npz_sha256={
                s: digest(dest / f"{s}.npz") for s in ("train", "validation", "heldout")
            },
        ),
    )


if __name__ == "__main__":
    main()
