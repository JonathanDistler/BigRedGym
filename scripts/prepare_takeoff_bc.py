"""Offline kick-up-teacher distillation into a CPG-free actor. No PPO training.

The frozen historical teacher is a data source only. The student sees the new
proprioceptive schema; CPG phase, reference, teacher action residual and controller
handoff state are never actor inputs. Closed-loop evaluation uses the student only.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from experiments.morphology.learning.environment import Assets, Environment
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import (
    TakeoffConfig,
    TakeoffEnvironment,
    inverse_target_action,
)
from experiments.morphology.task import metadata, write_json
from scripts.morphology_feasibility import save_csv


def capture_teacher(assets, output, count=12):
    from scripts.evaluate_kickup_handoff import Policy, matched_reset
    from scripts.search_kickup_reference import Experiment
    from gym.envs.go2.kickup_reference import KickReference
    from scripts.reproduce_transition_hold import KICK, REFERENCE

    checkpoint = Path(KICK)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    teacher = Policy(str(checkpoint), "go2frontyield", count)
    env = teacher.env
    reference = json.loads(Path(REFERENCE).read_text())
    teacher.reset()
    env.reference_parameters = KickReference(**reference)
    seeds = list(range(3000, 3000 + count))
    matched_reset(env, Experiment(), seeds)
    views = []
    for i in range(count):
        view = Environment(assets)
        view.m = env._backend._model_for_env(i)
        view.d = env._backend._datas[i]
        if [view.m.joint(j).name for j in range(1, view.m.njnt)] != view.names:
            raise ValueError("teacher/new-policy native joint ordering differs")
        view.mass = view.m.body_mass.sum()
        view.weight = view.mass * 9.81
        view.anchor = view.d.geom_xpos[view.foot_geoms[:2], :2].copy()
        views.append(view)
    source_frames = []
    summaries = []
    prefix_valid = np.ones(count, dtype=bool)
    maximum_pitch = np.zeros(count)
    maximum_clearance = np.zeros(count)
    saw_rear_loaded = np.zeros((count, 2), dtype=bool)
    source_lift = np.zeros(count, dtype=bool)
    compute = env._compute_torques
    counter = 0

    def capture():
        nonlocal counter
        torque = compute()
        for i, view in enumerate(views):
            row = view.measure()
            prefix_valid[i] &= not row["illegal_contacts"] and not row["self_contacts"]
            saw_rear_loaded[i] |= row["loads"][2:] / view.weight > 0.08
            good = (
                prefix_valid[i]
                and row["front_support"]
                and row["rear_clearance"] > 0.015
                and row["loads"][2:].sum() < 0.04 * view.weight
                and row["pitch"] > 0.12
            )
            source_lift[i] |= good and saw_rear_loaded[i].all()
            maximum_pitch[i] = max(
                maximum_pitch[i], row["pitch"] if prefix_valid[i] else 0.0
            )
            maximum_clearance[i] = max(
                maximum_clearance[i], row["rear_clearance"] if prefix_valid[i] else 0.0
            )
            if counter % round(0.02 / env.cfg.sim_dt) == 0:
                source_frames.append(
                    dict(
                        seed=seeds[i],
                        time=float(view.d.time),
                        qpos=view.d.qpos.copy(),
                        qvel=view.d.qvel.copy(),
                        torque=torque[i].detach().numpy().copy(),
                        prefix_valid=bool(prefix_valid[i]),
                        front_support=row["front_support"],
                        pitch=row["pitch"],
                        clearance=row["rear_clearance"],
                        anchor=view.anchor.copy(),
                        teacher_target=(env._standing_posture() + env.dof_pos_target)[i]
                        .detach()
                        .numpy()
                        .copy(),
                    )
                )
        counter += 1
        return torque

    env._compute_torques = capture
    try:
        with torch.inference_mode():
            for _ in range(round(1.4 / env.dt)):
                teacher.action()
                env.step()
    finally:
        env._compute_torques = compute
        env._backend.close()
    for i, seed in enumerate(seeds):
        summaries.append(
            dict(
                seed=seed,
                source_supported_lift=bool(source_lift[i]),
                max_pitch=maximum_pitch[i],
                max_clearance=maximum_clearance[i],
                collided_later=not bool(prefix_valid[i]),
            )
        )
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("original teacher checkpoint was modified")
    save_csv(output / "teacher_episodes.csv", summaries)
    write_json(
        output / "teacher_provenance.json",
        dict(
            checkpoint=str(checkpoint),
            sha256=digest,
            teacher_configuration=teacher.meta,
            seeds=seeds,
            reference=reference,
            scope="Historical kick-up prefix only; no claim of a five-second handstand",
            runtime_student_cpg=False,
        ),
    )
    np.savez_compressed(
        output / "teacher_native_frames.npz",
        qpos=np.stack([r["qpos"] for r in source_frames]),
        qvel=np.stack([r["qvel"] for r in source_frames]),
        torque=np.stack([r["torque"] for r in source_frames]),
        seed=np.array([r["seed"] for r in source_frames]),
        time=np.array([r["time"] for r in source_frames]),
    )
    return source_frames, summaries


def convert_frames(assets, frames, summaries, output):
    # Only a valid prefix from a source episode that actually lifts the rear feet.
    qualified = {r["seed"] for r in summaries if r["source_supported_lift"]}
    env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
    actors = []
    actions = []
    seeds = []
    times = []
    diagnostics = []
    for seed in sorted(qualified):
        records = sorted(
            [r for r in frames if r["seed"] == seed], key=lambda r: r["time"]
        )
        env.reset(1.0, seed)
        previous = env.nominal.copy()
        for frame in records:
            if not frame["prefix_valid"]:
                break
            env.d.qpos[:] = frame["qpos"]
            env.d.qvel[:] = frame["qvel"]
            env.d.time = frame["time"]
            mujoco.mj_forward(env.m, env.d)
            env.anchor = frame["anchor"].copy()
            env.target = previous.copy()
            env.steps = round(frame["time"] / env.cfg.dt)
            env.row = Environment.measure(env)
            # Torque-equivalent position avoids copying reference velocity feedforward
            # or different teacher gains into the student controller.
            equivalent = (
                env.d.qpos[env.qids]
                + (frame["torque"] + env.cfg.kd * env.d.qvel[env.vids]) / env.cfg.kp
            )
            action, projection = inverse_target_action(env, equivalent)
            actual_target = np.clip(
                env.nominal + env.cfg.action_scale * action, *env.bounds.T
            )
            actual_target = previous + np.clip(
                actual_target - previous,
                -0.7 * env.speed * env.cfg.dt,
                0.7 * env.speed * env.cfg.dt,
            )
            reproduced = np.clip(
                env.cfg.kp * (actual_target - env.d.qpos[env.qids])
                - env.cfg.kd * env.d.qvel[env.vids],
                -env.effort,
                env.effort,
            )
            error = float(np.max(abs(reproduced - frame["torque"])))
            actor, _ = env.observations()
            acceptable = (
                projection["action_projection_fraction"] == 0.0
                and error < 0.5
                and (env.d.qpos[env.qids] >= env.bounds[:, 0] - 0.001).all()
                and (env.d.qpos[env.qids] <= env.bounds[:, 1] + 0.001).all()
            )
            diagnostics.append(
                dict(
                    seed=seed,
                    time=frame["time"],
                    torque_error=error,
                    accepted=bool(acceptable),
                    **projection,
                )
            )
            # History follows the representable student command sequence, not a
            # hidden teacher residual or a CPG state.
            previous = actual_target.copy()
            if acceptable:
                actors.append(actor)
                actions.append(action)
                seeds.append(seed)
                times.append(frame["time"])
    save_csv(output / "conversion.csv", diagnostics)
    if not actors:
        raise RuntimeError("No representable collision-free demonstration frames")
    data = dict(
        observations=np.stack(actors),
        actions=np.stack(actions),
        seed=np.array(seeds),
        time=np.array(times),
    )
    np.savez_compressed(output / "bc_dataset.npz", **data)
    return data


def fit_and_evaluate(assets, data, output, updates):
    torch.manual_seed(42)
    net = ActorCritic(44, 74)
    optimizer = torch.optim.Adam(net.actor.parameters(), lr=3e-4)
    unique = np.unique(data["seed"])
    if len(unique) < 3:
        raise RuntimeError("Need episode-level train/validation split")
    heldout = set(unique[-max(2, len(unique) // 3) :])
    train = ~np.isin(data["seed"], list(heldout))
    validation = ~train
    x = torch.tensor(data["observations"])
    y = torch.tensor(data["actions"], dtype=torch.float32)
    targets = torch.atanh(y.clamp(-0.98, 0.98))
    train_ids = torch.tensor(np.flatnonzero(train))
    losses = []
    initial = {key: value.detach().clone() for key, value in net.state_dict().items()}
    for update in range(updates + 1):
        if update % 50 == 0 or update == updates:
            with torch.no_grad():
                prediction = net.actor(x).tanh()
                losses.append(
                    dict(
                        update=update,
                        train_mse=float((prediction[train] - y[train]).square().mean()),
                        validation_mse=float(
                            (prediction[validation] - y[validation]).square().mean()
                        ),
                    )
                )
        if update == updates:
            break
        ids = train_ids[torch.randint(len(train_ids), (min(128, len(train_ids)),))]
        loss = (net.actor(x[ids]) - targets[ids]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.actor.parameters(), 1.0)
        optimizer.step()
    save_csv(output / "bc_loss.csv", losses)
    # Actor-only initialization. The critic and Gaussian std are not cloned from
    # the reference-guided teacher and remain at their seeded initial values.
    torch.save(
        dict(
            actor=net.actor.state_dict(),
            source="historical_teacher_torque_equivalent",
            actor_dim=44,
            critic_dim=74,
            action_configuration=asdict(TakeoffConfig()),
            dataset_sha256=hashlib.sha256(
                (output / "bc_dataset.npz").read_bytes()
            ).hexdigest(),
            train_episode_seeds=unique[~np.isin(unique, list(heldout))].tolist(),
            validation_episode_seeds=sorted(int(s) for s in heldout),
        ),
        output / "bc_actor.pt",
    )
    if any(
        not torch.equal(initial[key], net.state_dict()[key])
        for key in initial
        if key.startswith("critic.") or key == "log_std"
    ):
        raise RuntimeError("BC changed critic or exploration parameters")
    results = []
    trace = []
    for scale in (1.0, 0.95, 0.9):
        env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
        for seed in (2000, 2001, 2002, 2003):
            actor, _ = env.reset(scale, seed)
            env.trace_enabled = True
            while env.steps * env.cfg.dt < 2.0:
                with torch.no_grad():
                    action = net.actor(torch.tensor(actor)).tanh().numpy()
                (actor, _), _, done, info = env.step(action)
                if done:
                    break
            results.append(dict(**info, evaluation_seed=seed))
            for row in env.physics_trace:
                trace.append(dict(**row, evaluation_seed=seed))
        print(f"CPG-free BC closed-loop evaluation scale={scale:.2f}", flush=True)
    save_csv(output / "bc_evaluation.csv", results)
    save_csv(output / "bc_physics.csv", trace)
    write_json(
        output / "bc_result.json",
        dict(
            offline_updates=updates,
            train_frames=int(train.sum()),
            validation_frames=int(validation.sum()),
            validation_episode_seeds=sorted(int(s) for s in heldout),
            validation_mse=losses[-1]["validation_mse"],
            genuine_takeoffs_by_scale={
                str(s): sum(r["takeoff"] for r in results if r["scale"] == s)
                for s in (1.0, 0.95, 0.9)
            },
            interpretation=(
                "Offline BC fit and measured closed-loop evaluation; "
                "fit alone cannot establish morphology transfer"
            ),
            ppo_steps=0,
            runtime_cpg=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    source.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/learning/takeoff.py"),
        Path("experiments/morphology/learning/environment.py"),
    ):
        shutil.copyfile(path, source / path.name)
    torch.set_num_threads(1)
    assets = Assets(args.output / "assets")
    write_json(
        args.output / "configuration.json",
        dict(**metadata(TakeoffConfig()), offline_bc=True, ppo_training=False),
    )
    frames, summaries = capture_teacher(assets, args.output)
    data = convert_frames(assets, frames, summaries, args.output)
    fit_and_evaluate(assets, data, args.output, args.updates)


if __name__ == "__main__":
    main()
