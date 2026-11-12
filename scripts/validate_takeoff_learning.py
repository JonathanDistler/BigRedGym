"""Audit PPO and validate takeoff reward and command authority without training."""

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import (
    TakeoffConfig,
    TakeoffEnvironment,
    inverse_target_action,
)
from experiments.morphology.task import metadata, write_json
from scripts.audit_morphology_dynamics import command, FEET
from scripts.morphology_feasibility import save_csv


AUTHORITY_CONFIGS = {
    "legacy": TakeoffConfig(
        action_scale=1.2, target_filter=0.25, terminate_on_goal=False
    ),
    "wider_filtered": TakeoffConfig(
        action_scale=1.6, target_filter=0.5, terminate_on_goal=False
    ),
    "direct_targets": TakeoffConfig(
        action_scale=2.0, target_filter=1.0, terminate_on_goal=False
    ),
}


def reference_action(env, crouch, duration, knee, previous):
    # The audit command generator is used only to test representability, never
    # installed as an inference reference in the learned actor.
    proxy = SimpleNamespace(
        t=env.steps * env.cfg.dt,
        names=env.names,
        qids=env.qids,
        jids=np.array([env.m.joint(n).id for n in env.names]),
        m=SimpleNamespace(
            opt=SimpleNamespace(timestep=env.cfg.dt), jnt_range=env.m.jnt_range
        ),
        measure=lambda: env.row,
        speed=env.speed,
    )
    desired, phase = command(
        proxy, env.assets.poses[env.scale], crouch, duration, knee, True, previous
    )
    action, error = inverse_target_action(env, desired)
    return action, desired, phase, error


def crouch_pose(assets, scale):
    from experiments.morphology.dynamics_audit import Audit

    # Reuse IK on the already compiled, unchanged model; no new model mutation.
    audit = object.__new__(Audit)
    audit.models = assets.models
    from experiments.morphology.task import Config

    audit.cfg = Config()
    audit.reset(scale, 0)
    q, error = audit.pose(0.27 * scale, feet=FEET, guess=assets.poses[scale])
    if error > 1e-6:
        raise RuntimeError("crouch IK failed")
    return q


def run_authority(assets, output, seeds):
    summaries = []
    traces = []
    for label, cfg in AUTHORITY_CONFIGS.items():
        env = TakeoffEnvironment(assets, cfg)
        for scale in (1.0, 0.95, 0.9):
            crouch = crouch_pose(assets, scale)
            for duration in (0.12, 0.24):
                for knee in (-1.8, -2.2):
                    for seed in seeds:
                        env.reset(scale, seed)
                        env.trace_enabled = True
                        previous = env.target.copy()
                        errors = []
                        for _ in range(90):
                            action, previous, phase, error = reference_action(
                                env, crouch, duration, knee, previous
                            )
                            errors.append(error)
                            _, _, done, info = env.step(action)
                            if done:
                                break
                        identity = dict(
                            configuration=label, duration=duration, knee=knee, seed=seed
                        )
                        for row in env.physics_trace:
                            traces.append(dict(**row, **identity))
                        summaries.append(
                            dict(
                                **info,
                                **identity,
                                max_target_error=max(e["target_error"] for e in errors),
                                projected_action_fraction=np.mean(
                                    [e["action_projection_fraction"] for e in errors]
                                ),
                                episode_reward_new=env.reward_sum,
                                **{
                                    f"episode_reward_{k}": v
                                    for k, v in env.terms_sum.items()
                                },
                            )
                        )
            print(f"authority {label} scale={scale:.2f}", flush=True)
    save_csv(output / "authority.csv", summaries)
    save_csv(output / "authority_physics.csv", traces)
    selected = [r for r in summaries if r["configuration"] == "direct_targets"]
    passed = all(
        any(
            r["scale"] == scale
            and r["takeoff"]
            and r["max_torque_ratio"] <= 1.0
            and r["max_speed_ratio"] <= 1.001
            and r["joint_limit_violation"] < 0.001
            for r in selected
        )
        for scale in (1.0, 0.95, 0.9)
    )
    return dict(
        passed=bool(passed),
        selected_configuration="direct_targets",
        successes_by_scale={
            str(s): sum(r["takeoff"] for r in selected if r["scale"] == s)
            for s in (1.0, 0.95, 0.9)
        },
        interpretation="Controller probes, not learned-policy performance",
    )


def audit_preserved(assets, root, output, seeds):
    summaries = []
    traces = []
    for run in sorted(root.glob("[AB]_seed*")):
        config = json.loads((run / "configuration.json").read_text())
        hp = config["protocol"]["ppo"]
        torch.manual_seed(0)
        net = ActorCritic(44, 74, hp["hidden_dims"], hp["initial_log_std"])
        checkpoint = run / "checkpoint_000024.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(saved["network"])
        net.eval()
        for scale in sorted(
            set(
                [
                    1.0,
                    saved["curriculum"]["protocol"]["scales"][
                        saved["curriculum"]["stage"]
                    ]
                    if config["method"] == "B"
                    else 1.0,
                ]
            )
        ):
            env = TakeoffEnvironment(
                assets,
                TakeoffConfig(
                    action_scale=1.2, target_filter=0.25, terminate_on_goal=False
                ),
            )
            for seed in seeds:
                actor, _ = env.reset(scale, seed)
                env.trace_enabled = True
                while True:
                    with torch.no_grad():
                        action = net.actor(torch.tensor(actor)).tanh().numpy()
                    (actor, _), _, done, info = env.step(action)
                    if done:
                        break
                identity = dict(
                    method=config["method"],
                    training_seed=config["training_seed"],
                    evaluation_seed=seed,
                )
                for row in env.physics_trace:
                    traces.append(dict(**row, **identity))
                trace = env.physics_trace
                rear = env.knee_indices
                result = dict(
                    **info,
                    **identity,
                    **{f"new_reward_{k}": v for k, v in env.terms_sum.items()},
                    rear_knee_positive_work=sum(
                        max(r[f"torque_{i}"] * r[f"velocity_{i}"], 0.0) * 0.002
                        for r in trace
                        for i in rear
                    ),
                    rear_knee_peak_extension_speed=max(
                        max(r[f"velocity_{i}"] for i in rear) for r in trace
                    ),
                    min_rear_load_fraction=min(
                        (r["load_RL"] + r["load_RR"]) / env.weight for r in trace
                    ),
                    checkpoint_sha256=hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                )
                summaries.append(result)
        print(f"preserved policy {run.name} audited", flush=True)
    save_csv(output / "preserved_policy_audit.csv", summaries)
    save_csv(output / "preserved_policy_physics.csv", traces)
    original = pd.concat(
        [pd.read_csv(run / "evaluation.csv") for run in root.glob("[AB]_seed*")]
    )
    final = original[
        (original.iteration == 24) & (original.scale == 1.0) & ~original.perturbation
    ]
    reward = final.groupby(["method", "training_seed"])[
        [c for c in final if c.startswith("episode_reward_")]
    ].mean()
    reward.to_csv(output / "original_reward_components.csv")
    return dict(
        replayed_checkpoints=4,
        replayed_episodes=len(summaries),
        genuine_takeoffs=sum(r["takeoff"] for r in summaries),
        reward_shortcut="Grounded pitch/support income without rear-foot lift",
    )


def revalidate_bc(assets, path, output):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(44, 74)
    net.actor.load_state_dict(saved["actor"])
    net.eval()
    results = []
    for scale in (1.0, 0.95, 0.9):
        env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
        for seed in (2000, 2001, 2002, 2003):
            actor, _ = env.reset(scale, seed)
            for _ in range(100):
                with torch.no_grad():
                    action = net.actor(torch.tensor(actor)).tanh().numpy()
                (actor, _), _, done, info = env.step(action)
                if done:
                    break
            results.append(dict(**info, evaluation_seed=seed))
    save_csv(output / "bc_revalidation.csv", results)
    write_json(
        output / "bc_revalidation.json",
        dict(
            actor_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            successes_by_scale={
                str(s): sum(r["takeoff"] for r in results if r["scale"] == s)
                for s in (1.0, 0.95, 0.9)
            },
            explanation="Student-only rollout; active-push certificate; no teacher",
        ),
    )


def negative_controls():
    from experiments.morphology.learning.takeoff import TakeoffTracker

    outcomes = []
    for name in (
        "standing",
        "lean",
        "single_rear_lift",
        "rear_fold_without_push",
        "flight_without_front_support",
        "passive_rear_stretch",
    ):
        tracker = TakeoffTracker()
        total = 0.0
        for i in range(500):
            feet = np.array(
                [
                    [0, 0.14, 0.022],
                    [0, -0.14, 0.022],
                    [-0.4, 0.14, 0.022],
                    [-0.4, -0.14, 0.022],
                ]
            )
            loads = np.ones(4) * 40
            pitch = 0.35 if name != "standing" else 0.0
            if i > 100:
                if name == "single_rear_lift":
                    feet[2, 2] = 0.07
                    loads[2] = 0
                if name in (
                    "rear_fold_without_push",
                    "flight_without_front_support",
                    "passive_rear_stretch",
                ):
                    feet[2:, 2] = 0.07
                    loads[2:] = 0
            row = dict(
                feet=feet,
                loads=loads,
                weight=160.0,
                front_support=name != "flight_without_front_support",
                front_drift=0.0,
                pitch=pitch,
                rear_clearance=float(feet[2:, 2].min() - 0.022),
            )
            terms = tracker.observe(
                row,
                i * 0.002,
                np.ones(2) if name == "passive_rear_stretch" else np.zeros(2),
                True,
            )
            total += sum(terms.values())
        outcomes.append(
            dict(control=name, positive_takeoff_reward=total, takeoff=tracker.confirmed)
        )
    return outcomes


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--previous-run",
        type=Path,
        default=Path("output/physical_morphology_ppo_90_v1"),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--bc-actor", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = a.output / "source"
    sources.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/learning/takeoff.py"),
        Path("experiments/morphology/learning/environment.py"),
        Path("experiments/morphology/learning/ppo.py"),
    ):
        shutil.copyfile(path, sources / path.name)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    write_json(
        a.output / "configuration.json",
        dict(
            **metadata(TakeoffConfig()),
            seeds=a.seeds,
            previous_run=str(a.previous_run),
            authority_configurations={
                k: asdict(v) for k, v in AUTHORITY_CONFIGS.items()
            },
            extended_training=False,
        ),
    )
    controls = negative_controls()
    write_json(a.output / "negative_controls.json", controls)
    old = audit_preserved(assets, a.previous_run, a.output, a.seeds)
    authority = run_authority(assets, a.output, a.seeds)
    if a.bc_actor:
        revalidate_bc(assets, a.bc_actor, a.output)
    write_json(
        a.output / "validation.json",
        dict(
            reward_semantic_sha256=hashlib.sha256(
                ast.dump(
                    ast.parse((sources / "takeoff.py").read_text()),
                    include_attributes=False,
                ).encode()
            ).hexdigest(),
            reward_negative_controls_passed=all(
                not r["takeoff"] and r["positive_takeoff_reward"] == 0 for r in controls
            ),
            action_authority=authority,
            previous_policy_audit=old,
            extended_training_launched=False,
            scope="Reward and authority checks only; PPO takeoff learning is untested",
        ),
    )


if __name__ == "__main__":
    main()
