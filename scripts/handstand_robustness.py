"""Bounded simulation-only audit, imitation, PPO and held-out evaluation CLI."""

import argparse
import json
import os
from pathlib import Path
import random
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from experiments.handstand_robustness.core import Environment, Policy, sha256
from gym.envs.go2.handstand_replay import build_model, load_trajectory

MILESTONES = {50, 250, 500, 1000, 2500, 5000}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_env(args, trajectory, seed=0, condition=None):
    return Environment(build_model(), trajectory, seed, condition)


def rollout(
    args, trajectory, policy=None, condition=None, seed=0, teacher=None, feedback=True
):
    env = make_env(args, trajectory, seed, condition)
    observations, labels = [], []
    frozen = env.observation()
    start = time.perf_counter()
    for k, reference in enumerate(trajectory["actions"]):
        obs = env.observation()
        if policy is not None:
            with torch.no_grad():
                action = policy(torch.tensor(obs if feedback else frozen)[None])[
                    0
                ].numpy()
        elif teacher is not None and k >= 69:
            # Validated 1.32s prefix, 0.05048s balance delay rounded by original loop.
            features = env.balance_features()
            action = np.clip(
                teacher["weight"] @ features[teacher["channels"]] + teacher["bias"],
                teacher["lower"],
                teacher["upper"],
            )
            t = k * 0.02 - 1.32
            duration = 0.64517335
            amp = max(0, min(1, t / 0.10, (duration - t) / 0.15))
            action[[1, 4]] += -0.0204364 * amp
            action[[2, 5]] += 0.10256115 * amp
        else:
            action = reference
        observations.append(obs)
        labels.append(np.clip(action, -1, 1))
        _, _, done = env.step(action)
        if done:
            break
    return (
        env.result()
        | dict(seconds=time.perf_counter() - start, control_steps=env.index),
        np.array(observations),
        np.array(labels),
    )


def legacy_teacher(path):
    state = torch.load(path, map_location="cpu", weights_only=False)["network"]
    return dict(
        weight=state["feedback.weight"].numpy(),
        bias=state["feedback.bias"].numpy(),
        channels=state["channels"].numpy(),
        lower=state["lower"].numpy(),
        upper=state["upper"].numpy(),
    )


def cases():
    result = {"nominal": {}}
    result.update(
        joint_position={"joint_position": 0.002},
        joint_velocity={"joint_velocity": 0.02},
        body_offset={"orientation": 0.002, "position": 0.002},
        friction={"friction": 0.9},
        mass={"mass": 1.05},
        mass_recompute_control={"recompute_constants": True},
        strength={"strength": 0.95},
        latency={"latency": 1},
    )
    for phase, start in [
        ("takeoff", 0.6),
        ("capture", 0.688),
        ("height_stabilization", 1.32),
        ("hold", 3.0),
    ]:
        for direction, wrench in [
            ("forward", [10, 0, 0, 0, 0, 0]),
            ("backward", [-10, 0, 0, 0, 0, 0]),
            ("lateral", [0, 10, 0, 0, 0, 0]),
            ("rotation", [0, 0, 0, 0, 1, 0]),
        ]:
            result[f"{phase}_{direction}"] = {
                "push": dict(
                    start=start,
                    duration=0.1,
                    wrench=wrench,
                    frame="world",
                    point="base_COM",
                )
            }
    return result


def evaluate(args, trajectory, policy=None, teacher=None):
    report = {}
    for name, condition in cases().items():
        if args.conditions and name not in args.conditions.split(","):
            continue
        trials = []
        for i in range(args.trials):
            try:
                r, _, _ = rollout(
                    args, trajectory, policy, condition, args.seed + 10000 + i, teacher
                )
            except ValueError as exc:
                r = dict(invalid_reset=str(exc), complete_success=False)
            trials.append(r)
        n = sum("invalid_reset" not in r for r in trials)
        s = sum(r["complete_success"] for r in trials)
        # Wilson 95% binomial interval; rejected resets do not count as rollouts.
        p = s / n if n else 0
        z = 1.96
        center = (p + z * z / (2 * n)) / (1 + z * z / n) if n else 0
        half = (
            z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            if n
            else 0
        )
        report[name] = dict(
            condition=condition,
            successes=s,
            trials=n,
            rejected_resets=len(trials) - n,
            wilson95=[center - half, center + half],
            results=trials,
        )
        print(name, s, "/", n, flush=True)
    return report


def checkpoint(path, policy, optimizer, update, args, extra=None):
    payload = dict(
        schema=1,
        policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),
        update=update,
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        torch_rng=torch.get_rng_state(),
        config=vars(args)
        | {
            "output": str(args.output),
            "trajectory": str(args.trajectory),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "teacher": str(args.teacher) if args.teacher else None,
            "feedback": str(args.feedback) if args.feedback else None,
        },
        curriculum=dict(stage="A", randomized=False),
        extra=extra,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def behavior_clone(args, trajectory):
    result, obs, labels = rollout(args, trajectory)
    if args.feedback:
        from experiments.handstand_robustness.feedback import FeedbackPolicy

        feedback = FeedbackPolicy(np.load(args.feedback, allow_pickle=False))
        collected_obs, collected_labels, episodes = [], [], []
        for i in range(8):
            condition = (
                {} if i == 0 else {"joint_position": 0.0002, "joint_velocity": 0.002}
            )
            row, xx, yy = rollout(args, trajectory, feedback, condition, args.seed + i)
            episodes.append(row)
            if row["complete_success"]:
                collected_obs.append(xx)
                collected_labels.append(yy)
        save_json(args.output / "feedback_collection.json", episodes)
        if not collected_obs:
            raise ValueError(
                "No verified successful feedback demonstrations; BC aborted"
            )
        obs, labels = np.concatenate(collected_obs), np.concatenate(collected_labels)
    np.savez(args.output / "demonstration.npz", observations=obs, actions=labels)
    policy = Policy()
    policy.mean.copy_(torch.tensor(obs.mean(0)))
    policy.std.copy_(torch.tensor(obs.std(0).clip(0.05)))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    x, y = torch.tensor(obs), torch.tensor(labels, dtype=torch.float32)
    for epoch in range(args.epochs):
        indices = torch.randperm(len(x))
        for chunk in indices.split(128):
            loss = (policy(x[chunk]) - y[chunk]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    live, _, _ = rollout(args, trajectory, policy)
    frozen, _, _ = rollout(args, trajectory, policy, feedback=False)
    with torch.no_grad():
        changed = x.clone()
        changed[:, 3] += 0.1
        sensitivity = float((policy(changed) - policy(x)).abs().mean())
    checkpoint(args.output / "bc.pt", policy, optimizer, 0, args)
    return dict(
        demonstration=result,
        final_mse=float((policy(x) - y).square().mean().detach()),
        nominal_live=live,
        frozen_observation=frozen,
        gyro_action_sensitivity=sensitivity,
        limitation=(
            "One nominal demonstration; no fabricated recovery labels. "
            "Sensitivity is not recovery proof."
        ),
    )


def train(args, trajectory):
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    start = 0
    history, best, stale = [], -float("inf"), 0
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(saved["policy"])
        if args.resume:
            if saved["config"]["trajectory_sha256"] != args.trajectory_sha256:
                raise ValueError("Resume reference hash mismatch")
            for key in ("seed", "envs", "rollout_steps"):
                if saved["config"][key] != getattr(args, key):
                    raise ValueError(f"Exact resume requires unchanged {key}")
            optimizer.load_state_dict(saved["optimizer"])
            start = saved["update"]
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            torch.set_rng_state(saved["torch_rng"])
            extra = saved.get("extra") or {}
            history, best, stale = (
                extra.get("history", []),
                extra.get("best", best),
                extra.get("stale", 0),
            )
    # Checkpoints occur at update boundaries. Fresh episodes each update make exact
    # continuation independent of unsaved simulator state.
    update = start
    for update in range(start + 1, args.updates + 1):
        records = []
        episode_returns = []
        for i in range(args.envs):
            env = make_env(args, trajectory, args.seed + update * args.envs + i)
            episode = []
            for k in range(len(trajectory["actions"])):
                obs = torch.tensor(env.observation())
                with torch.no_grad():
                    dist = policy.distribution(obs)
                    latent = dist.sample()
                    old_log = dist.log_prob(latent).sum()
                    value = policy.value(obs)
                _, reward, done = env.step(latent.tanh().numpy())
                episode.append([obs, latent, old_log, value, reward])
                if done:
                    break
            episode_returns.append(sum(row[4] for row in episode))
            gae, next_value = 0.0, 0.0
            for obs, latent, log, value, reward in reversed(episode):
                gae = reward + 0.99 * next_value - value.item() + 0.99 * 0.95 * gae
                records.append((obs, latent, log, gae, gae + value.item()))
                next_value = value.item()
        x = torch.stack([r[0] for r in records])
        actions = torch.stack([r[1] for r in records])
        logs = torch.stack([r[2] for r in records])
        advantages = torch.tensor([r[3] for r in records])
        returns = torch.tensor([r[4] for r in records])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(x)).split(args.rollout_steps):
                dist = policy.distribution(x[idx])
                ratio = (dist.log_prob(actions[idx]).sum(-1) - logs[idx]).exp()
                actor_loss = -torch.minimum(
                    ratio * advantages[idx], ratio.clamp(0.8, 1.2) * advantages[idx]
                ).mean()
                loss = (
                    actor_loss
                    + 0.5 * (policy.value(x[idx]) - returns[idx]).square().mean()
                    - 0.001 * dist.entropy().sum(-1).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        if (
            update % args.eval_every == 0
            or update in MILESTONES
            or update == args.updates
        ):
            validation, _, _ = rollout(args, trajectory, policy, seed=args.seed + 20000)
            score = (
                int(validation["complete_success"]),
                validation["strict_hold_seconds"],
            )
            scalar_score = score[0] * 100 + score[1]
            improved = scalar_score > best
            stale = 0 if improved else stale + 1
            best = max(best, scalar_score)
            history.append(
                dict(
                    update=update,
                    transitions=len(records),
                    mean_episode_reward=float(np.mean(episode_returns)),
                    optimizer_loss=float(loss.detach()),
                    validation=validation,
                )
            )
            print(json.dumps(history[-1]), flush=True)
            extra = dict(history=history, best=best, stale=stale)
            checkpoint(
                args.output / "latest.pt", policy, optimizer, update, args, extra
            )
            if improved:
                checkpoint(
                    args.output / "best.pt", policy, optimizer, update, args, extra
                )
            if update in MILESTONES:
                checkpoint(
                    args.output / f"update_{update:06d}.pt",
                    policy,
                    optimizer,
                    update,
                    args,
                    extra,
                )
            save_json(args.output / "curve.json", history)
            if stale >= args.patience:
                print("Early stop: strict validation did not improve", flush=True)
                break
    return dict(history=history, cumulative_update=update)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "mode",
        choices=["evaluate", "bc", "train", "benchmark", "teacher", "build-feedback"],
    )
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--trajectory-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--teacher", type=Path)
    source.add_argument("--feedback", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=4110)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--conditions", help="Optional comma-separated evaluation cases")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--envs", type=int, default=1)
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=128,
        help="PPO minibatch size; rollouts are complete episodes",
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.envs <= 64 or not 1 <= args.updates <= 5000:
        p.error("Use 1–64 serial environments and 1–5000 cumulative updates")
    if args.conditions and set(args.conditions.split(",")) - set(cases()):
        p.error("Unknown evaluation condition")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    code_paths = [
        Path(__file__),
        Path("experiments/handstand_robustness/core.py"),
        Path("experiments/handstand_robustness/feedback.py"),
        Path("gym/envs/go2/handstand_replay.py"),
    ]
    provenance = dict(
        source_sha256={str(path): sha256(path) for path in code_paths},
        configuration={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    model = build_model()
    trajectory = load_trajectory(args.trajectory, args.trajectory_sha256, model)
    policy = None
    if args.checkpoint:
        policy = Policy()
        policy.load_state_dict(
            torch.load(args.checkpoint, weights_only=False, map_location="cpu")[
                "policy"
            ]
        )
        policy.eval()
    if args.feedback and args.mode == "train":
        p.error("PPO accepts neural checkpoints only")
    if args.feedback:
        from experiments.handstand_robustness.feedback import FeedbackPolicy

        policy = FeedbackPolicy(np.load(args.feedback, allow_pickle=False))
    if args.mode == "build-feedback":
        from experiments.handstand_robustness.feedback import build

        gains = build(model, trajectory)
        np.savez(args.output / "controller.npz", **gains)
        result = dict(projection_residual_max=float(gains["projection_residual"].max()))
    elif args.mode == "train":
        if args.updates > 10:
            if policy is None:
                p.error("Long training requires a demonstrated checkpoint")
            health, _, _ = rollout(args, trajectory, policy)
            if not health["complete_success"]:
                p.error(
                    "Long training blocked: checkpoint lacks strict nominal success"
                )
        result = train(args, trajectory)
    elif args.mode == "bc":
        result = behavior_clone(args, trajectory)
    elif args.mode == "teacher":
        if not args.teacher:
            p.error("--teacher required")
        result = rollout(args, trajectory, teacher=legacy_teacher(args.teacher))[0]
    elif args.mode == "benchmark":
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        result = rollout(args, trajectory, policy)[0]
        result["cpu_percent_one_core"] = (
            100 * (time.process_time() - cpu_start) / (time.perf_counter() - wall_start)
        )
        result["transitions_per_second"] = result["control_steps"] / result["seconds"]
        result["peak_rss_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        result["backend"] = "native MuJoCo CPU, serial; includes strict 500Hz scoring"
    else:
        result = evaluate(
            args,
            trajectory,
            policy,
            legacy_teacher(args.teacher) if args.teacher else None,
        )
    save_json(
        args.output / f"{args.mode}.json",
        dict(
            result=result,
            reference_sha256=sha256(args.trajectory),
            checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
            seed=args.seed,
            torch_version=torch.__version__,
            provenance=provenance,
            teacher_sha256=sha256(args.teacher) if args.teacher else None,
            feedback_sha256=sha256(args.feedback) if args.feedback else None,
        ),
    )
    if args.mode == "evaluate":
        print(args.output / "evaluate.json")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
