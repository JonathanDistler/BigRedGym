"""Simulation-only takeoff diagnosis, bounded residual BC/PPO and exact resume."""

import argparse
import copy
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

from experiments.handstand_robustness.core import Policy, sha256
from experiments.handstand_robustness.feedback import FeedbackPolicy
from experiments.handstand_robustness.takeoff import ResidualPolicy, TakeoffEnvironment
from gym.envs.go2.handstand_replay import build_model, load_trajectory


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def env(trajectory, seed=0, condition=None, horizon=48):
    return TakeoffEnvironment(build_model(), trajectory, seed, condition, horizon)


def run(
    trajectory,
    actor=None,
    seed=0,
    condition=None,
    horizon=48,
    feedback=True,
    collect=False,
):
    if getattr(actor, "checkpoint_kind", None) == "capture_residual_v1":
        from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

        e = CaptureTrainingEnvironment(
            build_model(), trajectory, seed, condition, horizon
        )
    else:
        e = env(trajectory, seed, condition, horizon)
    data = []
    for k in range(horizon):
        obs = e.observation()
        with torch.no_grad():
            if actor is None:
                action = trajectory["actions"][k]
            elif isinstance(actor, ResidualPolicy):
                action = actor(torch.tensor(obs)[None], feedback=feedback)[0].numpy()
            else:
                action = actor(torch.tensor(obs)[None])[0].numpy()
        _, _, done = e.step(action)
        if collect:
            data.append((obs, action.copy()))
        if done:
            break
    return e.result(), data


def evaluate(trajectory, actor, trials=4, horizon=48, feedback=True):
    if (
        getattr(actor, "checkpoint_kind", None) == "capture_residual_v1"
        and horizon == 48
    ):
        horizon = int(actor.active_steps)
    rows = []
    for i in range(trials + 1):
        condition = {} if i == 0 else {"joint_position": 0.003, "joint_velocity": 0.03}
        row, _ = run(trajectory, actor, 90110 + i, condition, horizon, feedback)
        rows.append(row | dict(seed=90110 + i, condition=condition))
    return dict(
        nominal=rows[0],
        heldout_successes=sum(
            r.get("objective_success", r["takeoff_success"]) for r in rows[1:]
        ),
        heldout_trials=trials,
        full_maneuver_successes=sum(r["complete_success"] for r in rows),
        rows=rows,
    )


def load(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["policy"]
    if saved.get("kind") == "capture_residual_v1":
        from experiments.handstand_robustness.capture import CapturePolicy

        balance = {
            key: state["balance_" + key].numpy()
            for key in ["weight", "bias", "lower", "upper"]
        }
        p = CapturePolicy(
            state["actions"],
            state["centers"],
            balance,
            int(state["active_steps"]),
            float(state["bound"]),
        )
    elif saved.get("kind") == "takeoff_residual_v1":
        p = ResidualPolicy(state["actions"], state["centers"], float(state["bound"]))
    else:
        p = Policy()
    p.load_state_dict(state)
    return p, saved


def save(path, policy, optimizer, update, args, history, best, stale=0, select=False):
    contents = dict(
        kind=getattr(policy, "checkpoint_kind", "takeoff_residual_v1"),
        schema=1,
        policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),
        update=update,
        torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        reference_sha256=args.trajectory_sha256,
        curriculum=dict(
            task="standing_capture"
            if hasattr(policy, "active_steps")
            else "standing_takeoff",
            prefix_steps=int(policy.active_steps)
            if hasattr(policy, "active_steps")
            else 48,
            level=0 if update <= 5 else 1,
            intermediate_starts=False,
        ),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        history=history,
        best=best,
        stale=stale,
    )
    if select:
        policy._best_payload = copy.deepcopy(contents)
    contents["best_payload"] = getattr(policy, "_best_payload", None)
    temp = path.with_suffix(".tmp")
    torch.save(contents, temp)
    os.replace(temp, path)


class BoundedTeacher:
    def __init__(self, teacher, bound=0.05):
        self.teacher, self.bound = teacher, bound

    def __call__(self, obs):
        k = int(round(float(obs[0, 42]) * 400))
        prior = self.teacher.actions[k]
        result = self.teacher(obs).numpy()
        return torch.tensor(
            np.clip(prior + np.clip(result - prior, -self.bound, self.bound), -1, 1)
        )


def clone(args, t):
    reference, data = run(t, collect=True, horizon=320)
    if not reference["complete_success"]:
        raise ValueError("Reference regression failed")
    centers = np.array([row[0] for row in data])
    p = ResidualPolicy(t["actions"], centers)
    teacher = BoundedTeacher(FeedbackPolicy(np.load(args.teacher, allow_pickle=False)))
    xs, ys, episodes = [], [], []
    for i in range(args.demonstrations):
        condition = {} if i == 0 else {"joint_position": 0.002, "joint_velocity": 0.02}
        row, demonstration = run(t, teacher, args.seed + i, condition, collect=True)
        episodes.append(row | dict(seed=args.seed + i, condition=condition))
        # A successful takeoff prefix is not a handstand/recovery demonstration.
        if row["takeoff_success"] and row["finite"]:
            xs.extend(x for x, y in demonstration)
            ys.extend(y for x, y in demonstration)
    if len(xs) < 96:
        raise ValueError("Insufficient validated corrective takeoff demonstrations")
    x = torch.tensor(np.array(xs))
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    optimizer = torch.optim.Adam([p.gains], lr=2e-3)
    curve = []
    for epoch in range(args.epochs):
        for idx in torch.randperm(len(x)).split(256):
            loss = (
                (p(x[idx]) - y[idx]) / p.bound
            ).square().mean() + 1e-6 * p.gains.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            curve.append(dict(epoch=epoch, loss=float(loss.detach())))
    # PPO has its own fresh optimizer; complete optimizer state is saved thereafter.
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-4)
    save(args.output / "bc.pt", p, optimizer, 0, args, [], None)
    np.savez(
        args.output / "demonstrations.npz",
        observations=np.array(xs),
        actions=np.array(ys),
    )
    report = dict(
        teacher_validated=sum(r["takeoff_success"] for r in episodes),
        teacher_trials=len(episodes),
        teacher_episodes=episodes,
        curve=curve,
        student=evaluate(t, p, args.trials),
        phase_prior_only=evaluate(t, p, args.trials, feedback=False),
        labels=(
            "Corrective bounded feedback rollouts, accepted only for safe "
            "certified takeoff prefixes"
        ),
    )
    return report


def diagnose(args, t):
    p, _ = load(args.checkpoint)
    nominal, data = run(t, collect=True, horizon=320)
    x = torch.tensor(np.array([r[0] for r in data]))
    y = np.array([r[1] for r in data])
    with torch.no_grad():
        prediction = p(x).numpy()
    result, _ = run(t, p)
    return dict(
        nominal=nominal,
        student=result,
        demonstration_mse=float(np.mean((prediction - y) ** 2)),
        first24_action_rms=float(np.sqrt(np.mean((prediction[:24] - y[:24]) ** 2))),
        actor_observation_channels=43,
        normalization_mean=p.mean.tolist(),
        normalization_std=p.std.tolist(),
        action_scale=2.0,
        control_hz=50,
        initial_latent_std=p.log_std.exp().detach().tolist(),
        coverage=dict(
            standing_takeoff_samples=48,
            total_samples=len(data),
            perturbed_correction_samples=0,
        ),
        inference=(
            "Imitation distribution shift; insufficient corrective coverage. "
            "PPO has not trained this checkpoint."
        ),
    )


def validate_capture_continuation(args):
    proof_path = getattr(args, "capture_continuation_proof", None)
    if not proof_path:
        raise ValueError(
            "Capture updates >50 require objective-improvement and Molab proof"
        )
    proof = json.loads(Path(proof_path).read_text())

    def checked(key):
        entry = proof[key]
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Capture proof file hash mismatch")
        return json.loads(path.read_text())

    paired = checked("paired_full_report")
    reference = paired["result"]["reference"]["rows"][1:]
    student = paired["result"]["student"]["rows"][1:]
    if len(student) < 20 or len(reference) != len(student):
        raise ValueError("At least 20 paired held-out trials are required")
    for a, b in zip(reference, student):
        if (a["seed"], a["condition"], a["horizon_seconds"]) != (
            b["seed"],
            b["condition"],
            b["horizon_seconds"],
        ):
            raise ValueError(
                "Capture proof requires identical held-out starts and horizons"
            )
    if sum(r["complete_success"] for r in student) <= sum(
        r["complete_success"] for r in reference
    ):
        raise ValueError("No objective improvement over complete phase prior")
    provenance = paired["provenance"]
    if (
        provenance["checkpoint_sha256"] != sha256(args.checkpoint)
        or provenance["reference_sha256"] != args.trajectory_sha256
    ):
        raise ValueError("Capture proof checkpoint/reference mismatch")
    benchmark = checked("molab_benchmark")["result"]
    receipt = checked("molab_resume_receipt")
    if not (
        proof.get("runtime_location") == "molab"
        and benchmark
        and all(r["transitions_per_second"] > 0 for r in benchmark)
        and receipt.get("exact_resume") is True
        and receipt.get("checkpoint_kind") == "capture_residual_v1"
        and receipt.get("reference_sha256") == args.trajectory_sha256
    ):
        raise ValueError(
            "Capture proof lacks measured Molab throughput or exact resume"
        )
    # Evidence hashes protect accidental mixing; runtime location is a declared
    # provenance field, not remote attestation or a security boundary.


def train(args, t):
    p, saved = load(args.checkpoint)
    if not isinstance(p, ResidualPolicy):
        raise ValueError("Takeoff PPO requires validated residual BC checkpoint")
    make_env = env
    if getattr(p, "checkpoint_kind", None) == "capture_residual_v1":
        from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

        def make_env(trajectory, seed, condition):
            return CaptureTrainingEnvironment(
                build_model(), trajectory, seed, condition, int(p.active_steps)
            )

    capture_task = getattr(p, "checkpoint_kind", None) == "capture_residual_v1"
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-5 if capture_task else 1e-4)
    update = 0
    history = []
    best = None
    stale = 0
    if args.resume:
        if saved["reference_sha256"] != args.trajectory_sha256:
            raise ValueError("Resume reference mismatch")
        for key in ["seed", "envs", "rollout_steps"]:
            if saved["config"][key] != getattr(args, key):
                raise ValueError(f"Exact resume requires unchanged {key}")
        optimizer.load_state_dict(saved["optimizer"])
        update = saved["update"]
        history = saved["history"]
        best = saved["best"]
        stale = saved["stale"]
        torch.set_rng_state(saved["torch_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        selected = saved.get("best_payload")
        source_best = args.checkpoint.parent / "best.pt"
        if selected is None and source_best.exists():
            selected = torch.load(source_best, map_location="cpu", weights_only=False)
            selected = {k: v for k, v in selected.items() if k != "best_payload"}
        if selected is not None:
            p._best_payload = copy.deepcopy(selected)
            destination = args.output / "best.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(selected, temporary)
            os.replace(temporary, destination)
        elif best is not None:
            print(
                "Previous selected weights unavailable; selection restarts explicitly"
            )
            best = None
    if capture_task and args.updates > 50:
        validate_capture_continuation(args)
    if args.updates > 5:
        smoke = saved.get("history", [])
        healthy = all(r.get("ppo_healthy", r["finite"]) for r in smoke)
        if capture_task:
            healthy &= all(
                abs(r.get("final_approx_kl", r["diagnostics"]["approx_kl"])) < 0.05
                for r in smoke
            )
        if saved["update"] < 2 or not smoke or not healthy:
            raise ValueError(
                "Pilot >5 updates requires resumed healthy simulator smoke"
            )
        health = evaluate(t, p, 2)
        if (
            not health["nominal"].get(
                "objective_success", health["nominal"]["takeoff_success"]
            )
            or health["heldout_successes"] == 0
        ):
            raise ValueError("Pilot blocked: no held-out standing takeoff skill")
    start_time = time.perf_counter()
    start_cpu = time.process_time()
    transitions = 0
    for update in range(update + 1, args.updates + 1):
        records = []
        episodes = []
        reward_totals = {}
        qnoise, vnoise = (0.001, 0.01) if update <= 5 else (0.002, 0.02)
        for world in range(args.envs):
            episode_id = 0
            e = make_env(
                t,
                args.seed + update * 1000 + world * 10,
                {"joint_position": qnoise, "joint_velocity": vnoise},
            )
            trace = []
            for step in range(args.rollout_steps):
                obs = torch.tensor(e.observation())
                with torch.no_grad():
                    distribution = p.distribution(obs)
                    latent = distribution.sample()
                    log = distribution.log_prob(latent).sum()
                    value = p.value(obs)
                    action = p.action(obs, latent).numpy()
                _, reward, done = e.step(action)
                if not e.finite:
                    raise ValueError("Nonfinite simulator transition")
                trace.append((obs, latent, log, value, reward, done))
                for key, val in e.last_terms.items():
                    reward_totals[key] = reward_totals.get(key, 0.0) + val
                if done:
                    episodes.append(e.result())
                    episode_id += 1
                    e = make_env(
                        t,
                        args.seed + update * 1000 + world * 10 + episode_id,
                        {"joint_position": qnoise, "joint_velocity": vnoise},
                    )
                transitions += 1
            with torch.no_grad():
                next_value = float(p.value(torch.tensor(e.observation())))
            gae = 0.0
            for obs, latent, log, value, reward, done in reversed(trace):
                mask = 1.0 - float(done)
                gae = (
                    reward
                    + 0.99 * mask * next_value
                    - float(value)
                    + 0.99 * 0.95 * mask * gae
                )
                records.append((obs, latent, log, gae, gae + float(value)))
                next_value = float(value)
        x = torch.stack([r[0] for r in records])
        a = torch.stack([r[1] for r in records])
        old = torch.stack([r[2] for r in records])
        adv = torch.tensor([r[3] for r in records])
        returns = torch.tensor([r[4] for r in records])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        diagnostics = []
        kl_stopped = False
        for epoch in range(4):
            for idx in torch.randperm(len(x)).split(128):
                dist = p.distribution(x[idx])
                log = dist.log_prob(a[idx]).sum(-1)
                log_ratio = log - old[idx]
                ratio = log_ratio.exp()
                estimated_kl = ((ratio - 1) - log_ratio).mean()
                if capture_task and float(estimated_kl.detach()) > 0.02:
                    kl_stopped = True
                    break
                actor_loss = -torch.minimum(
                    ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]
                ).mean()
                critic_loss = (p.value(x[idx]) - returns[idx]).square().mean()
                loss = actor_loss + 0.5 * critic_loss
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
                if not bool(torch.isfinite(loss) and torch.isfinite(norm)):
                    raise ValueError("Nonfinite PPO loss/gradient")
                optimizer.step()
                diagnostics.append(
                    dict(
                        policy_loss=float(actor_loss.detach()),
                        critic_loss=float(critic_loss.detach()),
                        gradient_norm=float(norm),
                        approx_kl=float((old[idx] - log).mean().detach()),
                        clip_fraction=float(
                            ((ratio - 1).abs() > 0.2).float().mean().detach()
                        ),
                    )
                )
            if kl_stopped:
                break
        with torch.no_grad():
            final_log = p.distribution(x).log_prob(a).sum(-1)
            log_ratio = final_log - old
            final_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
        validation = evaluate(t, p, args.trials)
        score = (
            int(
                validation["nominal"].get(
                    "objective_success", validation["nominal"]["takeoff_success"]
                )
            ),
            validation["heldout_successes"],
            -sum(r["max_pitch_deg"] < 10 for r in validation["rows"]),
        )
        improved = best is None or tuple(score) > tuple(best)
        if improved:
            best = score
        stale = (
            0
            if validation["nominal"].get(
                "objective_success", validation["nominal"]["takeoff_success"]
            )
            else stale + 1
        )
        elapsed = time.perf_counter() - start_time
        row = dict(
            update=update,
            transitions=args.envs * args.rollout_steps,
            simulator_transitions_total=(
                sum(r["transitions"] for r in history) + args.envs * args.rollout_steps
            ),
            training_takeoffs=sum(r["takeoff_success"] for r in episodes),
            training_captures=sum(r.get("capture_success", False) for r in episodes),
            final_approx_kl=final_kl,
            kl_early_stopped=kl_stopped,
            ppo_healthy=bool(
                np.isfinite(final_kl) and (not capture_task or final_kl < 0.05)
            ),
            completed_training_episodes=len(episodes),
            reward_components=reward_totals,
            diagnostics={
                key: float(np.mean([d[key] for d in diagnostics]))
                for key in diagnostics[0]
            },
            finite=True,
            validation=validation,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            end_to_end_training_transitions_per_second=transitions / elapsed,
            cpu_percent_one_core=100 * (time.process_time() - start_cpu) / elapsed,
            curriculum_level=0 if update <= 5 else 1,
        )
        history.append(row)
        if improved:
            save(
                args.output / "best.pt",
                p,
                optimizer,
                update,
                args,
                history,
                best,
                stale,
                select=True,
            )
        save(
            args.output / "latest.pt", p, optimizer, update, args, history, best, stale
        )
        save(
            args.output / f"update_{update:06d}.pt",
            p,
            optimizer,
            update,
            args,
            history,
            best,
            stale,
        )
        write(args.output / "curve.json", history)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in [
                        "update",
                        "training_takeoffs",
                        "completed_training_episodes",
                        "diagnostics",
                        "peak_rss_MiB",
                    ]
                }
            ),
            flush=True,
        )
        if capture_task and not row["ppo_healthy"]:
            print("STOP: capture PPO KL exceeded healthy update bound", flush=True)
            break
        if stale >= 2:
            print("STOP: nominal takeoff lost in two validations", flush=True)
            break
    return dict(history=history, update=update)


def benchmark(args, t):
    p, _ = load(args.checkpoint)
    make_env = env
    if getattr(p, "checkpoint_kind", None) == "capture_residual_v1":
        from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

        def make_env(trajectory, seed):
            return CaptureTrainingEnvironment(
                build_model(), trajectory, seed, horizon=int(p.active_steps)
            )

    rows = []
    for count in args.env_counts:
        for length in args.lengths:
            worlds = [make_env(t, args.seed + i) for i in range(count)]
            start = time.perf_counter()
            cpu = time.process_time()
            completed = 0
            for k in range(length):
                observations = torch.tensor(np.array([e.observation() for e in worlds]))
                with torch.no_grad():
                    actions = p(observations).numpy()
                for i, e in enumerate(worlds):
                    _, _, done = e.step(actions[i])
                    completed += 1
                    if done:
                        worlds[i] = make_env(t, args.seed + i + k)
            elapsed = time.perf_counter() - start
            row = dict(
                envs=count,
                rollout_steps=length,
                transitions=completed,
                seconds=elapsed,
                transitions_per_second=completed / elapsed,
                cpu_percent_one_core=100 * (time.process_time() - cpu) / elapsed,
                peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                backend="native_mujoco_cpu_serial_vector; no parallel-worker claim",
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
            del worlds
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "mode", choices=["diagnose", "bc", "train", "evaluate", "benchmark"]
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument(
        "--trajectory-sha256",
        default="21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--capture-continuation-proof", type=Path)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument(
        "--allow-long-takeoff-continuation",
        action="store_true",
        help="Explicit opt-in for a prepared 51–500 update continuation",
    )
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4110)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--demonstrations", type=int, default=16)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--full-maneuver", action="store_true")
    parser.add_argument("--env-counts", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128])
    args = parser.parse_args()
    if (
        not 1 <= args.updates <= 500
        or not 1 <= args.envs <= 64
        or not 48 <= args.rollout_steps <= 128
    ):
        parser.error("Use updates 1–500, serial envs 1–64, rollout steps 48–128")
    if args.updates > 50 and not args.allow_long_takeoff_continuation:
        parser.error(
            "Updates above 50 require explicit --allow-long-takeoff-continuation"
        )
    if any(n < 1 or n > 64 for n in args.env_counts) or any(
        n not in [64, 128] for n in args.lengths
    ):
        parser.error("Benchmark envs 1–64 and lengths 64 or 128")
    if args.mode == "bc" and not args.teacher:
        parser.error("BC requires --teacher validated feedback archive")
    if args.mode != "bc" and not args.checkpoint:
        parser.error("--checkpoint required")
    if args.trials < 1:
        parser.error("At least one held-out trial required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    t = load_trajectory(args.trajectory, args.trajectory_sha256, build_model())
    started = time.perf_counter()
    if args.mode == "diagnose":
        result = diagnose(args, t)
    elif args.mode == "bc":
        result = clone(args, t)
    elif args.mode == "train":
        result = train(args, t)
    elif args.mode == "benchmark":
        result = benchmark(args, t)
    else:
        p, _ = load(args.checkpoint)
        result = evaluate(t, p, args.trials, 320 if args.full_maneuver else 48)
    write(
        args.output / f"{args.mode}.json",
        dict(
            result=result,
            reference_sha256=sha256(args.trajectory),
            checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
            teacher_sha256=sha256(args.teacher) if args.teacher else None,
            source_sha256={
                str(path): sha256(path)
                for path in [
                    Path(__file__),
                    Path("experiments/handstand_robustness/takeoff.py"),
                    Path("experiments/handstand_robustness/core.py"),
                ]
            },
            configuration={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            seconds=time.perf_counter() - started,
            peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        ),
    )
    print(args.output / f"{args.mode}.json")


if __name__ == "__main__":
    main()
