"""Matched A/full vs B/physical curriculum PPO. No CPG/posture curriculum."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from experiments.morphology.learning.environment import (
    Assets,
    Environment,
    LearningConfig,
)
from experiments.morphology.learning.ppo import (
    ActorCritic,
    PerformanceCurriculum,
    advantages,
    update,
)
from experiments.morphology.task import metadata, write_json
from scripts.morphology_feasibility import save_csv


def protocol(path):
    value = json.loads(Path(path).read_text())
    if "inherits" in value:
        base = json.loads((Path(path).parent / value["inherits"]).read_text())
        base.update(value)
        value = base
    if value["steps_per_iteration"] % value["num_envs"]:
        raise ValueError("rollout must divide evenly across environments")
    return value


def evaluate(
    net,
    env,
    seeds,
    scale,
    actual_steps,
    method,
    training_seed,
    iteration,
    perturb=False,
):
    episodes = []
    traces = []
    steps = 0
    for seed in seeds:
        actor, critic = env.reset(scale, seed)
        disturbed = False
        while True:
            if perturb and env.steps * env.cfg.dt >= 2.0 and not disturbed:
                env.d.qvel[4] += 0.3
                disturbed = True
            with torch.no_grad():
                action = net.actor(torch.tensor(actor)).tanh().numpy()
            (actor, critic), _, done, info = env.step(action)
            steps += 1
            record = dict(
                **info,
                method=method,
                training_seed=training_seed,
                evaluation_seed=seed,
                actual_steps=actual_steps,
                iteration=iteration,
                perturbation=perturb,
                perturbation_applied=disturbed,
            )
            traces.append(record)
            if done:
                episodes.append(
                    dict(
                        record,
                        **{f"episode_reward_{k}": v for k, v in env.terms_sum.items()},
                    )
                )
                break
    return episodes, traces, steps


class Run:
    def __init__(self, directory, method, seed, p, assets):
        self.directory = Path(directory)
        self.method = method
        self.seed = seed
        self.p = p
        self.directory.mkdir(parents=True, exist_ok=False)
        self.source = self.directory / "source"
        self.source.mkdir()
        for file in (
            Path(__file__),
            Path("experiments/morphology/learning/environment.py"),
            Path("experiments/morphology/learning/ppo.py"),
            Path("gym/envs/go2/leg_morphology.py"),
            Path("experiments/morphology/dynamics_audit.py"),
            Path("experiments/morphology/task.py"),
            Path("scripts/audit_morphology_dynamics.py"),
            Path("scripts/morphology_feasibility.py"),
        ):
            shutil.copyfile(file, self.source / file.name)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = ActorCritic(
            Environment.actor_dim,
            Environment.critic_dim,
            p["ppo"]["hidden_dims"],
            p["ppo"]["initial_log_std"],
            log_std_min=p["ppo"].get("log_std_min", -3.0),
        )
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=p["ppo"]["learning_rate"]
        )
        self.schedule = PerformanceCurriculum(method, p)
        self.cfg = LearningConfig(gamma=p["ppo"]["gamma"])
        self.envs = [Environment(assets, self.cfg) for _ in range(p["num_envs"])]
        self.eval_env = Environment(assets, self.cfg)
        self.actors = []
        self.critics = []
        for i, e in enumerate(self.envs):
            # Per-world reset streams avoid dependence on other worlds.
            a, c = e.reset(self.schedule.scale, self.episode_seed(i, 0))
            self.actors.append(a)
            self.critics.append(c)
        self.episode_counts = [0] * len(self.envs)
        self.actors = np.stack(self.actors)
        self.critics = np.stack(self.critics)
        self.actual_steps = self.iteration = self.evaluation_steps = 0
        self.full_scale_steps = 0
        self.initial_parameters = self.net.actor[0].weight.detach().clone()
        self.training = []
        self.episodes = []
        self.evaluations = []
        self.torch_state = torch.get_rng_state()
        self.start = time.monotonic()
        config = dict(
            **metadata(self.cfg),
            protocol=p,
            method=method,
            training_seed=seed,
            actor_observation_dim=Environment.actor_dim,
            critic_observation_dim=Environment.critic_dim,
            actor_observations="proprioception, targets, time, physical scale",
            critic_privileged="base velocity/height, COM, feet, loads, anchor drift",
            finite_horizon=True,
            timeout_bootstrap=False,
            reward_version="supported_kickup_v1_identical_all_morphologies",
            asset_hashes={
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(assets.directory).glob("*.urdf")
            }
            if hasattr(assets, "directory")
            else {},
        )
        write_json(self.directory / "configuration.json", config)

    def episode_seed(self, world, count):
        return int(
            np.random.SeedSequence([self.seed, world, count]).generate_state(1)[0]
        )

    def collect_and_update(self):
        torch.set_rng_state(self.torch_state)
        p = self.p
        hp = p["ppo"]
        n = len(self.envs)
        t = p["steps_per_iteration"] // n
        obs = []
        priv = []
        raws = []
        logs = []
        values = []
        rewards = []
        dones = []
        nextvals = []
        scale_steps = {s: 0 for s in p["scales"]}
        for _ in range(t):
            actor = torch.tensor(self.actors)
            critic = torch.tensor(self.critics)
            with torch.no_grad():
                raw, log, value = self.net.sample(actor, critic)
            obs.append(self.actors.copy())
            priv.append(self.critics.copy())
            raws.append(raw.numpy())
            logs.append(log.numpy())
            values.append(value.numpy())
            reward_step = []
            done_step = []
            next_actors = []
            next_critics = []
            terminal_critics = []
            for i, e in enumerate(self.envs):
                scale = e.scale
                (a, c), reward, done, info = e.step(raw[i].tanh().numpy())
                self.actual_steps += 1
                scale_steps[scale] = scale_steps.get(scale, 0) + 1
                self.full_scale_steps += int(scale == 1.0)
                reward_step.append(reward)
                done_step.append(done)
                terminal_critics.append(c)
                if done:
                    record = dict(
                        **info,
                        actual_steps=self.actual_steps,
                        iteration=self.iteration + 1,
                        method=self.method,
                        training_seed=self.seed,
                        world=i,
                        episode_seed=e.seed,
                        episode_index=self.episode_counts[i],
                    )
                    record.update(
                        {f"episode_reward_{k}": v for k, v in e.terms_sum.items()}
                    )
                    self.episodes.append(record)
                    self.schedule.completed(record, self.actual_steps)
                    self.episode_counts[i] += 1
                    a, c = e.reset(
                        self.schedule.scale,
                        self.episode_seed(i, self.episode_counts[i]),
                    )
                next_actors.append(a)
                next_critics.append(c)
            with torch.no_grad():
                nv = (
                    self.net.critic(torch.tensor(np.stack(terminal_critics)))
                    .squeeze(-1)
                    .numpy()
                )
            nextvals.append(nv)
            rewards.append(reward_step)
            dones.append(done_step)
            self.actors = np.stack(next_actors)
            self.critics = np.stack(next_critics)
        a, returns = advantages(
            np.array(rewards),
            np.array(values),
            np.array(nextvals),
            np.array(dones),
            hp["gamma"],
            hp["gae_lambda"],
        )

        def tensor(x):
            x = np.asarray(x, dtype=np.float32)
            return torch.tensor(x.reshape((-1,) + x.shape[2:]))

        batch = (
            tensor(obs),
            tensor(priv),
            tensor(raws),
            tensor(logs),
            tensor(returns),
            tensor(a),
        )
        stats = update(self.net, self.optimizer, batch, hp)
        self.torch_state = torch.get_rng_state()
        self.iteration += 1
        recent = [e for e in self.episodes if e["iteration"] == self.iteration]
        row = dict(
            iteration=self.iteration,
            actual_steps=self.actual_steps,
            physics_steps=self.actual_steps * 10,
            method=self.method,
            training_seed=self.seed,
            scale=self.schedule.scale,
            curriculum_stage=self.schedule.stage,
            full_scale_steps=self.full_scale_steps,
            completed_episodes=len(recent),
            wall_seconds=time.monotonic() - self.start,
            **stats,
        )
        for key in (
            "episode_reward",
            "episode_seconds",
            "max_supported_pitch",
            "max_pitch",
            "max_pitch_velocity",
            "takeoff",
            "rear_clearance",
            "front_support_fraction",
            "nonfoot_collision",
            "success",
            "recovery_seconds",
            "torque_saturation",
        ):
            row[key] = float(np.mean([e[key] for e in recent])) if recent else ""
        for name in (
            "pitch",
            "lift",
            "front_support",
            "stability",
            "recovery",
            "potential",
            "takeoff_bonus",
            "torque",
            "action_rate",
            "fall",
        ):
            key = f"episode_reward_{name}"
            row[key] = float(np.mean([e[key] for e in recent])) if recent else ""
        row["actor_parameter_change"] = float(
            torch.linalg.vector_norm(
                self.net.actor[0].weight.detach() - self.initial_parameters
            )
        )
        for scale in p["scales"]:
            row[f"steps_scale_{scale:.2f}"] = scale_steps.get(scale, 0)
        self.training.append(row)
        save_csv(self.directory / "training.csv", self.training)
        if self.episodes:
            save_csv(self.directory / "episodes.csv", self.episodes)
        write_json(self.directory / "curriculum_events.json", self.schedule.events)
        write_json(self.directory / "curriculum_windows.json", self.schedule.windows)
        print(
            f"{self.method} seed={self.seed} iter={self.iteration} "
            f"steps={self.actual_steps} "
            f"scale={self.schedule.scale:.2f} reward={row['episode_reward']} "
            f"pitch={row['max_supported_pitch']} KL={stats['post_update_kl']:.4f}",
            flush=True,
        )

    def checkpoint_and_evaluate(self, perturb=False):
        traces = []
        already_evaluated = any(
            r["actual_steps"] == self.actual_steps and not r["perturbation"]
            for r in self.evaluations
        )
        scales = [] if already_evaluated else sorted(set([1.0, self.schedule.scale]))
        for scale in scales:
            episodes, trace, count = evaluate(
                self.net,
                self.eval_env,
                self.p["evaluation_seeds"],
                scale,
                self.actual_steps,
                self.method,
                self.seed,
                self.iteration,
            )
            self.evaluations.extend(episodes)
            traces.extend(trace)
            self.evaluation_steps += count
        if perturb:
            episodes, trace, count = evaluate(
                self.net,
                self.eval_env,
                self.p["evaluation_seeds"],
                1.0,
                self.actual_steps,
                self.method,
                self.seed,
                self.iteration,
                True,
            )
            self.evaluations.extend(episodes)
            traces.extend(trace)
            self.evaluation_steps += count
        save_csv(self.directory / "evaluation.csv", self.evaluations)
        suffix = "_final" if perturb else ""
        if traces:
            save_csv(
                self.directory / f"evaluation_trace_{self.iteration:06d}{suffix}.csv",
                traces,
            )
        torch.save(
            dict(
                network=self.net.state_dict(),
                optimizer=self.optimizer.state_dict(),
                iteration=self.iteration,
                actual_steps=self.actual_steps,
                method=self.method,
                training_seed=self.seed,
                evaluation_steps=self.evaluation_steps,
                full_scale_steps=self.full_scale_steps,
                curriculum=asdict(self.schedule),
                torch_rng=self.torch_state,
                numpy_rng=self.rng.bit_generator.state,
                episode_counts=self.episode_counts,
                worlds=[e.state() for e in self.envs],
                actors=self.actors,
                critics=self.critics,
                training=self.training,
                episodes=self.episodes,
                evaluations=self.evaluations,
                initial_parameters=self.initial_parameters,
            ),
            self.directory / f"checkpoint_{self.iteration:06d}{suffix}.pt",
        )

    def restore(self, path):
        # Only own trusted experiment files. Pickled simulator/RNG state is required.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("method", self.method) != self.method
            or state.get("training_seed", self.seed) != self.seed
        ):
            raise ValueError("checkpoint method/seed mismatch")
        if state["curriculum"]["protocol"] != self.p:
            raise ValueError("checkpoint protocol mismatch")
        self.net.load_state_dict(state["network"])
        self.optimizer.load_state_dict(state["optimizer"])
        for key in (
            "iteration",
            "actual_steps",
            "evaluation_steps",
            "full_scale_steps",
            "episode_counts",
            "actors",
            "critics",
            "training",
            "episodes",
            "evaluations",
            "initial_parameters",
        ):
            setattr(self, key, state[key])
        self.schedule = PerformanceCurriculum(**state["curriculum"])
        for e, s in zip(self.envs, state["worlds"]):
            e.restore(s)
        self.rng.bit_generator.state = state["numpy_rng"]
        self.torch_state = state["torch_rng"]


def learning_gate(run):
    if run.schedule.stage > 0:
        return dict(
            passed=True,
            reason="measurable stage advancement criteria met",
            events=run.schedule.events,
        )
    stage_scale = run.p["scales"][0]
    records = [
        r
        for r in run.evaluations
        if r["scale"] == stage_scale and not r["perturbation"]
    ]
    if not records:
        return dict(passed=False, reason="no same-stage evaluation")
    initial = [r for r in records if r["iteration"] == 0]
    last_iterations = sorted(set(r["iteration"] for r in records))[-2:]
    final = [
        r for r in records if r["iteration"] in last_iterations and r["iteration"] > 0
    ]
    if not initial or not final:
        return dict(passed=False, reason="insufficient paired evaluation")
    change = {
        key: float(
            np.mean([r[key] for r in final]) - np.mean([r[key] for r in initial])
        )
        for key in (
            "takeoff",
            "max_supported_pitch",
            "rear_clearance",
            "episode_reward",
        )
    }
    front = float(np.mean([r["front_support_fraction"] for r in final]))
    passed = (
        change["takeoff"] >= 0.10
        or (
            change["max_supported_pitch"] >= 0.05236
            and change["rear_clearance"] >= 0.005
        )
    ) and front >= 0.5
    return dict(
        passed=bool(passed),
        deltas=change,
        final_front_support=front,
        comparison="same morphology and held-out seeds; reward alone insufficient",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/morphology/learning/protocol_90.json"),
    )
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    p = protocol(args.protocol)
    if args.smoke:
        p.update(
            training_seeds=[42],
            evaluation_seeds=[1000],
            num_envs=2,
            steps_per_iteration=64,
            diagnostic_iterations=1,
            evaluation_interval=1,
        )
        p["ppo"].update(epochs=1, batch_size=64)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", p)
    assets = Assets(args.output / "assets")
    assets.directory = args.output / "assets"
    torch.set_num_threads(1)
    decisions = []
    for seed in p["training_seeds"]:
        runs = []
        for method in ("A", "B"):
            run = Run(args.output / f"{method}_seed{seed}", method, seed, p, assets)
            run.checkpoint_and_evaluate()
            for _ in range(p["diagnostic_iterations"]):
                run.collect_and_update()
                if (
                    run.iteration % p["evaluation_interval"] == 0
                    or run.iteration == p["diagnostic_iterations"]
                ):
                    run.checkpoint_and_evaluate()
            runs.append(run)
        a, b = runs
        gate = learning_gate(b)
        decisions.append(dict(seed=seed, **gate))
        write_json(args.output / "diagnostic_decisions.json", decisions)
        if gate["passed"] and not args.diagnostic_only and not args.smoke:
            target = p["continuation_base_steps"]
            while b.actual_steps < target:
                for run in (b, a):
                    run.collect_and_update()
                    if run.iteration % p["evaluation_interval"] == 0:
                        run.checkpoint_and_evaluate()
                if b.schedule.scale == 1.0:
                    target = max(
                        target,
                        b.actual_steps
                        + max(0, p["min_full_scale_steps"] - b.full_scale_steps),
                    )
                    target = min(target, p["hard_cap_steps"])
            for run in (a, b):
                run.checkpoint_and_evaluate(perturb=True)
        else:
            for run in (a, b):
                # Final recovery evaluation uses the same full-size seeds.
                run.checkpoint_and_evaluate(perturb=True)
        assert a.actual_steps == b.actual_steps
        for run in (a, b):
            write_json(
                run.directory / "outcome.json",
                dict(
                    method=run.method,
                    seed=seed,
                    actual_steps=run.actual_steps,
                    evaluation_steps=run.evaluation_steps,
                    full_scale_steps=run.full_scale_steps,
                    final_scale=run.schedule.scale,
                    stages_reached=[p["scales"][0]]
                    + [e["scale"] for e in run.schedule.events]
                    if run.method == "B"
                    else [1.0],
                    curriculum_complete=run.method == "A"
                    or (
                        run.schedule.scale == 1.0
                        and run.full_scale_steps >= p["min_full_scale_steps"]
                    ),
                    diagnostic_gate=gate,
                    stalled=run.method == "B" and run.schedule.scale < 1.0,
                    stop_reason="diagnostic_gate_not_met"
                    if not gate["passed"]
                    else "diagnostic_only"
                    if args.diagnostic_only or args.smoke
                    else "matched_compute_budget",
                ),
            )
        print(f"paired seed={seed} diagnostic decision: {gate}", flush=True)
    write_json(
        args.output / "completion.json",
        dict(
            seeds=p["training_seeds"],
            decisions=decisions,
            physical_morphology=True,
            posture_curriculum=False,
            cpg=False,
            environment_steps=sum(
                json.loads(f.read_text())["actual_steps"]
                for f in args.output.glob("*/outcome.json")
            ),
        ),
    )


if __name__ == "__main__":
    main()
