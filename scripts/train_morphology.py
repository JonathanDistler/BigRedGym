"""Explicit opt-in paired PPO experiments. Does not run on import.

A: full-size; B: fixed short; C: performance-driven; D: fixed-iteration.
All methods use identical actor/critic, rewards, steps, and full-size evaluation.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from experiments.morphology.curriculum import Curriculum
from experiments.morphology.task import Config, Task, metadata, write_json
from scripts.morphology_feasibility import save_csv


class ActorCritic(nn.Module):
    def __init__(self, obs=45):
        super().__init__()

        def mlp(output):
            return nn.Sequential(
                nn.Linear(obs, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, output),
            )

        self.actor = mlp(12)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((12,), -0.5))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.exp())


def evaluate(net, task, seeds, iteration, method, training_seed):
    records, trajectories = [], []
    for perturb in (False, True):
        for seed in seeds:
            obs = task.reset(1.0, seed)
            disturbed = False
            trace = []
            with torch.no_grad():
                while not task.done:
                    if perturb and task.t >= 2 and not disturbed:
                        task.d.qvel[4] += 0.3
                        disturbed = True
                    action = net.actor(torch.tensor(obs)).numpy()
                    obs, _, _, row = task.step(action)
                    trace.append(
                        dict(
                            **row,
                            iteration=iteration,
                            method=method,
                            training_seed=training_seed,
                            seed=seed,
                            perturbation=perturb,
                        )
                    )
            records.append(
                dict(
                    method=method,
                    training_seed=training_seed,
                    evaluation_seed=seed,
                    iteration=iteration,
                    scale=1.0,
                    perturbation=perturb,
                    perturbation_applied=disturbed,
                    success=bool(task.last["success"]),
                    episode_seconds=task.t,
                    actual_control_steps=round(task.t / task.cfg.dt),
                    actual_physics_steps=round(task.t / task.m.opt.timestep),
                    reward=task.reward_sum,
                    best_hold=task.best_hold,
                    stability_fraction=np.mean([r["reward_stability"] for r in trace]),
                    saturation=np.mean([r["saturation"] for r in trace]),
                    termination=task.last["termination"],
                )
            )
            trajectories.extend(trace)
    return records, trajectories


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--method", choices=list("ABCD"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--steps-per-iteration", type=int, default=2048)
    p.add_argument("--eval-interval", type=int, default=25)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(1000, 1010)))
    p.add_argument("--start-scale", type=float, default=0.85)
    p.add_argument("--screening", type=Path)
    p.add_argument("--fixed-interval", type=int)
    a = p.parse_args()
    if min(a.iterations, a.steps_per_iteration, a.eval_interval) < 1:
        p.error("budgets and intervals must be positive")
    if a.method in ("C", "D"):
        import json

        if not a.screening:
            p.error("C/D require --screening with demonstrated candidate_scales")
        candidates = json.loads(a.screening.read_text())["candidate_scales"]
        if not candidates:
            p.error("No improved feasibility demonstrated; refine screening first")
        a.start_scale = min(candidates)
    stage_count = len([s for s in (0.8, 0.85, 0.9, 0.95, 1.0) if s >= a.start_scale])
    if a.fixed_interval is None:
        a.fixed_interval = max(1, a.iterations // stage_count)
    if a.fixed_interval < 1:
        p.error("fixed interval must be positive")
    if a.method == "D" and (stage_count - 1) * a.fixed_interval >= a.iterations:
        p.error("D must reach full scale before its training budget ends")
    a.output.mkdir(parents=True, exist_ok=False)
    # Save executable source, including untracked changes, with every new run.
    sources = a.output / "source"
    sources.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/task.py"),
        Path("experiments/morphology/curriculum.py"),
        Path("gym/envs/go2/leg_morphology.py"),
    ):
        shutil.copyfile(path, sources / path.name)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = ActorCritic()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    cfg = Config()
    task = Task(a.output / "assets", cfg)
    evaluation = Task(a.output / "evaluation_assets", cfg)
    schedule = Curriculum(a.method, a.start_scale, fixed_interval=a.fixed_interval)
    hp = dict(
        gamma=0.99,
        gae_lambda=0.95,
        clip=0.2,
        epochs=5,
        batch_size=256,
        learning_rate=3e-4,
        entropy_weight=0.01,
        value_weight=0.5,
        max_grad_norm=1.0,
        architecture=[256, 256, 128],
        activation="ELU",
        initial_log_std=-0.5,
    )
    write_json(
        a.output / "configuration.json",
        dict(
            **metadata(cfg),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            ppo=hp,
            curriculum=asdict(schedule),
            control_steps_per_second=100,
            physics_steps_per_control=5,
            experiment="morphology PPO",
        ),
    )
    obs = task.reset(schedule.scale, int(rng.integers(2**31)))
    training, episodes, evaluations = [], [], []
    terms_sum = {}
    actual_steps = 0
    for iteration in range(a.iterations + 1):
        if iteration % a.eval_interval == 0 or iteration == a.iterations:
            records, traces = evaluate(
                net, evaluation, a.eval_seeds, iteration, a.method, a.seed
            )
            evaluations.extend(records)
            save_csv(a.output / "evaluation.csv", evaluations)
            save_csv(a.output / f"evaluation_trajectories_{iteration:06d}.csv", traces)
            torch.save(
                dict(
                    model=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iteration=iteration,
                    actual_steps=actual_steps,
                    curriculum=asdict(schedule),
                    torch_rng=torch.get_rng_state(),
                    numpy_rng=rng.bit_generator.state,
                ),
                a.output / f"model_{iteration:06d}.pt",
            )
        if iteration == a.iterations:
            break
        schedule.iteration(iteration)
        buffer = []
        for _ in range(a.steps_per_iteration):
            tensor = torch.tensor(obs)
            with torch.no_grad():
                dist = net.distribution(tensor)
                raw_action = dist.sample()
                value = net.critic(tensor).item()
                logprob = dist.log_prob(raw_action).sum().item()
            next_obs, reward, done, row = task.step(raw_action.numpy())
            timeout = row["termination"] == "timeout"
            with torch.no_grad():
                next_value = (
                    net.critic(torch.tensor(next_obs)).item()
                    if not done or timeout
                    else 0.0
                )
            buffer.append(
                (obs, raw_action.numpy(), reward, value, logprob, done, next_value)
            )
            actual_steps += 1
            for key, val in row.items():
                if key.startswith("reward_"):
                    terms_sum[key] = terms_sum.get(key, 0.0) + val
            obs = next_obs
            if done:
                episodes.append(
                    dict(
                        iteration=iteration + 1,
                        actual_steps=actual_steps,
                        method=a.method,
                        training_seed=a.seed,
                        scale=task.scale,
                        reward=task.reward_sum,
                        episode_seconds=task.t,
                        success=bool(row["success"]),
                        termination=row["termination"],
                        best_hold=task.best_hold,
                        **{
                            k: v / max(round(task.t / cfg.dt), 1)
                            for k, v in terms_sum.items()
                        },
                    )
                )
                terms_sum = {}
                schedule.completed(task.scale, row["success"], iteration + 1)
                obs = task.reset(schedule.scale, int(rng.integers(2**31)))
        advantages = np.zeros(len(buffer), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(buffer))):
            _, _, reward, value, _, done, next_value = buffer[t]
            delta = reward + hp["gamma"] * next_value - value
            gae = delta + hp["gamma"] * hp["gae_lambda"] * (not done) * gae
            advantages[t] = gae
        observations = torch.tensor(np.stack([b[0] for b in buffer]))
        actions = torch.tensor(np.stack([b[1] for b in buffer]))
        oldlogs = torch.tensor([b[4] for b in buffer])
        returns = torch.tensor(
            advantages + np.array([b[3] for b in buffer], dtype=np.float32)
        )
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / advantages.std(
            unbiased=False
        ).clamp(min=1e-8)
        updates = []
        for _ in range(hp["epochs"]):
            for ids in torch.randperm(len(buffer)).split(hp["batch_size"]):
                dist = net.distribution(observations[ids])
                newlog = dist.log_prob(actions[ids]).sum(-1)
                logratio = newlog - oldlogs[ids]
                ratio = logratio.exp()
                loss_policy = -torch.minimum(
                    ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
                ).mean()
                loss_value = (
                    (net.critic(observations[ids]).flatten() - returns[ids])
                    .square()
                    .mean()
                )
                loss = (
                    loss_policy
                    + hp["value_weight"] * loss_value
                    - hp["entropy_weight"] * dist.entropy().sum(-1).mean()
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite PPO loss")
                optimizer.zero_grad()
                loss.backward()
                norm = nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
                optimizer.step()
                updates.append(
                    [
                        loss_policy.item(),
                        loss_value.item(),
                        ((ratio - 1) - logratio).mean().item(),
                        (torch.abs(ratio - 1) > hp["clip"]).float().mean().item(),
                        norm.item(),
                    ]
                )
        stats = np.mean(updates, axis=0)
        recent = [e for e in episodes if e["iteration"] == iteration + 1]
        training.append(
            dict(
                iteration=iteration + 1,
                actual_steps=actual_steps,
                physics_steps=actual_steps * 5,
                method=a.method,
                training_seed=a.seed,
                scale=schedule.scale,
                curriculum_stage=schedule.stage,
                completed_episodes=len(recent),
                mean_reward=np.mean([e["reward"] for e in recent]) if recent else "",
                episode_seconds=np.mean([e["episode_seconds"] for e in recent])
                if recent
                else "",
                policy_loss=stats[0],
                value_loss=stats[1],
                approx_kl=stats[2],
                clip_fraction=stats[3],
                gradient_norm=stats[4],
            )
        )
        save_csv(a.output / "training.csv", training)
        if episodes:
            save_csv(a.output / "episodes.csv", episodes)
        write_json(a.output / "curriculum_events.json", schedule.events)
        print(
            f"iteration={iteration + 1} steps={actual_steps} "
            f"scale={schedule.scale:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
