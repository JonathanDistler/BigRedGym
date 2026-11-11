"""Short, separately saved PPO diagnostic around an immutable balance teacher."""

import argparse
import json
import random
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from torch import nn
from gym.envs.go2.frozen_balance_residual import FrozenBalanceResidual, ResidualAdapter
from scripts.train import get_train_args, setup
from scripts.train_kickup_recovery import frontier_configuration
from scripts.reproduce_transition_hold import BALANCE
from scripts.optimize_posture_transition import digest
from scripts.evaluate_frozen_residual import evaluate
from scripts.evaluate_kickup_handoff import dump


def gae(rewards, values, next_values, terminated, dones, gamma=0.998, lam=0.95):
    """Bootstrap from terminal observations on timeouts; never across resets."""
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        td = rewards[t] + gamma * next_values[t] * ~terminated[t] - values[t]
        tail = td + gamma * lam * ~dones[t] * tail
        advantages[t] = tail
    return advantages, advantages + values


def validate_baseline_parity(parity_path, commands_path):
    parity = json.loads(parity_path.read_text())
    commands = json.loads(commands_path.read_text())
    expected = dict(
        exact_complete_outcome_parity=True,
        near_vertical=14,
        validation_plus_heldout=30,
        states=123,
    )
    if any(parity.get(k) != v for k, v in expected.items()):
        raise ValueError("Complete zero-residual outcome parity must pass")
    if (
        not commands.get("exact_command_and_controller_parity")
        or commands.get("states") != 285
        or not commands.get("maximum_errors")
        or any(commands["maximum_errors"].values())
    ):
        raise ValueError("Exact command and CPG parity must pass")
    summary = json.loads((parity_path.parent / "summary.json").read_text())
    if (
        summary["baseline_sha256"] != digest(BALANCE)
        or summary["zero_correction_max_target_error_rad"] != 0
    ):
        raise ValueError("Parity baseline hash or target commands differ")
    for split in ("validation", "heldout"):
        path = f"datasets/recovery_frontier_v4/{split}.npz"
        if summary[split]["dataset_sha256"] != digest(path):
            raise ValueError("Parity evaluation dataset differs")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, choices=range(1, 26), default=10)
    parser.add_argument("--dataset", default="datasets/recovery_frontier_v4/train.npz")
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--commands-parity", type=Path, required=True)
    a = parser.parse_args()
    validate_baseline_parity(a.parity, a.commands_parity)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, runner = frontier_configuration(a.dataset)
    # Stage 2 mixes measured recoverables and nearby frontier, retains 70% stable.
    # Advancement continues to require two successful measured episode batches.
    cfg.handstand_curriculum.initial_level = 2
    args = get_train_args(
        [
            "--task",
            "go2kickup_recovery",
            "--backend",
            "mujoco",
            "--device",
            "cpu",
            "--num_envs",
            "16",
            "--headless",
            "--disable_wandb",
            "--warm_start",
            BALANCE,
            "--seed",
            "42",
            "--experiment_name",
            "frozen_residual_diagnostic",
        ]
    )
    _, r = setup(args, cfg, runner)
    e = r.env
    model = FrozenBalanceResidual(r.alg.actor)
    adapter = ResidualAdapter(SimpleNamespace(env=e, runner=r), model)
    # New value function: no inherited normalization or old-objective targets.
    critic = nn.Sequential(
        nn.Linear(63, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
    )
    actor_opt = torch.optim.Adam(model.recovery.parameters(), lr=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    original_hash = digest(BALANCE)
    teacher = {k: v.clone() for k, v in model.balance.state_dict().items()}
    rows = []
    episode_rewards = {}
    episode_duration = torch.zeros(e.num_envs)
    dump(
        a.output_dir / "configuration.json",
        dict(
            iterations=a.iterations,
            seed=42,
            dataset=a.dataset,
            dataset_sha256=digest(a.dataset),
            baseline_sha256=original_hash,
            parity=str(a.parity),
            commands_parity=str(a.commands_parity),
            actor_lr=1e-5,
            clip=0.05,
            target_kl=0.002,
            correction_limit_rad=0.04,
            observation_noise=False,
            initial_stage=2,
            rollout_steps=256,
        ),
    )

    def save(it):
        path = a.output_dir / f"model_{it}.pt"
        torch.save(
            dict(
                controller=model.state_dict(),
                critic=critic.state_dict(),
                actor_optimizer=actor_opt.state_dict(),
                critic_optimizer=critic_opt.state_dict(),
                baseline_sha256=original_hash,
                iteration=it,
                curriculum=e.training_state_dict(),
                rng=torch.get_rng_state(),
            ),
            path,
        )
        return path

    try:
        save(0)
        for it in range(1, a.iterations + 1):
            completed_episodes = []
            batch = {
                k: []
                for k in (
                    "obs",
                    "critic",
                    "latent",
                    "logp",
                    "reward",
                    "value",
                    "next",
                    "terminated",
                    "done",
                )
            }
            terms_sum = {}
            correction_sum = activation = 0.0
            with torch.no_grad():
                for _ in range(256):
                    obs = r.get_obs(r.actor_cfg["obs"]).clone()
                    co = r.get_obs(r.critic_cfg["obs"]).clone()
                    dist = model.distribution(obs)
                    latent = dist.sample()
                    adapter.apply(latent)
                    value = critic(co).squeeze(-1)
                    e.step()
                    terminated = e.terminated.clone()
                    done = terminated | e.timed_out
                    # Read the terminal observation BEFORE resetting worlds.
                    nv = critic(r.get_obs(r.critic_cfg["obs"])).squeeze(-1)
                    terms = r.get_rewards(
                        r.critic_cfg["reward"]["weights"],
                        modifier=e.dt,
                        mask=~terminated,
                    )
                    terms.update(
                        r.get_rewards(
                            r.critic_cfg["reward"]["termination_weight"],
                            modifier=e.dt,
                            mask=terminated,
                        )
                    )
                    penalty = adapter.last.square().mean(-1) * 10 * e.dt
                    reward = sum(terms.values()) - penalty
                    episode_duration += e.dt
                    for key, value in {**terms, "correction_cost": -penalty}.items():
                        episode_rewards.setdefault(key, torch.zeros(e.num_envs))
                        episode_rewards[key] += value
                    for i in done.nonzero().flatten().tolist():
                        completed_episodes.append(
                            dict(
                                duration_s=float(episode_duration[i]),
                                terminated=bool(terminated[i]),
                                timed_out=bool(e.timed_out[i]),
                                longest_hold_s=float(e._best_hold[i]),
                                reward_components={
                                    k: float(v[i]) for k, v in episode_rewards.items()
                                },
                            )
                        )
                    episode_duration[done] = 0
                    for value in episode_rewards.values():
                        value[done] = 0
                    correction_sum += float(adapter.last.square().mean())
                    activation += float(
                        (adapter.last.abs().amax(-1) > 1e-6).float().mean()
                    )
                    for k, v in terms.items():
                        terms_sum[k] = terms_sum.get(k, 0.0) + float(v.mean())
                    for k, v in dict(
                        obs=obs,
                        critic=co,
                        latent=latent,
                        logp=dist.log_prob(latent).sum(-1),
                        reward=reward,
                        value=value,
                        next=nv,
                        terminated=terminated,
                        done=done,
                    ).items():
                        batch[k].append(v.clone())
                    e._reset_idx(done)
                    adapter.reset(done)
            b = {k: torch.stack(v) for k, v in batch.items()}
            adv, returns = gae(
                b["reward"], b["value"], b["next"], b["terminated"], b["done"]
            )
            advantage_std = float(adv.std())
            adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).flatten()
            obs, co = b["obs"].flatten(0, 1), b["critic"].flatten(0, 1)
            latent, old_logp = b["latent"].flatten(0, 1), b["logp"].flatten()
            with torch.no_grad():
                old_mean = model.recovery(obs).clone()
            for _ in range(4):
                prediction = critic(co).squeeze(-1)
                value_loss = (prediction - returns.flatten()).square().mean()
                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()
            for _ in range(2):
                dist = model.distribution(obs)
                ratio = (dist.log_prob(latent).sum(-1) - old_logp).exp()
                surrogate = -torch.minimum(
                    ratio * adv, ratio.clamp(0.95, 1.05) * adv
                ).mean()
                # Penalize latent means even behind the gate, avoiding hidden drift.
                regularizer = 0.01 * model.recovery(obs).square().mean()
                actor_opt.zero_grad()
                (surrogate + regularizer).backward()
                nn.utils.clip_grad_norm_(model.recovery.parameters(), 0.25)
                actor_opt.step()
                with torch.no_grad():
                    kl = (
                        (
                            (model.recovery(obs) - old_mean).square()
                            / (2 * model.latent_std.square())
                        )
                        .sum(-1)
                        .mean()
                    )
                if kl > 0.002:
                    break
            assert all(
                torch.equal(v, teacher[k])
                for k, v in model.balance.state_dict().items()
            )
            assert digest(BALANCE) == original_hash
            checkpoint = save(it)
            row = dict(
                iteration=it,
                kl=float(kl),
                surrogate=float(surrogate.detach()),
                value_loss=float(value_loss.detach()),
                advantage_std=advantage_std,
                reward_components=terms_sum,
                completed_episodes=completed_episodes,
                residual_rms_rad=(correction_sum / 256) ** 0.5,
                activation_frequency=activation / 256,
                entropy=float(dist.entropy().sum(-1).mean()),
                clip_fraction=float(((ratio.detach() - 1).abs() > 0.05).float().mean()),
                curriculum=e.training_state_dict(),
            )
            # Preserve completed-update metrics even if evaluation is interrupted.
            rows.append(row)
            dump(a.output_dir / "diagnostic.json", rows)
            if it in (1, 2, 5, 10, 15, 20, 25):
                rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
                try:
                    result = evaluate(checkpoint, a.output_dir / f"evaluation_{it}")
                finally:
                    torch.set_rng_state(rng[0])
                    np.random.set_state(rng[1])
                    random.setstate(rng[2])
                row["near_vertical_successes"] = result["retention"]["successes"]
                row["recovery_successes"] = sum(
                    result[s]["all_states"]["successes"]
                    for s in ("validation", "heldout")
                )
                if row["near_vertical_successes"] < 13:
                    row["early_stop"] = (
                        "Retention below 13/16; no repeatable gain established"
                    )
            dump(a.output_dir / "diagnostic.json", rows)
            print(
                "residual iteration", it, row.get("near_vertical_successes"), flush=True
            )
            if "early_stop" in row:
                break
    finally:
        e._backend.close()


if __name__ == "__main__":
    main()
