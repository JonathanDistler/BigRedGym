"""Fresh-copy legacy PPO audit or actor-frozen critic control; never resumes it."""

import argparse
import hashlib
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.train import get_train_args, setup, train
from scripts.train_kickup_recovery import frontier_configuration
from scripts.reproduce_transition_hold import BALANCE
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.recovery_nominal_fixture import evaluate_nominal


def stats(x):
    x = x.detach()
    return dict(
        mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max())
    )


def evaluate_preserving_rng(checkpoint, output):
    states = torch.get_rng_state(), np.random.get_state(), random.getstate()
    policy = Policy(str(checkpoint), "go2kickup_recovery", 16, True)
    try:
        with torch.inference_mode():
            return evaluate_nominal(policy, output)
    finally:
        policy.env._backend.close()
        torch.set_rng_state(states[0])
        np.random.set_state(states[1])
        random.setstate(states[2])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--critic-only", action="store_true")
    parser.add_argument("--iterations", type=int, choices=(1, 2, 5, 10), default=10)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, runner = frontier_configuration("datasets/recovery_frontier_v3/train.npz")
    name = "legacy_balance_audit_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%f"
    )
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
            "--max_iterations",
            str(a.iterations),
            "--save_interval",
            "1",
            "--disable_wandb",
            "--headless",
            "--warm_start",
            BALANCE,
            "--experiment_name",
            name,
            "--seed",
            "42",
        ]
    )
    train_cfg, policy = setup(args, cfg, runner)
    alg = policy.alg
    actor_before = {k: v.detach().clone() for k, v in alg.actor.state_dict().items()}
    rows = []
    original_critic = alg.update_critic
    original_update = alg.update

    def critic(data):
        rows[-1].update(
            raw_advantages=stats(data["advantages"]),
            returns=stats(data["returns"]),
            values=stats(data["values"]),
        )
        return original_critic(data)

    alg.update_critic = critic
    if a.critic_only:

        def frozen_update(data):
            alg.mean_surrogate_loss = 0.0

        alg.update_actor = frozen_update

    def update(data):
        with torch.no_grad():
            alg.actor.update_distribution(data["actor_obs"])
            mu = alg.actor.action_mean.detach().clone()
            sigma = alg.actor.action_std.detach().clone()
            logp = alg.actor.get_actions_log_prob(data["actions"]).detach().clone()
        row = dict(
            iteration=policy.it,
            rewards=stats(data["rewards"]),
            dones=int(data["dones"].sum()),
            stored_terminated=int(data["terminated"].sum()),
            timeouts=int(data["timed_out"].sum()),
            actor_optimizer_steps_before=max(
                (int(s["step"]) for s in alg.optimizer.state.values() if "step" in s),
                default=0,
            ),
        )
        rows.append(row)
        original_update(data)
        with torch.no_grad():
            alg.actor.update_distribution(data["actor_obs"])
            new_mu, new_sigma = alg.actor.action_mean, alg.actor.action_std
            kl = (
                torch.log(new_sigma / sigma)
                + (sigma.square() + (mu - new_mu).square()) / (2 * new_sigma.square())
                - 0.5
            ).sum(-1)
            ratio = (alg.actor.get_actions_log_prob(data["actions"]) - logp).exp()
            row.update(
                analytic_kl=float(kl.mean()),
                approximate_kl=float(((ratio - 1) - ratio.log()).mean()),
                clip_fraction=float(
                    ((ratio - 1).abs() > alg.clip_param).float().mean()
                ),
                action_mean_rms_change=float((new_mu - mu).square().mean().sqrt()),
                entropy=float(alg.actor.entropy.mean()),
                normalized_advantages=stats(data["advantages"]),
                value_loss=alg.mean_value_loss,
                surrogate_loss=alg.mean_surrogate_loss,
                actor_learning_rate=alg.optimizer.param_groups[0]["lr"],
                critic_learning_rate=alg.critic_optimizer.param_groups[0]["lr"],
            )
        policy.save()
        if policy.it in (1, 2, 5, 10):
            dest = a.output_dir / f"iteration_{policy.it}"
            dest.mkdir()
            probe = evaluate_preserving_rng(
                Path(train_cfg.log_dir) / f"model_{policy.it}.pt", dest
            )
            row["near_vertical_successes"] = probe["successes"]
            if probe["successes"] < 12 and not a.critic_only:
                row["early_stop"] = "Near-vertical retention below 12/16"
                dump(
                    a.output_dir / "audit.json",
                    dict(
                        log_dir=train_cfg.log_dir,
                        critic_only=a.critic_only,
                        updates=rows,
                    ),
                )
                raise StopIteration("Actor regression: stop this fresh-copy audit")
        dump(
            a.output_dir / "audit.json",
            dict(log_dir=train_cfg.log_dir, critic_only=a.critic_only, updates=rows),
        )

    alg.update = update
    original_hash = hashlib.sha256(Path(BALANCE).read_bytes()).hexdigest()
    try:
        try:
            train(train_cfg, policy)
        except StopIteration:
            pass
        if a.critic_only:
            assert all(
                torch.equal(v, actor_before[k])
                for k, v in alg.actor.state_dict().items()
            )
        assert hashlib.sha256(Path(BALANCE).read_bytes()).hexdigest() == original_hash
    finally:
        policy.env._backend.close()


if __name__ == "__main__":
    main()
