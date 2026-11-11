"""Record the first legacy rollout without performing any optimizer updates."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train import get_train_args, setup, train
from scripts.train_kickup_recovery import frontier_configuration
from scripts.reproduce_transition_hold import BALANCE
from scripts.evaluate_kickup_handoff import dump
from scripts.optimize_posture_transition import digest
from scripts.audit_balance_collapse import stats


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    cfg, runner = frontier_configuration("datasets/recovery_frontier_v3/train.npz")
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
            "1",
            "--disable_wandb",
            "--headless",
            "--warm_start",
            BALANCE,
            "--experiment_name",
            "frozen_stage_zero_reward_audit",
            "--seed",
            "42",
        ]
    )
    config, r = setup(args, cfg, runner)
    teacher = {k: v.clone() for k, v in r.alg.actor.state_dict().items()}
    original = r.get_rewards
    components = {}

    def rewards(weights, modifier=1, mask=None):
        result = original(weights, modifier, mask)
        if modifier == r.env.dt:
            for key, value in result.items():
                components.setdefault(key, []).append(value.detach().clone())
        return result

    def stop(data):
        dump(
            a.output,
            dict(
                seed=42,
                actor_updated=False,
                critic_updated=False,
                checkpoint_sha256=digest(BALANCE),
                rewards=stats(data["rewards"]),
                dones=int(data["dones"].sum()),
                timeouts=int(data["timed_out"].sum()),
                components={
                    k: dict(
                        per_transition=stats(torch.stack(v)),
                        mean_integrated_per_environment=float(
                            torch.stack(v).sum(0).mean()
                        ),
                    )
                    for k, v in components.items()
                },
                curriculum=r.env.training_state_dict(),
            ),
        )
        raise StopIteration

    r.get_rewards = rewards
    r.alg.update = stop
    try:
        try:
            train(config, r)
        except StopIteration:
            pass
        assert all(
            torch.equal(v, teacher[k]) for k, v in r.alg.actor.state_dict().items()
        )
    finally:
        r.env._backend.close()


if __name__ == "__main__":
    main()
