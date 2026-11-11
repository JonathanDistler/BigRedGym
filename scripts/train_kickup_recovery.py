"""Explicit CPG warm-start recipe; --validate performs one PPO iteration only."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gym.envs.go2.go2kickup_recovery_config import (
    Go2KickupRecoveryCfg,
    Go2KickupRecoveryRunnerCfg,
)
from gym.utils.original_cfg import checkpoint_configs
from gym.utils.helpers import class_to_dict
from gym.utils.robot_metrics import configure_contact_cpg
from scripts.train import get_train_args, setup, train
from scripts.reproduce_transition_hold import BALANCE


def configuration(dataset):
    cfg, runner = Go2KickupRecoveryCfg(), Go2KickupRecoveryRunnerCfg()
    saved_cfg, saved_runner = checkpoint_configs(BALANCE, "go2balance")
    for name in ("control", "scaling", "contact_cpg"):
        setattr(cfg, name, copy.deepcopy(getattr(saved_cfg, name)))
    for name in ("actor", "critic"):
        for key, value in class_to_dict(getattr(saved_runner, name)).items():
            if key != "reward":
                setattr(getattr(runner, name), key, copy.deepcopy(value))
    configure_contact_cpg(
        argparse.Namespace(
            cpg=True, contact_modes=["stance", "stance", "swing", "swing"]
        ),
        cfg,
        runner,
    )
    cfg.recovery.dataset = str(Path(dataset).resolve())
    return cfg, runner


def frontier_configuration(dataset):
    """Measured reset curriculum; fresh namespaces leave legacy defaults intact."""
    cfg, runner = configuration(dataset)
    cfg.recovery = SimpleNamespace(**class_to_dict(cfg.recovery))
    cfg.recovery.frontier_curriculum = True
    cfg.handstand_curriculum = SimpleNamespace(
        **class_to_dict(cfg.handstand_curriculum)
    )
    cfg.handstand_curriculum.min_episodes = 64
    cfg.handstand_curriculum.required_batches = 2
    cfg.handstand_curriculum.success_fraction = 0.8
    runner.algorithm = SimpleNamespace(**class_to_dict(runner.algorithm))
    runner.algorithm.learning_rate = 1e-4
    runner.algorithm.lr_range = [1e-5, 3e-4]
    runner.algorithm.clip_param = 0.1
    runner.critic = SimpleNamespace(**class_to_dict(runner.critic))
    reward = runner.critic.reward
    weights = dict(reward["weights"])
    weights.update(recovery_com_offset=0.5, recovery_com_velocity=1.0)
    runner.critic.reward = SimpleNamespace(
        weights=SimpleNamespace(**weights),
        termination_weight=SimpleNamespace(**reward["termination_weight"]),
    )
    return cfg, runner


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--iterations", type=int, choices=(50, 200), default=50)
    p.add_argument("--frontier-curriculum", action="store_true")
    p.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume into a new run; --iterations specifies the final total",
    )
    a = p.parse_args()
    if a.resume_checkpoint and a.validate:
        p.error("--resume-checkpoint cannot be combined with --validate")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = a.output_dir / "warm_start.pt"
    shutil.copy2(BALANCE, checkpoint)
    original_hash = hashlib.sha256(Path(BALANCE).read_bytes()).hexdigest()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    cfg, runner = (
        frontier_configuration(a.dataset)
        if a.frontier_curriculum
        else configuration(a.dataset)
    )
    if a.validate:
        cfg.handstand_curriculum.initial_level = 2 if a.frontier_curriculum else 4
    name = "go2kickup_recovery_" + datetime.now(timezone.utc).strftime(
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
            "1" if a.validate else str(a.iterations),
            "--save_interval",
            "25",
            "--disable_wandb",
            "--headless",
            "--warm_start",
            str(checkpoint),
            "--experiment_name",
            name,
            "--seed",
            "42",
        ]
    )
    torch.set_num_threads(1)
    train_cfg, policy = setup(args, cfg, runner)
    try:
        if a.resume_checkpoint:
            saved = torch.load(
                a.resume_checkpoint, weights_only=True, map_location="cpu"
            )
            if not 0 < saved["iter"] < a.iterations:
                raise ValueError("Resume iteration must be below the requested total")
            effective = json.loads(
                (a.resume_checkpoint.parent / "effective_config.json").read_text()
            )
            if effective["task"] != "go2kickup_recovery":
                raise ValueError("Only recovery runs can be resumed")
            for key in ("actor", "critic", "algorithm"):
                if effective["train"][key] != class_to_dict(getattr(train_cfg, key)):
                    raise ValueError(f"Resume {key} configuration differs")
            if effective["env"]["recovery"] != class_to_dict(cfg.recovery):
                raise ValueError("Resume dataset/curriculum configuration differs")
            policy.load(str(a.resume_checkpoint))
            # Restoring curriculum resets worlds in inference mode. Burn-in
            # must use the same mode when updating the resulting buffers.
            policy.burn_in_normalization = torch.inference_mode()(
                policy.burn_in_normalization
            )
            policy.alg.learning_rate = policy.alg.optimizer.param_groups[0]["lr"]
            policy.num_learning_iterations = a.iterations - policy.it
            # Preserve the paired evaluation series in the new run directory.
            for iteration in (0, 25):
                source = a.resume_checkpoint.parent / f"model_{iteration}.pt"
                if iteration < policy.it and source.exists():
                    shutil.copy2(source, Path(train_cfg.log_dir) / source.name)
        assert policy.num_steps_per_env == 256
        assert policy.alg_cfg["batch_size"] == 1024
        report = dict(
            task="go2kickup_recovery",
            checkpoint_sha256=original_hash,
            rollout_steps_per_env=policy.num_steps_per_env,
            num_envs=16,
            transitions=4096,
            optimizer_batch_size=1024,
            algorithm=policy.alg_cfg,
            actor=policy.actor_cfg,
            train=class_to_dict(train_cfg),
            validation=a.validate,
            resume_checkpoint=str(a.resume_checkpoint) if a.resume_checkpoint else None,
            resume_note="Optimizer/curriculum restored; native worlds and RNG restart"
            if a.resume_checkpoint
            else None,
        )
        (a.output_dir / "configuration.json").write_text(json.dumps(report, indent=2))
        observed = []
        if a.validate:
            original_rewards = policy.get_rewards

            def checked_rewards(*args, **kwargs):
                terms = original_rewards(*args, **kwargs)
                for value in terms.values():
                    if not torch.isfinite(value).all():
                        raise RuntimeError("Nonfinite rollout reward")
                observed.append({k: float(v.mean()) for k, v in terms.items()})
                return terms

            policy.get_rewards = checked_rewards
        train(train_cfg, policy)
        for network in (policy.alg.actor, policy.alg.critic):
            assert all(torch.isfinite(p).all() for p in network.parameters())
        assert torch.isfinite(policy.env.root_states).all()
        assert torch.isfinite(torch.tensor(policy.alg.mean_value_loss))
        assert torch.isfinite(torch.tensor(policy.alg.mean_surrogate_loss))
        (a.output_dir / "result.json").write_text(
            json.dumps(
                dict(
                    iterations=policy.it,
                    finite_parameters=True,
                    finite_native_state=True,
                    finite_reward_calls=len(observed),
                    value_loss=policy.alg.mean_value_loss,
                    surrogate_loss=policy.alg.mean_surrogate_loss,
                    log_dir=train_cfg.log_dir,
                    curriculum=policy.env.training_state_dict(),
                ),
                indent=2,
            )
        )
        assert hashlib.sha256(Path(BALANCE).read_bytes()).hexdigest() == original_hash
    finally:
        policy.env._backend.close()


if __name__ == "__main__":
    main()
