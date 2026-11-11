"""Paired full-episode reward audit on the actual stage-two reset sampler."""

import argparse
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.train import get_train_args, setup
from scripts.train_kickup_recovery import frontier_configuration
from scripts.evaluate_frozen_residual import load_controller
from gym.envs.go2.frozen_balance_residual import ResidualAdapter
from scripts.evaluate_kickup_handoff import dump
from scripts.reproduce_transition_hold import BALANCE


@torch.no_grad()
def episodes(checkpoint):
    cfg, runner = frontier_configuration("datasets/recovery_frontier_v4/train.npz")
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
            "--disable_wandb",
            "--headless",
            "--warm_start",
            BALANCE,
            "--seed",
            "42",
            "--experiment_name",
            "frozen_residual_episode_evaluation",
        ]
    )
    _, r = setup(args, cfg, runner)
    e = r.env
    try:
        # Each paired trial gets the full eight-second horizon from its reset.
        e.episode_length_buf.zero_()
        e._hold_time.zero_()
        e._best_hold.zero_()
        fingerprints = [
            hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()
            for d in e._backend._datas
        ]
        model = load_controller(SimpleNamespace(runner=r), checkpoint)
        adapter = ResidualAdapter(SimpleNamespace(env=e, runner=r), model)
        active = torch.ones(16, dtype=torch.bool)
        totals = {}
        results = [None] * 16
        for step in range(round(8.1 / e.dt)):
            adapter.apply()
            e.step()
            terms = r.get_rewards(
                r.critic_cfg["reward"]["weights"], e.dt, ~e.terminated
            )
            terms.update(
                r.get_rewards(
                    r.critic_cfg["reward"]["termination_weight"], e.dt, e.terminated
                )
            )
            terms["additional_correction_cost"] = (
                -10 * e.dt * adapter.last.square().mean(-1)
            )
            for key, value in terms.items():
                totals.setdefault(key, torch.zeros(16)).add_(value * active)
            done = (e.terminated | e.timed_out) & active
            for i in done.nonzero().flatten().tolist():
                results[i] = dict(
                    slot=i,
                    duration_s=(step + 1) * e.dt,
                    success=bool(e._best_hold[i] >= 5 - 1e-6 and not e.terminated[i]),
                    longest_hold_s=float(e._best_hold[i]),
                    terminated=bool(e.terminated[i]),
                    timed_out=bool(e.timed_out[i]),
                    reward_components={k: float(v[i]) for k, v in totals.items()},
                )
            active &= ~done
            if not active.any():
                break
            reset_mask = (e.terminated | e.timed_out).clone()
            e._reset_idx(reset_mask)
            adapter.reset(reset_mask)
        assert all(row is not None for row in results)
        return dict(
            initial_state_sha256=fingerprints,
            successes=sum(row["success"] for row in results),
            episodes=results,
        )
    finally:
        e._backend.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    original = episodes(None)
    residual = episodes(a.checkpoint)
    assert original["initial_state_sha256"] == residual["initial_state_sha256"]
    dump(
        a.output,
        dict(
            original=original,
            residual=residual,
            note="Deterministic full eight-second paired trials, no optimizer updates, "
            "no observation noise or exploration; independent of the stochastic "
            "training episode records.",
        ),
    )


if __name__ == "__main__":
    main()
