"""Controlled noise ablation on demonstrated resets; never changes a checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-noise-only", action="store_true")
    p.add_argument("--horizon", type=float, default=6.5)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.dataset, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    ids = np.flatnonzero(data["stage_0"])
    ids = ids[np.linspace(0, len(ids) - 1, 16).astype(int)]
    samples = [sample_from_arrays(data, int(i)) for i in ids]
    results = {}
    torch.set_num_threads(1)
    for scale in (1.0,) if a.training_noise_only else (1.0, 0.5):
        policy = Policy(BALANCE, "go2balance", 16, True)
        try:
            policy.runner.alg.actor.std.mul_(scale)

            def action():
                r = policy.runner
                obs = (
                    r.get_noisy_obs(r.actor_cfg["obs"], r.actor_cfg["noise"])
                    if a.training_noise_only
                    else r.get_obs(r.actor_cfg["obs"])
                )
                r.set_actions(r.actor_cfg["actions"], r.alg.actor.act(obs), False)

            policy.action = action
            torch.manual_seed(12345)
            outcomes, _ = replay_batch(policy, samples, a.horizon)
            results[str(scale)] = dict(
                successes=sum(r["success"] for r in outcomes),
                count=16,
                collision_free_successes=sum(
                    r["success"] and not r["collision"] for r in outcomes
                ),
                mean_hold_s=float(np.mean([r["hold_s"] for r in outcomes])),
                outcomes=outcomes,
            )
            print(
                "noise scale",
                scale,
                results[str(scale)]["successes"],
                "/16",
                flush=True,
            )
        finally:
            policy.env._backend.close()
    dump(
        a.output,
        dict(
            sample_indices=ids.tolist(),
            conditional_deterministic_successes=16,
            noise_ablations=results,
            observation_noise=a.training_noise_only,
            interpretation=(
                "Single noise draw per reset; exploratory evidence. "
                "No significance claim or training change."
            ),
        ),
    )


if __name__ == "__main__":
    main()
