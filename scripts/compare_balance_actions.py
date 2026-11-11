"""Paired actor/target comparisons on every preserved frontier snapshot."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import install_samples, sample_from_arrays
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE
from scripts.optimize_posture_transition import digest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    p = Policy(BALANCE, "go2balance", 16, True)
    observations, references = [], []
    try:
        for split in ("train", "validation", "heldout"):
            with np.load(f"datasets/recovery_frontier_v4/{split}.npz") as f:
                arrays = {k: f[k] for k in f.files}
            for start in range(0, len(arrays["seed"]), 16):
                samples = [
                    sample_from_arrays(arrays, i)
                    for i in range(start, min(start + 16, len(arrays["seed"])))
                ]
                install_samples(p, samples)
                observations.append(
                    p.runner.get_obs(p.runner.actor_cfg["obs"])[: len(samples)].clone()
                )
                references.append(p.env._standing_posture()[: len(samples)].clone())
        obs, reference = torch.cat(observations), torch.cat(references)
        original = p.runner.alg.actor
        baseline = original.act_inference(obs)
        limits = p.env.dof_pos_limits[p.env.actuated_dof_indices]
        results = {}
        for checkpoint in a.checkpoints:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            actor = copy.deepcopy(original)
            actor.load_state_dict(saved["actor_state_dict"], strict=True)
            action = actor.act_inference(obs)
            change = 0.25 * (action - baseline)
            target = reference + 0.25 * action
            kl = (
                torch.log(actor.std / original.std)
                + (original.std.square() + (baseline - action).square())
                / (2 * actor.std.square())
                - 0.5
            ).sum(-1)
            results[str(checkpoint)] = dict(
                sha256=digest(checkpoint),
                target_change_rms_rad=float(change.square().mean().sqrt()),
                target_change_max_rad=float(change.abs().max()),
                target_limit_violation_fraction=float(
                    ((target < limits[:, 0]) | (target > limits[:, 1])).float().mean()
                ),
                mean_kl_from_original=float(kl.mean()),
                action_std=actor.std.tolist(),
                actor_optimizer_lr=saved["optimizer_state_dict"]["param_groups"][0][
                    "lr"
                ],
                actor_optimizer_max_step=max(
                    int(v["step"])
                    for v in saved["optimizer_state_dict"]["state"].values()
                    if "step" in v
                )
                if saved["optimizer_state_dict"]["state"]
                else 0,
            )
        dump(
            a.output,
            dict(states=len(obs), baseline_sha256=digest(BALANCE), comparisons=results),
        )
    finally:
        p.env._backend.close()


if __name__ == "__main__":
    main()
