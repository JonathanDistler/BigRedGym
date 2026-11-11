"""Measure learned correction versus available gated authority on saved states."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import install_samples, sample_from_arrays
from scripts.evaluate_frozen_residual import load_controller
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE
from scripts.optimize_posture_transition import digest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    p = Policy(BALANCE, "go2balance", 16, True)
    model = load_controller(p, a.checkpoint)
    results = {}
    try:
        for split in ("train", "validation", "heldout"):
            with np.load(f"datasets/recovery_frontier_v4/{split}.npz") as f:
                arrays = {k: f[k] for k in f.files}
            obs = []
            for start in range(0, len(arrays["seed"]), 16):
                samples = [
                    sample_from_arrays(arrays, i)
                    for i in range(start, min(start + 16, len(arrays["seed"])))
                ]
                install_samples(p, samples)
                obs.append(
                    p.runner.get_obs(p.runner.actor_cfg["obs"])[: len(samples)].clone()
                )
            obs = torch.cat(obs)
            gate = model.gate(obs)
            correction = model.correction(obs)
            results[split] = {}
            for k in range(3):
                mask = torch.as_tensor(arrays["stratum"] == k)
                g, c = gate[mask], correction[mask]
                results[split][str(k)] = dict(
                    count=int(mask.sum()),
                    zero_authority_fraction=float((g == 0).float().mean()),
                    gate_mean=float(g.mean()),
                    available_bound_mean_rad=float((g * model.max_correction).mean()),
                    learned_correction_rms_rad=float(c.square().mean().sqrt()),
                    learned_correction_max_rad=float(c.abs().max()),
                    latent_mean_rms=float(
                        model.recovery(obs[mask]).square().mean().sqrt()
                    ),
                )
        dump(
            a.output,
            dict(
                checkpoint=str(a.checkpoint),
                sha256=digest(a.checkpoint),
                splits=results,
                observations="59 original deployable actor inputs",
                limitation="No base linear velocity, height, COM position, "
                "or contact force "
                "is directly observed by this feedforward residual or gate. "
                "Gate-zero fractions are initial-state measurements, not "
                "whole-rollout activation fractions.",
            ),
        )
    finally:
        p.env._backend.close()


if __name__ == "__main__":
    main()
