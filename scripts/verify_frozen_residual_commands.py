"""Lockstep MuJoCo verification of zero-head commands, phases and memory."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.frozen_balance_residual import FrozenBalanceResidual
from gym.envs.go2.recovery_frontier import install_samples, sample_from_arrays
from scripts.evaluate_frozen_residual import attach
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    baseline = Policy(BALANCE, "go2balance", 16, True)
    wrapped = Policy(BALANCE, "go2balance", 16, True)
    attach(wrapped, FrozenBalanceResidual(wrapped.runner.alg.actor))
    fields = (
        "dof_pos_target",
        "leg_phase",
        "_cpg_load",
        "dof_vel_target",
        "tau_ff",
        "dof_pos_history",
        "root_states",
    )
    maxima = dict.fromkeys(fields, 0.0)
    count = 0
    try:
        for split in ("train", "validation", "heldout"):
            with np.load(f"datasets/recovery_frontier_v4/{split}.npz") as f:
                arrays = {k: f[k] for k in f.files}
            for start in range(0, len(arrays["seed"]), 16):
                samples = [
                    sample_from_arrays(arrays, i)
                    for i in range(start, min(start + 16, len(arrays["seed"])))
                ]
                install_samples(baseline, samples)
                install_samples(wrapped, samples)
                for _ in range(20):
                    baseline.action()
                    wrapped.action()
                    for field in fields:
                        x, y = getattr(baseline.env, field), getattr(wrapped.env, field)
                        error = float((x - y).abs().max())
                        maxima[field] = max(maxima[field], error)
                        assert torch.equal(x, y), (split, start, field, error)
                    x = baseline.env._standing_posture() + baseline.env.dof_pos_target
                    y = wrapped.env._standing_posture() + wrapped.env.dof_pos_target
                    assert torch.equal(x, y)
                    baseline.env.step()
                    wrapped.env.step()
                count += len(samples)
        dump(
            a.output,
            dict(
                states=count,
                control_steps_per_state=20,
                exact_command_and_controller_parity=True,
                maximum_errors=maxima,
            ),
        )
    finally:
        baseline.env._backend.close()
        wrapped.env._backend.close()


if __name__ == "__main__":
    main()
