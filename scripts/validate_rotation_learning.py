"""Executable pre-training checks; no policy optimization."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.rotation import pendulum_state, potential_increment
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from experiments.morphology.task import write_json
from scripts.rotation_learning_tools import load_actor, environment, action
from scripts.run_takeoff_ppo import REWARD_TERMS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    root = a.diagnosis
    checks = {}
    s = pendulum_state(np.array([-0.2, 0, 0.3]), np.array([0.1, 0, 0]), 10.0, 0.5, 0.4)
    checks["angle_convention"] = (
        np.pi / 2 < s["pendulum_theta"] < np.pi and s["pendulum_omega"] > 0
    )
    checks["energy_barrier"] = bool(
        np.isclose(s["potential_barrier"], 10 * 9.81 * (np.hypot(0.2, 0.3) - 0.3))
    )
    checks["terminal_potential"] = potential_increment(0.8, 100, 0.995, True) == -0.8
    checks["stationary_potential_nonpositive"] = (
        potential_increment(0.4, 0.4, 0.995) < 0
    )
    net, _, p = load_actor(
        Path(
            "output/multimorph_bc_ppo_continuation_20261111_v2/A_seed42/checkpoint_000030.pt"
        )
    )
    assets = Assets(root / "validation_assets")
    new = environment(assets, p, True)
    old = TakeoffEnvironment(
        assets, TakeoffConfig(**dict(p["environment"], terminate_on_goal=False))
    )
    obs, _ = new.reset(1.0, 63000)
    old.reset(1.0, 63000)
    errors = []
    takeoff_parity = []
    physical = []
    finite = []
    while True:
        act = action(net, obs)
        (obs, _), reward, done, info = new.step(act)
        _, original, old_done, old_info = old.step(act)
        physical.append(np.array_equal(new.d.qpos, old.d.qpos) and done == old_done)
        takeoff_parity.append(info["takeoff"] == old_info["takeoff"])
        errors.append(
            abs(original - sum(info[f"reward_{key}"] for key in REWARD_TERMS))
        )
        finite.append(np.isfinite(reward))
        if done:
            break
    checks.update(
        physics_parity=all(physical),
        takeoff_reward_unchanged=max(errors) < 1e-10,
        takeoff_certificate_unchanged=all(takeoff_parity),
        finite_new_rewards=bool(all(finite)),
    )
    labels = np.load(root / "bc_phase_labels.npz")["weights"]
    checks["phase_weights_bounded"] = bool(np.all((labels >= 0.1) & (labels <= 1)))
    trajectory = pd.read_csv(root / "A_seed42_scale1.00_reset61000_trajectory.csv")
    f = trajectory[trajectory.takeoff & ~trajectory.nonfoot_collision]
    measured = np.diff(f.momentum_y) / np.diff(f.time)
    predicted = (f.front_moment_y + f.rear_moment_y).to_numpy()[:-1]
    residual = float(np.sqrt(np.mean((measured - predicted) ** 2)))
    checks["contact_moment_sign_and_scale"] = residual < 3.0
    result = dict(
        passed=bool(all(checks.values())),
        checks=checks,
        max_takeoff_reward_error=max(errors),
        momentum_rate_residual_rms=residual,
        rotation_source_sha256=hashlib.sha256(
            Path("experiments/morphology/learning/rotation.py").read_bytes()
        ).hexdigest(),
        model_note="Approximate pendulum residual is reported, not asserted exact.",
    )
    write_json(root / "validation.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
