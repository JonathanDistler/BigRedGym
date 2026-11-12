"""Disclose runner-up failures and test frozen models under changed actions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.pendulum_models import enrich, validate
from scripts.run_pendulum_reference import (
    compatible_actor,
    new_env,
    rollout,
    GATE_RESETS,
)
from scripts.morphology_feasibility import save_csv
from experiments.morphology.task import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    a = parser.parse_args()
    r = a.run
    torch.set_num_threads(1)
    assets = Assets(r / "challenge_assets")
    original = pd.read_csv(r / "robustness_cases.csv")
    details = []
    errors = []
    forecasts = []
    one_steps = []
    interactions = 0
    for seed in (42, 43):
        net, p, _ = compatible_actor(r, seed, assets)
        env = new_env(assets, p)
        ref = pd.read_csv(r / f"reference_seed{seed}.csv")
        model = json.loads((r / f"model_seed{seed}.json").read_text())
        changed = []
        for rank in (1, 2):
            candidate = json.loads((r / f"seed{seed}_candidate{rank}.json").read_text())
            for reset in GATE_RESETS:
                world = torch.load(
                    r / f"seed{seed}_reset{reset}_start.pt",
                    weights_only=False,
                    map_location="cpu",
                )
                stats, trace, _ = rollout(env, net, world, candidate, ref, True)
                interactions += stats["steps"]
                prior = original[
                    (original.training_seed == seed)
                    & (original["rank"] == rank)
                    & (original.reset == reset)
                ].iloc[0]
                assert np.isclose(stats["peak_pitch"], prior.peak_pitch, atol=1e-10)
                d = pd.DataFrame(trace)
                tag = f"challenge_seed{seed}_rank{rank}_reset{reset}"
                d.to_csv(r / f"{tag}_trajectory.csv", index=False)
                e = enrich(d, env.mass)
                e.to_csv(r / f"{tag}_model.csv", index=False)
                changed.append((tag, e))
                invalid = d[
                    d.nonfoot_collision
                    | d.self_collision
                    | (d.max_speed_ratio > 1.001)
                    | (d.joint_violation > 0.001)
                ]
                details.append(
                    dict(
                        training_seed=seed,
                        rank=rank,
                        reset=reset,
                        **stats,
                        nonfoot_collision=bool(d.nonfoot_collision.any()),
                        self_collision=bool(d.self_collision.any()),
                        max_speed_ratio=float(d.max_speed_ratio.max()),
                        max_joint_violation=float(d.joint_violation.max()),
                        max_torque_ratio=float(d.torque_ratio.max()),
                        first_invalid_time=float(invalid.time.iloc[0] - d.time.iloc[0])
                        if len(invalid)
                        else -1.0,
                        mean_front_moment=float(d.front_moment_y.mean()),
                        mean_rear_moment=float(d.rear_moment_y.mean()),
                    )
                )
        e, f, o = validate(changed, model)
        errors.extend(e)
        forecasts.extend(f)
        one_steps.extend(o)
    save_csv(r / "failure_diagnostics.csv", details)
    save_csv(r / "changed_action_model_errors.csv", errors)
    save_csv(r / "changed_action_forecasts.csv", forecasts)
    save_csv(r / "changed_action_one_step.csv", one_steps)
    write_json(
        r / "challenge_accounting.json",
        dict(
            interactions=interactions,
            ppo_interactions=0,
            model_refit=False,
            reference_reoptimization=False,
        ),
    )


if __name__ == "__main__":
    main()
