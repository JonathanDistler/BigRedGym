"""Prepare a 10–20-update pilot; execution requires robust supported progression.

Uses existing RotationRun/RotationEnvironment without changing their rewards,
observations or actuator interface. Default is preparation only.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.task import write_json
from scripts.run_rotation_learning import migrate, evaluate_run
from scripts.morphology_feasibility import save_csv


def supported_gate(root):
    path = root / "frontier/validation_cases.csv"
    if not path.exists():
        return False
    d = pd.read_csv(path)
    required = {
        "improved",
        "cross45",
        "full_valid",
        "nonfoot_collision",
        "self_collision",
        "max_speed_ratio",
        "max_joint_violation",
        "max_torque_ratio",
        "contact_gap",
        "front_drift",
        "slip_distance",
    }
    if not required.issubset(d.columns):
        return False
    d["eligible"] = (
        d.improved
        & d.cross45
        & d.full_valid
        & ~d.nonfoot_collision
        & ~d.self_collision
        & (d.max_speed_ratio <= 1.001)
        & (d.max_joint_violation < 0.001)
        & (d.max_torque_ratio <= 1.00001)
        & (d.contact_gap <= 0.02000001)
        & (d.front_drift <= 0.04)
        & (d.slip_distance <= 0.02)
    )
    return all(
        any(
            len(g) == 10 and g.reset.nunique() == 10 and g.eligible.mean() >= 0.8
            for _, g in d[d.seed == seed].groupby("label")
        )
        for seed in (42, 43)
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--validation-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10, choices=range(10, 21))
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    passed = supported_gate(a.validation_run)
    plan = dict(
        updates=a.updates,
        maximum_updates=20,
        training_seeds=[42, 43],
        steps_per_update=1024,
        planned_ppo_interactions=2 * a.updates * 1024,
        bc_weight=2.0,
        bc_floor=2.0,
        learning_rate=1e-5,
        source="output/multimorph_bc_ppo_continuation_20261111_v2",
        source_update=30,
        environment="Existing RotationEnvironment; no rebuild",
        runner="Existing RotationRun/migrate",
        gate_passed=bool(passed),
        terminal_recovery="Separate endpoint required before extended training",
        execute_requested=a.execute,
        wandb="disabled",
    )
    write_json(a.output / "pilot_plan.json", plan)
    if not a.execute or not passed:
        write_json(
            a.output / "completion.json",
            dict(
                ppo_interactions=0,
                actor_loaded=False,
                reason="prepare_only"
                if not a.execute
                else "supported_progression_gate_failed",
            ),
        )
        print("Pilot prepared; no training. Supported progression gate:", passed)
        return
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    runs = []
    records = []
    traces = a.output / "traces"
    traces.mkdir()
    cfg = json.loads(
        Path("experiments/morphology/learning/protocol_rotation.json").read_text()
    )
    cfg["rotation_experiment"]["phase_labels"] = (
        "output/rotation_diagnosis_20261111_v1/bc_phase_labels.npz"
    )
    cfg["evaluation_seeds"] = list(range(67500, 67508))
    cfg["scales"] = [1.0]
    for seed in (42, 43):
        runs.append(
            migrate(
                a.output / f"A_seed{seed}",
                "A",
                seed,
                Path(plan["source"]),
                copy.deepcopy(cfg),
                assets,
            )
        )
    reason = "completed bounded pilot"
    for offset in range(a.updates + 1):
        if offset:
            for run in runs:
                run.collect_and_update()
        if offset % 2 == 0 or offset == a.updates:
            for run in runs:
                run.checkpoint_and_evaluate()
                records.extend(
                    evaluate_run(run, "global", cfg["evaluation_seeds"], traces)
                )
            save_csv(a.output / "probes.csv", records)
            recent = [r for r in records if r["iteration"] == 30 + offset]
            retention = all(
                np.mean(
                    [
                        r["takeoff"]
                        for r in recent
                        if r["training_seed"] == seed
                        and r["stochastic"] == noise
                        and r["scale"] == 1.0
                    ]
                )
                >= 0.75
                for seed in (42, 43)
                for noise in (False, True)
            )
            divergence = offset and any(
                not np.isfinite(r.training[-1]["post_update_kl"])
                or r.training[-1]["post_update_kl"] > 0.03
                or r.training[-1]["clip_fraction"] > 0.3
                for r in runs
            )
            if not retention or divergence:
                reason = "matched stop: retention or divergence"
                break
    write_json(
        a.output / "completion.json",
        dict(
            ppo_interactions=sum(r.actual_steps - 30720 for r in runs),
            reason=reason,
            extended_training=False,
        ),
    )


if __name__ == "__main__":
    main()
