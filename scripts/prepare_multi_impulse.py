# ruff: noqa: E501
"""Validate actual handstand resets and collect native teacher supervision."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from experiments.multi_impulse.core import MultiConfig, MultiEnvironment
from experiments.multi_impulse.balance import equilibrium, BalanceTeacher
from experiments.front_catch.data import raw_state, validate_state, sha
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.physics import World, feasible_state
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(
        "output/multi_impulse_diagnosis_v1/B_best.pt",
        weights_only=False,
        map_location="cpu",
    )
    parent = saved["runs"]["B_seed42"]
    cfg = MultiConfig(**parent["config"])
    cfg.kp = 60.0
    cfg.kd = 3.0
    cfg.capture_enabled = False
    cfg.balance_training = True
    cfg.seconds = 6.2
    cfg.target_base_height = 0.57
    assets = Assets(a.output / "assets")
    q, target, meta = equilibrium(assets, cfg)
    cfg.target_height = float(meta["geometry"]["com"][2])
    e = MultiEnvironment(assets, cfg)
    teacher = BalanceTeacher(e, q, target)
    world = World(assets, cfg)
    bank = {
        split: {k: [] for k in ("near_vertical", "75", "60", "45")}
        for split in ("train", "evaluation")
    }
    records = []
    observations = []
    actions = []
    for split, base, count in [("train", 70000, 12), ("evaluation", 71000, 6)]:
        for i in range(count):
            pitch = [90, 88, 92, 86, 89, 91][i % 6]
            omega = [0, 0.2, -0.2, 0.35, -0.35, 0.1][i % 6]
            good, details = feasible_state(
                world,
                q[e.qids],
                base + i,
                pitch=pitch,
                omega=omega,
                com_shift=(i % 3 - 1) * 0.004,
            )
            if not good:
                records.append(
                    dict(
                        split=split, seed=base + i, initial_valid=False, details=details
                    )
                )
                continue
            try:
                raw_state(
                    e,
                    "near_vertical",
                    base + i,
                    qpos=world.d.qpos.copy(),
                    qvel=world.d.qvel.copy(),
                    target=target,
                )
            except ValueError as exc:
                records.append(
                    dict(
                        split=split, seed=base + i, initial_valid=False, error=str(exc)
                    )
                )
                continue
            state = e.state()
            frames = []
            hold = 0.0
            for step in range(310):
                obs, _ = e.observations()
                action = teacher.command(e)
                frames.append((obs.copy(), action.copy()))
                _, _, done, info = e.step(action)
                hold = max(hold, e.multi_best_hold)
                if done:
                    break
            recoverable = hold >= 5.0
            record = dict(
                split=split,
                seed=base + i,
                initial_valid=True,
                pitch=pitch,
                omega=omega,
                teacher_hold=hold,
                recoverable_5s=recoverable,
                termination=info["termination"],
            )
            records.append(record)
            print("balance probe", record, flush=True)
            if recoverable:
                bank[split]["near_vertical"].append(dict(state=state, source=record))
                if split == "train":
                    observations.extend(f[0] for f in frames)
                    actions.extend(f[1] for f in frames)
    # Lower-angle proposals must pass a native 100ms safety probe. They are
    # transient states, not certified recoverable states, and cannot advance a
    # curriculum until the learned policy demonstrates recovery from them.
    for split, base in [("train", 72000), ("evaluation", 73000)]:
        for j, pitch in enumerate((80, 75, 65, 60, 50, 45)):
            good, _ = feasible_state(world, q[e.qids], base + j, pitch=pitch, omega=0.3)
            if not good:
                continue
            try:
                raw_state(
                    e,
                    "partial",
                    base + j,
                    qpos=world.d.qpos.copy(),
                    qvel=world.d.qvel.copy(),
                    target=target,
                )
            except ValueError:
                continue
            state = e.state()
            valid = True
            for step in range(5):
                _, _, done, _ = e.step(teacher.command(e))
                valid &= validate_state(e)[0] and not done
            tier = "75" if pitch >= 75 else "60" if pitch >= 60 else "45"
            if valid:
                bank[split][tier].append(
                    dict(
                        state=state,
                        source=dict(
                            pitch=pitch, probe_seconds=0.1, recovery_certified=False
                        ),
                    )
                )
    if not bank["train"]["near_vertical"] or not bank["evaluation"]["near_vertical"]:
        raise RuntimeError(
            "No independently validated recoverable handstand resets; do not fabricate a bank"
        )
    torch.save(bank, a.output / "balance_bank.pt")
    np.savez_compressed(
        a.output / "balance_demonstrations.npz",
        observations=np.stack(observations),
        actions=np.stack(actions),
    )
    np.savez_compressed(
        a.output / "teacher.npz",
        q=q,
        target=target,
        K=teacher.K,
        indices=teacher.indices,
    )
    dump(a.output / "validation.json", records)
    dump(
        a.output / "manifest.json",
        dict(
            config=cfg.__dict__,
            equilibrium=meta,
            teacher_closed_loop_max_eigenvalue=teacher.max_eigenvalue,
            bank_counts={s: {k: len(v) for k, v in g.items()} for s, g in bank.items()},
            bank_sha256=sha(a.output / "balance_bank.pt"),
            demonstrations_sha256=sha(a.output / "balance_demonstrations.npz"),
            observation_time_denominator=8.0,
            teacher_role="Offline validator/supervision only; one learned actor deployed",
            original_B_sha256=sha("output/multi_impulse_diagnosis_v1/B_best.pt"),
        ),
    )


if __name__ == "__main__":
    main()
