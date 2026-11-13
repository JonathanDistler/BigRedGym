"""Capture certified actor prefixes and revalidate archived states; no search."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.front_catch.data import (
    INITIAL,
    network,
    config,
    validate_state,
    raw_state,
    state_id,
    sha,
)
from experiments.front_catch.core import CatchEnvironment
from experiments.morphology.learning.environment import Assets
from scripts.run_com_energy_phase import dump
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path, default=INITIAL)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    net, saved = network(a.initial)
    env = CatchEnvironment(assets, config(saved))
    kinds = ("loaded", "pre_liftoff", "post_liftoff", "partial", "near_vertical")
    bank = {split: {k: [] for k in kinds} for split in ("train", "evaluation")}
    records = []
    frames = []
    seed_frames = []
    ids = set()

    def admit(split, kind, source):
        valid, details = validate_state(env)
        key = state_id(env)
        record = dict(split=split, kind=kind, source=source, state_id=key, **details)
        records.append(record)
        if valid and key not in ids:
            ids.add(key)
            bank[split][kind].append(
                dict(state=env.state(), source=source, state_id=key)
            )

    for seed in range(95000, 95006):
        obs, _ = env.reset(1.0, seed)
        split = "train" if seed < 95004 else "evaluation"
        seen = set()
        last = None
        while True:
            with torch.no_grad():
                action = net.actor(torch.tensor(obs)).tanh().numpy()
            before_obs = obs.copy()
            before_phase = env.phase.progress
            (obs, _), _, done, info = env.step(action)
            valid, _ = validate_state(env)
            if valid and not env.nonfoot_collision:
                frames.append((before_obs, action, before_phase))
                seed_frames.append(seed)
                if (
                    "loaded" not in seen
                    and 0.18 <= env.d.time <= 0.3
                    and env.row["front_support"]
                ):
                    admit(split, "loaded", f"certified_actor_seed{seed}")
                    seen.add("loaded")
                if not env.tracker.confirmed and env.d.time >= 0.6:
                    last = env.state()
                if env.tracker.confirmed and "post_liftoff" not in seen:
                    if last is not None:
                        current = env.state()
                        env.restore(last)
                        admit(split, "pre_liftoff", f"certified_actor_seed{seed}")
                        env.restore(current)
                    admit(split, "post_liftoff", f"certified_actor_seed{seed}")
                    seen.add("post_liftoff")
                if (
                    env.tracker.confirmed
                    and env.row["pitch"] >= np.deg2rad(30)
                    and "partial" not in seen
                ):
                    admit(split, "partial", f"certified_actor_seed{seed}")
                    seen.add("partial")
            if done:
                break
        print("capture", seed, info["certified_standing_takeoff"], flush=True)
    archive = Path("output/recoverability_maps_20261112/recoverable_states.npz")
    old = np.load(archive)
    labels = __import__("json").loads(
        archive.with_name("validated_region.json").read_text()
    )
    for i, label in enumerate(labels):
        split = "train" if label["seed"] % 3 else "evaluation"
        valid, _ = raw_state(
            env,
            "near_vertical",
            i,
            integration=old[f"{i}_integration"],
            target=old[f"{i}_target"],
        )
        if abs(env.momentum_sample["com_height"] - label["features"]["com_z"]) > 1e-7:
            raise ValueError("Archive geometry mismatch")
        if valid and env.row["front_support"]:
            admit(split, "near_vertical", f"{archive}:{i}")
    # A small, fixed set of old trajectories; verify native scale and recorded COM.
    paths = [
        Path(
            "output/pendulum_guided_rotation_v2/seed42_candidate0_reset65106_states.npz"
        ),
        Path("output/dynamic_pass_through_v1/seed42_rank0_reset67203_states.npz"),
    ]
    imported = []
    for path in paths:
        if not path.exists():
            continue
        data = np.load(path)
        table = pd.read_csv(str(path).replace("_states.npz", "_trajectory.csv"))
        candidates = []
        for j, time in enumerate(data["time"]):
            row = table.iloc[int(np.argmin(abs(table.time - time)))]
            if (
                row.scale != 1.0
                or abs(row.time - time) > 0.003
                or not bool(row.takeoff)
            ):
                continue
            torque = np.array([row[f"torque_{k}"] for k in range(12)])
            valid, details = raw_state(
                env,
                "partial",
                int(j),
                qpos=data["qpos"][j],
                qvel=data["qvel"][j],
                target=data["target"][j],
                time=float(time),
                torque=torque,
            )
            matched = abs(env.momentum_sample["com_height"] - row.com_z) < 1e-7
            if (
                valid
                and matched
                and details["front_support"]
                and 0.55 <= details["pitch"] <= 1.25
                and details["rear_clearance"] > 0.015
            ):
                candidates.append((details["pitch"], env.state(), j))
        for rank, (pitch, state, index) in enumerate(
            sorted(candidates, reverse=True, key=lambda x: x[0])[:4]
        ):
            env.restore(state)
            admit("train", "partial", f"{path}:{index}")
            imported.append(dict(path=str(path), index=index, pitch=pitch))
    for split in bank:
        for kind in kinds:
            if not bank[split][kind]:
                raise RuntimeError(f"No valid {split} {kind} resets")
    # Final-phase state/action labels from the independently validated balance policy.
    roles = ["takeoff"] * len(frames)
    for split in ("train", "evaluation"):
        for item in bank[split]["near_vertical"]:
            env.restore_start(item["state"], "near_vertical")
            actor, _ = env.observations()
            action = np.clip((env.target - env.nominal) / env.cfg.action_scale, -1, 1)
            frames.append((actor, action, 0.8))
            seed_frames.append(95000 if split == "train" else 95005)
            roles.append("balance")
    # Full bank and dataset are immutable inputs to all matched conditions.
    torch.save(bank, a.output / "reset_bank.pt")
    np.savez_compressed(
        a.output / "bc_demonstrations.npz",
        observations=np.stack([f[0] for f in frames]),
        actions=np.stack([f[1] for f in frames]),
        phase=np.array([f[2] for f in frames]),
        seed=np.array(seed_frames),
        role=np.array(roles),
    )
    save_csv(a.output / "state_validation.csv", records)
    dump(
        a.output / "manifest.json",
        dict(
            initial=str(a.initial),
            initial_sha256=sha(a.initial),
            bank_sha256=sha(a.output / "reset_bank.pt"),
            dataset_sha256=sha(a.output / "bc_demonstrations.npz"),
            bank_counts={
                split: {k: len(v) for k, v in groups.items()}
                for split, groups in bank.items()
            },
            imported_trajectories=imported,
            reference_energy=env.mass * 9.81 * env.cfg.target_height,
            physical_scale=1.0,
            model_mass=env.mass,
            foot_penetration_tolerance=0.0025,
            note=(
                "Native soft foot compression allowed; nonfoot contacts rejected. "
                "Intermediate starts never count as standing success."
            ),
        ),
    )


if __name__ == "__main__":
    main()
