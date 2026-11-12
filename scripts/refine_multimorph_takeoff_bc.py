"""Collect perturbed teacher prefixes; BC labels follow the actual target history."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from scripts.prepare_multimorph_takeoff_bc import (
    adapted_action,
    crouch_pose,
    fit,
    SCALES,
)
from scripts.morphology_feasibility import save_csv
from experiments.morphology.task import write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--latent-std", type=float, default=0.01)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    gate = json.loads((a.source / "teacher_gate.json").read_text())
    if not gate["passed"]:
        raise ValueError("Teacher gate failed")
    assets = Assets(a.output / "assets")
    extra = []
    episodes = []
    for scale in SCALES:
        env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
        crouch = crouch_pose(assets, scale)
        for seed in range(51000, 51016):
            obs, _ = env.reset(scale, seed)
            rng = np.random.default_rng(seed + int(scale * 1000))
            frames = []
            for _ in range(70):
                teacher, error = adapted_action(
                    env,
                    crouch,
                    gate["parameters"][str(scale)],
                    gate["source_reference"],
                )
                applied = np.clip(teacher + rng.normal(0, 0.01, 12), -1, 1)
                frame = dict(
                    observations=obs.copy(),
                    actions=teacher.copy(),
                    qpos=env.d.qpos.copy(),
                    qvel=env.d.qvel.copy(),
                    target=env.target.copy(),
                    contacts=env.row["loads"].copy(),
                    com=env.row["com"].copy(),
                    time=env.steps * env.cfg.dt,
                    seed=seed,
                    scale=scale,
                )
                (obs, _), _, done, info = env.step(applied)
                frame.update(
                    torque=env.d.qfrc_applied[env.vids].copy(),
                    commanded_target=env.target.copy(),
                )
                frames.append(frame)
                if (
                    info["takeoff"]
                    and info["genuine_flight_pitch"] >= np.deg2rad(25)
                    and info["genuine_airborne_seconds"] >= 0.06
                ):
                    break
                if done:
                    break
            accepted = bool(
                info["takeoff"]
                and info["genuine_flight_pitch"] >= np.deg2rad(20)
                and info["genuine_airborne_seconds"] >= 0.04
                and not info["nonfoot_collision"]
                and not info["self_collision"]
                and info["max_speed_ratio"] <= 1.001
                and info["joint_limit_violation"] < 0.001
            )
            episodes.append(dict(**info, seed=seed, accepted=accepted))
            if accepted:
                extra.extend(frames)
        print(
            "PERTURBED TEACHER",
            scale,
            sum(r["accepted"] for r in episodes if r["scale"] == scale),
            flush=True,
        )
    save_csv(a.output / "perturbed_teacher_episodes.csv", episodes)
    old = np.load(a.source / "bc_dataset.npz")
    data = {
        k: np.concatenate([old[k], np.stack([r[k] for r in extra])]) for k in old.files
    }
    np.savez_compressed(a.output / "bc_dataset.npz", **data)
    shutil.copyfile(a.source / "teacher_gate.json", a.output / "teacher_gate.json")
    write_json(
        a.output / "augmentation.json",
        dict(
            accepted_episodes=sum(r["accepted"] for r in episodes),
            attempted_episodes=len(episodes),
            frames=len(extra),
            source=str(a.source),
            action_noise_std=0.01,
            label=(
                "Unperturbed feedback-teacher action at actual noisy rollout "
                "state/target history; accepted collision-free certified "
                "prefixes only; same source-episode split"
            ),
        ),
    )
    fit(
        assets,
        a.output,
        a.updates,
        initial_actor=a.source / "bc_actor.pt",
        latent_std=a.latent_std,
        selection_start=56000,
        final_start=57000,
    )


if __name__ == "__main__":
    main()
