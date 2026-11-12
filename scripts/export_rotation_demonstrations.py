"""Replay bounded-search prefixes as optional same-actor demonstration data."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.task import write_json
from scripts.rotation_learning_tools import action, environment, load_actor
from scripts.morphology_feasibility import save_csv


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    a = parser.parse_args()
    torch.set_num_threads(1)
    assets = Assets(a.search / "demo_assets")
    summaries = []
    contexts = json.loads((a.diagnosis / "contexts.json").read_text())
    search = {
        r["context"]: r
        for r in json.loads((a.search / "search_summary.json").read_text())
    }
    for context in contexts:
        tag = context["tag"]
        saved = torch.load(context["path"], weights_only=False, map_location="cpu")
        net, _, p = load_actor(Path(saved["checkpoint"]))
        planned = np.load(a.search / f"{tag}_search_best.npz")["actions"]
        for seed in (
            [61000, 61002, 61003, 61004]
            if search[tag]["optional_demo_valid"]
            else [61000]
        ):
            env = environment(assets, p)
            obs, _ = env.reset(saved["scale"], seed)
            env.record_rotation_trace = True
            frames = []
            post_index = 0
            post = False
            while True:
                if env.tracker.confirmed:
                    post = True
                if post and post_index == len(planned):
                    break
                act = planned[post_index] if post else action(net, obs)
                frames.append(
                    dict(
                        observations=obs.copy(),
                        actions=act.copy(),
                        qpos=env.d.qpos.copy(),
                        qvel=env.d.qvel.copy(),
                        target=env.target.copy(),
                        time=float(env.d.time),
                        certified_takeoff=bool(env.tracker.confirmed),
                        scale=env.scale,
                        seed=seed,
                    )
                )
                (obs, _), _, done, info = env.step(act)
                if post:
                    post_index += 1
                if done:
                    break
            valid = (
                not info["nonfoot_collision"]
                and not info["self_collision"]
                and info["max_speed_ratio"] <= 1.001
                and info["joint_limit_violation"] < 0.001
            )
            qualifies = bool(
                valid
                and info["max_pitch"] >= np.deg2rad(45)
                and env.tracker.confirmed
                and post_index == len(planned)
            )
            if seed == 61000:
                np.savez_compressed(
                    a.search / f"{tag}_replayed_prefix.npz",
                    **{key: np.array([r[key] for r in frames]) for key in frames[0]},
                )
                if qualifies:
                    np.savez_compressed(
                        a.search / f"{tag}_optional_rotation_demo.npz",
                        **{
                            key: np.array([r[key] for r in frames]) for key in frames[0]
                        },
                    )
            summaries.append(
                dict(
                    context=tag,
                    evaluation_seed=seed,
                    valid=bool(valid),
                    takeoff=info["takeoff"],
                    peak_pitch=info["max_pitch"],
                    final_pitch=info["pitch"],
                    final_omega=info["pitch_velocity"],
                    nonfoot_collision=info["nonfoot_collision"],
                    post_steps=post_index,
                    optional_demonstration=qualifies,
                    interactions=len(frames),
                )
            )
    save_csv(a.search / "demonstration_replay.csv", summaries)
    write_json(
        a.search / "demonstration_note.json",
        dict(
            used_in_pilot=False,
            inference="Optional actor demonstration; no separate inference controller.",
            horizon="Half-second search prefixes; not validated handstands.",
        ),
    )


if __name__ == "__main__":
    main()
