"""Recover dynamic-progress candidates from saved CEM scores without re-search."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.task import write_json
from scripts.rotation_learning_tools import load_actor
from scripts.run_pendulum_reference import new_env
from scripts.search_dynamic_rotation import TRAIN, TEST, execute
from scripts.morphology_feasibility import save_csv


def reconstruct(root, seed):
    data = pd.read_csv(root / "search_cases.csv")
    data = data[data.seed == seed]
    population = json.loads((root / "protocol.json").read_text())["population"]
    generations = json.loads((root / "protocol.json").read_text())["generations"]
    warm = np.load(
        f"output/rotation_search_20261111_v2/A_seed{seed}_scale1.00_reset61000_search_best.npz"
    )["residual"]
    mean = np.r_[warm, warm[-1:]]
    std = np.full_like(mean, 0.07)
    rng = np.random.default_rng(74000 + seed)
    bank = {}
    for generation in range(generations):
        candidates = [
            dict(mode="residual", knots=k)
            for k in rng.normal(mean, std, (population, 4, 6))
        ]
        candidates[0] = dict(mode="residual", knots=mean.copy())
        candidates[1] = dict(mode="residual", knots=np.zeros((4, 6)))
        candidates[2] = dict(mode="hold")
        ranked = []
        for index, candidate in enumerate(candidates):
            rows = data[
                (data.generation == generation) & (data.candidate == index)
            ].sort_values("reset")
            values = rows[
                "exploration_score" if generation < 2 else "final_score"
            ].to_numpy()
            score = float(np.mean(values) - 0.5 * np.std(values))
            bank[(generation, index)] = candidate
            if candidate["mode"] == "residual":
                ranked.append((score, candidate["knots"]))
        elite = np.stack(
            [c for _, c in sorted(ranked, key=lambda r: r[0], reverse=True)[:6]]
        )
        mean = 0.3 * mean + 0.7 * elite.mean(0)
        std = np.maximum(0.015, 0.3 * std + 0.7 * elite.std(0))
    return bank, data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    a = parser.parse_args()
    out = a.run / "frontier"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(out / "assets")
    records = []
    replays = []
    count = 0
    plans = []
    for seed in (42, 43):
        bank, data = reconstruct(a.run, seed)
        net, _, p = load_actor(
            Path(
                f"output/multimorph_bc_ppo_continuation_20261111_v2/A_seed{seed}/checkpoint_000030.pt"
            )
        )
        env = new_env(assets, p)
        # Preselect using development data only: best collision-free upward
        # progression, highest safe-prefix progression, and best worst-reset peak.
        native = data[
            ~data.nonfoot_collision
            & ~data.self_collision
            & (data.max_speed_ratio <= 1.001)
            & (data.max_joint_violation < 0.001)
        ]
        selected = []
        if len(native[native.cross45]):
            row = (
                native[native.cross45]
                .sort_values("valid_prefix_peak", ascending=False)
                .iloc[0]
            )
            selected.append((int(row.generation), int(row.candidate)))
        row = data.sort_values("valid_prefix_peak", ascending=False).iloc[0]
        selected.append((int(row.generation), int(row.candidate)))
        worst = (
            data.groupby(["generation", "candidate"])
            .valid_prefix_peak.min()
            .sort_values(ascending=False)
        )
        selected.append(tuple(map(int, worst.index[0])))
        selected = list(dict.fromkeys(selected))
        for label, (generation, index) in enumerate(selected):
            candidate = bank[(generation, index)]
            tag = f"seed{seed}_dynamic{label}"
            write_json(
                out / f"{tag}.json",
                dict(
                    mode=candidate["mode"],
                    knots=np.asarray(candidate["knots"]).tolist(),
                    source_generation=generation,
                    source_candidate=index,
                ),
            )
            # Verify the recovered actions against every original development
            # rollout before drawing any conclusion from new validation resets.
            for reset in TRAIN:
                state = torch.load(
                    a.run / f"seed{seed}_reset{reset}_start.pt",
                    weights_only=False,
                    map_location="cpu",
                )
                result, trace, controls = execute(
                    env, net, state, candidate, record=True
                )
                count += result["steps"]
                old = data[
                    (data.generation == generation)
                    & (data.candidate == index)
                    & (data.reset == reset)
                ].iloc[0]
                if not np.isclose(result["peak_pitch"], old.peak_pitch, atol=1e-6):
                    raise ValueError(
                        "CEM reconstruction did not reproduce the saved trajectory"
                    )
                replays.append(dict(seed=seed, label=label, reset=reset, **result))
                if reset == 61000 or (
                    not result["nonfoot_collision"] and result["cross45"]
                ):
                    save_csv(out / f"{tag}_development{reset}_trajectory.csv", trace)
                    np.savez_compressed(
                        out / f"{tag}_development{reset}_states.npz",
                        **{
                            key: np.array([r[key] for r in controls])
                            for key in controls[0]
                        },
                    )
            for reset in TEST:
                state = torch.load(
                    a.run / f"seed{seed}_reset{reset}_start.pt",
                    weights_only=False,
                    map_location="cpu",
                )
                result, trace, controls = execute(
                    env, net, state, candidate, record=True
                )
                count += result["steps"]
                records.append(dict(seed=seed, label=label, reset=reset, **result))
                save_csv(out / f"{tag}_reset{reset}_trajectory.csv", trace)
                np.savez_compressed(
                    out / f"{tag}_reset{reset}_states.npz",
                    **{
                        key: np.array([r[key] for r in controls]) for key in controls[0]
                    },
                )
            plans.append(
                dict(seed=seed, label=label, generation=generation, candidate=index)
            )
    save_csv(out / "development_reproduction.csv", replays)
    save_csv(out / "validation_cases.csv", records)
    # Collision-free dynamic progression is logged independently of front
    # support and terminal recovery; it is never relabeled as controlled success.
    native_progress = [
        r
        for r in records + replays
        if r["cross45"]
        and r["progress_after45_degrees"] >= 5
        and not r["nonfoot_collision"]
        and not r["self_collision"]
        and r["max_speed_ratio"] <= 1.001
        and r["max_joint_violation"] < 0.001
    ]
    if native_progress:
        write_json(
            out / "pilot_preparation.json",
            dict(
                status="prepared_but_disabled",
                updates=10,
                maximum_updates=20,
                training_seeds=[42, 43],
                existing_environment="experiments.morphology.learning.rotation.RotationEnvironment",
                existing_runner="scripts.run_rotation_learning.RotationRun",
                source_update=30,
                bc_initial_coefficient=2.0,
                bc_floor=2.0,
                abrupt_bc_removal=False,
                learning_rate=1e-5,
                steps_per_update=1024,
                evaluations_every=2,
                launch_enabled=False,
                blocking_requirements=[
                    "Robust front-foot support/recontact",
                    "Separate terminal recovery audit",
                ],
                checkpoint_template="output/multimorph_bc_ppo_continuation_20261111_v2/A_seed{seed}/checkpoint_000030.pt",
                note="Original BC prefix retained; development data remain unapproved.",
            ),
        )
    terminal_robust = all(
        any(
            np.mean(
                [
                    r["improved"] and r["terminal_recovery_60"]
                    for r in records
                    if r["seed"] == seed and r["label"] == label
                ]
            )
            >= 0.8
            for label in {r["label"] for r in records if r["seed"] == seed}
        )
        for seed in (42, 43)
    )
    supported_robust = all(
        any(
            np.mean(
                [
                    r["improved"]
                    for r in records
                    if r["seed"] == seed and r["label"] == label
                ]
            )
            >= 0.8
            for label in {r["label"] for r in records if r["seed"] == seed}
        )
        for seed in (42, 43)
    )
    write_json(
        out / "completion.json",
        dict(
            selections=plans,
            verification_and_validation_steps=count,
            ppo_interactions=0,
            native_progress_cases=len(native_progress),
            controlled_improvement_cases=sum(r["improved"] for r in records),
            terminal_recovery_cases=sum(r["terminal_recovery_60"] for r in records),
            molab_go=bool(supported_robust),
            terminal_recovery_go=bool(terminal_robust),
            note="Verified CEM reconstruction; no reoptimization.",
        ),
    )


if __name__ == "__main__":
    main()
