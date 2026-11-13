"""Matched A/B/C momentum diagnostic. Hard cap: four PPO updates, no extended mode."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.com_energy_phase.data import sha
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import advantages, update
from experiments.momentum_phase.core import MomentumEnvironment
from experiments.momentum_phase.tools import network, configuration, evaluate
from scripts.run_com_energy_phase import dump
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--updates", type=int, default=2, choices=range(0, 5))
    p.add_argument("--steps", type=int, default=256, choices=(128, 256))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    a = p.parse_args()
    gate = json.loads((a.validation / "validation.json").read_text())
    if not gate["passed"] or not gate["certified_initialization"]:
        raise ValueError("Physics gate failed")
    for file, digest in gate["source_sha256"].items():
        if sha(file) != digest:
            raise ValueError(f"Validation source changed: {file}")
    initial = Path(gate["initial_checkpoint"])
    if sha(initial) != gate["initial_sha256"]:
        raise ValueError("Initialization hash changed")
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit_tests/test_momentum_phase.py",
            "tests/unit_tests/test_com_energy_phase.py",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    (out / "validation_tests.txt").write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError("Pre-PPO tests failed")
    assets = Assets(out / "assets")
    ref = json.loads((a.validation / "reference.json").read_text())
    envelope = json.loads((a.validation / "momentum_envelope.json").read_text())
    _, parent = network(initial)
    times = parent["config"]["phase_times"]
    data = np.load(a.validation / "takeoff_demonstrations.npz")
    balance = np.load(a.validation / "balance_demonstrations.npz")
    hp = json.loads(
        Path("experiments/morphology/learning/protocol_takeoff.json").read_text()
    )["ppo"]
    hp.update(
        epochs=2,
        batch_size=64,
        bc_weight=0.02,
        bc_latent_scale=0.05,
        bc_batch_size=128,
        bc_phase_fraction=1.0,
    )
    source = out / "source"
    source.mkdir()
    for f in [
        Path(__file__),
        Path("scripts/validate_momentum_phase.py"),
        *Path("experiments/momentum_phase").glob("*.py"),
    ]:
        shutil.copyfile(f, source / f.name)
    dump(
        out / "protocol.json",
        dict(
            updates=a.updates,
            steps=a.steps,
            seeds=a.seeds,
            evaluation_seeds=[94000, 94001],
            checkpoint=str(initial),
            checkpoint_sha256=sha(initial),
            original_bc_sha256=gate["initial_bc_sha256"],
            validation=str(a.validation),
            validation_sha256=sha(a.validation / "validation.json"),
            offline_takeoff_sha256=sha(a.validation / "takeoff_demonstrations.npz"),
            offline_balance_sha256=sha(a.validation / "balance_demonstrations.npz"),
            common_architecture=dict(
                actor=61, critic=91, hidden=[256, 256, 128], actions=12
            ),
            phase_filter=False,
            geometry=1.0,
            ppo=hp,
            source_sha256={str(f): sha(f) for f in source.glob("*.py")},
        ),
    )
    all_results = []
    for seed in a.seeds:
        for condition in "ABC":
            directory = out / f"{condition}_seed{seed}"
            directory.mkdir()
            cfg = configuration(ref, times, condition, envelope)
            dump(directory / "config.json", asdict(cfg))
            env = MomentumEnvironment(assets, cfg)
            ev = MomentumEnvironment(assets, copy.deepcopy(cfg))
            net, _ = network(initial, seed)
            optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
            train = data["seed"] < 93004
            x = data["observations"][train]
            y = data["actions"][train]
            phase = data["phase"][train]
            weights = np.where(
                phase < 0.4, 1.0, 0.1 if condition == "A" else 0.025
            ).astype(np.float32)
            # Weak later supervision uses only the best measured, safe prefix.
            later_path = a.validation / "selected_rotation_demonstrations.npz"
            if condition in ("B", "C") and later_path.exists():
                later = np.load(later_path)
                x = np.concatenate([x, later["observations"]])
                y = np.concatenate([y, later["actions"]])
                weights = np.r_[
                    weights, np.full(len(later["actions"]), 0.05, np.float32)
                ]
            # Terminal labels come from successful archived balance continuations.
            if condition == "C":
                x = np.concatenate([x, balance["observations"]])
                y = np.concatenate([y, balance["actions"]])
                weights = np.r_[
                    weights, np.full(len(balance["actions"]), 0.15, np.float32)
                ]
            demos = (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32),
            )
            dump(
                directory / "supervision.json",
                dict(
                    early_weight=1.0,
                    swing_weight=0.1 if condition == "A" else 0.025,
                    balance_weight=0.15 if condition == "C" else 0.0,
                    train_frames=len(x),
                    heldout_takeoff_seeds=[93004, 93005],
                    balance_scope="Offline balance labels; no transition demo",
                ),
            )
            torch.manual_seed(seed + 100000)
            obs, critic = env.reset(1.0, seed * 1000)
            episode = 0
            training = []
            for iteration in range(a.updates + 1):
                rows = evaluate(net, ev, [94000, 94001], directory, f"{iteration:03}")
                all_results.extend(
                    dict(
                        **r,
                        condition=condition,
                        training_seed=seed,
                        iteration=iteration,
                    )
                    for r in rows
                )
                torch.save(
                    dict(
                        network=net.state_dict(),
                        optimizer=optimizer.state_dict(),
                        environment=copy.deepcopy(env.state()),
                        rng=torch.get_rng_state(),
                        config=asdict(cfg),
                        iteration=iteration,
                        actual_steps=iteration * a.steps,
                        initialization_sha256=gate["initial_sha256"],
                    ),
                    directory / f"checkpoint_{iteration:03}.pt",
                )
                print(
                    condition,
                    seed,
                    "evaluation",
                    iteration,
                    "pitch",
                    np.rad2deg(np.mean([r["peak_certified_pitch"] for r in rows])),
                    flush=True,
                )
                if iteration == a.updates:
                    break
                records = []
                episode_rows = []
                for _ in range(a.steps):
                    with torch.no_grad():
                        raw, log, value = net.sample(
                            torch.tensor(obs), torch.tensor(critic)
                        )
                    (nextobs, nextcritic), reward, done, info = env.step(
                        raw.tanh().numpy()
                    )
                    with torch.no_grad():
                        nv = float(net.critic(torch.tensor(nextcritic)).squeeze())
                    records.append(
                        (
                            obs,
                            critic,
                            raw.numpy(),
                            float(log),
                            float(value),
                            reward,
                            done,
                            nv,
                            env.trace[-1],
                        )
                    )
                    obs, critic = nextobs, nextcritic
                    if done:
                        episode_rows.append(dict(**info, episode=episode))
                        episode += 1
                        obs, critic = env.reset(1.0, seed * 1000 + episode)
                adv, ret = advantages(
                    np.array([r[5] for r in records])[:, None],
                    np.array([r[4] for r in records])[:, None],
                    np.array([r[7] for r in records])[:, None],
                    np.array([r[6] for r in records])[:, None],
                    hp["gamma"],
                    hp["gae_lambda"],
                )
                batch = tuple(
                    torch.tensor(
                        np.asarray([r[j] for r in records]), dtype=torch.float32
                    )
                    for j in (0, 1, 2, 3)
                ) + (
                    torch.tensor(ret[:, 0], dtype=torch.float32),
                    torch.tensor(adv[:, 0], dtype=torch.float32),
                )
                stats = update(net, optimizer, batch, hp, demos)
                record = dict(
                    iteration=iteration + 1,
                    actual_steps=(iteration + 1) * a.steps,
                    **stats,
                )
                for key in (
                    "reward",
                    "L_com",
                    "L_support",
                    "L_target",
                    "r_momentum",
                    "potential",
                    "K_system",
                    "mechanical_energy",
                    "rear_power",
                    "calf_clearance",
                    "terminal_rate",
                ):
                    record[f"mean_{key}"] = float(np.mean([r[8][key] for r in records]))
                for key in (
                    "momentum_tracking",
                    "angular_impulse",
                    "upward_energy",
                    "useful_rotation_energy",
                    "terminal",
                    "com_support",
                ):
                    record[f"sum_reward_{key}"] = float(
                        sum(r[8][f"reward_{key}"] for r in records)
                    )
                training.append(record)
                save_csv(directory / "training.csv", training)
                save_csv(
                    directory / f"training_trace_{iteration + 1:03}.csv",
                    [r[8] for r in records],
                )
                if episode_rows:
                    save_csv(
                        directory / f"episodes_{iteration + 1:03}.csv", episode_rows
                    )
    save_csv(out / "comparison.csv", all_results)
    dump(
        out / "completion.json",
        dict(
            extended_training=False,
            training_interactions=len(a.seeds) * 3 * a.updates * a.steps,
            note="Evaluation and offline validation are separate; native geometry.",
        ),
    )


if __name__ == "__main__":
    main()
