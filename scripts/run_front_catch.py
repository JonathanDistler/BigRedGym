"""Matched parallel A/B/C PPO with exact update-boundary resume."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from experiments.front_catch.data import network, config, sha, Curriculum
from experiments.front_catch.vector import Vector
from experiments.front_catch.learning import collect, evaluate, hp_defaults
from experiments.front_catch.io import append_csv, atomic_checkpoint, source_hashes
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import update
from scripts.run_com_energy_phase import dump


def checkpoint(
    out, runs, vector, curriculum, iteration, protocol, evaluation_steps, history
):
    values = {}
    for mode, run in runs.items():
        values[mode] = dict(
            network=run["net"].state_dict(),
            optimizer=run["optimizer"].state_dict(),
            torch_rng=run["rng"],
            actors=run["actors"],
            critics=run["critics"],
            worlds=vector.state(mode),
            config=asdict(run["cfg"]),
            steps=run["steps"],
        )
    path = out / f"checkpoint_{iteration:06}.pt"
    atomic_checkpoint(
        path,
        dict(
            runs=values,
            curriculum=curriculum,
            iteration=iteration,
            protocol=protocol,
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            evaluation_steps=evaluation_steps,
            history=history,
        ),
    )
    return path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation", type=Path)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--num-envs", type=int, choices=(16, 32))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--updates", type=int, default=10, choices=range(1, 21))
    p.add_argument("--horizon", type=int, default=32, choices=(4, 8, 16, 32, 64, 128))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=2)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--suite", choices=("original", "anticipatory_v1"), default="original")
    a = p.parse_args()
    os.environ.update(
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    torch.set_num_threads(1)
    started = time.monotonic()
    out = a.output
    restored = None
    if a.resume:
        restored = torch.load(a.resume, weights_only=False, map_location="cpu")
        protocol = restored["protocol"]
        if a.resume.parent.resolve() != out.resolve():
            raise ValueError("Resume into the original directory; no silent forks")
        paths = sorted(out.glob("checkpoint_*.pt"))
        if paths[-1].resolve() != a.resume.resolve():
            raise ValueError(
                "Resume requires latest checkpoint; older checkpoints are preserved"
            )
        if protocol["source_sha256"] != source_hashes():
            raise ValueError("Source changed; refuse inexact resume")
        if protocol["torch_version"] != torch.__version__:
            raise ValueError("PyTorch runtime changed; refuse inexact resume")
        if protocol["mujoco_version"] != mujoco.__version__:
            raise ValueError("MuJoCo runtime changed")
        preparation = Path(protocol["preparation"])
        n = protocol["num_envs"]
        updates = protocol["updates"]
        horizon = protocol["horizon"]
        seeds = protocol["seeds"]
    else:
        if a.preparation is None:
            p.error("--preparation is required for a new pilot")
        preparation = a.preparation
        n = a.num_envs
        if n is None:
            if a.benchmark is None:
                p.error("Supply --benchmark selection.json or benchmarked --num-envs")
            selected = json.loads(a.benchmark.read_text())
            n = selected["num_envs"]
            a.workers = selected["workers"]
        updates = 1 if a.smoke else a.updates
        horizon = 4 if a.smoke else a.horizon
        seeds = [42] if a.smoke else a.seeds
        if n * horizon * updates * 3 * len(seeds) > 200000:
            p.error(
                "Pilot exceeds 200k transitions; use a separately reviewed experiment"
            )
        out.mkdir(parents=True, exist_ok=False)
    session_index = 0
    while True:
        session = out / "sessions" / f"session_{session_index:04}"
        try:
            session.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            session_index += 1
    history = (
        restored["history"]
        if restored
        else dict(training=[], episodes=[], evaluation=[], trace_files=[])
    )
    manifest = json.loads((preparation / "manifest.json").read_text())
    for key, file in [
        ("bank_sha256", "reset_bank.pt"),
        ("dataset_sha256", "bc_demonstrations.npz"),
    ]:
        if sha(preparation / file) != manifest[key]:
            raise ValueError("Prepared data changed: " + file)
    if sha(manifest["initial"]) != manifest["initial_sha256"]:
        raise ValueError("Preserved actor checkpoint changed")
    if restored and protocol["preparation_sha256"] != sha(
        preparation / "manifest.json"
    ):
        raise ValueError("Preparation manifest changed")
    bank = torch.load(
        preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    demos = np.load(preparation / "bc_demonstrations.npz")
    train = demos["seed"] < 95004
    weights = np.where(demos["phase"][train] < 0.4, 1.0, 0.025).astype(np.float32)
    if "role" in demos:
        weights[demos["role"][train] == "balance"] = 0.15
    demonstrations = (
        torch.tensor(demos["observations"][train], dtype=torch.float32),
        torch.tensor(demos["actions"][train], dtype=torch.float32),
        torch.tensor(weights),
    )
    hp = hp_defaults()
    assets = Assets(out / "assets")
    runs = {}
    configs = {}
    lane_seeds = {}
    for seed in seeds:
        for condition in "ABC":
            mode = f"{condition}_seed{seed}"
            net, parent = network(manifest["initial"])
            cfg = config(parent, condition)
            cfg.comparison_suite = protocol.get("comparison_suite", "original") if restored else a.suite
            optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
            torch.manual_seed(seed + 100000)
            runs[mode] = dict(
                net=net,
                optimizer=optimizer,
                cfg=cfg,
                rng=torch.get_rng_state(),
                steps=0,
            )
            configs[mode] = cfg
            lane_seeds[mode] = seed
    configs["evaluation"] = config(parent, "A")
    configs["evaluation"].comparison_suite = next(iter(configs.values())).comparison_suite
    lane_seeds["evaluation"] = 0
    if not restored:
        protocol = dict(
            comparison_suite=a.suite,
            condition_labels=(dict(A="original_fixed_support", B="early_reposition", C="early_reposition_rear_sweep")
                              if a.suite == "anticipatory_v1" else dict(A="fixed", B="slide", C="catch")),
            preparation=str(preparation),
            preparation_sha256=sha(preparation / "manifest.json"),
            num_envs=n,
            workers=a.workers,
            updates=updates,
            horizon=horizon,
            seeds=seeds,
            eval_every=a.eval_every,
            checkpoint_every=a.checkpoint_every,
            source_sha256=source_hashes(),
            initial_sha256=manifest["initial_sha256"],
            model_mass=manifest["model_mass"],
            hp=hp,
            mujoco_version=mujoco.__version__,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            standing_evaluation_seeds=[96000, 96001, 96002, 96003],
            wandb=False,
            geometry=1.0,
            training_budget=n * horizon * updates * 3 * len(seeds),
            smoke=a.smoke,
        )
        dump(out / "protocol.json", protocol)
        source = out / "source"
        source.mkdir()
        for f in protocol["source_sha256"]:
            destination = source / f
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, destination)
    else:
        hp = protocol["hp"]
    curriculum = restored["curriculum"] if restored else Curriculum()
    iteration = restored["iteration"] if restored else 0
    evaluation_steps = restored["evaluation_steps"] if restored else 0
    with Vector(assets, configs, bank, n, seed=lane_seeds, workers=a.workers) as vector:
        for mode, run in runs.items():
            if restored:
                saved = restored["runs"][mode]
                run["net"].load_state_dict(saved["network"])
                run["optimizer"].load_state_dict(saved["optimizer"])
                run["rng"] = saved["torch_rng"]
                run["steps"] = saved["steps"]
                run["actors"], run["critics"] = vector.restore(mode, saved["worlds"])
                if not np.array_equal(run["actors"], saved["actors"]):
                    raise ValueError("Restored observations differ")
            else:
                run["actors"], run["critics"] = vector.reset(mode, curriculum.mixture)
        if restored:
            np.random.set_state(restored["numpy_rng"])
            torch.set_rng_state(restored["torch_rng"])

        def evaluations(index):
            nonlocal evaluation_steps
            grouped = []
            for mode, run in runs.items():
                rows, traces, count = evaluate(
                    run["net"], vector, "evaluation", run["cfg"].catch_mode, bank
                )
                evaluation_steps += count
                rows = [
                    dict(**r, mode=mode, iteration=index, actual_steps=run["steps"])
                    for r in rows
                ]
                history["evaluation"].extend(rows)
                append_csv(session / "evaluation.csv", rows)
                folder = session / "evaluation_traces"
                folder.mkdir(exist_ok=True)
                from scripts.morphology_feasibility import save_csv

                path = folder / f"{mode}_{index:06}.csv"
                if path.exists():
                    raise FileExistsError(path)
                save_csv(path, traces)
                history["trace_files"].append(str(path))
                standing = [r for r in rows if r["evaluation_type"] == "standing"]
                grouped.append(standing)
                print(
                    "evaluation",
                    mode,
                    index,
                    "pitch",
                    np.rad2deg(np.mean([r["peak_swing_pitch"] for r in standing])),
                    flush=True,
                )
            curriculum.update(grouped, index)

        if not restored:
            evaluations(0)
            checkpoint(
                out, runs, vector, curriculum, 0, protocol, evaluation_steps, history
            )
        last_checkpoint = iteration
        while iteration < updates:
            if a.max_seconds and time.monotonic() - started >= a.max_seconds:
                break
            for mode, run in runs.items():
                torch.set_rng_state(run["rng"])
                t = time.monotonic()
                batch, actor, critic, episodes, samples = collect(
                    run["net"],
                    vector,
                    mode,
                    run["actors"],
                    run["critics"],
                    horizon,
                    curriculum.mixture,
                    hp,
                )
                collection_seconds = time.monotonic() - t
                t = time.monotonic()
                stats = update(run["net"], run["optimizer"], batch, hp, demonstrations)
                update_seconds = time.monotonic() - t
                run["rng"] = torch.get_rng_state()
                run["actors"] = actor
                run["critics"] = critic
                run["steps"] += n * horizon
                record = dict(
                    mode=mode,
                    iteration=iteration + 1,
                    actual_steps=run["steps"],
                    collection_seconds=collection_seconds,
                    steps_per_second=n * horizon / collection_seconds,
                    ppo_update_seconds=update_seconds,
                    standing_probability=curriculum.mixture[0],
                    **stats,
                )
                for key in (
                    "reward",
                    "peak_swing_pitch",
                    "front_recontacts",
                    "backward_mean",
                    "front_impulse",
                    "rear_impulse",
                    "capture_error",
                ):
                    record["mean_" + key] = float(np.mean([r[key] for r in samples]))
                for term in (
                    "rotation_progress",
                    "recoverability",
                    "useful_placement",
                    "stable_recatch",
                    "terminal",
                    "impact",
                    "early_placement",
                    "rear_sweep",
                ):
                    record["sum_reward_" + term] = float(
                        sum(s.get("reward_" + term, 0.0) for s in samples)
                    )
                history["training"].append(record)
                append_csv(session / "training.csv", [record])
                if episodes:
                    episode_records = [
                        dict(**r, mode=mode, iteration=iteration + 1) for r in episodes
                    ]
                    history["episodes"].extend(episode_records)
                    append_csv(session / "episodes.csv", episode_records)
                print(
                    "update",
                    mode,
                    iteration + 1,
                    "steps",
                    run["steps"],
                    "KL",
                    stats["post_update_kl"],
                    flush=True,
                )
            iteration += 1
            if iteration % protocol["eval_every"] == 0 or iteration == updates:
                evaluations(iteration)
            if iteration % protocol["checkpoint_every"] == 0 or iteration == updates:
                checkpoint(
                    out,
                    runs,
                    vector,
                    curriculum,
                    iteration,
                    protocol,
                    evaluation_steps,
                    history,
                )
                last_checkpoint = iteration
        if last_checkpoint != iteration:
            checkpoint(
                out,
                runs,
                vector,
                curriculum,
                iteration,
                protocol,
                evaluation_steps,
                history,
            )
    path = session / f"status_{iteration:06}.json"
    if not path.exists():
        dump(
            path,
            dict(
                completed=iteration == updates,
                iteration=iteration,
                total_training_steps=sum(r["steps"] for r in runs.values()),
                evaluation_steps=evaluation_steps,
                curriculum_stage=curriculum.stage,
                curriculum_events=curriculum.events,
                resume_command=(
                    f"{sys.executable} scripts/run_front_catch.py "
                    f"--output {out} --resume {out}/checkpoint_{iteration:06}.pt "
                    f"--workers {a.workers}"
                ),
            ),
        )


if __name__ == "__main__":
    main()
