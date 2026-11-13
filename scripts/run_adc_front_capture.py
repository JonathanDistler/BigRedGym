"""Bounded matched A/B/C/D PPO; updates is an absolute cumulative target."""

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    WANDB_MODE="disabled",
    WANDB_DISABLED="true",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
import mujoco
import numpy as np
import torch
from experiments.front_catch.data import network, config, sha
from experiments.front_catch.vector import Vector
from experiments.front_catch.learning import collect, evaluate, hp_defaults
from experiments.front_catch.io import atomic_checkpoint, append_csv, source_hashes
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import update
from experiments.adc_front_capture.curriculum import (
    ActuatorCurriculum,
    ContactCurriculum,
)
from scripts.run_com_energy_phase import dump


def sources():
    values = source_hashes()
    for root in (
        "experiments/adc_front_capture",
        "experiments/morphology",
        "gym/envs/go2",
        "gym/envs/base",
        "gym/utils",
    ):
        for p in Path(root).rglob("*.py"):
            values[str(p)] = sha(p)
    values["scripts/run_adc_front_capture.py"] = sha(__file__)
    return values


def initial(path, mode=None):
    """Explicit lane import; fine-tuning initialization is distinct from resume."""
    saved = torch.load(path, weights_only=False, map_location="cpu")
    if "runs" not in saved:
        return network(path)
    if mode is None or mode not in saved["runs"]:
        raise ValueError(
            "Combined checkpoint needs --initial-mode; available: "
            + str(list(saved["runs"]))
        )
    from experiments.morphology.learning.ppo import ActorCritic

    net = ActorCritic(73, 103)
    lane = saved["runs"][mode]
    net.load_state_dict(lane["network"])
    return net, lane


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation", type=Path)
    p.add_argument("--initial", type=Path)
    p.add_argument("--initial-mode")
    p.add_argument("--resume", type=Path)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--interaction-budget", type=int, default=800000)
    p.add_argument("--num-envs", type=int, choices=(16, 32), default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--conditions", nargs="+", choices=list("ABCD"), default=list("ABCD")
    )
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--initial-stiffness", type=float, default=60.0)
    p.add_argument("--probe", type=Path)
    p.add_argument(
        "--progress-rule", choices=("episode_ema", "recovery"), default="episode_ema"
    )
    p.add_argument("--nominal-finetune", action="store_true")
    p.add_argument("--phase-bc", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.workers < 0 or a.horizon < 1 or a.checkpoint_every < 1 or a.eval_every < 1:
        p.error("Invalid positive control argument")
    started = time.monotonic()
    out = a.output
    restored = None
    if a.resume:
        restored = torch.load(a.resume, weights_only=False, map_location="cpu")
        protocol = restored["protocol"]
        a.nominal_finetune = a.nominal_finetune or restored.get(
            "nominal_finetune", False
        )
        if (
            a.resume.parent.resolve() != out.resolve()
            or sorted(out.glob("checkpoint_*.pt"))[-1].resolve() != a.resume.resolve()
        ):
            raise ValueError(
                "Resume from latest checkpoint in original output directory"
            )
        if protocol["source_sha256"] != sources():
            raise ValueError("Source changed: inexact resume refused")
        if (
            protocol["torch_version"] != torch.__version__
            or protocol["mujoco_version"] != mujoco.__version__
        ):
            raise ValueError("Runtime changed")
        # Extend the target explicitly; keep PPO horizon and epochs fixed.
        target = a.updates if a.updates is not None else restored["target_updates"]
        if target < restored["iteration"]:
            raise ValueError("Target precedes restored iteration")
        preparation = Path(protocol["preparation"])
        n = protocol["num_envs"]
        horizon = protocol["horizon"]
        seeds = protocol["seeds"]
        conditions = protocol["conditions"]
        hp = protocol["hp"]
    else:
        if a.preparation is None:
            p.error("--preparation required")
        target = 30 if a.updates is None else a.updates
        if target < 1:
            p.error("updates must be positive")
        preparation = a.preparation
        n = a.num_envs
        horizon = a.horizon
        seeds = a.seeds
        conditions = a.conditions
        if len(set(conditions)) != len(conditions) or len(set(seeds)) != len(seeds):
            p.error("Duplicate conditions/seeds")
        hp = hp_defaults()
        manifest = json.loads((preparation / "manifest.json").read_text())
        init = a.initial or Path(manifest["initial"])
        net, parent = initial(init, a.initial_mode)
        nominal_k = float(parent["config"]["kp"])
        nominal_b = float(np.mean(parent["config"]["kd"]))
        if a.initial_stiffness > nominal_k:
            if not a.probe:
                p.error("Elevated stiffness requires a measured --probe JSON")
            probe = json.loads(a.probe.read_text())
            if (
                probe["initial_sha256"] != sha(init)
                or probe["nominal_k"] != nominal_k
                or a.initial_stiffness not in probe["accepted_stiffness"]
            ):
                p.error("Probe does not validate this initialization/gain")
        protocol = dict(
            preparation=str(preparation),
            preparation_sha256=sha(preparation / "manifest.json"),
            initial=str(init),
            initial_sha256=sha(init),
            initial_mode=a.initial_mode,
            num_envs=n,
            horizon=horizon,
            seeds=seeds,
            conditions=conditions,
            hp=hp,
            eval_every=a.eval_every,
            checkpoint_every=a.checkpoint_every,
            nominal_k=nominal_k,
            nominal_b=nominal_b,
            initial_stiffness=a.initial_stiffness,
            progress_rule=a.progress_rule,
            phase_bc=a.phase_bc,
            balance_demonstrations=False,
            source_sha256=sources(),
            torch_version=torch.__version__,
            mujoco_version=mujoco.__version__,
            numpy_version=np.__version__,
            normalization="none (fixed feature scaling)",
            observation_dimensions=[73, 103],
            evaluation_dynamics="original nominal",
            condition_labels=dict(
                A="anticipatory baseline",
                B="ADC only",
                C="capture only",
                D="ADC + capture",
            ),
            wandb=False,
        )
        out.mkdir(parents=True, exist_ok=False)
        dump(out / "protocol.json", protocol)
        for f in protocol["source_sha256"]:
            dst = out / "source" / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
    budget = n * horizon * target * len(seeds) * len(conditions)
    if budget > a.interaction_budget:
        p.error(
            f"{budget} cumulative interactions exceeds budget {a.interaction_budget}"
        )
    manifest = json.loads((preparation / "manifest.json").read_text())
    if sha(preparation / "manifest.json") != protocol["preparation_sha256"]:
        raise ValueError("Preparation changed")
    for key, name in [
        ("bank_sha256", "reset_bank.pt"),
        ("dataset_sha256", "bc_demonstrations.npz"),
    ]:
        if sha(preparation / name) != manifest[key]:
            raise ValueError("Preparation input changed")
    if sha(protocol["initial"]) != protocol["initial_sha256"]:
        raise ValueError("Initialization changed")
    bank = torch.load(
        preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    demos = np.load(preparation / "bc_demonstrations.npz")
    train = demos["seed"] < 95004
    if "role" in demos:
        train &= demos["role"] != "balance"  # Archived targets are not validated holds.
    weights = np.where(demos["phase"][train] < 0.4, 1.0, 0.025).astype(np.float32)
    if protocol["phase_bc"]:
        weights = np.repeat(weights[:, None], 12, axis=1)
        # Earlier foreleg freedom; retain strong rear-joint push-off supervision.
        _, parent = initial(protocol["initial"], protocol["initial_mode"])
        assets_names = Assets(out / "assets").metadata[1.0]["names"]
        front = [
            i for i, name in enumerate(assets_names) if name.startswith(("FL_", "FR_"))
        ]
        weights[:, front] *= 0.15
        weights[demos["phase"][train] >= 0.8] = 0.005
    demonstrations = (
        torch.tensor(demos["observations"][train], dtype=torch.float32),
        torch.tensor(demos["actions"][train], dtype=torch.float32),
        torch.tensor(weights),
    )
    runs = {}
    configs = {}
    lane_seeds = {}
    for seed in seeds:
        for condition in conditions:
            mode = f"{condition}_seed{seed}"
            net, parent = initial(protocol["initial"], protocol["initial_mode"])
            cfg = config(parent, "B")
            cfg.adc_enabled = True
            cfg.capture_enabled = condition in "CD"
            cfg.comparison_suite = "anticipatory_v1"
            optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
            torch.manual_seed(seed + 100000)
            runs[mode] = dict(
                net=net,
                optimizer=optimizer,
                cfg=cfg,
                rng=torch.get_rng_state(),
                steps=0,
                actuator=ActuatorCurriculum(
                    protocol["nominal_k"],
                    protocol["nominal_b"],
                    protocol["initial_stiffness"]
                    if condition in "BD"
                    else protocol["nominal_k"],
                    protocol["progress_rule"],
                ),
                contact=ContactCurriculum(),
            )
            configs[mode] = cfg
            lane_seeds[mode] = seed
            if restored:
                saved = restored["runs"][mode]
                cfg.kp = saved["config"]["kp"]
                cfg.kd = np.asarray(saved["config"]["kd"])
                runs[mode]["actuator"] = saved["actuator"]
                runs[mode]["contact"] = saved["contact"]
    for mode, run in runs.items():
        evaluation_mode = "evaluation_" + mode
        configs[evaluation_mode] = copy.deepcopy(run["cfg"])
        configs[evaluation_mode].kp = protocol["nominal_k"]
        configs[evaluation_mode].kd = protocol["nominal_b"]
        lane_seeds[evaluation_mode] = 0
    iteration = restored["iteration"] if restored else 0
    history = (
        restored["history"]
        if restored
        else dict(training=[], episodes=[], evaluation=[])
    )
    evaluation_steps = restored["evaluation_steps"] if restored else 0
    session = out / "sessions" / f"{len(list((out / 'sessions').glob('*'))):04}"
    session.mkdir(parents=True, exist_ok=False)
    dump(
        session / "budget.json",
        dict(
            target_updates=target,
            cumulative_budget=budget,
            nominal_finetune=a.nominal_finetune,
        ),
    )
    with Vector(
        Assets(out / "assets"), configs, bank, n, seed=lane_seeds, workers=a.workers
    ) as vector:
        for mode, run in runs.items():
            if restored:
                saved = restored["runs"][mode]
                run["net"].load_state_dict(saved["network"])
                run["optimizer"].load_state_dict(saved["optimizer"])
                run["rng"] = saved["torch_rng"]
                run["steps"] = saved["steps"]
                run["actors"], run["critics"] = vector.restore(mode, saved["worlds"])
                if not (
                    np.array_equal(run["actors"], saved["actors"])
                    and np.array_equal(run["critics"], saved["critics"])
                ):
                    raise ValueError("Restore observation mismatch")
            else:
                run["actors"], run["critics"] = vector.reset(
                    mode, run["contact"].mixture(mode[0] in "CD", bank)
                )
        if restored:
            np.random.set_state(restored["numpy_rng"])
            torch.set_rng_state(restored["torch_rng"])

        def evaluate_all(index):
            nonlocal evaluation_steps
            for mode, run in runs.items():
                rows, traces, count = evaluate(
                    run["net"], vector, "evaluation_" + mode, "B", bank
                )
                evaluation_steps += count
                rows = [
                    dict(**r, mode=mode, iteration=index, actual_steps=run["steps"])
                    for r in rows
                ]
                history["evaluation"].extend(rows)
                append_csv(session / f"evaluation_{mode}.csv", rows)
                if mode[0] in "CD":
                    run["contact"].update(rows, index)
                standing = [r for r in rows if r["evaluation_type"] == "standing"]
                rate = np.mean([r["certified_standing_takeoff"] for r in standing])
                counts = history.setdefault("collapse_counts", {})
                counts[mode] = counts.get(mode, 0) + 1 if rate < 0.5 else 0
                if counts[mode] >= 2:
                    history["halt_reason"] = "certified takeoff collapse in " + mode
                print(
                    "eval",
                    mode,
                    index,
                    "pitch",
                    np.rad2deg(
                        np.mean([r["peak_collision_free_pitch"] for r in standing])
                    ),
                    "takeoff",
                    np.mean([r["certified_standing_takeoff"] for r in standing]),
                    flush=True,
                )

        def save(index):
            values = {}
            for mode, r in runs.items():
                worlds = vector.state(mode)
                # Worker-local gains are not part of Environment.state; save explicitly.
                cfg = asdict(r["cfg"])
                values[mode] = dict(
                    network=r["net"].state_dict(),
                    optimizer=r["optimizer"].state_dict(),
                    torch_rng=r["rng"],
                    steps=r["steps"],
                    actors=r["actors"],
                    critics=r["critics"],
                    worlds=worlds,
                    config=cfg,
                    actuator=r["actuator"],
                    contact=r["contact"],
                )
            atomic_checkpoint(
                out / f"checkpoint_{index:06}.pt",
                dict(
                    runs=values,
                    iteration=index,
                    protocol=protocol,
                    target_updates=target,
                    nominal_finetune=a.nominal_finetune,
                    history=history,
                    evaluation_steps=evaluation_steps,
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                ),
            )

        if not restored:
            evaluate_all(0)
            save(0)
        last = iteration
        while iteration < target:
            if (out / "STOP").exists() or (
                a.max_seconds and time.monotonic() - started >= a.max_seconds
            ):
                break
            for mode, r in runs.items():
                k = (
                    protocol["nominal_k"]
                    if a.nominal_finetune
                    else r["actuator"].stiffness
                )
                gains = vector.dynamics(
                    mode, k, protocol["nominal_k"], protocol["nominal_b"]
                )
                r["cfg"].kp = k
                r["cfg"].kd = np.asarray(gains[0]["kd"])
                torch.set_rng_state(r["rng"])
                t = time.monotonic()
                batch, actor, critic, episodes, samples = collect(
                    r["net"],
                    vector,
                    mode,
                    r["actors"],
                    r["critics"],
                    horizon,
                    r["contact"].mixture(mode[0] in "CD", bank),
                    hp,
                )
                stats = update(r["net"], r["optimizer"], batch, hp, demonstrations)
                r["rng"] = torch.get_rng_state()
                r["actors"] = actor
                r["critics"] = critic
                r["steps"] += n * horizon
                if mode[0] in "BD" and not a.nominal_finetune:
                    r["actuator"].update(episodes, iteration + 1)
                record = dict(
                    mode=mode,
                    iteration=iteration + 1,
                    actual_steps=r["steps"],
                    steps_per_second=n * horizon / (time.monotonic() - t),
                    actual_kp=k,
                    actual_kd_mean=float(np.mean(gains[0]["kd"])),
                    **stats,
                )
                for term in sorted(
                    {key for row in samples for key in row if key.startswith("reward_")}
                ):
                    record["sum_" + term] = float(
                        sum(row.get(term, 0) for row in samples)
                    )
                history["training"].append(record)
                # Different rewards have separate CSV schemas by condition.
                append_csv(session / f"training_{mode}.csv", [record])
                if episodes:
                    rows = [
                        dict(**e, mode=mode, iteration=iteration + 1) for e in episodes
                    ]
                    history["episodes"].extend(rows)
                    append_csv(session / f"episodes_{mode}.csv", rows)
                print(
                    "update",
                    mode,
                    iteration + 1,
                    "steps",
                    r["steps"],
                    "KL",
                    stats["post_update_kl"],
                    "K",
                    k,
                    flush=True,
                )
            iteration += 1
            if iteration % protocol["eval_every"] == 0 or iteration == target:
                evaluate_all(iteration)
            if (
                iteration % protocol["checkpoint_every"] == 0
                or iteration == target
                or history.get("halt_reason")
            ):
                save(iteration)
                last = iteration
            if history.get("halt_reason"):
                break
        if last != iteration:
            save(iteration)
    dump(
        session / "status.json",
        dict(
            iteration=iteration,
            target_updates=target,
            total_training_steps=sum(r["steps"] for r in runs.values()),
            evaluation_steps=evaluation_steps,
            completed=iteration == target,
        ),
    )


if __name__ == "__main__":
    main()
