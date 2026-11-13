"""Matched B0/B1/B2/B3, explicit bootstrap accounting, exact boundary resume."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    WANDB_MODE="disabled",
    WANDB_DISABLED="true",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
import torch
import numpy as np
import mujoco
from experiments.multi_impulse.core import MultiConfig
from experiments.multi_impulse.network import initialize, apply_warm
from experiments.multi_impulse.vector import Vector
from experiments.adc_front_capture.curriculum import ActuatorCurriculum
from experiments.front_catch.learning import collect, hp_defaults
from experiments.front_catch.data import sha
from experiments.front_catch.io import atomic_checkpoint, append_csv
from experiments.morphology.learning.environment import Assets
from experiments.multi_impulse.learning import safe_update
from scripts.run_adc_front_capture import sources as old_sources
from scripts.run_com_energy_phase import dump


def sources():
    result = old_sources()
    for p in Path("experiments/multi_impulse").glob("*.py"):
        result[str(p)] = sha(p)
    result["scripts/run_multi_impulse.py"] = sha(__file__)
    return result


def evaluate(net, vector, mode, bank, seconds):
    specs = [dict(kind="standing", seed=s) for s in (96000, 96001, 96002, 96003)]
    for tier in ("near_vertical", "75", "60", "45"):
        if bank["balance"]["evaluation"][tier]:
            specs.append(dict(kind="balance_" + tier, index=0, seed=98000 + len(specs)))
    specs += [None] * (vector.n - len(specs))
    actor, _ = vector.evaluation_reset(mode, "B", specs)
    active = np.array([spec is not None for spec in specs])
    rows = []
    steps = 0
    for _ in range(round(seconds / 0.02) + 1):
        if not active.any():
            break
        with torch.no_grad():
            actions = net.actor(torch.tensor(actor)).tanh().numpy()
        results = vector.evaluation_step(mode, actions, active)
        steps += int(active.sum())
        for i, r in enumerate(results):
            if active[i] and r[3]:
                rows.append(
                    dict(
                        **r[5],
                        evaluation_seed=specs[i]["seed"],
                        evaluation_type=specs[i]["kind"],
                    )
                )
                active[i] = False
        actor = np.stack([r[0] for r in results])
    if active.any():
        raise RuntimeError("Evaluation horizon mismatch")
    return rows, steps


def vector_score(rows):
    standing = [r for r in rows if r["evaluation_type"] == "standing"]
    return [
        float(np.mean([r[k] for r in standing]))
        for k in (
            "certified_standing_takeoff",
            "standing_success",
            "handstand_seconds",
            "seconds_above75",
            "seconds_above60",
            "seconds_above45",
            "front_support_fraction",
            "peak_collision_free_pitch",
        )
    ]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--preparation", type=Path, default=Path("output/multi_impulse_preparation_v1")
    )
    p.add_argument("--balance-initial", type=Path)
    p.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Absolute additional-study update target; parent was update 30",
    )
    p.add_argument("--num-envs", type=int, choices=(16, 32), default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=["B0", "B1", "B2", "B3"],
        default=["B0", "B1", "B2", "B3"],
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--interaction-budget", type=int, default=220000)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--initial-stiffness", type=float, default=60.0)
    p.add_argument("--adc-probe", type=Path)
    p.add_argument(
        "--progress-rule", choices=["episode_ema", "recovery"], default="recovery"
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    out = a.output
    restored = (
        torch.load(a.resume, weights_only=False, map_location="cpu")
        if a.resume
        else None
    )
    if restored:
        if a.updates is None:
            a.updates = restored["target_updates"]
        protocol = restored["protocol"]
        if (
            protocol["sources"] != sources()
            or protocol["torch"] != torch.__version__
            or protocol["mujoco"] != mujoco.__version__
        ):
            raise ValueError("Source/runtime changed")
        latest = max(
            int(path.stem.split("_")[1])
            for pattern in ("checkpoint_*.pt", "boundary_*.pt")
            for path in out.glob(pattern)
        )
        if (
            a.resume.parent.resolve() != out.resolve()
            or int(a.resume.stem.split("_")[1]) != latest
        ):
            raise ValueError("Use latest complete training boundary")
    else:
        if a.updates is None:
            a.updates = 100
        if a.updates < 1 or a.horizon < 1:
            p.error("Positive update/horizon required")
        out.mkdir(parents=True, exist_ok=False)
        protocol = dict(
            preparation=str(a.preparation),
            preparation_sha=sha(a.preparation / "manifest.json"),
            initial="output/multi_impulse_diagnosis_v1/B_best.pt",
            balance_initial=str(a.balance_initial) if a.balance_initial else None,
            n=a.num_envs,
            horizon=a.horizon,
            seeds=a.seeds,
            conditions=a.conditions,
            seconds=a.seconds,
            hp=hp_defaults(),
            eval_every=a.eval_every,
            checkpoint_every=a.checkpoint_every,
            sources=sources(),
            torch=torch.__version__,
            mujoco=mujoco.__version__,
            nominal=[60.0, 3.0],
            observation_dims=[93, 123],
            parent_updates=30,
            bootstrap=(
                "Balance supervision versus equal-count baseline PPO warmup; "
                "main PPO interactions matched"
            ),
            initial_stiffness=a.initial_stiffness,
            progress_rule=a.progress_rule,
            wandb=False,
        )
        protocol["initial_sha"] = sha(protocol["initial"])
        if a.initial_stiffness > 60:
            if not a.adc_probe:
                p.error("Elevated stiffness requires a fresh B-specific probe")
            probe = json.loads(a.adc_probe.read_text())
            if (
                probe["initial_sha256"] != protocol["initial_sha"]
                or a.initial_stiffness not in probe["accepted_stiffness"]
            ):
                p.error("Unvalidated initial gain")
        protocol["balance_initial_sha"] = (
            sha(a.balance_initial) if a.balance_initial else None
        )
        dump(out / "protocol.json", protocol)
    if (
        a.updates
        * protocol["n"]
        * protocol["horizon"]
        * len(protocol["conditions"])
        * len(protocol["seeds"])
        > a.interaction_budget
    ):
        p.error("Explicit interaction budget exceeded")
    prep = Path(protocol["preparation"])
    manifest = json.loads((prep / "manifest.json").read_text())
    if sha(prep / "manifest.json") != protocol["preparation_sha"]:
        raise ValueError("Preparation changed")
    if sha(protocol["initial"]) != protocol["initial_sha"]:
        raise ValueError("Initial changed")
    if (
        protocol["balance_initial"]
        and sha(protocol["balance_initial"]) != protocol["balance_initial_sha"]
    ):
        raise ValueError("Balance initial changed")
    bank = torch.load(
        "output/adc_front_capture_preparation_v1/reset_bank.pt",
        weights_only=False,
        map_location="cpu",
    )
    bank["balance"] = torch.load(
        prep / "balance_bank.pt", weights_only=False, map_location="cpu"
    )
    if sha(prep / "balance_bank.pt") != manifest["bank_sha256"]:
        raise ValueError("Balance bank changed")
    parent = torch.load(protocol["initial"], weights_only=False, map_location="cpu")[
        "runs"
    ]["B_seed42"]
    warm = (
        torch.load(protocol["balance_initial"], weights_only=False, map_location="cpu")
        if protocol["balance_initial"]
        else None
    )
    demos = np.load("output/adc_front_capture_preparation_v1/bc_demonstrations.npz")
    ids = (demos["seed"] < 95004) & (demos["role"] != "balance")
    demo_obs = np.pad(demos["observations"][ids], ((0, 0), (0, 20)))
    weights = np.repeat(
        np.where(demos["phase"][ids] < 0.4, 1.0, 0.025)[:, None], 12, axis=1
    ).astype(np.float32)
    names = Assets(out / "assets").metadata[1.0]["names"]
    weights[:, [i for i, n in enumerate(names) if n.startswith(("FL_", "FR_"))]] *= 0.15
    demonstrations = (
        torch.tensor(demo_obs, dtype=torch.float32),
        torch.tensor(demos["actions"][ids], dtype=torch.float32),
        torch.tensor(weights),
    )
    runs = {}
    configs = {}
    seeds = {}
    for seed in protocol["seeds"]:
        for condition in protocol["conditions"]:
            mode = f"{condition}_seed{seed}"
            enabled = condition != "B0"
            net, opt = initialize(parent, enabled)
            if warm and condition in ("B1", "B3"):
                apply_warm(net, opt, warm)
            cfg = MultiConfig(**parent["config"])
            cfg.corrective = condition in ("B2", "B3")
            cfg.balance_training = condition in ("B1", "B3")
            cfg.context_enabled = enabled
            cfg.capture_enabled = False
            cfg.seconds = protocol["seconds"]
            cfg.target_base_height = 0.57
            torch.manual_seed(seed + 100000)
            runs[mode] = dict(
                net=net,
                opt=opt,
                cfg=cfg,
                rng=torch.get_rng_state(),
                steps=0,
                stage=0,
                adc=ActuatorCurriculum(
                    initial_k=protocol["initial_stiffness"],
                    rule=protocol["progress_rule"],
                ),
            )
            if cfg.balance_training:
                for param in net.actor.base.parameters():
                    param.requires_grad_(False)
            configs[mode] = cfg
            seeds[mode] = seed
            configs["eval_" + mode] = copy.deepcopy(cfg)
            configs["eval_" + mode].kp = 60.0
            configs["eval_" + mode].kd = 3.0
            seeds["eval_" + mode] = 0
    bootstrap_frames = warm.get("source_data_frames", 0) if warm else 0
    total_exposure = (
        a.updates * protocol["n"] * protocol["horizon"] + bootstrap_frames
    ) * len(runs)
    if total_exposure > a.interaction_budget:
        p.error("Budget excludes required equal-count warmup")
    if not restored and bootstrap_frames:
        warm_modes = {
            mode: copy.deepcopy(r["cfg"])
            for mode, r in runs.items()
            if not r["cfg"].balance_training
        }
        with Vector(
            Assets(out / "assets"), warm_modes, bank, 1, seed=seeds, workers=0
        ) as warm_vector:
            for mode in warm_modes:
                r = runs[mode]
                torch.set_rng_state(r["rng"])
                aa, cc = warm_vector.reset(
                    mode, [0.35, 0.156, 0.1495, 0.1495, 0.0975, 0.0975]
                )
                batch, _, _, _, _ = collect(
                    r["net"],
                    warm_vector,
                    mode,
                    aa,
                    cc,
                    bootstrap_frames,
                    [0.35, 0.156, 0.1495, 0.1495, 0.0975, 0.0975],
                    protocol["hp"],
                )
                safe_update(r["net"], r["opt"], batch, protocol["hp"], demonstrations)
                r["rng"] = torch.get_rng_state()
        for r in runs.values():
            r["steps"] = bootstrap_frames
    iteration = restored["iteration"] if restored else 0
    history = (
        restored["history"]
        if restored
        else dict(training=[], episodes=[], evaluation=[], selection=[])
    )
    session = out / "sessions" / f"{len(list((out / 'sessions').glob('*'))):04}"
    session.mkdir(parents=True)
    dump(
        session / "budget.json",
        dict(target=a.updates, training_interaction_cap=a.interaction_budget),
    )
    with Vector(
        Assets(out / "assets"),
        configs,
        bank,
        protocol["n"],
        seed=seeds,
        workers=a.workers,
    ) as v:

        def mixture(mode, index):
            if runs[mode]["cfg"].balance_training:
                return (
                    [0, 0, 0, 0, 0, 1]
                    if runs[mode]["stage"] == 0
                    else [
                        (0.3, 0.5, 0.7, 1.0)[runs[mode]["stage"]],
                        0,
                        0,
                        0,
                        0,
                        1 - (0.3, 0.5, 0.7, 1.0)[runs[mode]["stage"]],
                    ]
                )
            return [0.35, 0.156, 0.1495, 0.1495, 0.0975, 0.0975]

        for mode, r in runs.items():
            if restored:
                old = restored["runs"][mode]
                r["net"].load_state_dict(old["network"])
                r["opt"].load_state_dict(copy.deepcopy(old["optimizer"]))
                r["rng"] = old["rng"]
                r["steps"] = old["steps"]
                r["stage"] = old["stage"]
                r["adc"] = old["adc"]
                for param in r["net"].actor.base.parameters():
                    param.requires_grad_(
                        not r["cfg"].balance_training or r["stage"] > 0
                    )
                v.curriculum(mode, r["stage"])
                r["a"], r["c"] = v.restore(mode, old["worlds"])
                if not (
                    np.array_equal(r["a"], old["a"])
                    and np.array_equal(r["c"], old["c"])
                ):
                    raise ValueError("Inexact observation restoration")
            else:
                r["a"], r["c"] = v.reset(mode, mixture(mode, 0))
        if restored:
            torch.set_rng_state(restored["torch_rng"])
            np.random.set_state(restored["numpy_rng"])

        def save(index, name=None):
            payload = dict(
                iteration=index,
                target_updates=a.updates,
                protocol=protocol,
                history=history,
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                runs={},
            )
            for mode, r in runs.items():
                payload["runs"][mode] = dict(
                    network=r["net"].state_dict(),
                    optimizer=r["opt"].state_dict(),
                    config=asdict(r["cfg"]),
                    rng=r["rng"],
                    steps=r["steps"],
                    stage=r["stage"],
                    adc=r["adc"],
                    worlds=v.state(mode),
                    a=r["a"],
                    c=r["c"],
                )
            path = out / (name or f"checkpoint_{index:06}.pt")
            atomic_checkpoint(path, payload)
            return path

        def evaluations(index):
            for mode, r in runs.items():
                rows, steps = evaluate(
                    r["net"], v, "eval_" + mode, bank, protocol["seconds"]
                )
                rows = [
                    dict(**row, mode=mode, iteration=index, actual_steps=r["steps"])
                    for row in rows
                ]
                history["evaluation"].extend(rows)
                append_csv(session / f"evaluation_{mode}.csv", rows)
                score = vector_score(rows)
                record = dict(
                    mode=mode, iteration=index, score=score, evaluation_steps=steps
                )
                old = [
                    row
                    for row in history["selection"]
                    if row["mode"] == mode and row["score"][0] >= 0.75
                ]
                record["nondominated"] = score[0] >= 0.75 and not any(
                    all(x >= y for x, y in zip(row["score"][1:], score[1:]))
                    for row in old
                )
                history["selection"].append(record)
                # Recovery, not reward, gates reset expansion; empty tiers block it.
                bal = [
                    row for row in rows if row["evaluation_type"].startswith("balance_")
                ]
                if (
                    r["cfg"].balance_training
                    and r["stage"] < 3
                    and bal
                    and np.mean([row["intermediate_success"] for row in bal]) >= 0.5
                ):
                    nexttier = ("75", "60", "45")[r["stage"]]
                    if (
                        bank["balance"]["train"][nexttier]
                        and bank["balance"]["evaluation"][nexttier]
                    ):
                        r["stage"] += 1
                        for param in r["net"].actor.base.parameters():
                            param.requires_grad_(True)
                        v.curriculum(mode, r["stage"])
                print(
                    "evaluation",
                    mode,
                    index,
                    "pitch",
                    np.rad2deg(score[-1]),
                    "hold",
                    score[2],
                    "balance_hold",
                    [round(row["handstand_seconds"], 3) for row in bal],
                    flush=True,
                )

        if not restored:
            evaluations(0)
            save(0)
        last = iteration
        while iteration < a.updates:
            if (out / "STOP").exists() or (
                a.max_seconds and time.monotonic() - started > a.max_seconds
            ):
                break
            for mode, r in runs.items():
                torch.set_rng_state(r["rng"])
                t = time.monotonic()
                gains = v.dynamics(mode, r["adc"].stiffness, 60.0, 3.0)
                actual_kp = r["adc"].stiffness
                batch, actor, critic, episodes, samples = collect(
                    r["net"],
                    v,
                    mode,
                    r["a"],
                    r["c"],
                    protocol["horizon"],
                    mixture(mode, iteration),
                    protocol["hp"],
                )
                stats = safe_update(
                    r["net"], r["opt"], batch, protocol["hp"], demonstrations
                )
                r["adc"].update(episodes, iteration + 1)
                r["a"], r["c"] = actor, critic
                r["rng"] = torch.get_rng_state()
                r["steps"] += protocol["n"] * protocol["horizon"]
                record = dict(
                    mode=mode,
                    iteration=iteration + 1,
                    actual_steps=r["steps"],
                    stage=r["stage"],
                    kp=actual_kp,
                    kd=float(np.mean([g["kd"] for g in gains])),
                    steps_per_second=protocol["n"]
                    * protocol["horizon"]
                    / (time.monotonic() - t),
                    **stats,
                )
                history["training"].append(record)
                append_csv(session / f"training_{mode}.csv", [record])
                if episodes:
                    rows = [
                        dict(**row, mode=mode, iteration=iteration + 1)
                        for row in episodes
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
                    flush=True,
                )
            iteration += 1
            # A training boundary is durable BEFORE evaluation. A failed evaluator
            # cannot discard completed PPO updates; final checkpoint adds outcomes.
            if iteration % protocol["eval_every"] == 0 or iteration == a.updates:
                save(iteration, f"boundary_{iteration:06}.pt")
                evaluations(iteration)
            if iteration % protocol["checkpoint_every"] == 0 or iteration == a.updates:
                path = save(iteration)
                last = iteration
                milestones = out / "milestones"
                milestones.mkdir(exist_ok=True)
                for row in history["selection"]:
                    if row["iteration"] == iteration and row["nondominated"]:
                        link = milestones / f"{row['mode']}_{iteration:06}.pt"
                        if not link.exists():
                            os.link(path, link)
        if last != iteration:
            save(iteration)
    dump(
        session / "status.json",
        dict(
            iteration=iteration,
            parent_updates=30,
            completed=iteration == a.updates,
            total_training_interactions=sum(r["steps"] for r in runs.values()),
            bootstrap_frames=warm.get("source_data_frames", 0)
            + warm.get("dagger_frames", 0)
            if warm
            else 0,
        ),
    )


if __name__ == "__main__":
    main()
