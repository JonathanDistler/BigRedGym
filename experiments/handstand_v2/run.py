"""Self-contained opt-in local/Molab runner. Run from repository root."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import torch
import curriculum as legacy
from experiments.handstand_v2.environment import V2Environment
from experiments.handstand_v2.measurement import STAGES
from experiments.morphology.learning.environment import Assets

ORIGINAL = Path(
    "output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt"
)
OLD = Path("output/molab_imports/go2_handstand_curriculum_60_v1/checkpoints")


def dump_json(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        return str(x)

    Path(path).write_text(json.dumps(value, indent=2, default=convert))


def source_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    manifest = Path("handstand_v2_source_manifest.json")
    if manifest.exists():
        return json.loads(manifest.read_text())["commit"]
    return "unknown: use source.zip hashes"


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


@torch.random.fork_rng(devices=[])
def evaluate(net, assets, cfg, label, out, seeds, plot=True, old=False, stage=1):
    rows = []
    for stochastic in (False, True):
        for seed in seeds:
            torch.manual_seed(seed + 1000000)
            env = V2Environment(assets, cfg)
            env.set_stage(stage)
            actor, _ = env.reset(1.0, seed)
            cumulative = []
            while True:
                with torch.no_grad():
                    dist = net.distribution(torch.tensor(actor))
                    raw = dist.sample() if stochastic else dist.loc
                (actor, _), reward, done, info = env.step(raw.tanh().numpy())
                cumulative.append(
                    dict(
                        time=env.d.time,
                        **{k: v for k, v in info.items() if k.startswith("reward_")},
                    )
                )
                if done:
                    break
            m = env.measurement
            rows.append(
                dict(
                    label=label,
                    seed=seed,
                    stochastic=stochastic,
                    takeoff=info["takeoff"],
                    peak=math.degrees(m.peak),
                    loaded=math.degrees(m.loaded_peak),
                    certified=math.degrees(m.certified_peak),
                    peak_time=m.peak_time,
                    first_invalid=m.first_invalid,
                    max_gap=m.max_gap,
                    slip=m.slip,
                    recovered=m.recovered,
                    front_recovery_seconds=m.best_front_recovery,
                    stable_handstand_seconds=m.best_handstand,
                    catch=m.best_catch,
                    crossings=m.crossings,
                    dynamic_crossings=m.dynamic_crossings,
                    flight_displacement_m=m.flight_displacement,
                    useful_feedback=sum(r["v2_progress"] for r in env.trace),
                    collision=info["nonfoot_collision"] or info["self_collision"],
                    termination=info["termination"],
                )
            )
            stem = f"{label}_{seed}_{int(stochastic)}"
            write_csv(out / f"{stem}_physics.csv", env.trace)
            write_csv(out / f"{stem}_rewards.csv", cumulative)
            # Replay legacy rewards on the identical reset and policy noise stream.
            if old:
                legacy_env = legacy.HandstandCurriculumEnvironment(assets, cfg)
                legacy_env.set_stage(1)
                a, _ = legacy_env.reset(1.0, seed)
                torch.manual_seed(seed + 1000000)
                traces = []
                while True:
                    with torch.no_grad():
                        dist = net.distribution(torch.tensor(a))
                        raw = dist.sample() if stochastic else dist.loc
                    (a, _), r, d, i = legacy_env.step(raw.tanh().numpy())
                    traces.append(
                        dict(
                            time=legacy_env.d.time,
                            reward=r,
                            **{k: v for k, v in i.items() if k.startswith("reward_")},
                        )
                    )
                    if d:
                        break
                write_csv(out / f"{stem}_legacy_rewards.csv", traces)
            if plot and seed == seeds[0]:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                tr = env.trace
                t = [r["time"] for r in tr]
                fig, ax = plt.subplots(6, 1, figsize=(11, 16), sharex=True)
                groups = [
                    ("pitch", "pitch_velocity"),
                    ("rear_clearance", "gap"),
                    ("load_FL", "load_FR"),
                    ("loaded_slip", "front_drift"),
                    ("support_momentum_y", "pendulum_omega"),
                    ("v2_progress", "v2_velocity"),
                ]
                for axis, keys in zip(ax, groups):
                    for k in keys:
                        axis.plot(t, [r.get(k, 0) for r in tr], label=k)
                    axis.legend()
                    axis.grid()
                ax[-1].set_xlabel("seconds")
                fig.tight_layout()
                fig.savefig(out / f"{stem}.png")
                plt.close(fig)
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "mode", choices=["diagnose", "smoke", "pilot", "extended", "playback"]
    )
    p.add_argument("--checkpoint", type=Path, default=ORIGINAL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--gate", type=Path)
    p.add_argument("--replication-gate", type=Path)
    p.add_argument("--authorize-extended", action="store_true")
    p.add_argument("--training-seed", type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    net, hp, cfg, source = legacy.load_source(ORIGINAL)
    if args.checkpoint != ORIGINAL:
        net.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
                "network"
            ]
        )
    if args.mode in ("smoke", "pilot", "extended"):
        hp["learning_rate"] = min(hp["learning_rate"], 1e-5)
        hp["bc_weight"] = 0.0
    assets = Assets(out / "assets")
    seeds = list(range(70400, 70400 + args.seeds))
    if args.mode == "playback":
        import mujoco.viewer

        env = V2Environment(assets, cfg)
        a, _ = env.reset(1.0, seeds[0])
        with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
            import time

            while viewer.is_running():
                with torch.no_grad():
                    action = net.actor(torch.tensor(a)).tanh().numpy()
                (a, _), _, done, _ = env.step(action)
                viewer.sync()
                time.sleep(cfg.dt)
                if done:
                    a, _ = env.reset(1.0, seeds[0])
        return
    manifest = dict(
        commit=source_commit(),
        original_sha256=legacy.sha256(ORIGINAL),
        args=vars(args),
        stages=STAGES,
        environment=vars(cfg),
        hp=hp,
        torch=torch.__version__,
    )
    dump_json(out / "manifest.json", manifest)
    # Snapshot sources, including the uncommitted legacy curriculum.
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root in ("experiments", "scripts", "gym", "resources", "learning"):
            for f in Path(root).rglob("*"):
                if (
                    f.is_file()
                    and "__pycache__" not in str(f)
                    and f.suffix not in (".pyc",)
                ):
                    z.write(f)
        z.write("curriculum.py")
        z.write("pyproject.toml")
    manifest["source_zip_sha256"] = legacy.sha256(out / "source.zip")
    dump_json(out / "manifest.json", manifest)
    if args.mode == "diagnose":
        rows = evaluate(net, assets, cfg, "original", out, seeds, old=True)
        for update in (4, 20, 44):
            ckpt = next(OLD.glob(f"policy_update_{update:04d}_*.pt"))
            net.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=False)["network"]
            )
            rows += evaluate(net, assets, cfg, f"update{update}", out, seeds, old=True)
        dump_json(out / "comparison.json", rows)
    else:
        # Tests precede every optimizer run, including on Molab.
        subprocess.run(
            [
                os.sys.executable,
                "-m",
                "pytest",
                "-q",
                "experiments/handstand_v2/test_measurement.py",
            ],
            check=True,
        )
        if args.mode in ("pilot", "extended"):
            if not args.gate:
                raise ValueError("--gate must identify a passing preceding run")
            gate = json.loads(args.gate.read_text())
            if gate.get("pilot_go") is not True:
                raise ValueError("preceding gate did not pass")
            if args.mode == "extended":
                if not args.replication_gate:
                    raise ValueError("Independent training replication gate required")
                replication = json.loads(args.replication_gate.read_text())
                if not (
                    gate.get("extended_go") is True
                    and replication.get("extended_go") is True
                    and gate.get("training_seed") != replication.get("training_seed")
                ):
                    raise ValueError("No replicated supported gain")
        if args.mode == "extended" and not args.authorize_extended:
            raise ValueError("Extended training requires separate user authorization")
        baseline = evaluate(net, assets, cfg, "before", out, seeds)
        if (
            min(
                np.mean([r["takeoff"] for r in baseline if r["stochastic"] == s])
                for s in (False, True)
            )
            < 0.9
        ):
            raise RuntimeError("Takeoff baseline below 90%; no PPO")
        if sum(r["useful_feedback"] for r in baseline) <= 0:
            raise RuntimeError("No useful progress feedback; no PPO")
        hp["learning_rate"] = min(hp["learning_rate"], 1e-5)
        hp["bc_weight"] = 0.0
        optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
        envs = [V2Environment(assets, cfg) for _ in range(4)]
        initial = [
            e.reset(1.0, legacy.event_seed(args.training_seed, i, 0))
            for i, e in enumerate(envs)
        ]
        actors = np.stack([a for a, c in initial])
        critics = np.stack([c for a, c in initial])
        counts = [0] * 4
        stage = 1
        training = []
        passing = 0
        budget = {"smoke": 2, "pilot": 10, "extended": 100}[args.mode]
        for iteration in range(1, budget + 1):
            actors, critics, completed, stats = legacy.ppo_iteration(
                net,
                optimizer,
                hp,
                envs,
                actors,
                critics,
                args.training_seed,
                counts,
                1024,
                stage,
            )
            training.append(dict(iteration=iteration, stage=stage, **stats))
            print(training[-1], flush=True)
            if not all(
                np.isfinite(v) for v in stats.values() if isinstance(v, (float, int))
            ):
                raise RuntimeError("nonfinite PPO")
            if stats["post_update_kl"] > 0.03:
                break
            torch.save(
                dict(network=net.state_dict(), hp=hp, iteration=iteration, stage=stage),
                out / f"policy_{iteration:04d}.pt",
            )
            if iteration % 2 and iteration != budget:
                continue
            probe = evaluate(
                net,
                assets,
                cfg,
                f"probe{iteration}",
                out,
                seeds,
                plot=False,
                stage=stage,
            )
            retention = min(
                np.mean([r["takeoff"] for r in probe if r["stochastic"] == s])
                for s in (False, True)
            )
            if retention < 0.9:
                break
            # Require two passing windows; keep target fixed within each training block.
            target, hold = STAGES[stage][1:]
            rates = [
                np.mean(
                    [
                        target in r["crossings"] and (hold == 0 or r["catch"] >= hold)
                        for r in probe
                        if r["stochastic"] == s
                    ]
                )
                for s in (False, True)
            ]
            passing = passing + 1 if min(rates) >= 0.8 else 0
            if passing >= 2 and stage < len(STAGES) - 1:
                stage += 1
                passing = 0
                initial = [
                    e.reset(
                        1.0, legacy.event_seed(args.training_seed, i, counts[i] + 1)
                    )
                    for i, e in enumerate(envs)
                ]
                actors = np.stack([a for a, c in initial])
                critics = np.stack([c for a, c in initial])
        after = evaluate(net, assets, cfg, "after", out, seeds)
        dump_json(out / "comparison.json", dict(before=baseline, after=after))
        dump_json(out / "training.json", training)
        takeoff = min(
            np.mean([r["takeoff"] for r in after if r["stochastic"] == s])
            for s in (False, True)
        )
        useful = sum(r["useful_feedback"] for r in after) > 0
        gain = min(
            np.mean([r["certified"] for r in after if r["stochastic"] == s])
            - np.mean([r["certified"] for r in baseline if r["stochastic"] == s])
            for s in (False, True)
        )
        dump_json(
            out / "gate.json",
            dict(
                training_seed=args.training_seed,
                completed_updates=len(training),
                source_sha256=legacy.sha256(ORIGINAL),
                tests_passed=True,
                pilot_go=bool(takeoff >= 0.9 and useful and len(training) == budget),
                extended_go=bool(takeoff >= 0.9 and gain > 1.0),
                supported_gain_degrees=gain,
                takeoff=takeoff,
                note=(
                    "Extended mode needs user authorization and "
                    "supported gain replicated with another training seed."
                ),
            ),
        )
    import shutil

    shutil.make_archive(str(out) + "_artifacts", "zip", out)
    print("Artifacts:", str(out) + "_artifacts.zip", flush=True)


if __name__ == "__main__":
    main()
