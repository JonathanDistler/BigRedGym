"""Bounded, matched A/B/C/D COM-energy diagnostic; no extended-training mode."""

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.com_energy_phase.core import EnergyConfig, EnergyEnvironment, Stages
from experiments.com_energy_phase.data import reference, align, sha
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic, advantages, update
from scripts.morphology_feasibility import save_csv


def dump(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        )
    )


def actor_initialization(net, saved):
    state = copy.deepcopy(saved["actor"])
    if net.actor[0].in_features == 47:
        w = state["0.weight"]
        state["0.weight"] = torch.cat([w, torch.zeros((len(w), 3))], 1)
    net.actor.load_state_dict(state)


def evaluate(net, env, seeds, out, label):
    rows = []
    traces = []
    for seed in seeds:
        obs, _ = env.reset(1.0, seed)
        while True:
            with torch.no_grad():
                action = net.actor(torch.tensor(obs)).tanh().numpy()
            (obs, _), _, done, info = env.step(action)
            if done:
                break
        rows.append(dict(**info, seed=seed, checkpoint=label))
        traces.extend(dict(**s, seed=seed, checkpoint=label) for s in env.trace)
    save_csv(out / f"evaluation_{label}.csv", rows)
    save_csv(out / f"trace_{label}.csv", traces)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intermediate-fraction", type=float, default=0.0)
    parser.add_argument("--updates", type=int, default=2, choices=range(0, 5))
    parser.add_argument("--steps", type=int, default=128, choices=(64, 128, 256))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument(
        "--bc", type=Path, default=Path("output/multimorph_bc_20261111_v2/bc_actor.pt")
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/multimorph_bc_20261111_v2/bc_dataset.npz"),
    )
    parser.add_argument(
        "--reference", type=Path, default=Path("output/recoverability_maps_20261112")
    )
    args = parser.parse_args()
    if not 0 <= args.intermediate_fraction <= 1:
        parser.error("intermediate-fraction must be in [0,1]")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    # Tests run before any PPO mutation; preserve the exact validation output.
    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit_tests/test_com_energy_phase.py",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    (out / "validation.txt").write_text(validation.stdout + validation.stderr)
    if validation.returncode:
        raise RuntimeError("COM-energy validation failed; inspect validation.txt")
    torch.set_num_threads(1)
    assets = Assets(out / "assets")
    probe = EnergyEnvironment(assets, EnergyConfig())
    ref = reference(probe, args.reference)
    dump(out / "reference.json", ref)
    reset_bank = np.load(args.reference / "recoverable_states.npz")
    data = np.load(args.dataset)
    bc = torch.load(args.bc, weights_only=False, map_location="cpu")
    if sha(args.dataset) != bc["dataset_sha256"]:
        raise ValueError("BC dataset provenance mismatch")
    encoding, weights, events, times = align(data, ref["mass"])
    dump(out / "demonstration_events.json", events)
    np.savez_compressed(out / "phase_alignment.npz", encoding=encoding, weights=weights)
    source = out / "source"
    source.mkdir()
    for file in [Path(__file__), *Path("experiments/com_energy_phase").glob("*.py")]:
        shutil.copyfile(file, source / file.name)
    hp = json.loads(
        Path("experiments/morphology/learning/protocol_takeoff.json").read_text()
    )["ppo"]
    hp.update(
        epochs=2,
        batch_size=64,
        bc_weight=0.02,
        bc_batch_size=128,
        bc_latent_scale=0.05,
        bc_phase_fraction=1.0,
    )
    manifest = dict(
        args=vars(args),
        bc_sha256=sha(args.bc),
        dataset_sha256=sha(args.dataset),
        phase_times=times,
        training_steps_per_arm=args.updates * args.steps,
        evaluation_seeds=[91000, 91001],
        source_sha256={str(f): sha(f) for f in source.glob("*.py")},
        full_geometry=1.0,
        runtime_cpg=False,
        parent_commit="059ac39",
        scope="software smoke and short diagnostic only",
    )
    dump(out / "manifest.json", manifest)
    all_results = []
    for seed in args.seeds:
        for arm in "ABCD":
            directory = out / f"{arm}_seed{seed}"
            directory.mkdir()
            cfg = EnergyConfig(
                energy_enabled=arm != "A",
                phase_enabled=arm in "CD",
                phase_times=times,
                target_height=ref["height"],
                target_rotation=tuple(ref["rotation"]),
                energy_tau=0.04 if arm == "D" else 0.0,
                orientation_tau=0.04 if arm == "D" else 0.0,
                action_tau=0.02 if arm == "D" else 0.0,
            )
            dump(directory / "config.json", asdict(cfg))
            env = EnergyEnvironment(assets, cfg)
            evaluation = EnergyEnvironment(assets, copy.deepcopy(cfg))
            torch.manual_seed(seed)
            net = ActorCritic(
                47 if cfg.phase_enabled else 44,
                77 if cfg.phase_enabled else 74,
                log_std=np.log(0.05),
            )
            actor_initialization(net, bc)
            # Match critic initialization too, with zero new phase columns.
            torch.manual_seed(seed)
            common = ActorCritic(44, 74, log_std=np.log(0.05))
            critic = copy.deepcopy(common.critic.state_dict())
            if cfg.phase_enabled:
                w = critic["0.weight"]
                critic["0.weight"] = torch.cat([w, torch.zeros((len(w), 3))], 1)
            net.critic.load_state_dict(critic)
            # New phase columns start at zero, preserving every historical actor output.
            x = data["observations"].astype(np.float32)
            if cfg.phase_enabled:
                x = np.c_[x, encoding].astype(np.float32)
            # Episode split matches the original BC dataset's held-out last four seeds.
            train = data["seed"] < np.unique(data["seed"])[-4]
            demos = (
                torch.tensor(x[train]),
                torch.tensor(data["actions"][train], dtype=torch.float32),
                torch.tensor(
                    weights[train] if cfg.phase_enabled else np.ones(train.sum()),
                    dtype=torch.float32,
                ),
            )
            # Supervised initialization learns only the new phase input columns.
            if cfg.phase_enabled:
                opt = torch.optim.Adam([net.actor[0].weight], lr=1e-5)
                for _ in range(8):
                    ids = torch.randint(len(demos[0]), (128,))
                    pred = net.actor(demos[0][ids])
                    target = torch.atanh(demos[1][ids].clamp(-0.999, 0.999))
                    loss = ((pred - target).square().mean(-1) * demos[2][ids]).mean()
                    opt.zero_grad()
                    loss.backward()
                    net.actor[0].weight.grad[:, :44] = 0
                    opt.step()
            torch.manual_seed(seed + 100000)
            optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
            stages = Stages()
            obs, priv = env.reset(1.0, seed * 1000)
            episode = 0
            training = []
            for iteration in range(args.updates + 1):
                results = evaluate(
                    net, evaluation, [91000, 91001], directory, f"{iteration:03}"
                )
                all_results.extend(
                    dict(**r, arm=arm, training_seed=seed, iteration=iteration)
                    for r in results
                )
                checkpoint = dict(
                    network=net.state_dict(),
                    optimizer=optimizer.state_dict(),
                    environment=copy.deepcopy(env.state()),
                    config=asdict(cfg),
                    rng=torch.get_rng_state(),
                    curriculum=stages,
                    iteration=iteration,
                    manifest=manifest,
                )
                torch.save(checkpoint, directory / f"checkpoint_{iteration:03}.pt")
                if iteration == args.updates:
                    break
                records = []
                episodes = []
                for _ in range(args.steps):
                    with torch.no_grad():
                        raw, log, value = net.sample(
                            torch.tensor(obs), torch.tensor(priv)
                        )
                    (nextobs, nextpriv), reward, done, info = env.step(
                        raw.tanh().numpy()
                    )
                    with torch.no_grad():
                        nv = float(net.critic(torch.tensor(nextpriv)).squeeze())
                    records.append(
                        (
                            obs,
                            priv,
                            raw.numpy(),
                            float(log),
                            float(value),
                            reward,
                            done,
                            nv,
                        )
                    )
                    obs, priv = nextobs, nextpriv
                    if done:
                        stages.observe(info)
                        cfg.stage = stages.stage
                        episodes.append(info)
                        episode += 1
                        reset_seed = seed * 1000 + episode
                        rng = np.random.default_rng(reset_seed)
                        if (
                            stages.stage >= 3
                            and rng.random() < args.intermediate_fraction
                        ):
                            index = int(rng.choice(ref["checked_indices"]))
                            obs, priv = env.reset_intermediate(
                                reset_bank[f"{index}_integration"],
                                reset_seed,
                                dict(source=str(args.reference), index=index),
                            )
                        else:
                            obs, priv = env.reset(1.0, reset_seed)
                a, returns = advantages(
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
                    torch.tensor(returns[:, 0], dtype=torch.float32),
                    torch.tensor(a[:, 0], dtype=torch.float32),
                )
                stats = update(net, optimizer, batch, hp, demos)
                training.append(
                    dict(
                        iteration=iteration + 1,
                        steps=(iteration + 1) * args.steps,
                        stage=stages.stage,
                        **stats,
                    )
                )
                save_csv(directory / "training.csv", training)
                dump(
                    directory / "curriculum.json",
                    dict(stage=stages.stage, events=stages.events),
                )
                print(
                    arm, seed, iteration + 1, "KL", stats["post_update_kl"], flush=True
                )
    save_csv(out / "comparison.csv", all_results)
    dump(
        out / "completion.json",
        dict(
            extended_training=False,
            updates=args.updates,
            training_interactions=len(args.seeds) * 4 * args.updates * args.steps,
        ),
    )


if __name__ == "__main__":
    main()
