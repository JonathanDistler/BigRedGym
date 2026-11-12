"""Opt-in, bounded 2x2 rotation PPO pilot; no extended launch path."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.rotation import (
    RotationConfig,
    RotationEnvironment,
    ROTATION_TERMS,
)
from experiments.morphology.task import write_json
from scripts.run_takeoff_ppo import TakeoffRun, REWARD_TERMS
from scripts.morphology_feasibility import save_csv


class RotationRun(TakeoffRun):
    config_class = RotationConfig
    environment_class = RotationEnvironment
    reward_terms = REWARD_TERMS + ROTATION_TERMS

    def attach_phases(self, path):
        labels = np.load(path)["weights"]
        data = np.load(self.p["bc_regularization"]["dataset"])
        mask = data["seed"] < self.p["bc_regularization"]["validation_seed_start"]
        if labels.shape != (len(mask),):
            raise ValueError("Phase labels have wrong length")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["dataset_sha256"] != self.p["bc_regularization"]["dataset_sha256"]:
            raise ValueError("Phase labels belong to another dataset")
        self.bc_demonstrations = (
            *self.bc_demonstrations[:2],
            torch.tensor(labels[mask]),
        )


def migrate(directory, method, seed, source, p, assets):
    config = json.loads(
        (source / f"{method}_seed{seed}" / "configuration.json").read_text()
    )
    old = config["protocol"]
    run = RotationRun(
        directory,
        method,
        seed,
        copy.deepcopy(old),
        assets,
        Path(config["bc_initialization"]["path"]),
    )
    checkpoint = source / f"{method}_seed{seed}" / "checkpoint_000030.pt"
    run.restore(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert all(
        torch.equal(v, run.net.state_dict()[k]) for k, v in saved["network"].items()
    )
    assert torch.equal(run.torch_state, saved["torch_rng"])
    optimizer = run.optimizer.state_dict()
    assert optimizer["param_groups"] == saved["optimizer"]["param_groups"]
    assert all(
        torch.equal(v, optimizer["state"][k][name])
        for k, value in saved["optimizer"]["state"].items()
        for name, v in value.items()
    )
    run.p = copy.deepcopy(p)
    run.schedule.protocol = run.p
    run.cfg = RotationConfig(**p["environment"])
    for env in [*run.envs, run.eval_env]:
        env.cfg = run.cfg
        env.measure()
        env._rotation_phi = env.rotation_potential()
        env.reward_sum = 0.0
        env.terms_sum = {}
    run.attach_phases(Path(p["rotation_experiment"]["phase_labels"]))
    updated = json.loads((directory / "configuration.json").read_text())
    updated.update(
        protocol=run.p,
        config=asdict(run.cfg),
        reward_version="rotation_v1",
        migration=dict(
            source_checkpoint=str(checkpoint),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            network_optimizer_rng_preserved=True,
            critic_note="Old critic retained; new objective changes value targets.",
            reward_accumulators="Cleared; physical/tracker/episode states retained.",
        ),
    )
    write_json(directory / "configuration.json", updated)
    return run


def evaluate_run(run, anchor, seeds, trace_directory):
    rows = []
    traces = []
    env = RotationEnvironment(run.envs[0].assets, run.cfg)
    for scale in sorted(set([1.0, run.schedule.scale])):
        for stochastic in (False, True):
            for seed in seeds:
                obs, _ = env.reset(scale, seed)
                env.record_rotation_trace = True
                rng = torch.Generator().manual_seed(seed + 1000000)
                while True:
                    with torch.no_grad():
                        distribution = run.net.distribution(torch.tensor(obs))
                        latent = distribution.loc
                        if stochastic:
                            latent = latent + distribution.scale * torch.randn(
                                12, generator=rng
                            )
                    (obs, _), _, done, info = env.step(latent.tanh().numpy())
                    if done:
                        break
                flight = [
                    r
                    for r in env.rotation_trace
                    if r["takeoff"]
                    and not r["nonfoot_collision"]
                    and not r["self_collision"]
                ]
                peak = max(env.rotation_trace, key=lambda r: r["pitch"])
                lift = flight[0] if flight else None
                rows.append(
                    dict(
                        **info,
                        anchor=anchor,
                        method=run.method,
                        training_seed=run.seed,
                        iteration=run.iteration,
                        actual_steps=run.actual_steps,
                        pilot_steps=run.actual_steps - 30720,
                        stochastic=stochastic,
                        evaluation_seed=seed,
                        liftoff_momentum=lift["momentum_y"] if lift else float("nan"),
                        liftoff_energy=lift["rotational_energy"]
                        if lift
                        else float("nan"),
                        liftoff_deficit=lift["energy_deficit"]
                        if lift
                        else float("nan"),
                        peak_momentum=peak["momentum_y"],
                        peak_omega=peak["pitch_velocity"],
                        mean_front_moment=float(
                            np.mean([r["front_moment_y"] for r in flight])
                        )
                        if flight
                        else float("nan"),
                    )
                )
                rows[-1].update(
                    {
                        f"episode_reward_{key}": value
                        for key, value in env.terms_sum.items()
                    }
                )
                if scale == 1.0 and not stochastic and seed == seeds[0]:
                    for r in env.rotation_trace:
                        traces.append(
                            dict(
                                **r,
                                anchor=anchor,
                                method=run.method,
                                training_seed=run.seed,
                                iteration=run.iteration,
                                evaluation_seed=seed,
                            )
                        )
    name = f"{anchor}_{run.method}_seed{run.seed}"
    save_csv(
        trace_directory / f"{name}_update{run.iteration:06d}_trajectory.csv", traces
    )
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/morphology/learning/protocol_rotation.json"),
    )
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--resume-update", type=int)
    a = parser.parse_args()
    if not 1 <= a.updates <= 10:
        raise ValueError("Pilot is restricted to 1–10 updates")
    validation = json.loads((a.diagnosis / "validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("Rotation model/reward validation failed")
    if (
        validation["rotation_source_sha256"]
        != hashlib.sha256(
            Path("experiments/morphology/learning/rotation.py").read_bytes()
        ).hexdigest()
    ):
        raise ValueError("Rotation code changed since validation")
    p = json.loads(a.protocol.read_text())
    p["rotation_experiment"]["phase_labels"] = str(a.diagnosis / "bc_phase_labels.npz")
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "protocol.json", p)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    traces = a.output / "traces"
    traces.mkdir()
    records = []
    outcomes = []
    start = 30 if a.resume_update is None else a.resume_update
    for seed in (42, 43):
        runs = []
        for anchor in ("global", "phase"):
            for method in ("A", "B"):
                cfg = copy.deepcopy(p)
                if anchor == "phase":
                    cfg["bc_regularization"].update(
                        phase_decay_start_update=30, phase_decay_updates=4
                    )
                directory = a.output / anchor / f"{method}_seed{seed}"
                if a.resume_update is None:
                    run = migrate(directory, method, seed, a.source, cfg, assets)
                else:
                    checkpoint = (
                        a.source
                        / anchor
                        / f"{method}_seed{seed}"
                        / f"checkpoint_{start:06d}.pt"
                    )
                    saved = torch.load(
                        checkpoint, weights_only=False, map_location="cpu"
                    )
                    cfg = saved["curriculum"]["protocol"]
                    run = RotationRun(
                        directory,
                        method,
                        seed,
                        cfg,
                        assets,
                        Path("output/multimorph_bc_20261111_v2/bc_actor.pt"),
                    )
                    run.attach_phases(Path(cfg["rotation_experiment"]["phase_labels"]))
                    run.restore(checkpoint)
                runs.append((anchor, run))
        # Within each geometry/seed the anchor ablation begins identically.
        for method in ("A", "B"):
            pair = [r for _, r in runs if r.method == method]
            assert all(
                torch.equal(v, pair[1].net.state_dict()[k])
                for k, v in pair[0].net.state_dict().items()
            )
        baseline = {}
        reason = "completed bounded rotation pilot"
        for offset in range(a.updates + 1):
            if offset:
                for _, run in runs:
                    run.collect_and_update()
                if any(
                    not np.isfinite(r.training[-1]["post_update_kl"])
                    or r.training[-1]["post_update_kl"] > 0.03
                    or r.training[-1]["clip_fraction"] > 0.3
                    for _, r in runs
                ):
                    reason = "matched quartet stopped for divergence"
                    break
            if offset % 2 == 0 or offset == a.updates:
                regression = False
                for anchor, run in runs:
                    run.checkpoint_and_evaluate()
                    new = evaluate_run(run, anchor, p["evaluation_seeds"], traces)
                    records.extend(new)
                    rate = float(
                        np.mean(
                            [
                                r["takeoff"]
                                for r in new
                                if r["scale"] == 1 and not r["stochastic"]
                            ]
                        )
                    )
                    sampled = float(
                        np.mean(
                            [
                                r["takeoff"]
                                for r in new
                                if r["scale"] == 1 and r["stochastic"]
                            ]
                        )
                    )
                    key = (anchor, run.method)
                    if offset == 0:
                        baseline[key] = (rate, sampled)
                    regression |= (
                        rate < baseline[key][0] - 0.25
                        or sampled < baseline[key][1] - 0.25
                    )
                    print(
                        f"PROBE {anchor} {run.method} seed={seed} "
                        f"update={run.iteration} takeoff={rate}/{sampled}",
                        flush=True,
                    )
                save_csv(a.output / "probes.csv", records)
                if regression:
                    reason = "matched quartet stopped for takeoff regression"
                    break
        for anchor, run in runs:
            run.checkpoint_and_evaluate()
            outcomes.append(
                dict(
                    anchor=anchor,
                    method=run.method,
                    seed=seed,
                    update=run.iteration,
                    actual_steps=run.actual_steps,
                    additional_steps=run.actual_steps - 30720,
                    scale=run.schedule.scale,
                    events=run.schedule.events,
                    reason=reason,
                )
            )
        write_json(a.output / "outcome.json", outcomes)
    write_json(
        a.output / "completion.json", dict(outcomes=outcomes, extended_training=False)
    )


if __name__ == "__main__":
    main()
