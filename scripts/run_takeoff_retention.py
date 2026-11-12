"""Bounded matched BC retention diagnostic; never launches extended training."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from experiments.morphology.task import write_json
from scripts.run_takeoff_ppo import TakeoffRun, check_validation
from scripts.run_physical_morphology_ppo import protocol
from scripts.morphology_feasibility import save_csv


def probe(run, seeds):
    records = []
    # Complete horizon: five-second balance is observable; no goal termination.
    env = TakeoffEnvironment(
        run.envs[0].assets,
        TakeoffConfig(**dict(run.p["environment"], terminate_on_goal=False)),
    )
    for scale in sorted(set([1.0, run.schedule.scale])):
        for stochastic in (False, True):
            for seed in seeds:
                rng = torch.Generator().manual_seed(seed + 1000000)
                actor, _ = env.reset(scale, seed)
                while True:
                    with torch.no_grad():
                        latent = run.net.actor(torch.tensor(actor))
                        if stochastic:
                            latent = latent + run.net.distribution(
                                torch.tensor(actor)
                            ).scale * torch.randn(12, generator=rng)
                        action = latent.tanh().numpy()
                    (actor, _), _, done, info = env.step(action)
                    if done:
                        break
                records.append(
                    dict(
                        **info,
                        method=run.method,
                        training_seed=run.seed,
                        iteration=run.iteration,
                        evaluation_seed=seed,
                        stochastic=stochastic,
                    )
                )
    return records


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bc-actor", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--updates", type=int, default=20, choices=range(10, 26))
    p.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/morphology/learning/protocol_takeoff.json"),
    )
    p.add_argument("--bc-gate", type=Path)
    a = p.parse_args()
    gate = check_validation(a.validation)
    cfg = protocol(a.protocol)
    if a.bc_gate:
        import json

        bc_gate = json.loads(a.bc_gate.read_text())
        if not bc_gate["passed"]:
            raise ValueError("Shared BC closed-loop gate failed; PPO not launched")
        import hashlib

        if (
            bc_gate.get("actor_sha256")
            != hashlib.sha256(a.bc_actor.read_bytes()).hexdigest()
        ):
            raise ValueError("BC actor differs from gated actor")
        if abs(np.exp(cfg["ppo"]["initial_log_std"]) - bc_gate["latent_std"]) > 1e-8:
            raise ValueError("Exploration std differs from gated BC evaluation")
    cfg.update(
        training_seeds=a.training_seeds,
        evaluation_seeds=cfg.get(
            "retention_evaluation_seeds", list(range(31000, 31008))
        ),
        evaluation_interval=2,
        base_training_steps=a.updates * 1024,
    )
    a.output.mkdir(parents=True, exist_ok=False)
    write_json(a.output / "protocol.json", cfg)
    write_json(a.output / "validation_gate.json", gate)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    assets.directory = a.output / "assets"
    all_records = []
    outcomes = []
    for seed in a.training_seeds:
        runs = [
            TakeoffRun(
                a.output / f"{m}_seed{seed}",
                m,
                seed,
                copy.deepcopy(cfg),
                assets,
                a.bc_actor,
            )
            for m in ("A", "B")
        ]
        baseline = {}
        regressions = {m: 0 for m in ("A", "B")}
        reason = "completed bounded diagnostic"
        for iteration in range(a.updates + 1):
            if iteration:
                for run in runs:
                    run.collect_and_update()
                if any(
                    not np.isfinite(r.training[-1]["post_update_kl"])
                    or r.training[-1]["post_update_kl"] > 0.03
                    or r.training[-1]["clip_fraction"] > 0.3
                    for r in runs
                ):
                    reason = "excessive policy divergence"
                    break
            if iteration % 2 == 0 or iteration == a.updates:
                for run in runs:
                    run.checkpoint_and_evaluate()
                    records = probe(run, cfg["evaluation_seeds"])
                    all_records.extend(records)
                    # Retention gate on original geometry, where BC has a reliable mean.
                    rate = np.mean(
                        [
                            r["takeoff"]
                            for r in records
                            if r["scale"] == 1.0 and not r["stochastic"]
                        ]
                    )
                    if iteration == 0:
                        baseline[run.method] = float(rate)
                    regressions[run.method] = (
                        regressions[run.method] + 1
                        if rate < baseline[run.method] - 0.25
                        else 0
                    )
                    print(
                        f"PROBE seed={seed} arm={run.method} update={iteration} "
                        f"full deterministic takeoff={rate:.3f}",
                        flush=True,
                    )
                save_csv(a.output / "probes.csv", all_records)
                if max(regressions.values()) >= 3:
                    reason = (
                        "persistent full-scale deterministic takeoff regression "
                        "(three evaluations)"
                    )
                    break
        for run in runs:
            if not (run.directory / f"checkpoint_{run.iteration:06d}.pt").exists():
                run.checkpoint_and_evaluate()
            outcomes.append(
                dict(
                    seed=seed,
                    method=run.method,
                    updates=run.iteration,
                    scale=run.schedule.scale,
                    events=run.schedule.events,
                    stop_reason=reason,
                    baseline_full_scale_takeoff=baseline[run.method],
                )
            )
        write_json(a.output / "outcome.json", outcomes)
    write_json(
        a.output / "completion.json",
        dict(bounded_diagnostic=True, extended_training=False, outcomes=outcomes),
    )


if __name__ == "__main__":
    main()
