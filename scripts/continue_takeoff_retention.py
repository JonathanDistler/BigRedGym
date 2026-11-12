"""Bounded paired continuation using the unchanged checkpoint protocol."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.task import write_json
from scripts.morphology_feasibility import save_csv
from scripts.run_takeoff_ppo import TakeoffRun, semantic_hash
from scripts.run_takeoff_retention import probe


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-update", type=int, default=10)
    parser.add_argument("--additional-updates", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.additional_updates <= 20:
        raise ValueError("Only bounded continuations of 1–20 updates are supported")
    p = json.loads((args.source / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", p)
    torch.set_num_threads(1)
    assets = Assets(args.output / "assets")
    records = []
    outcomes = []
    audits = []
    for seed in (42, 43):
        runs = []
        for method in ("A", "B"):
            directory = args.source / f"{method}_seed{seed}"
            checkpoint = directory / f"checkpoint_{args.start_update:06d}.pt"
            config = json.loads((directory / "configuration.json").read_text())
            if config["reward_semantic_sha256"] != semantic_hash(
                "experiments/morphology/learning/takeoff.py"
            ):
                raise ValueError("Reward/environment differs from source experiment")
            bc = Path(config["bc_initialization"]["path"])
            if (
                hashlib.sha256(bc.read_bytes()).hexdigest()
                != config["bc_initialization"]["sha256"]
            ):
                raise ValueError("BC initialization hash mismatch")
            run = TakeoffRun(
                args.output / directory.name, method, seed, copy.deepcopy(p), assets, bc
            )
            run.restore(checkpoint)
            if run.iteration != args.start_update:
                raise ValueError("Checkpoint update differs from requested start")
            shutil.copy2(checkpoint, run.directory / checkpoint.name)
            state = torch.load(checkpoint, weights_only=False, map_location="cpu")
            assert all(
                torch.equal(v, run.net.state_dict()[k])
                for k, v in state["network"].items()
            )
            assert torch.equal(run.torch_state, state["torch_rng"])
            assert run.rng.bit_generator.state == state["numpy_rng"]
            assert np.array_equal(run.actors, state["actors"])
            assert np.array_equal(run.critics, state["critics"])
            assert (
                run.optimizer.state_dict()["param_groups"]
                == state["optimizer"]["param_groups"]
            )
            assert all(
                torch.equal(v, run.optimizer.state_dict()["state"][k][name])
                for k, value in state["optimizer"]["state"].items()
                for name, v in value.items()
                if torch.is_tensor(v)
            )
            audits.append(
                dict(
                    method=method,
                    seed=seed,
                    checkpoint=str(checkpoint),
                    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    network_equal=True,
                    optimizer_equal=True,
                    torch_rng_equal=True,
                    numpy_rng_equal=True,
                    observations_equal=True,
                    normalization="Fixed observations; no running normalization",
                )
            )
            runs.append(run)
        write_json(args.output / "restore_audit.json", audits)
        regressions = {"A": 0, "B": 0}
        baseline = {}
        reason = f"completed {args.additional_updates} additional updates"
        for offset in range(args.additional_updates + 1):
            if offset:
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
            if offset % 5 == 0 or offset == args.additional_updates:
                for run in runs:
                    if offset:
                        run.checkpoint_and_evaluate()
                    rows = probe(run, p["evaluation_seeds"])
                    for row in rows:
                        row["actual_steps"] = run.actual_steps
                    records.extend(rows)
                    rate = float(
                        np.mean(
                            [
                                r["takeoff"]
                                for r in rows
                                if r["scale"] == 1 and not r["stochastic"]
                            ]
                        )
                    )
                    if offset == 0:
                        baseline[run.method] = rate
                    regressions[run.method] = (
                        regressions[run.method] + 1
                        if rate < baseline[run.method] - 0.25
                        else 0
                    )
                    print(
                        f"PROBE seed={seed} arm={run.method} "
                        f"update={run.iteration} takeoff={rate}",
                        flush=True,
                    )
                save_csv(args.output / "probes.csv", records)
                if max(regressions.values()) >= 3:
                    reason = "persistent deterministic takeoff regression"
                    break
        for run in runs:
            if not (run.directory / f"checkpoint_{run.iteration:06d}.pt").exists():
                run.checkpoint_and_evaluate()
            outcomes.append(
                dict(
                    seed=seed,
                    method=run.method,
                    updates=run.iteration,
                    actual_steps=run.actual_steps,
                    scale=run.schedule.scale,
                    events=run.schedule.events,
                    stop_reason=reason,
                )
            )
        write_json(args.output / "outcome.json", outcomes)
    write_json(
        args.output / "completion.json",
        dict(outcomes=outcomes, extended_training=False),
    )


if __name__ == "__main__":
    main()
