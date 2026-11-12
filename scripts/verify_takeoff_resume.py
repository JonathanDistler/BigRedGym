"""Verify all paired checkpoint state in a clean checkout without training."""

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
from experiments.morphology.task import write_json
from scripts.run_takeoff_ppo import TakeoffRun, semantic_hash


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", type=int, default=30)
    args = parser.parse_args()
    p = json.loads((args.run / "protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(args.output / "assets")
    audits = []
    for seed in (42, 43):
        for method in ("A", "B"):
            directory = args.run / f"{method}_seed{seed}"
            config = json.loads((directory / "configuration.json").read_text())
            assert config["reward_semantic_sha256"] == semantic_hash(
                "experiments/morphology/learning/takeoff.py"
            )
            checkpoint = directory / f"checkpoint_{args.update:06d}.pt"
            state = torch.load(checkpoint, weights_only=False, map_location="cpu")
            run = TakeoffRun(
                args.output / directory.name,
                method,
                seed,
                copy.deepcopy(p),
                assets,
                Path(config["bc_initialization"]["path"]),
            )
            run.restore(checkpoint)
            assert run.iteration == args.update
            assert asdict(run.schedule) == state["curriculum"]
            assert all(
                torch.equal(v, run.net.state_dict()[k])
                for k, v in state["network"].items()
            )
            opt = run.optimizer.state_dict()
            assert opt["param_groups"] == state["optimizer"]["param_groups"]
            assert all(
                torch.equal(v, opt["state"][k][name])
                for k, value in state["optimizer"]["state"].items()
                for name, v in value.items()
            )
            assert torch.equal(run.torch_state, state["torch_rng"])
            assert run.rng.bit_generator.state == state["numpy_rng"]
            assert np.array_equal(run.actors, state["actors"])
            assert np.array_equal(run.critics, state["critics"])
            assert run.episode_counts == state["episode_counts"]
            assert all(
                np.allclose(
                    e.state()["integration"], s["integration"], atol=1e-10, rtol=1e-10
                )
                for e, s in zip(run.envs, state["worlds"])
            )
            audits.append(
                dict(
                    method=method,
                    seed=seed,
                    update=args.update,
                    actual_steps=run.actual_steps,
                    checkpoint_sha256=hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    network_optimizer_curriculum_rng_observations_worlds_verified=True,
                    normalization="Fixed transformations; no running statistics",
                    training_interactions=0,
                )
            )
    write_json(args.output / "verification.json", audits)
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
