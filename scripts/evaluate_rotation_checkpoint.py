"""Fresh matched full-horizon rotation evaluation; zero optimizer updates."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.task import write_json
from scripts.run_rotation_learning import RotationRun, evaluate_run
from scripts.morphology_feasibility import save_csv


def main():
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--checkpoint", type=Path)
    parser.add_argument("--update", type=int, default=36)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(64000, 64008))
    )
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    paths = (
        [a.checkpoint]
        if a.checkpoint
        else sorted(a.run.glob(f"*/*_seed*/checkpoint_{a.update:06d}.pt"))
    )
    if a.checkpoint is None:
        paths = [
            p
            for p in paths
            if any(p.parent.name.endswith(f"seed{s}") for s in a.training_seeds)
        ]
    if not paths:
        raise ValueError("No checkpoints found")
    traces = a.output / "traces"
    traces.mkdir()
    rows = []
    audits = []
    for index, path in enumerate(paths):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        p = saved["curriculum"]["protocol"]
        method = saved["method"]
        seed = saved["training_seed"]
        anchor = path.parent.parent.name
        run = RotationRun(
            a.output / f"restored_{index}",
            method,
            seed,
            p,
            assets,
            Path("output/multimorph_bc_20261111_v2/bc_actor.pt"),
        )
        run.attach_phases(Path(p["rotation_experiment"]["phase_labels"]))
        run.restore(path)
        assert all(
            torch.equal(v, run.net.state_dict()[k]) for k, v in saved["network"].items()
        )
        opt = run.optimizer.state_dict()
        assert opt["param_groups"] == saved["optimizer"]["param_groups"]
        assert all(
            torch.equal(v, opt["state"][k][name])
            for k, value in saved["optimizer"]["state"].items()
            for name, v in value.items()
        )
        assert torch.equal(run.torch_state, saved["torch_rng"])
        assert run.rng.bit_generator.state == saved["numpy_rng"]
        assert asdict(run.schedule) == saved["curriculum"]
        assert np.array_equal(run.actors, saved["actors"]) and np.array_equal(
            run.critics, saved["critics"]
        )
        assert all(
            np.allclose(
                e.state()["integration"], s["integration"], rtol=1e-10, atol=1e-10
            )
            for e, s in zip(run.envs, saved["worlds"])
        )
        audits.append(
            dict(
                checkpoint=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                network_optimizer_curriculum_rng_observations_worlds_verified=True,
                training_interactions=0,
            )
        )
        rows.extend(evaluate_run(run, anchor, a.seeds, traces))
        save_csv(a.output / "probes.csv", rows)
        write_json(a.output / "restore_verification.json", audits)
        print(path, flush=True)


if __name__ == "__main__":
    main()
