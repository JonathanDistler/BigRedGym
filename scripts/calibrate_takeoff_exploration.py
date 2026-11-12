"""Frozen BC policy exploration calibration. Simulation only; no optimization."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import TakeoffEnvironment, TakeoffConfig
from experiments.morphology.task import write_json
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bc-actor", required=True, type=Path)
    p.add_argument(
        "--stds", type=float, nargs="+", default=[0.0, 0.05, 0.10, float(np.exp(-1.5))]
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002, 2003])
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.9])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    net = ActorCritic(44, 74)
    net.actor.load_state_dict(
        torch.load(a.bc_actor, weights_only=False, map_location="cpu")["actor"]
    )
    records = []
    for scale in a.scales:
        env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
        for std in a.stds:
            for seed in a.seeds:
                rng = torch.Generator().manual_seed(seed)
                actor, _ = env.reset(scale, seed)
                for _ in range(100):
                    with torch.no_grad():
                        latent = net.actor(torch.tensor(actor))
                        action = (
                            (latent + std * torch.randn(12, generator=rng))
                            .tanh()
                            .numpy()
                        )
                    (actor, _), _, done, info = env.step(action)
                    if done:
                        break
                records.append(dict(**info, latent_std=std, evaluation_seed=seed))
            print(f"scale={scale:.2f} latent_std={std:.3f} measured", flush=True)
    save_csv(a.output / "calibration.csv", records)
    write_json(
        a.output / "configuration.json",
        dict(
            actor=str(a.bc_actor),
            seconds=2.0,
            scales=a.scales,
            seeds=a.seeds,
            latent_stds=a.stds,
            optimizer_steps=0,
            scope="Exploratory action-noise calibration, not PPO learning",
        ),
    )


if __name__ == "__main__":
    main()
