"""Measure cumulative distribution drift on fixed demonstration states and replays."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--run", type=Path, default=Path("output/takeoff_retention_20261111_v1")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = np.load("output/takeoff_bc_investigation_v1/bc_dataset.npz")
    x = torch.tensor(data["observations"], dtype=torch.float32)
    assets = Assets(a.output / "assets")
    rows = []
    traces = []
    episodes = []
    for folder in sorted(f for f in a.run.glob("*_seed*") if f.is_dir()):
        initial = torch.load(
            folder / "checkpoint_000000.pt", weights_only=False, map_location="cpu"
        )
        net = ActorCritic(44, 74)
        net.load_state_dict(initial["network"])
        with torch.no_grad():
            mu0 = net.actor(x)
            std0 = net.distribution(x).scale
        for path in sorted(folder.glob("checkpoint_*.pt")):
            state = torch.load(path, weights_only=False, map_location="cpu")
            net.load_state_dict(state["network"])
            with torch.no_grad():
                mu = net.actor(x)
                std = net.distribution(x).scale
                kl = (
                    torch.log(std / std0)
                    + (std0.square() + (mu0 - mu).square()) / (2 * std.square())
                    - 0.5
                )
                for j in range(12):
                    rows.append(
                        dict(
                            run=folder.name,
                            iteration=state["iteration"],
                            joint=j,
                            latent_mean_shift_rms=float(
                                (mu[:, j] - mu0[:, j]).square().mean().sqrt()
                            ),
                            bounded_mean_shift_rms=float(
                                (mu[:, j].tanh() - mu0[:, j].tanh())
                                .square()
                                .mean()
                                .sqrt()
                            ),
                            cumulative_gaussian_kl=float(kl[:, j].mean()),
                            effective_latent_std=float(std[:, j].mean()),
                            raw_log_std=float(net.log_std[j]),
                            bounded_noise_std=float(
                                (std[:, j] * (1 - mu[:, j].tanh().square())).mean()
                            ),
                        )
                    )
            if state["iteration"] not in (0, 4, 8):
                continue
            for scale in [1.0] if state["method"] == "A" else [0.9, 1.0]:
                for stochastic in (False, True):
                    env = TakeoffEnvironment(
                        assets, TakeoffConfig(terminate_on_goal=False)
                    )
                    obs, _ = env.reset(scale, 54000)
                    env.trace_enabled = True
                    rng = torch.Generator().manual_seed(1054000)
                    for _ in range(100):
                        with torch.no_grad():
                            mu = net.actor(torch.tensor(obs))
                            sd = net.distribution(torch.tensor(obs)).scale
                            sampled = (
                                mu + sd * torch.randn(12, generator=rng)
                                if stochastic
                                else mu
                            )
                        (obs, _), _, done, info = env.step(sampled.tanh().numpy())
                        if done:
                            break
                    identity = dict(
                        run=folder.name,
                        iteration=state["iteration"],
                        evaluation_seed=54000,
                        stochastic=stochastic,
                    )
                    episodes.append(dict(**info, **identity))
                    traces.extend(dict(**r, **identity) for r in env.physics_trace)
        print("Analyzed", folder.name, flush=True)
        save_csv(a.output / "distribution_drift.csv", rows)
        save_csv(a.output / "replay_episodes.csv", episodes)
        save_csv(a.output / "replay_physics.csv", traces)


if __name__ == "__main__":
    main()
