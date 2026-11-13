"""Matched frozen-actor reward-only/action-only smoothing ablation (no PPO)."""

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.com_energy_phase.core import EnergyConfig, EnergyEnvironment
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from scripts.run_com_energy_phase import evaluate, dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    net = ActorCritic(47, 77)
    net.load_state_dict(saved["network"])
    net.eval()
    assets = Assets(a.output / "assets")
    for label, reward_tau, action_tau in [
        ("instant", 0.0, 0.0),
        ("reward_ema", 0.04, 0.0),
        ("action_ema", 0.0, 0.02),
        ("both", 0.04, 0.02),
    ]:
        out = a.output / label
        out.mkdir()
        cfg = EnergyConfig(**copy.deepcopy(saved["config"]))
        cfg.energy_tau = cfg.orientation_tau = reward_tau
        cfg.action_tau = action_tau
        dump(out / "config.json", asdict(cfg))
        env = EnergyEnvironment(assets, cfg)
        evaluate(net, env, [91000, 91001], out, "frozen")
    dump(
        a.output / "scope.json",
        dict(ppo_updates=0, checkpoint=str(a.checkpoint), actor_fixed=True),
    )


if __name__ == "__main__":
    main()
