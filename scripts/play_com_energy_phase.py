"""Deterministic standalone checkpoint playback using the training phase encoder."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.com_energy_phase.core import EnergyConfig, EnergyEnvironment
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from scripts.run_com_energy_phase import evaluate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[91000, 91001])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    cfg = EnergyConfig(**saved["config"])
    phase = cfg.phase_enabled
    net = ActorCritic(47 if phase else 44, 77 if phase else 74)
    net.load_state_dict(saved["network"])
    net.eval()
    env = EnergyEnvironment(Assets(a.output / "assets"), cfg)
    evaluate(net, env, a.seeds, a.output, "playback")


if __name__ == "__main__":
    main()
