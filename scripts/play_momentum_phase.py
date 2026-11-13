"""Standalone single-actor checkpoint playback with the training estimator and phase."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.momentum_phase.core import MomentumConfig, MomentumEnvironment
from experiments.momentum_phase.tools import network, evaluate
from experiments.morphology.learning.environment import Assets


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[94000, 94001])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    net, saved = network(a.checkpoint)
    net.eval()
    cfg = MomentumConfig(**saved["config"])
    env = MomentumEnvironment(Assets(a.output / "assets"), cfg)
    evaluate(net, env, a.seeds, a.output, "playback")


if __name__ == "__main__":
    main()
