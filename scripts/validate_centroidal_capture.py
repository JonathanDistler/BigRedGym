"""Continue a verified planned prefix with the unmodified single actor."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions
from experiments.unified_go2.core import Environment, load
from experiments.retained_go2.actor import Student
import torch
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    starts = load(a.run / "starts.pt")
    cfg, data, bank, assets = setup(starts["args"]["preparation"], a.run)
    net = Student()
    net.load_state_dict(load(starts["args"]["checkpoint"])["network"])
    result = {}
    for label in ["baseline", "best"]:
        saved = load(a.run / (label + ".pt"))
        e = Environment(assets, copy.deepcopy(cfg))
        e.capture_centers = data["centers"]
        e.restore(copy.deepcopy(saved["final"]))
        obs, _ = e.observations()
        commands = []
        for _ in range(300):
            u = actions(net, obs[None])[0]
            commands.append(u)
            (obs, _), _, done, info = e.step(u)
            if done:
                break
        result[label] = dict(
            continuation_seconds=len(commands) * 0.02,
            termination=info["termination"],
            five_second_success=bool(e.ever_hold_success),
            certified_takeoff=bool(e.tracker.confirmed),
            collision=bool(e.nonfoot_collision or e.self_collision),
            terminal_pitch=float(np.rad2deg(e.momentum_sample["pitch"])),
            terminal_velocity=e.momentum_sample["pitch_velocity"],
        )
        torch.save(
            dict(final=e.state(), actions=commands),
            a.run / (label + "_continuation.pt"),
        )
    (a.run / "continuation.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
