"""Evaluate a trusted takeoff checkpoint on training and original geometry."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import TakeoffEnvironment, TakeoffConfig
from scripts.run_physical_morphology_ppo import evaluate
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(40000, 40032)))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    s = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    cfg = json.loads((a.checkpoint.parent / "configuration.json").read_text())
    net = ActorCritic(
        44,
        74,
        cfg["protocol"]["ppo"]["hidden_dims"],
        log_std_min=cfg["protocol"]["ppo"].get("log_std_min", -3.0),
    )
    net.load_state_dict(s["network"])
    torch.set_num_threads(1)
    env = TakeoffEnvironment(
        Assets(a.output / "assets"),
        TakeoffConfig(**dict(cfg["protocol"]["environment"], terminate_on_goal=False)),
    )
    c = s["curriculum"]
    scale = 1.0 if s["method"] == "A" else c["protocol"]["scales"][c["stage"]]
    rows = []
    for scale in sorted(set([scale, 1.0])):
        episodes, traces, _ = evaluate(
            net,
            env,
            a.seeds,
            scale,
            s["actual_steps"],
            s["method"],
            s["training_seed"],
            s["iteration"],
        )
        rows.extend(episodes)
        save_csv(a.output / f"trace_{scale:.2f}.csv", traces)
    save_csv(a.output / "evaluation.csv", rows)


if __name__ == "__main__":
    main()
