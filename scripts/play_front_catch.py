"""Single-actor playback with separately labelled intermediate resets."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.core import CatchConfig
from experiments.front_catch.anticipatory import environment_class
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--mode", default="C_seed42")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[96000, 96001, 96002, 96003])
    p.add_argument(
        "--start-kind",
        default="standing",
        choices=(
            "standing",
            "loaded",
            "pre_liftoff",
            "post_liftoff",
            "partial",
            "near_vertical",
        ),
    )
    p.add_argument("--preparation", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    run = saved["runs"][a.mode] if "runs" in saved else saved
    net = ActorCritic(73, 103)
    if "network" in run:
        net.load_state_dict(run["network"])
    else:
        net.actor.load_state_dict(run["actor"])
    cfg = CatchConfig(**run["config"])
    cfg.full_trace = True
    env = environment_class(cfg)(Assets(a.output / "assets"), cfg)
    rows = []
    traces = []
    bank = None
    if a.start_kind != "standing":
        if a.preparation is None:
            p.error("Intermediate playback requires --preparation")
        bank = torch.load(
            a.preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
        )
    for seed in a.seeds:
        if bank:
            states = bank["evaluation"][a.start_kind]
            state = states[seed % len(states)]["state"]
            obs, _ = env.restore_start(state, a.start_kind)
        else:
            obs, _ = env.reset(1.0, seed)
        while True:
            with torch.no_grad():
                action = net.actor(torch.tensor(obs)).tanh().numpy()
            (obs, _), _, done, info = env.step(action)
            if done:
                break
        rows.append(dict(**info, evaluation_seed=seed))
        traces.extend(dict(**s, evaluation_seed=seed) for s in env.trace)
    save_csv(a.output / "evaluation.csv", rows)
    save_csv(a.output / "trace.csv", traces)


if __name__ == "__main__":
    main()
