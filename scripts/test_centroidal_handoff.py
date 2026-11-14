"""Matched continuous standing replays with offline balance takeover times."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions, render_episode
from experiments.unified_go2.core import Environment, load
from experiments.balance_feedback.actor import BalanceActor
from experiments.centroidal_capture.core import metrics
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = load(a.trajectory)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    cfg.seconds = 12.0
    balance = BalanceActor()
    balance.load_state_dict(
        load(Path("output/retained_go2_preparation_v3/balance.pt"))["network"]
    )
    rows = []
    best_score = -1e9
    best = None
    for handoff in [1.16, 1.2, 1.24, 1.28, 1.32, 1.36, 1.4]:
        e = Environment(assets, copy.deepcopy(cfg))
        e.capture_centers = data["centers"]
        e.restore(copy.deepcopy(saved["standing"]))
        obs, _ = e.observations()
        commands = []
        states = [e.state()["integration"]]
        trace = []
        stable = best_stable = 0.0
        for k in range(round(handoff / 0.02) + 310):
            u = (
                saved["standing_actions"][k]
                if k < round(handoff / 0.02)
                else actions(balance, obs[None], "balance")[0]
            )
            commands.append(u)
            (obs, _), _, done, info = e.step(u)
            x = metrics(e, True)
            trace.append(x)
            states.append(e.state()["integration"])
            valid = (
                x["safe"]
                and x["both_front"]
                and abs(x["pitch"] - np.pi / 2) < 0.2
                and abs(x["pitch_velocity"]) < 1
                and np.linalg.norm(x["com_velocity"]) < 0.6
                and x["rear_clearance"] > 0.08
            )
            stable = stable + 0.02 if valid else 0.0
            best_stable = max(best_stable, stable)
            if done:
                break
        row = dict(
            handoff=handoff,
            legal=bool(e._momentum_prefix_safe),
            stable_capture_seconds=best_stable,
            strict_hold_seconds=float(e.multi_best_hold),
            five_second_success=bool(e.ever_hold_success),
            certified_takeoff=bool(e.tracker.confirmed),
            termination=info["termination"],
            end_time=float(e.d.time),
            collision=bool(e.nonfoot_collision or e.self_collision),
        )
        rows.append(row)
        print(row, flush=True)
        score = 100 * e.multi_best_hold + best_stable
        if score > best_score:
            best_score = score
            best = dict(
                result=row,
                standing=saved["standing"],
                initial=saved["standing"],
                final=copy.deepcopy(e.state()),
                standing_actions=np.array(commands),
                standing_states=np.array(states),
                rows=trace,
            )
        torch.save(
            dict(result=row, actions=commands, final=e.state()),
            a.output / f"handoff_{handoff:.2f}.pt",
        )
    torch.save(best, a.output / "best.pt")
    (a.output / "comparison.json").write_text(json.dumps(rows, indent=2))
    if a.render:
        render_episode(
            e,
            best["standing_states"],
            a.output / "best.gif",
            "Offline balance takeover trial",
        )


if __name__ == "__main__":
    main()
