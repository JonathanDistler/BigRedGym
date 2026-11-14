"""Cold standing-start replay of a saved capture trial; no intermediate restore."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, render_episode, actions
from experiments.balance_feedback.actor import BalanceActor
from experiments.unified_go2.core import Environment, load
from experiments.centroidal_capture.core import metrics
import torch
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--continue-balance", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    saved = load(a.trajectory)
    e.restore(copy.deepcopy(saved["standing"]))
    states = [e.state()["integration"]]
    rows = []
    for u in saved["standing_actions"]:
        _, _, done, info = e.step(u)
        states.append(e.state()["integration"])
        rows.append(metrics(e, True))
        if done:
            break
    error = float(np.max(abs(e.state()["integration"] - saved["final"]["integration"])))
    if a.continue_balance and not done:
        balance = BalanceActor()
        balance.load_state_dict(
            load(Path("output/retained_go2_preparation_v3/balance.pt"))["network"]
        )
        obs, _ = e.observations()
        commands = list(saved["standing_actions"])
        for _ in range(310):
            u = actions(balance, obs[None], "balance")[0]
            commands.append(u)
            (obs, _), _, done, info = e.step(u)
            states.append(e.state()["integration"])
            rows.append(metrics(e, True))
            if done:
                break
    else:
        commands = list(saved["standing_actions"])
    result = dict(
        final_state_error=error,
        certified_takeoff=bool(e.tracker.confirmed),
        five_second_success=bool(e.ever_hold_success),
        strict_hold_seconds=float(e.multi_best_hold),
        termination=info["termination"],
        final_time=float(e.d.time),
    )
    torch.save(
        dict(
            rows=rows,
            states=states,
            standing_states=np.array(states),
            standing_actions=np.array(commands),
            standing=saved["standing"],
            final=e.state(),
        ),
        a.output / "replay.pt",
    )
    (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(result)
    if a.render:
        render_episode(e, states, a.output / "replay.gif", "Cold standing replay")
    if error > 1e-10:
        raise RuntimeError("Replay diverged")


if __name__ == "__main__":
    main()
