"""Exact standing replay with 500 Hz kinematics and solver-step external wrench."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup
from scripts.diagnose_centroidal_capture import used_wrench
from experiments.unified_go2.core import Environment, load
from experiments.centroidal_capture.core import metrics
import mujoco
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    saved = load(a.trajectory)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    e.restore(copy.deepcopy(saved["standing"]))
    rows = []
    pending = {}
    orig_step = mujoco.mj_step
    orig_measure = e.measure

    def step(m, d):
        orig_step(m, d)
        pending["moment"], pending["force"] = used_wrench(e)

    def measure():
        row = orig_measure()
        if pending:
            x = metrics(e, True)
            x["used_moment"] = pending["moment"].tolist()
            x["used_force"] = pending["force"].tolist()
            x["qpos"] = e.d.qpos.copy()
            x["qvel"] = e.d.qvel.copy()
            x["collision_pairs"] = [
                (
                    e.m.body(int(e.m.geom_bodyid[c.geom1])).name,
                    e.m.body(int(e.m.geom_bodyid[c.geom2])).name,
                )
                for c in e.d.contact
                if c.dist < 0
                and (
                    e.m.geom_bodyid[c.geom1] not in [0, *e.feet]
                    or e.m.geom_bodyid[c.geom2] not in [0, *e.feet]
                )
            ]
            rows.append(x)
            pending.clear()
        return row

    mujoco.mj_step = step
    e.measure = measure
    try:
        for u in saved["standing_actions"]:
            _, _, done, info = e.step(u)
            if done:
                break
    finally:
        mujoco.mj_step = orig_step
        del e.measure
    torch.save(dict(rows=rows, final=e.state()), a.output / "audit.pt")
    # Report support intervals and impact velocities without using pitch alone.
    first = next(
        (i for i, x in enumerate(rows) if x["time"] > 0.9 and x["both_front"]), None
    )
    longest = run = 0.0
    strict = e.multi_best_hold
    for x in rows:
        valid = (
            x["safe"]
            and x["both_front"]
            and x["orientation_error"] < 0.2
            and np.linalg.norm(x["com_velocity"]) < 0.6
            and abs(x["pitch_velocity"]) < 1
            and x["rear_clearance"] > 0.08
        )
        run = run + e.m.opt.timestep if valid else 0.0
        longest = max(longest, run)
    summary = dict(
        exact_state_error=float(
            np.max(abs(e.state()["integration"] - saved["final"]["integration"]))
        ),
        first_both_front=rows[first]["time"] if first else None,
        first_contact_pitch=float(np.rad2deg(rows[first]["pitch"])) if first else None,
        pre_contact_foot_velocity=rows[first - 1]["foot_velocity"][:2]
        if first
        else None,
        loose_stability_seconds=longest,
        strict_hold_seconds=float(strict),
        five_second_success=bool(e.ever_hold_success),
        termination=info["termination"],
        terminal_time=float(e.d.time),
        terminal_collision_pairs=rows[-1]["collision_pairs"],
    )
    if first:
        end = next((i for i, x in enumerate(rows) if x["time"] >= 1.32), len(rows) - 1)
        summary["contact_braking_impulse_to_132"] = np.sum(
            [
                np.array(x["used_moment"]) * e.m.opt.timestep
                for x in rows[first : end + 1]
            ],
            axis=0,
        ).tolist()
        summary["contact_delta_H_to_132"] = (
            np.array(rows[end]["H"]) - np.array(rows[first - 1]["H"])
        ).tolist()
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
