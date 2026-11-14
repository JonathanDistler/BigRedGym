"""Kinematic contact options only: never used as a trajectory or reset."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup
from experiments.unified_go2.core import Environment, load
from experiments.morphology.dynamics_audit import ground_clearance
import mujoco
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--diagnosis", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.diagnosis
    )
    e = Environment(assets, cfg)
    records = []
    for label in ["61", "83"]:
        saved = load(a.diagnosis / (label + "_balance_dt1.pt"))
        for time in [0.8, 0.9, 1.0, 1.1, 1.14]:
            state = saved["states"][round(time / 0.02) - 1]
            for leg, foot in [("FL", e.foot_geoms[0]), ("FR", e.foot_geoms[1])]:
                d = mujoco.MjData(e.m)
                mujoco.mj_setState(e.m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
                mujoco.mj_forward(e.m, d)
                original = d.qpos.copy()
                indices = [
                    i
                    for i, n in enumerate(e.names)
                    if n.startswith(leg) and ("thigh" in n or "calf" in n)
                ]
                qs = e.qids[indices]
                limits = e.bounds[indices]
                speed = e.speed[indices]
                nonfeet = [
                    g
                    for g in range(e.m.ngeom)
                    if e.m.body(int(e.m.geom_bodyid[g])).name.startswith(leg)
                    and e.m.geom_bodyid[g] != e.feet[0 if leg == "FL" else 1]
                    and (e.m.geom_contype[g] or e.m.geom_conaffinity[g])
                ]
                best = None
                lowest = 1e9
                for q1 in np.linspace(*limits[0], 41):
                    for q2 in np.linspace(*limits[1], 41):
                        d.qpos[qs] = [q1, q2]
                        mujoco.mj_kinematics(e.m, d)
                        if min(ground_clearance(e.m, d, g) for g in nonfeet) < 0.002:
                            continue
                        gap = d.geom_xpos[foot, 2] - 0.022
                        lowest = min(lowest, gap)
                        if -0.003 <= gap <= 0.006:
                            travel = float(
                                np.max(abs(d.qpos[qs] - original[qs]) / speed)
                            )
                            if (
                                best is None
                                or travel < best["minimum_speed_limited_time"]
                            ):
                                best = dict(
                                    minimum_speed_limited_time=travel,
                                    q=[q1, q2],
                                    gap=float(gap),
                                    foot=d.geom_xpos[foot].tolist(),
                                )
                records.append(
                    dict(
                        seed=label,
                        time=time,
                        leg=leg,
                        actual_q=original[qs].tolist(),
                        minimum_sampled_gap=float(lowest),
                        contact_option=best,
                    )
                )
    (a.diagnosis / "reachability.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
