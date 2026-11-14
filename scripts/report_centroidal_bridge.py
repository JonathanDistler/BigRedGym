"""Replay saved commands, audit at every physics step, plot physical accounting."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup
from experiments.unified_go2.core import Environment, load
from experiments.centroidal_bridge.physics import audit
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    starts = load(a.run / "starts.pt")
    cfg, data, bank, assets = setup(starts["args"]["preparation"], a.run)
    summaries = {}
    for label in ["baseline", "best"]:
        saved = load(a.run / (label + ".pt"))
        e = Environment(assets, copy.deepcopy(cfg))
        e.capture_centers = data["centers"]
        e.restore(copy.deepcopy(starts["standing"]))
        audits = []
        original = e.measure

        def measure():
            row = original()
            x = audit(e)
            x["pitch"] = e.momentum_sample["pitch"]
            x["pitch_velocity"] = e.momentum_sample["pitch_velocity"]
            x["safe"] = bool(e._momentum_prefix_safe)
            audits.append(x)
            return row

        e.measure = measure
        # Continuous standing start, replay the unchanged prefix then planned suffix.
        for u in list(starts["prefix_actions"]) + list(saved["actions"]):
            e.step(u)
        del e.measure
        torch.save(
            dict(audits=audits, physics=e.physics_trace, final=e.state()),
            a.run / (label + "_audit.pt"),
        )
        t = np.array([x["time"] for x in audits])
        h = np.array([x["H"] for x in audits])
        moment = np.array([x["moment"] for x in audits])
        dt = np.diff(t)
        # Forward Euler matches the semi-implicit MuJoCo velocity integration.
        impulse = np.cumsum(moment[:-1] * dt[:, None], axis=0)
        residual = h[1:] - h[0] - impulse
        axis = e.pitch_axis
        groups = {}
        for leg in ["FL", "FR", "RL", "RR", "other"]:
            groups[leg] = np.array(
                [
                    sum(
                        (
                            np.array(c["moment"]) @ axis
                            for c in x["contacts"]
                            if (
                                c["body"].startswith(leg)
                                if leg != "other"
                                else not c["body"].startswith(("FL", "FR", "RL", "RR"))
                            )
                        ),
                        0.0,
                    )
                    for x in audits
                ]
            )
        front = groups["FL"] + groups["FR"]
        s = e.physics_trace
        ts = np.array([x["time"] for x in s])
        summary = dict(
            max_subtree_error=max(x["subtree_error"] for x in audits),
            delta_H=(h[-1] - h[0]).tolist(),
            external_impulse=impulse[-1].tolist(),
            residual=residual[-1].tolist(),
            max_residual_norm=float(np.linalg.norm(residual, axis=1).max()),
            front_positive_impulse=float(np.sum(np.maximum(front[:-1], 0) * dt)),
            front_negative_impulse=float(np.sum(np.minimum(front[:-1], 0) * dt)),
            standing_replay_state_error=float(
                np.max(np.abs(e.state()["integration"] - saved["final"]["integration"]))
            ),
            peak_safe_pitch=max(np.rad2deg(x["pitch"]) for x in audits if x["safe"]),
        )
        for deg in [50, 55, 60, 70, 80]:
            summary["safe_seconds_above_" + str(deg)] = float(
                sum(
                    dd
                    for x, dd in zip(audits[:-1], dt)
                    if x["safe"] and np.rad2deg(x["pitch"]) >= deg
                )
            )
        peak = max((x for x in audits if x["safe"]), key=lambda x: x["pitch"])
        summary["peak_time"] = peak["time"]
        summary["peak_H_pitch"] = float(np.array(peak["H"]) @ axis)
        summary["peak_front_moment"] = float(
            front[np.argmax([x["pitch"] if x["safe"] else -10 for x in audits])]
        )
        summaries[label] = summary
        fig, axes = plt.subplots(5, 2, figsize=(14, 17), layout="constrained")
        ax = axes.ravel()
        ax[0].plot(t, [np.rad2deg(x["pitch"]) for x in audits], label="pitch deg")
        ax[0].plot(t, [x["pitch_velocity"] for x in audits], label="pitch rate rad/s")
        ax[1].plot(t, h @ axis, label="H COM")
        ax[1].plot(t[1:], h[0] @ axis + impulse @ axis, label="H0 + external impulse")
        ax[2].plot(
            t[1:],
            np.cumsum(np.maximum(front[:-1], 0) * dt),
            label="positive front impulse",
        )
        ax[2].plot(
            t[1:],
            np.cumsum(np.minimum(front[:-1], 0) * dt),
            label="negative front impulse",
        )
        for leg in ["FL", "FR"]:
            ax[3].plot(t, groups[leg], label=leg + " moment")
        for leg in ["RL", "RR", "other"]:
            ax[4].plot(t, groups[leg], label=leg + " moment")
        for name, prefix in [
            ("trunk", None),
            ("rear", ("RL", "RR")),
            ("front", ("FL", "FR")),
        ]:
            values = [
                sum(
                    (
                        np.array(v) @ axis
                        for k, v in x["links"].items()
                        if (
                            not k.startswith(("FL", "FR", "RL", "RR"))
                            if prefix is None
                            else k.startswith(prefix)
                        )
                    ),
                    0.0,
                )
                for x in audits
            ]
            ax[5].plot(t, values, label=name + " H")
        for key in ["com_support_x", "com_height"]:
            ax[6].plot(ts, [x[key] for x in s], label=key)
        for key in ["mechanical_energy", "potential", "K_system"]:
            ax[7].plot(ts, [x[key] for x in s], label=key)
        for leg in ["FL", "FR"]:
            for joint in ["thigh", "calf"]:
                for suffix in ["q", "target"]:
                    key = f"{leg}_{joint}_joint_{suffix}"
                    ax[8].plot(ts, [x[key] for x in s], label=key)
        ax[9].plot(t[1:], residual @ axis, label="COM impulse residual")
        ax[9].plot(
            ts,
            [x["L_support"] - x["L_com"] for x in s],
            label="support translational contribution",
        )
        for aa in ax:
            aa.legend(fontsize=7)
            aa.set_xlabel("time s")
            aa.grid(alpha=0.3)
        fig.savefig(a.run / (label + "_physics.png"), dpi=140)
        plt.close(fig)
    (a.run / "accounting.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
