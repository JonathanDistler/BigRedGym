"""Summarize frozen-actor smoothing ablations and verify reward-only equivalence."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    root = a.directory
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    instant = pd.read_csv(root / "instant/trace_frozen.csv")
    for label in ["instant", "reward_ema", "action_ema", "both"]:
        d = pd.read_csv(root / label / "trace_frozen.csv")
        e = pd.read_csv(root / label / "evaluation_frozen.csv")
        err = (
            d[[f"requested_torque_{i}" for i in range(12)]].values
            - d[[f"torque_{i}" for i in range(12)]].values
        )
        rows.append(
            dict(
                variant=label,
                takeoff_rate=e.takeoff.mean(),
                pitch_degrees=np.rad2deg(e.genuine_flight_pitch.mean()),
                max_target_reward=d.r_target.max(),
                requested_actual_torque_rms=float(np.sqrt(np.mean(err**2))),
                positive_work=float(
                    d.groupby("seed").positive_actuator_work.last().mean()
                ),
            )
        )
        g = d[d.seed == 91000]
        for ax, key in zip(axes.flat, ["pitch", "r_target", "target_8", "torque_8"]):
            ax.plot(g.time, g[key], label=label)
            ax.set(xlabel="time (s)", ylabel=key)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        if label == "reward_ema":
            if not np.array_equal(
                d[[f"joint_{i}" for i in range(12)]].values,
                instant[[f"joint_{i}" for i in range(12)]].values,
            ):
                raise ValueError("Reward-only smoothing changed frozen-actor physics")
    table = pd.DataFrame(rows)
    table.to_csv(root / "summary.csv", index=False)
    fig.savefig(root / "smoothing_ablation.png", dpi=150)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
