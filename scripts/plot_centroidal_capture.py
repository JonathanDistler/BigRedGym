"""Plot saved capture/diagnostic measurements without additional simulation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.unified_go2.core import load
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    saved = load(a.trajectory)
    rows = saved["rows"]
    t = np.array([r["time"] for r in rows])
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), layout="constrained")
    ax = axes.ravel()
    ax[0].plot(t, [r["pitch"] * 180 / np.pi for r in rows], label="pitch deg")
    ax[0].plot(t, [r["pitch_velocity"] for r in rows], label="pitch rate rad/s")
    for key, panel in [("omega", 1), ("H", 2), ("com", 3), ("com_velocity", 4)]:
        for i, c in enumerate("xyz"):
            ax[panel].plot(t, [r[key][i] for r in rows], label=key + " " + c)
    for i, leg in enumerate(["FL", "FR"]):
        ax[5].plot(t, [r["gaps"][i] for r in rows], label=leg + " gap m")
        for j, c in enumerate("xyz"):
            ax[6].plot(
                t, [r["foot_velocity"][i][j] for r in rows], label=leg + " v" + c
            )
    for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        ax[7].plot(t, [r["loads"][i] for r in rows], label=leg + " load N")
    for i in [1, 2, 4, 5]:
        ax[8].plot(t, [r["q"][i] for r in rows], label=str(i) + " angle")
        ax[8].plot(t, [r["target"][i] for r in rows], label=str(i) + " target", ls="--")
    for key in ["front_moment", "rear_moment"]:
        ax[9].plot(t, [r[key] for r in rows], label=key + " Nm")
    for key in ["torque_ratio", "speed_ratio", "capture_distance"]:
        ax[10].plot(t, [r[key] for r in rows], label=key)
    for group in ["body", "thigh", "calf"]:
        ax[11].plot(
            t, [r["clearances"][group] for r in rows], label=group + " clearance"
        )
    for panel in ax:
        panel.legend(fontsize=7)
        panel.grid(alpha=0.3)
        panel.set_xlabel("time s")
    fig.savefig(a.output, dpi=130)


if __name__ == "__main__":
    main()
