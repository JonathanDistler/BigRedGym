"""Plot matched-seed pilot trajectories against the zero-residual baseline."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--pilot-dir", type=Path, required=True)
    p.add_argument("--zero-dir", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.pilot_dir / "summary.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    sources = [("zero", args.zero_dir, "black")] + [
        (k, args.pilot_dir / f"iteration_{k}", color)
        for k, color in [
            ("0", "tab:gray"),
            ("50", "tab:blue"),
            ("100", "tab:orange"),
            ("200", "tab:green"),
        ]
        if k in summary
    ]
    for label, root, color in sources:
        data = [
            list(csv.DictReader((root / f"trial_{seed}.csv").open()))
            for seed in range(3000, 3016)
        ]
        t = np.array([float(x["time"]) for x in data[0]])

        def field(key):
            return np.array([[float(row[key]) for row in episode] for episode in data])

        clear = field("clearance_min") * 1000
        axes[0, 0].plot(t, clear.mean(0), color=color, label=label)
        axes[0, 1].plot(t, field("front_drift").mean(0) * 100, color=color)
        axes[1, 0].plot(t, np.rad2deg(field("pitch_rad")).mean(0), color=color)
        axes[1, 1].plot(t, field("pitch_rate").mean(0), color=color)
        residual = np.stack([abs(field(f"residual_{j}")) for j in range(6, 12)]).mean(
            (0, 1)
        )
        axes[2, 0].plot(t, residual, color=color)
        torque = np.stack([abs(field(f"torque_{j}")) for j in (8, 11)]).mean((0, 1))
        axes[2, 1].plot(t, torque, color=color)
    for ax, title in zip(
        axes.flat,
        [
            "Mean minimum rear clearance (mm)",
            "Mean maximum front drift (cm)",
            "Mean pitch angle (degrees)",
            "Mean pitch angular velocity (rad/s)",
            "Mean absolute rear residual (rad)",
            "Mean absolute rear knee torque (N m)",
        ],
    ):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvspan(0.777, 0.978, color="gray", alpha=0.12)
    axes[0, 0].axhline(30, color="black", linestyle=":")
    axes[0, 1].axhline(4, color="black", linestyle=":")
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time (s); shaded = rear push")
    fig.tight_layout()
    fig.savefig(args.pilot_dir / "comparison.png", dpi=160)
    rows = [
        "| Checkpoint | Takeoff /16 | Mean peak min clearance (mm) "
        "| Mean max front drift (cm) | Mean peak pitch (deg) "
        "| Mean terminal pitch (deg) | Rear push residual (rad) | Curriculum /16 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, r in summary.items():
        rows.append(
            f"| {key} | {r['valid_takeoff']} | {r['mean_peak_min_clearance_mm']:.2f} "
            f"| {r['mean_max_front_drift_cm']:.2f} "
            f"| {np.rad2deg(r['mean_peak_pitch_rad']):.2f} "
            f"| {np.rad2deg(r['mean_terminal_pitch_rad']):.2f} "
            f"| {r['mean_push_abs_rear_residual_rad']:.5f} "
            f"| {r['curriculum_success']} |"
        )
    (args.pilot_dir / "comparison.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
