"""Summarize independent trials and plot one matched baseline/reference pair."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    validation = json.loads((root / "validation.json").read_text())
    summary = {}
    for label, results in validation.items():

        def values(key):
            return np.asarray([r[key] for r in results])

        summary[label] = dict(
            trials=len(results),
            success_10mm=int(values("success_10mm").sum()),
            success_30mm=int(values("success_30mm").sum()),
            peak_clearance_mm_range=(
                values("peak_clearance_m")[
                    [
                        np.argmin(values("peak_clearance_m")),
                        np.argmax(values("peak_clearance_m")),
                    ]
                ]
                * 1000
            ).tolist(),
            mean_supported_peak_clearance_mm=float(
                values("supported_peak_clearance_m").mean() * 1000
            ),
            mean_max_front_drift_cm=float(values("max_front_drift_m").mean() * 100),
            max_front_drift_cm=float(values("max_front_drift_m").max() * 100),
            min_verified_30mm_hold_s=float(values("verified_30mm_hold_s").min()),
            mean_peak_pitch_rate_rad_s=float(values("peak_pitch_rate_rad_s").mean()),
            mean_peak_pitch_momentum_Nms=float(
                values("peak_pitch_momentum_Nms").mean()
            ),
            mean_rear_push_impulse_Ns=values("rear_push_impulse_Ns").mean(0).tolist(),
            mean_rear_excess_push_impulse_Ns=values("rear_excess_push_impulse_Ns")
            .mean(0)
            .tolist(),
            front_contact_retention=float(values("front_contact_retention").mean()),
            max_front_contact_loss_s=float(values("front_contact_loss_max_s").max()),
            body_contact_trials=int(values("body_contact").sum()),
            max_torque_Nm=values("peak_torque_Nm").max(0).tolist(),
            max_saturation_fraction=values("saturation_fraction").max(0).tolist(),
            max_velocity_limit_ratio=float(values("peak_velocity_limit_ratio").max()),
            mean_push_tracking_rmse_rad=values("push_tracking_rmse_rad")
            .mean(0)
            .tolist(),
        )
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    events = []
    for result in validation["optimized"]:
        with (root / f"optimized_{result['seed']}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for i, row in enumerate(rows[1:], 1):
            if (
                min(float(row["clearance_2"]), float(row["clearance_3"])) >= 0.01
                and row["front_contact"] == "True"
                and max(float(row["force_2"]), float(row["force_3"]))
                < 0.015 * result["body_weight_N"]
            ):
                dt = float(row["time"]) - float(rows[i - 1]["time"])
                events.append(
                    dict(
                        seed=result["seed"],
                        time_s=float(row["time"]),
                        rear_upward_velocity_m_s=[
                            (
                                float(row[f"clearance_{j}"])
                                - float(rows[i - 1][f"clearance_{j}"])
                            )
                            / dt
                            for j in (2, 3)
                        ],
                    )
                )
                break
    (root / "liftoff_events.json").write_text(json.dumps(events, indent=2))
    seed = validation["optimized"][0]["seed"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True)
    for label, color in [("baseline", "tab:blue"), ("optimized", "tab:green")]:
        rows = list(csv.DictReader((root / f"{label}_{seed}.csv").open()))

        def a(key):
            return np.array([float(r[key]) for r in rows])

        t = a("time")
        axes[0, 0].plot(t, 1000 * a("clearance_min"), color=color, label=label)
        axes[0, 1].plot(t, 100 * a("front_drift"), color=color)
        axes[1, 0].plot(t, a("pitch_rate"), color=color)
        axes[1, 1].plot(t, a("pitch_momentum"), color=color)
        axes[2, 0].plot(t, a("com_x"), color=color, label=label + " COM")
        axes[2, 0].plot(
            t, a("cop_x"), color=color, linestyle="--", label=label + " COP"
        )
        axes[2, 1].plot(t, a("force_2"), color=color, label=label + " RL")
        axes[2, 1].plot(
            t, a("force_3"), color=color, linestyle="--", label=label + " RR"
        )
        axes[3, 0].plot(t, a("desired_8"), color=color, label=label + " desired")
        axes[3, 0].plot(
            t, a("actual_8"), color=color, linestyle="--", label=label + " actual"
        )
        axes[3, 1].plot(t, a("torque_8"), color=color, label=label + " RL")
        axes[3, 1].plot(
            t, a("torque_11"), color=color, linestyle="--", label=label + " RR"
        )
    for ax, title in zip(
        axes.flat,
        [
            "Minimum rear clearance (mm)",
            "Maximum front drift (cm)",
            "Pitch angular velocity (rad/s)",
            "Pitch angular momentum about COM (N m s)",
            "World COM / COP x (m)",
            "Rear normal forces (N)",
            "Rear-left knee tracking (rad)",
            "Rear knee torques (N m)",
        ],
    ):
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.axvline(0.777, color="gray", alpha=0.4)
        ax.axvline(0.978, color="gray", alpha=0.4)
    axes[0, 0].axhline(30, color="black", linestyle=":", label="30 mm")
    axes[0, 1].axhline(4, color="black", linestyle=":")
    for ax in [axes[0, 0], axes[2, 0], axes[3, 0]]:
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(
        f"Matched MuJoCo standing start {seed}; gray lines = optimized push window"
    )
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
