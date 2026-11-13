"""Measured diagnostic figures and compact per-arm aggregates."""

import argparse
import json
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
    comparison = pd.read_csv(root / "comparison.csv")
    table = comparison.groupby(["arm", "iteration"]).agg(
        takeoff_rate=("takeoff", "mean"),
        peak_pitch_rad=("genuine_flight_pitch", "mean"),
        handstand_rate=("success", "mean"),
        collision_rate=("nonfoot_collision", "mean"),
        front_support=("front_support_fraction", "mean"),
    )
    table["peak_pitch_degrees"] = np.rad2deg(table.peak_pitch_rad)
    table.to_csv(root / "summary.csv")
    final = int(comparison.iteration.max())
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    keys = [
        "com_height",
        "potential",
        "kinetic",
        "momentum_y",
        "pitch_velocity",
        "r_target",
        "pendulum_length",
        "work",
        "orientation_error",
    ]
    extras = []
    for arm in "ABCD":
        paths = list(root.glob(f"{arm}_seed*/trace_{final:03}.csv"))
        for index, path in enumerate(paths):
            d = pd.read_csv(path)
            d = d[d.seed == 91000]
            for ax, key in zip(axes.flat, keys):
                ax.plot(
                    d.time,
                    d[key],
                    label=f"{arm} seed {path.parent.name.split('seed')[-1]}",
                )
                ax.set(xlabel="time (s)", ylabel=key)
                ax.grid(alpha=0.2)
            extras.append(
                dict(
                    arm=arm,
                    seed=path.parent.name,
                    peak_potential=d.potential.max(),
                    peak_total_energy=d.total_energy.max(),
                    peak_momentum=d.momentum_y.max(),
                    peak_joint_target_rate=float(
                        np.max(
                            abs(np.diff(d[[f"target_{i}" for i in range(12)]], axis=0))
                        )
                        / 0.02
                    ),
                    joint_target_acceleration_rms=float(
                        np.sqrt(
                            np.mean(
                                np.diff(
                                    d[[f"target_{i}" for i in range(12)]], n=2, axis=0
                                )
                                ** 2
                            )
                        )
                        / 0.02**2
                    ),
                    joint_tracking_rms=float(
                        np.sqrt(
                            np.mean(
                                (
                                    d[[f"target_{i}" for i in range(12)]].values
                                    - d[[f"joint_{i}" for i in range(12)]].values
                                )
                                ** 2
                            )
                        )
                    ),
                    final_actuator_work=d.work.iloc[-1],
                    peak_rear_clearance=d.rear_clearance.max(),
                    max_com_check=d.com_check.max(),
                    max_momentum_check=d.momentum_check.max(),
                )
            )
    axes[0, 0].legend(fontsize=7)
    fig.savefig(root / "physics_rewards.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(extras).to_csv(root / "physical_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for arm in "ABCD":
        path = sorted(root.glob(f"{arm}_seed*/trace_{final:03}.csv"))[0]
        d = pd.read_csv(path)
        d = d[d.seed == 91000]
        axes[0].plot(np.rad2deg(d.pitch), d.pitch_velocity, label=arm)
        axes[1].plot(d.time, d.target_8, label=f"{arm} target")
        axes[1].plot(d.time, d.joint_8, ls="--", alpha=0.6, label=f"{arm} actual")
        axes[2].plot(d.time, d.torque_8, label=arm)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].set(xlabel="body pitch (degrees)", ylabel="pitch velocity (rad/s)")
    axes[0].axvspan(30, 35, alpha=0.12, color="gray")
    axes[1].set(xlabel="time (s)", ylabel="RL calf (rad)")
    axes[2].set(xlabel="time (s)", ylabel="RL calf torque (Nm)")
    fig.savefig(root / "phase_portrait_actions.png", dpi=150)
    plt.close(fig)
    print(table.to_string())
    print(json.loads((root / "reference.json").read_text())["height"])


if __name__ == "__main__":
    main()
