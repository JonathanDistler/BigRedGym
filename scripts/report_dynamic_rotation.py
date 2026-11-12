"""Concise pass-through evidence, separate support and terminal outcomes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "figures"
    out.mkdir(exist_ok=True)
    old = pd.read_csv(r / "historical_reassessment.csv")
    old.to_csv(out / "historical_metrics.csv", index=False)
    d = pd.read_csv(r / "frontier/validation_cases.csv")
    d["pitch_degrees"] = np.rad2deg(d.peak_pitch)
    d["native_valid"] = (
        ~d.nonfoot_collision
        & ~d.self_collision
        & (d.max_speed_ratio <= 1.001)
        & (d.max_torque_ratio <= 1.00001)
        & (d.max_joint_violation < 0.001)
    )
    d.to_csv(out / "frontier_per_reset.csv", index=False)
    d.groupby(["seed", "label"])[
        [
            "cross45",
            "cross60",
            "pitch_degrees",
            "omega45",
            "native_valid",
            "full_valid",
            "improved",
            "terminal_recovery_60",
            "contact_gap",
            "front_drift",
        ]
    ].agg(["mean", "std", "min", "max"]).to_csv(out / "frontier_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for name, path in [
        (
            "Earlier 45.19 degree sequence",
            r / "old_A_seed42_scale1.00_reset61000_trajectory.csv",
        ),
        (
            "New native-valid progression",
            r / "frontier/seed42_dynamic0_reset67203_trajectory.csv",
        ),
        (
            "New collision-limited progression",
            r / "frontier/seed42_dynamic1_reset67200_trajectory.csv",
        ),
    ]:
        f = pd.read_csv(path)
        t = f.time - f.time.iloc[0]
        axes[0, 0].plot(t, np.rad2deg(f.pitch), label=name)
        axes[0, 1].plot(t, f.pitch_velocity, label=name)
        axes[0, 2].plot(np.rad2deg(f.pitch), f.pitch_velocity, label=name)
        support = f.support_valid
        axes[1, 0].plot(t, support.astype(float), label=name)
        axes[1, 1].plot(t, f.front_drift, label=name)
        axes[1, 2].plot(t, f.front_moment_y, label=name)
    axes[0, 0].axhline(45, ls="--", c="gray")
    axes[0, 0].axhline(60, ls=":", c="gray")
    for ax, y in zip(
        axes.flat,
        [
            "Pitch (degrees)",
            "Pitch velocity (rad/s)",
            "Pitch velocity (rad/s)",
            "Both front feet loaded",
            "Front-foot displacement (m)",
            "Front moment about COM (N m)",
        ],
    ):
        ax.set(ylabel=y, xlabel="Time since saved start (s)")
    axes[0, 2].set_xlabel("Pitch (degrees)")
    axes[0, 0].legend(fontsize=6)
    save(fig, out / "dynamic_progression_and_support")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    chosen = d[(d.seed == 42) & (d.label == 0)]
    axes[0].plot(chosen.reset, chosen.pitch_degrees, "o-")
    axes[0].axhline(45, ls="--", c="gray")
    axes[0].axhline(60, ls=":", c="gray")
    axes[1].plot(chosen.reset, chosen.omega45, "o-", label="Body pitch rate")
    axes[1].plot(
        chosen.reset, chosen.at45_com_omega, "o-", label="Relative COM angular rate"
    )
    axes[1].plot(
        chosen.reset,
        chosen.at45_required_com_omega,
        "--",
        label="Conditional COM coast requirement",
    )
    axes[2].plot(
        chosen.reset, chosen.contact_gap, "o-", label="Longest front-support gap (s)"
    )
    axes[2].plot(chosen.reset, chosen.front_drift, "o-", label="Front displacement (m)")
    axes[0].set(ylabel="Peak pitch (degrees)", xlabel="Fresh reset")
    axes[1].set(ylabel="Rate at 45 degrees (rad/s)", xlabel="Fresh reset")
    axes[2].set(xlabel="Fresh reset", ylabel="Support-loss/displacement measures")
    axes[1].legend(fontsize=6)
    axes[2].legend(fontsize=6)
    save(fig, out / "fresh_reset_rotation_and_recovery")
    s = pd.read_csv(r / "search_cases.csv")
    s["peak_degrees"] = np.rad2deg(s.valid_prefix_peak)
    fig, ax = plt.subplots(figsize=(8, 4))
    for seed, g in s.groupby("seed"):
        summary = g.groupby("generation").peak_degrees.max()
        ax.plot(summary.index, summary.values, marker="o", label=f"seed {seed}")
    ax.axhline(45, ls="--", c="gray")
    ax.axhline(60, ls=":", c="gray")
    ax.set(
        xlabel="Offline CEM generation",
        ylabel="Maximum supported valid-prefix pitch (degrees)",
    )
    ax.legend()
    save(fig, out / "search_progress_not_terminal_success")


if __name__ == "__main__":
    main()
