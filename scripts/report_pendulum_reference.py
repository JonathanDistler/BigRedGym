"""Offline pendulum prediction, contact, and robust-reference evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, path):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + extension), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    r = a.run
    out = r / "figures"
    out.mkdir(exist_ok=True)
    errors = pd.read_csv(r / "model_errors.csv")
    errors.groupby(["model", "kind"]).mean(numeric_only=True).to_csv(
        out / "model_summary.csv"
    )
    changed = r / "changed_action_model_errors.csv"
    if changed.exists():
        pd.read_csv(changed).groupby(["model", "kind"]).mean(numeric_only=True).to_csv(
            out / "changed_action_model_summary.csv"
        )
    f = pd.read_csv(r / "model_forecasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, d in f.groupby("model"):
        time = d.time - d.start_time
        grouped = (
            pd.DataFrame(
                dict(
                    time=np.round(time, 5),
                    theta_error=np.rad2deg(d.theta_predicted - d.theta_measured),
                    omega_error=d.omega_predicted - d.omega_measured,
                )
            )
            .groupby("time")
            .agg(["mean", "std"])
        )
        for ax, key in zip(axes, ("theta_error", "omega_error")):
            ax.plot(grouped.index, grouped[key]["mean"], label=name)
            ax.fill_between(
                grouped.index,
                grouped[key]["mean"] - grouped[key]["std"],
                grouped[key]["mean"] + grouped[key]["std"],
                alpha=0.12,
            )
    axes[0].set(xlabel="Forecast horizon (s)", ylabel="COM angle error (degrees)")
    axes[1].set(xlabel="Forecast horizon (s)", ylabel="COM angular-rate error (rad/s)")
    axes[0].legend(fontsize=8)
    save(fig, out / "01_heldout_model_forecasts")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (seed, name) in zip(
        axes.flat,
        [
            (42, "fixed"),
            (42, "configuration_inertia"),
            (43, "fixed"),
            (43, "configuration_inertia"),
        ],
    ):
        d = f[(f.model == name) & f.rollout.str.startswith(f"seed{seed}_")]
        if d.empty:
            ax.text(
                0.5,
                0.5,
                "No continuous eligible 100 ms interval",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set(
                title=f"seed {seed}, {name}",
                xlabel="Time (s)",
                ylabel="COM angle (degrees)",
            )
            continue
        start = d.start_time.min()
        d = d[d.start_time == start]
        # Multiple resets can share a start; show one measured input sequence.
        d = d[d.rollout == d.rollout.iloc[0]]
        ax.plot(
            d.time - d.start_time,
            np.rad2deg(d.theta_measured),
            label="Measured COM angle",
        )
        ax.plot(
            d.time - d.start_time,
            np.rad2deg(d.theta_predicted),
            "--",
            label="Initial state + commands forecast",
        )
        ax.set(
            title=f"seed {seed}, {name}",
            xlabel="Time (s)",
            ylabel="COM angle from upright (degrees)",
        )
        ax.legend(fontsize=7)
    save(fig, out / "02_measured_vs_model")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, seed in enumerate((42, 43)):
        d = pd.read_csv(r / f"seed{seed}_reset65010_model.csv")
        d = d[d.takeoff & ~d.nonfoot_collision]
        axes[row, 0].plot(d.time, np.rad2deg(d.pitch), label="Body pitch")
        axes[row, 0].plot(d.time, np.rad2deg(d.theta), label="COM angle from upright")
        axes[row, 1].plot(d.time, d.length, label="COM to material-foot pivot length")
        axes[row, 1].plot(d.time, d.inertia_com_y, label="COM pitch inertia (kg m²)")
        axes[row, 2].plot(
            d.time,
            d.torque_identified_configuration_inertia,
            label="Inverse-identified effective torque",
        )
        axes[row, 2].plot(
            d.time,
            d.physical_effective_torque,
            "--",
            label="Contact pivot moment minus shape-momentum rate",
        )
        for ax in axes[row]:
            ax.set(xlabel="Time (s)", title=f"seed {seed}")
            ax.legend(fontsize=7)
    save(fig, out / "03_angles_length_and_effective_torque")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, seed in enumerate((42, 43)):
        ref = pd.read_csv(r / f"reference_seed{seed}.csv")
        candidate = pd.read_csv(r / f"seed{seed}_candidate0_reset65100_trajectory.csv")
        baseline = pd.read_csv(r / f"baseline_seed{seed}_reset65100_trajectory.csv")
        for label, d in [("Candidate", candidate), ("Original policy", baseline)]:
            t = d.time - d.time.iloc[0]
            axes[row, 0].plot(t, np.rad2deg(d.pitch), label=label)
            axes[row, 1].plot(t, d.pitch_velocity, label=label)
            axes[row, 2].plot(np.rad2deg(d.pitch), d.pitch_velocity, label=label)
        axes[row, 0].plot(
            ref.time, np.rad2deg(ref.pitch), "--", label="Proposed reference"
        )
        axes[row, 1].plot(
            ref.time, ref.pitch_velocity, "--", label="Proposed reference"
        )
        axes[row, 0].set(
            xlabel="Time since valid start (s)", ylabel="Body pitch (degrees)"
        )
        axes[row, 1].set(xlabel="Time (s)", ylabel="Body pitch velocity (rad/s)")
        axes[row, 2].set(
            xlabel="Body pitch (degrees)", ylabel="Body pitch velocity (rad/s)"
        )
        for ax in axes[row]:
            ax.set_title(f"seed {seed}")
            ax.legend(fontsize=7)
    save(fig, out / "04_reference_tracking_and_phase_portrait")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, seed in enumerate((42, 43)):
        d = pd.read_csv(r / f"seed{seed}_candidate0_reset65100_trajectory.csv")
        t = d.time - d.time.iloc[0]
        axes[row, 0].plot(t, d.front_moment_y, label="Front moment about COM")
        axes[row, 0].plot(t, d.rear_moment_y, label="Rear moment about COM")
        axes[row, 1].plot(t, d.FL_load, label="FL ground reaction")
        axes[row, 1].plot(t, d.FR_load, label="FR ground reaction")
        axes[row, 2].plot(
            t, np.hypot(d.FL_velocity_x, d.FL_velocity_y), label="FL slip speed"
        )
        axes[row, 2].plot(
            t, np.hypot(d.FR_velocity_x, d.FR_velocity_y), label="FR slip speed"
        )
        for ax in axes[row]:
            ax.set(xlabel="Time (s)", title=f"seed {seed}")
            ax.legend(fontsize=7)
        axes[row, 0].set_ylabel("N m")
        axes[row, 1].set_ylabel("N")
        axes[row, 2].set_ylabel("m/s")
    save(fig, out / "05_contacts_and_slip")
    cases = pd.read_csv(r / "robustness_cases.csv")
    cases["peak_degrees"] = np.rad2deg(cases.peak_pitch)
    cases.groupby(["training_seed", "rank"]).agg(
        success_rate=("success", "mean"),
        mean_peak=("peak_degrees", "mean"),
        max_peak=("peak_degrees", "max"),
        mean_hold=("hold_seconds", "mean"),
        max_hold=("hold_seconds", "max"),
        unsafe_rate=("unsafe", "mean"),
    ).to_csv(out / "robustness_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for (seed, rank), d in cases.groupby(["training_seed", "rank"]):
        label = f"seed {seed}, rank {rank}"
        axes[0].scatter(d.reset, d.peak_degrees, label=label, s=15)
        axes[1].scatter(d.reset, d.hold_seconds, label=label, s=15)
        axes[2].scatter(d.reset, d.success.astype(float), label=label, s=15)
    axes[0].axhline(45, ls="--", c="gray")
    axes[1].axhline(0.3, ls="--", c="gray")
    for ax, y in zip(
        axes,
        ("Peak pitch (degrees)", "Low-speed in-band hold (s)", "All-criteria success"),
    ):
        ax.set(xlabel="Nearby-reset seed", ylabel=y)
    axes[0].legend(fontsize=6)
    save(fig, out / "06_all_robustness_cases")
    search = pd.read_csv(r / "search_cases.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for seed, d in search.groupby("training_seed"):
        g = d.groupby("generation")[["peak_pitch", "hold_seconds"]].max()
        axes[0].plot(
            g.index, np.rad2deg(g.peak_pitch), marker="o", label=f"seed {seed}"
        )
        axes[1].plot(g.index, g.hold_seconds, marker="o", label=f"seed {seed}")
    axes[0].axhline(45, ls="--", c="gray")
    axes[1].axhline(0.3, ls="--", c="gray")
    axes[0].set(
        xlabel="Offline CEM generation",
        ylabel="Maximum pitch, including invalid cases (degrees)",
    )
    axes[1].set(xlabel="Offline CEM generation", ylabel="Maximum low-speed hold (s)")
    axes[0].legend()
    save(fig, out / "07_search_peaks_are_not_success")
    feasibility = []
    for seed in (42, 43):
        ref = pd.read_csv(r / f"reference_seed{seed}.csv")
        model = json.loads((r / f"model_seed{seed}.json").read_text())
        for name, limits in model["torque_envelope"].items():
            torque = ref[f"torque_{name}"]
            feasibility.append(
                dict(
                    seed=seed,
                    model=name,
                    required_min=float(torque.min()),
                    required_max=float(torque.max()),
                    identified_low=limits[0],
                    identified_high=limits[1],
                    outside_fraction=float(
                        ((torque < limits[0]) | (torque > limits[1])).mean()
                    ),
                )
            )
    pd.DataFrame(feasibility).to_csv(out / "reference_torque_envelope.csv", index=False)


if __name__ == "__main__":
    main()
