"""Measured morphology BC, teacher and old-PPO regression figures."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--bc", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--regression", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--noise", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.bc / "bc_evaluation.csv")
    metrics = [
        "takeoff",
        "max_pitch",
        "rear_clearance",
        "genuine_flight_pitch",
        "genuine_airborne_seconds",
        "certified_work_min",
        "front_support_fraction",
        "nonfoot_collision",
        "max_pitch_velocity",
    ]
    summary = d.groupby(["scale", "stochastic"])[metrics].agg(["mean", "std", "count"])
    summary.to_csv(a.output / "bc_summary.csv")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for ax, metric in zip(axes.flat, metrics):
        for sampled, g in d.groupby("stochastic"):
            stats = g.groupby("scale")[metric].mean()
            ax.plot(
                stats.index,
                stats,
                "o-",
                label="sampled" if sampled else "deterministic",
            )
        ax.set(xlabel="Physical leg scale", ylabel=metric, xticks=[0.9, 0.95, 1.0])
    axes[0, 0].legend()
    save(fig, a.output, "bc_behavior")
    losses = pd.read_csv(a.bc / "bc_loss.csv")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.semilogy(losses["update"], losses.train_mse, label="training")
    ax.semilogy(
        losses["update"], losses.validation_mse, label="held-out source episodes"
    )
    ax.set(xlabel="Additional supervised updates", ylabel="Bounded-action MSE")
    ax.legend()
    save(fig, a.output, "bc_loss")
    teacher = pd.read_csv(a.teacher / "teacher_episodes.csv")
    teacher.groupby("scale")[metrics + ["accepted"]].mean().to_csv(
        a.output / "teacher_summary.csv"
    )
    drift = pd.read_csv(a.regression / "distribution_drift.csv")
    drift.to_csv(a.output / "old_ppo_joint_drift.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for run, g in drift.groupby("run"):
        s = g.groupby("iteration")
        axes[0].plot(s.cumulative_gaussian_kl.sum(), label=run)
        axes[1].plot(s.bounded_mean_shift_rms.mean(), label=run)
        axes[2].plot(s.effective_latent_std.mean(), label=run)
    for ax, label in zip(
        axes,
        [
            "Cumulative Gaussian KL on fixed demo states",
            "Joint-mean bounded-action RMS drift",
            "Effective latent noise std",
        ],
    ):
        ax.set(xlabel="PPO update", ylabel=label)
    axes[0].legend(fontsize=7)
    save(fig, a.output, "old_ppo_distribution_drift")
    f = pd.read_csv(a.regression / "replay_physics.csv")
    tracking = []
    for keys, g in f[(f.time >= 0.4) & (f.time <= 0.8)].groupby(
        ["run", "iteration", "scale", "stochastic"]
    ):
        tracking.append(
            dict(zip(["run", "iteration", "scale", "stochastic"], keys))
            | dict(
                rear_knee_target_error=float(
                    np.sqrt(
                        np.mean(
                            np.r_[
                                (g.target_8 - g.joint_8) ** 2,
                                (g.target_11 - g.joint_11) ** 2,
                            ]
                        )
                    )
                ),
                rear_left_peak_load=float(g.load_RL.max()),
                rear_right_peak_load=float(g.load_RR.max()),
            )
        )
    pd.DataFrame(tracking).to_csv(a.output / "old_ppo_tracking.csv", index=False)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    for iteration in (0, 4, 8):
        for sampled in (False, True):
            g = f[
                (f.run == "B_seed42")
                & (f.scale == 1.0)
                & (f.iteration == iteration)
                & (f.stochastic == sampled)
            ]
            label = f"update {iteration} " + ("sampled" if sampled else "mean")
            col = int(sampled)
            axes[0, col].plot(g.time, g.joint_8, label=label)
            axes[0, col].plot(g.time, g.target_8, ls=":", alpha=0.5)
            axes[1, col].plot(g.time, g.load_RL, label=label)
            axes[2, col].plot(g.time, g.load_FR, label=label)
    for ax in axes.flat:
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=6)
    for col in (0, 1):
        axes[0, col].set_ylabel("Rear knee actual/target (rad)")
        axes[1, col].set_ylabel("Rear-left normal force (N)")
        axes[2, col].set_ylabel("Front-right normal force (N)")
    save(fig, a.output, "old_ppo_contact_tracking")
    if a.noise:
        noise = pd.read_csv(a.noise / "calibration.csv")
        noise.groupby(["scale", "latent_std"])[
            ["takeoff", "front_support_fraction", "genuine_flight_pitch"]
        ].agg(["mean", "std", "count"]).to_csv(a.output / "noise_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for scale, group in noise.groupby("scale"):
            grouped = group.groupby("latent_std")
            for ax, metric in zip(axes, ["takeoff", "front_support_fraction"]):
                ax.plot(grouped[metric].mean(), "o-", label=f"{scale:.0%}")
                ax.set(xlabel="Latent noise std", ylabel=metric, ylim=(-0.02, 1.02))
        axes[0].legend()
        save(fig, a.output, "frozen_actor_noise")


if __name__ == "__main__":
    main()
