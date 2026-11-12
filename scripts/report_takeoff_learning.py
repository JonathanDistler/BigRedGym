"""Measured takeoff reward/authority/BC validation figures, plus optional PPO data."""

import argparse
import json
from pathlib import Path
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, root, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(root / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--bc", type=Path)
    p.add_argument("--ppo", type=Path)
    p.add_argument("--exploration", type=Path)
    p.add_argument("--fresh-exploration", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(a.validation / "original_reward_components.csv")
    original.groupby("method").mean(numeric_only=True).to_csv(
        a.output / "v1_reward_summary.csv"
    )
    terms = [c for c in original if c.startswith("episode_reward_")]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    original.groupby("method")[terms].mean().T.rename(
        index=lambda x: x.removeprefix("episode_reward_")
    ).plot.bar(ax=ax)
    ax.set(
        ylabel="Mean full-scale evaluation episode reward component",
        xlabel="Original v1 reward term",
    )
    ax.tick_params(axis="x", rotation=30)
    save(fig, a.output, "01_original_reward_shortcut")
    audit = pd.read_csv(a.validation / "preserved_policy_audit.csv")
    audit.to_csv(a.output / "policy_audit.csv", index=False)
    physics = pd.read_csv(a.validation / "preserved_policy_physics.csv")
    f = physics[
        (physics["method"] == "B")
        & (physics.training_seed == 42)
        & (physics.scale == 1.0)
        & (physics.evaluation_seed == 1000)
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i, leg in ((8, "RL"), (11, "RR")):
        axes[0, 0].plot(f.time, f[f"joint_{i}"], label=f"{leg} actual")
        axes[0, 0].plot(f.time, f[f"target_{i}"], "--", label=f"{leg} target")
        axes[0, 1].plot(f.time, f[f"velocity_{i}"], label=leg)
        axes[1, 0].plot(f.time, f[f"load_{leg}"], label=leg)
        axes[1, 1].plot(f.time, f[f"torque_{i}"], label=leg)
    axes[1, 1].axhline(45.43, color="k", ls=":", label="Calf effort bound")
    axes[1, 1].axhline(-45.43, color="k", ls=":")
    axes[2, 0].plot(f.time, np.rad2deg(f.pitch), label="Body pitch")
    axes[2, 1].plot(f.time, f.rear_clearance, label="Minimum rear-foot clearance")
    for ax, label in zip(
        axes.flat,
        (
            "Rear knee position (rad)",
            "Rear knee velocity (rad/s)",
            "Rear normal force (N)",
            "Applied knee torque (Nm)",
            "Body pitch (deg)",
            "Rear-foot clearance (m)",
        ),
    ):
        ax.set(xlabel="Time (s)", ylabel=label)
        ax.legend(fontsize=7)
    save(fig, a.output, "02_trained_policy_motor_audit")
    authority = pd.read_csv(a.validation / "authority.csv")
    grouped = authority.groupby(["configuration", "scale"]).agg(
        takeoff_rate=("takeoff", "mean"),
        flight_pitch=("genuine_flight_pitch", "max"),
        rear_clearance=("rear_clearance", "max"),
        target_error=("max_target_error", "max"),
        projection=("projected_action_fraction", "mean"),
        torque=("max_torque_ratio", "max"),
        speed=("max_speed_ratio", "max"),
        joint_violation=("joint_limit_violation", "max"),
    )
    grouped.to_csv(a.output / "authority_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for scale, f in grouped.reset_index().groupby("scale"):
        f = f.set_index("configuration").loc[
            ["legacy", "wider_filtered", "direct_targets"]
        ]
        for ax, key, label in zip(
            axes,
            ("takeoff_rate", "flight_pitch", "target_error"),
            (
                "Reference-probe takeoff rate",
                "Maximum genuine-flight pitch (rad)",
                "Maximum target representation error (rad)",
            ),
        ):
            ax.plot(range(3), f[key], "o-", label=f"{scale:.2f}")
            ax.set(
                xticks=range(3), xticklabels=["legacy", "wider", "direct"], ylabel=label
            )
    axes[0].legend(title="Physical scale")
    save(fig, a.output, "03_action_authority")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    groups = [
        audit.episode_reward.to_numpy(),
        authority[authority.takeoff].episode_reward_new.to_numpy(),
        authority[~authority.takeoff].episode_reward_new.to_numpy(),
    ]
    ax.boxplot(
        groups,
        tick_labels=[
            "v1 actors: no takeoff\n8-second audits",
            "Certified reference takeoffs\nup to 1.8 seconds",
            "Failed reference attempts\nup to 1.8 seconds",
        ],
    )
    ax.set(ylabel="New takeoff reward return")
    save(fig, a.output, "04_takeoff_reward_separation")
    missing = []
    if a.bc:
        loss = pd.read_csv(a.bc / "bc_loss.csv")
        evaluation = pd.read_csv(
            a.validation / "bc_revalidation.csv"
            if (a.validation / "bc_revalidation.csv").exists()
            else a.bc / "bc_evaluation.csv"
        )
        loss.to_csv(a.output / "bc_loss.csv", index=False)
        evaluation.to_csv(a.output / "bc_evaluation.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        axes[0].semilogy(loss["update"], loss.train_mse, label="Training episodes")
        axes[0].semilogy(loss["update"], loss.validation_mse, label="Held-out episodes")
        axes[0].set(xlabel="Offline BC gradient updates", ylabel="Bounded action MSE")
        axes[0].legend()
        stats = evaluation.groupby("scale").agg(
            rate=("takeoff", "mean"),
            trials=("takeoff", "size"),
            successes=("takeoff", "sum"),
        )
        stats.to_csv(a.output / "bc_transfer_summary.csv")
        axes[1].plot(stats.index, stats.rate, "o-")
        for scale, r in stats.iterrows():
            axes[1].annotate(
                f"{int(r.successes)}/{int(r.trials)}",
                (scale, r.rate),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
            )
        axes[1].set(
            xlabel="Evaluation physical scale",
            ylabel="CPG-free BC supported takeoff rate",
            ylim=(-0.05, 1.18),
        )
        save(fig, a.output, "05_BC_fit_and_closed_loop_transfer")
    else:
        missing.append("BC results")
    if a.ppo:
        if json.loads((a.ppo / "completion.json").read_text()).get("software_smoke"):
            missing.append("Extended PPO learning curves: only software smoke was run")
        runs = sorted(a.ppo.glob("[AB]_seed*"))
        train = pd.concat([pd.read_csv(r / "training.csv") for r in runs])
        evaluation = pd.concat([pd.read_csv(r / "evaluation.csv") for r in runs])
        train.to_csv(a.output / "ppo_training.csv", index=False)
        evaluation.to_csv(a.output / "ppo_evaluation.csv", index=False)
        budgets = (
            train.groupby(["training_seed", "method"])
            .actual_steps.max()
            .unstack("method")
        )
        if not (budgets.A == budgets.B).all():
            raise ValueError("Unmatched training interaction budget")
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for method, f in train.groupby("method"):
            for ax, key, label in zip(
                axes.flat,
                (
                    "post_update_kl",
                    "clip_fraction",
                    "takeoff",
                    "genuine_flight_pitch",
                    "rear_clearance",
                    "scale",
                ),
                (
                    "Post-update KL",
                    "PPO clipping fraction",
                    "Training takeoff fraction",
                    "Genuine flight pitch (rad)",
                    "Rear-foot clearance (m)",
                    "Physical morphology scale",
                ),
            ):
                mean = f.groupby("actual_steps")[key].mean()
                ax.plot(mean.index, mean, "o-", label=method)
                ax.set(xlabel="Actual training steps", ylabel=label)
        axes[0, 0].legend(title="Condition")
        fig.suptitle(
            "Software smoke only"
            if json.loads((a.ppo / "completion.json").read_text()).get("software_smoke")
            else "Matched takeoff PPO",
            y=1.02,
        )
        save(fig, a.output, "06_PPO_metrics")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for (method, scale), f in evaluation.groupby(["method", "scale"]):
            stats = f.groupby("actual_steps")[
                ["takeoff", "genuine_flight_pitch"]
            ].mean()
            axes[0].plot(
                stats.index, stats.takeoff, "o-", label=f"{method}, scale {scale:.2f}"
            )
            axes[1].plot(
                stats.index,
                stats.genuine_flight_pitch,
                "o-",
                label=f"{method}, scale {scale:.2f}",
            )
        axes[0].set(xlabel="Actual training steps", ylabel="Certified takeoff rate")
        axes[1].set(
            xlabel="Actual training steps", ylabel="Genuine-flight rotation (rad)"
        )
        axes[0].legend(fontsize=7)
        save(fig, a.output, "07_training_and_full_scale_evaluation")
    else:
        missing.append("New-reward PPO learning curves; extended experiment not run")
    if a.exploration:
        noise = pd.read_csv(a.exploration / "calibration.csv")
        stats = noise.groupby(["scale", "latent_std"]).agg(
            takeoff_rate=("takeoff", "mean"),
            takeoffs=("takeoff", "sum"),
            trials=("takeoff", "size"),
            front_support=("front_support_fraction", "mean"),
        )
        stats.to_csv(a.output / "exploration_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for scale, f in stats.reset_index().groupby("scale"):
            axes[0].plot(
                f.latent_std, f.takeoff_rate, "o-", label=f"{scale:.2f}: exploratory"
            )
            axes[1].plot(f.latent_std, f.front_support, "o-", label=f"{scale:.2f}")
        if a.fresh_exploration:
            fresh = pd.read_csv(a.fresh_exploration / "calibration.csv")
            fs = fresh.groupby(["scale", "latent_std"]).agg(
                takeoffs=("takeoff", "sum"),
                trials=("takeoff", "size"),
                rate=("takeoff", "mean"),
                support=("front_support_fraction", "mean"),
            )
            fs.to_csv(a.output / "exploration_fresh_summary.csv")
            for (scale, std), r in fs.iterrows():
                axes[0].scatter(
                    [std],
                    [r.rate],
                    marker="x",
                    s=80,
                    label=f"{scale:.2f}: fresh {int(r.takeoffs)}/{int(r.trials)}",
                )
        axes[0].set(
            xlabel="Gaussian latent std",
            ylabel="Frozen-policy genuine takeoff rate",
            ylim=(-0.05, 1.1),
        )
        axes[1].set(xlabel="Gaussian latent std", ylabel="Front-support fraction")
        axes[0].legend(fontsize=6)
        axes[1].legend(fontsize=7)
        save(fig, a.output, "08_exploration_calibration")
    validation = json.loads((a.validation / "validation.json").read_text())
    report = dict(
        validation=validation,
        missing=missing,
        measured_only=True,
        distinction=(
            "Reference probes: feasibility. BC: initialization. "
            "PPO smoke: update validation, not extended training."
        ),
        uncertainty=(
            "Authority: command grid and two reset seeds. BC: one actor "
            "and four reset seeds per scale. No superiority claim."
        ),
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copyfile(__file__, a.output / "report_source.py")


if __name__ == "__main__":
    main()
