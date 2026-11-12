"""Plot actual retention observations; variability is across training seeds."""

import argparse
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--step-axes", action="store_true")
    a = p.parse_args()
    out = a.run / "figures"
    out.mkdir(exist_ok=True)
    if (a.run / "probes.csv").exists():
        data = pd.read_csv(a.run / "probes.csv")
    else:
        data = pd.concat(
            [pd.read_csv(f) for f in a.run.glob("*_seed*/evaluation.csv")],
            ignore_index=True,
        )
        data["stochastic"] = False
    metrics = [
        "takeoff",
        "max_pitch",
        "max_pitch_velocity",
        "rear_clearance",
        "genuine_airborne_seconds",
        "certified_work_min",
        "front_support_fraction",
        "nonfoot_collision",
        "success",
        "genuine_flight_pitch",
    ]
    x = "actual_steps" if a.step_axes else "iteration"
    per = (
        data.groupby(["method", "training_seed", x, "scale", "stochastic"])[metrics]
        .mean()
        .reset_index()
    )
    per.to_csv(out / "per_seed.csv", index=False)
    summary = per.groupby(["method", x, "scale", "stochastic"])[metrics].agg(
        ["mean", "std"]
    )
    summary.to_csv(out / "summary.csv")
    labels = {
        "takeoff": "Certified takeoff rate",
        "max_pitch": "Peak pitch (rad)",
        "max_pitch_velocity": "Peak pitch velocity (rad/s)",
        "rear_clearance": "Episode peak rear clearance (m)",
        "genuine_airborne_seconds": "Certified flight duration (s)",
        "certified_work_min": "Minimum rear-leg certified work (J)",
        "front_support_fraction": "Front support fraction",
        "nonfoot_collision": "Nonfoot collision rate",
        "success": "Five-second handstand success rate",
        "genuine_flight_pitch": "Certified flight pitch (rad)",
    }
    fig, axes = plt.subplots(4, 3, figsize=(13, 13))
    for ax, metric in zip(axes.flat, metrics):
        for (method, scale, stochastic), group in per.groupby(
            ["method", "scale", "stochastic"]
        ):
            g = group.groupby(x)[metric].agg(["mean", "std", "count"]).fillna(0)
            label = f"{method} {scale:.0%} " + ("sample" if stochastic else "mean")
            ax.plot(g.index, g["mean"], marker="o", markersize=3, label=label)
            ax.fill_between(
                g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.12
            )
        ax.set(
            xlabel="Training transitions" if a.step_axes else "PPO updates",
            ylabel=labels[metric],
        )
        if metric in (
            "takeoff",
            "front_support_fraction",
            "nonfoot_collision",
            "success",
        ):
            ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"retention.{ext}", dpi=180)
    plt.close(fig)
    training = pd.concat(
        [pd.read_csv(f) for f in a.run.glob("*_seed*/training.csv")], ignore_index=True
    )
    training.to_csv(out / "training.csv", index=False)
    if "demo_action_shift_rms" in training:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for (method, seed), g in training.groupby(["method", "training_seed"]):
            for ax, key in zip(
                axes.flat,
                [
                    "bc_weight",
                    "bc_loss",
                    "demo_action_shift_rms",
                    "demo_cumulative_gaussian_kl",
                ],
            ):
                ax.plot(g[x], g[key], label=f"{method} seed {seed}")
                ax.set(
                    xlabel="Training transitions" if a.step_axes else "PPO updates",
                    ylabel=key,
                )
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"bc_anchor_drift.{ext}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for (method, seed), g in training.groupby(["method", "training_seed"]):
        for ax, key in zip(axes, ["post_update_kl", "clip_fraction"]):
            ax.plot(g[x], g[key], label=f"{method} seed {seed}")
            ax.set(
                xlabel="Training transitions" if a.step_axes else "PPO updates",
                ylabel=key,
            )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"stability.{ext}", dpi=180)

    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3))
    for (method, seed), g in training.groupby(["method", "training_seed"]):
        ax.step(g.actual_steps, g.scale, where="post", label=f"{method} seed {seed}")
    ax.set(
        xlabel="Training transitions",
        ylabel="Physical leg scale",
        yticks=[0.9, 0.95, 1.0],
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"curriculum.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
