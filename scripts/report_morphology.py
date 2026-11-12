"""Regenerate measured-data figures (PNG/PDF) and CSV tables. No synthetic data."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def export(fig, output, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def curves(ax, frame, y, ylabel):
    for method, f in frame.groupby("method"):
        # First aggregate trials within each training seed; seeds are independent units.
        per_seed = f.groupby(["training_seed", "iteration"])[y].mean().reset_index()
        stats = per_seed.groupby("iteration")[y].agg(["mean", "std", "count"])
        x, mean = stats.index.to_numpy(), stats["mean"].to_numpy()
        ax.plot(x, mean, label=method)
        if (stats["count"] > 1).any():
            sd = stats["std"].fillna(0).to_numpy()
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.18)
    ax.set(xlabel="PPO iteration", ylabel=ylabel)
    ax.legend(title="Method")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--feasibility", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    dynamics = pd.read_csv(a.feasibility / "dynamics.csv")
    dynamics.to_csv(a.output / "dynamics.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax, (y, label) in zip(
        axes.flat,
        [
            ("pendulum_length", "COM to front support (m)"),
            ("com_z", "Whole-body COM height (m)"),
            ("pitch_inertia_com", "Pitch inertia at COM (kg m²)"),
            ("com_x", "Whole-body COM x (m)"),
            ("pitch_inertia_support", "Pitch inertia at support (kg m²)"),
            ("mass", "Robot mass (kg)"),
        ],
    ):
        ax.plot(dynamics.scale, dynamics[y], "o-")
        ax.set(xlabel="Thigh/calf scale", ylabel=label)
    export(fig, a.output, "04_dynamics")
    attempts = pd.read_csv(a.feasibility / "attempts.csv")
    summary = attempts.groupby(["probe", "scale"]).agg(
        success_rate=("success", "mean"),
        max_pitch_rad=("max_pitch", "max"),
        mean_episode_s=("episode_seconds", "mean"),
        mean_rear_impulse_Ns=("rear_impulse", "mean"),
        mean_push_impulse_Ns=("push_impulse", "mean"),
        mean_excess_push_impulse_Ns=("excess_push_impulse", "mean"),
        peak_rear_torque_Nm=("peak_rear_torque", "max"),
        mean_saturation=("saturation", "mean"),
        mean_front_support=("front_support_fraction", "mean"),
        max_rear_clearance_m=("max_rear_clearance", "max"),
    )
    summary.to_csv(a.output / "feasibility_summary.csv")
    trajectory = pd.read_csv(a.feasibility / "trajectories.csv")
    # Predeclared representative probe, same seed and command at every scale.
    selected = trajectory[
        (trajectory.probe == "push")
        & (trajectory.seed == 0)
        & np.isclose(trajectory.amplitude, 1)
        & np.isclose(trajectory.duration, 0.24)
    ]
    fig, axes = plt.subplots(3, 2, figsize=(9, 7))
    for ax, (y, label) in zip(
        axes.flat,
        [
            ("pitch", "Body pitch (rad)"),
            ("pitch_velocity", "Pitch velocity (rad/s)"),
            ("com_x", "COM x (m)"),
            ("com_z", "COM z (m)"),
            ("rear_clearance", "Rear-foot clearance (m)"),
            ("front_support", "Both front feet loaded"),
        ],
    ):
        for scale, f in selected.groupby("scale"):
            ax.plot(f.time, f[y], label=f"{scale:.2f}")
        ax.set(xlabel="Time (s)", ylabel=label)
    axes[0, 0].legend(title="Scale", ncol=2)
    export(fig, a.output, "05_feasibility_trajectories")
    missing = []
    if a.runs:
        configs = [json.loads((r / "configuration.json").read_text()) for r in a.runs]
        expected = configs[0]
        seen = set()
        for config in configs:
            for key in ("config", "ppo", "source_sha256"):
                if config[key] != expected[key]:
                    raise ValueError(f"Incompatible comparison: {key}")
            for key in (
                "iterations",
                "steps_per_iteration",
                "eval_seeds",
                "eval_interval",
            ):
                if config["args"][key] != expected["args"][key]:
                    raise ValueError(f"Unmatched experimental parameter: {key}")
            identity = (config["args"]["method"], config["args"]["seed"])
            if identity in seen:
                raise ValueError("Duplicate method/training seed")
            seen.add(identity)
        seeds_by_method = {}
        for method, seed in seen:
            seeds_by_method.setdefault(method, set()).add(seed)
        if any(
            seeds != next(iter(seeds_by_method.values()))
            for seeds in seeds_by_method.values()
        ):
            raise ValueError("Methods must use the same training seeds")
        train = pd.concat([pd.read_csv(r / "training.csv") for r in a.runs])
        evaluation = pd.concat([pd.read_csv(r / "evaluation.csv") for r in a.runs])
        train.to_csv(a.output / "training.csv", index=False)
        evaluation.to_csv(a.output / "evaluation.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        curves(axes[0], train, "mean_reward", "Mean completed episode reward")
        curves(axes[1], train, "episode_seconds", "Mean episode survival (s)")
        export(fig, a.output, "01_training")
        standing = evaluation[~evaluation.perturbation]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        curves(ax, standing, "success", "Five-second success at full scale")
        ax.set_ylim(-0.05, 1.05)
        export(fig, a.output, "02_success")
        fig, ax = plt.subplots(figsize=(5, 3))
        curves(ax, train, "scale", "Normalized thigh/calf length")
        for r in a.runs:
            for event in json.loads((r / "curriculum_events.json").read_text()):
                ax.axvline(event["iteration"], color="k", alpha=0.2, linewidth=0.6)
        export(fig, a.output, "03_schedule")
        # Matched final iteration only; reject unlike training budgets.
        budgets = train.groupby(["method", "training_seed"]).actual_steps.max()
        if budgets.nunique() != 1:
            raise ValueError(
                "Final ablation requires equal actual environment-step budgets"
            )
        final = evaluation[evaluation.iteration == evaluation.iteration.max()]
        final = (
            final.groupby(["method", "training_seed", "perturbation"])[
                ["success", "episode_seconds", "stability_fraction", "saturation"]
            ]
            .mean()
            .reset_index()
        )
        final.to_csv(a.output / "final_per_seed.csv", index=False)
        table = final.groupby(["method", "perturbation"])[
            ["success", "episode_seconds", "stability_fraction", "saturation"]
        ].agg(["mean", "std", "count"])
        table.to_csv(a.output / "final_summary.csv")
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        normal = final[~final.perturbation]
        for ax, y in zip(axes, ("success", "episode_seconds", "stability_fraction")):
            stat = normal.groupby("method")[y].agg(["mean", "std"])
            ax.bar(stat.index, stat["mean"], yerr=stat["std"].fillna(0), capsize=3)
            ax.set(xlabel="Method", ylabel=y.replace("_", " "))
        export(fig, a.output, "06_ablation")
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, y in zip(axes, ("approx_kl", "clip_fraction", "gradient_norm")):
            curves(ax, train, y, y.replace("_", " "))
        export(fig, a.output, "optimization")
        # Saved full-scale policy attempt, same first evaluation seed.
        traces = pd.concat(
            [
                pd.read_csv(sorted(r.glob("evaluation_trajectories_*.csv"))[-1])
                for r in a.runs
            ]
        )
        traces = traces[(traces.seed == traces.seed.min()) & ~traces.perturbation]
        traces.to_csv(a.output / "representative_policy_trajectories.csv", index=False)
        fig, axes = plt.subplots(3, 2, figsize=(9, 7))
        for ax, y in zip(
            axes.flat,
            (
                "pitch",
                "pitch_velocity",
                "com_x",
                "com_z",
                "rear_clearance",
                "front_support",
            ),
        ):
            for (method, seed), f in traces.groupby(["method", "training_seed"]):
                ax.plot(f.time, f[y], label=f"{method}, seed {seed}")
            ax.set(xlabel="Time (s)", ylabel=y.replace("_", " "))
        axes[0, 0].legend()
        export(fig, a.output, "05_policy_trajectories")
    else:
        missing = [
            "01_training",
            "02_success",
            "03_schedule",
            "05_policy_trajectories",
            "06_ablation",
        ]
    (a.output / "report.json").write_text(
        json.dumps(
            dict(
                missing_figures=missing,
                uncertainty=(
                    "Error bands: std across training seeds for n > 1. "
                    "No seed variability inferred for n=1."
                ),
                note="Probe results only; missing PPO data are omitted.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
