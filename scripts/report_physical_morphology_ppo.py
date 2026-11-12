"""Regenerate measured A/full versus B/physical-curriculum PPO figures and tables."""

import argparse
import hashlib
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
LABELS = {"A": "A: fixed 100%", "B": "B: physical curriculum"}


def export(fig, root, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def curve(ax, frame, y, label, scale=1.0):
    for method, f in frame.groupby("method"):
        # Evaluation trials average within a training seed first.
        per = f.groupby(["training_seed", "actual_steps"])[y].mean().reset_index()
        stats = per.groupby("actual_steps")[y].agg(["mean", "std", "count"])
        x = stats.index.to_numpy()
        mean = stats["mean"].to_numpy() * scale
        ax.plot(x, mean, label=LABELS[method])
        if (stats["count"] > 1).any():
            sd = stats["std"].where(stats["count"] > 1).to_numpy() * scale
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.18)
    ax.set(xlabel="Actual training control steps", ylabel=label)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    runs = sorted(a.input.glob("[AB]_seed*"))
    configs = [json.loads((r / "configuration.json").read_text()) for r in runs]
    expected = configs[0]
    for c in configs:
        for key in (
            "config",
            "protocol",
            "actor_observation_dim",
            "critic_observation_dim",
            "reward_version",
        ):
            if c[key] != expected[key]:
                raise ValueError(f"unmatched configuration: {key}")
    train = pd.concat(
        [pd.read_csv(r / "training.csv") for r in runs], ignore_index=True
    )
    evaluation = pd.concat(
        [pd.read_csv(r / "evaluation.csv") for r in runs], ignore_index=True
    )
    episode_files = [r / "episodes.csv" for r in runs if (r / "episodes.csv").exists()]
    episodes = (
        pd.concat([pd.read_csv(f) for f in episode_files], ignore_index=True)
        if episode_files
        else evaluation.iloc[:0].copy()
    )
    # No duplicate evaluation trials may inflate seed counts.
    keys = [
        "method",
        "training_seed",
        "iteration",
        "scale",
        "evaluation_seed",
        "perturbation",
    ]
    if evaluation.duplicated(keys).any():
        raise ValueError("duplicate evaluation trial")
    budgets = (
        train.groupby(["method", "training_seed"]).actual_steps.max().unstack("method")
    )
    if not (budgets.A == budgets.B).all():
        raise ValueError("unmatched A/B interaction budgets")
    full = evaluation[(evaluation.scale == 1.0) & ~evaluation.perturbation].copy()
    train.to_csv(a.output / "training.csv", index=False)
    evaluation.to_csv(a.output / "evaluation.csv", index=False)
    episodes.to_csv(a.output / "episodes.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    curve(axes[0], train, "episode_reward", "Training episode reward")
    curve(axes[1], train, "episode_seconds", "Training episode survival (s)")
    axes[0].legend()
    export(fig, a.output, "01_training_reward_survival")
    metrics = [
        ("max_supported_pitch", "Mean maximum supported pitch (deg)", 180 / np.pi),
        ("max_pitch_velocity", "Mean maximum pitch velocity (rad/s)", 1),
        ("takeoff", "Supported rear-foot takeoff rate", 1),
        ("rear_clearance", "Mean maximum rear-foot clearance (m)", 1),
        ("success", "Five-second handstand success rate", 1),
        ("recovery_seconds", "Longest near-vertical recovery (s)", 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (key, label, scale) in zip(axes.flat, metrics):
        curve(ax, full, key, label, scale)
    axes[0, 0].legend()
    fig.suptitle("Every checkpoint evaluated at actual 100% Go2 geometry", y=1.01)
    export(fig, a.output, "02_full_scale_performance")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, label) in zip(
        axes,
        [
            ("front_support_fraction", "Front-foot support fraction"),
            ("nonfoot_collision", "Nonfoot-collision episode rate"),
            ("torque_saturation", "Actuator saturation fraction"),
        ],
    ):
        curve(ax, full, key, label)
    axes[0].legend()
    export(fig, a.output, "03_full_scale_contacts_constraints")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    curve(ax, train, "scale", "Physical thigh/calf length scale")
    events = []
    windows = []
    for r in runs:
        config = json.loads((r / "configuration.json").read_text())
        for event in json.loads((r / "curriculum_events.json").read_text()):
            events.append(
                dict(
                    **event,
                    method=config["method"],
                    training_seed=config["training_seed"],
                )
            )
            ax.axvline(event["actual_steps"], color="k", alpha=0.25, lw=0.7)
        for window in json.loads((r / "curriculum_windows.json").read_text()):
            windows.append(
                dict(
                    **window,
                    method=config["method"],
                    training_seed=config["training_seed"],
                )
            )
    ax.set_ylim(0.78, 1.02)
    ax.legend()
    export(fig, a.output, "04_physical_curriculum")
    (a.output / "curriculum_events.json").write_text(json.dumps(events, indent=2))
    if windows:
        pd.DataFrame(windows).to_csv(a.output / "curriculum_windows.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (key, label) in zip(
        axes.flat,
        [
            ("post_update_kl", "Post-update approximate KL"),
            ("clip_fraction", "PPO clipping fraction"),
            ("gradient_norm", "Gradient norm before clipping"),
            ("action_mean_abs", "Mean absolute policy action"),
            ("action_std", "Base Gaussian action std"),
            ("actor_parameter_change", "Actor parameter change (L2)"),
        ],
    ):
        curve(ax, train, key, label)
    axes[0, 0].legend()
    export(fig, a.output, "05_policy_optimization")
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    terms = [
        "pitch",
        "lift",
        "front_support",
        "stability",
        "recovery",
        "potential",
        "takeoff_bonus",
        "torque",
        "action_rate",
        "fall",
    ]
    for ax, term in zip(axes.flat, terms):
        curve(ax, train, "episode_reward_" + term, term.replace("_", " "))
    axes[0, 0].legend(fontsize=7)
    export(fig, a.output, "06_reward_components")
    # Own-stage diagnostic is separate from the primary full-size evaluation.
    current = evaluation[(evaluation["method"] == "B") & ~evaluation.perturbation]
    initial = expected["protocol"]["scales"][0]
    current = current[current.scale == initial]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, label, scale) in zip(axes, [metrics[0], metrics[2], metrics[3]]):
        curve(ax, current, key, label, scale)
    axes[0].legend()
    export(fig, a.output, "07_start_morphology_learning_diagnostic")
    common_steps = int(budgets.to_numpy().min())
    eligible = full[full.actual_steps <= common_steps]
    endpoint = int(eligible.actual_steps.max())
    final = full[full.actual_steps == endpoint]
    columns = [
        "episode_reward",
        "max_supported_pitch",
        "max_pitch",
        "max_pitch_velocity",
        "takeoff",
        "rear_clearance",
        "front_support_fraction",
        "nonfoot_collision",
        "success",
        "handstand_seconds",
        "recovery_seconds",
        "torque_saturation",
        "max_speed_ratio",
        "joint_limit_violation",
        "episode_seconds",
        "com_x",
        "com_y",
        "com_z",
        "com_speed",
    ]
    seed_summary = final.groupby(["method", "training_seed"])[columns].mean()
    seed_summary.to_csv(a.output / "final_full_scale_per_seed.csv")
    seed_summary.groupby("method").agg(["mean", "std", "count"]).to_csv(
        a.output / "final_full_scale_summary.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, key in zip(axes, ("takeoff", "max_supported_pitch", "success")):
        stat = seed_summary.groupby("method")[key].agg(["mean", "std"])
        multiplier = 180 / np.pi if key == "max_supported_pitch" else 1
        ax.bar(
            [LABELS[x] for x in stat.index],
            stat["mean"] * multiplier,
            yerr=stat["std"].fillna(0) * multiplier,
            capsize=4,
        )
        ax.set(
            ylabel="Supported pitch (deg)"
            if key == "max_supported_pitch"
            else key.replace("_", " ")
        )
        ax.tick_params(axis="x", rotation=15)
    export(fig, a.output, "08_final_full_scale_comparison")
    # Same held-out trajectory seed and same training seed for each method.
    representative = []
    training_seed = min(c["training_seed"] for c in configs)
    eval_seed = expected["protocol"]["evaluation_seeds"][0]
    for r in runs:
        c = json.loads((r / "configuration.json").read_text())
        if c["training_seed"] != training_seed:
            continue
        candidates = [
            f for f in r.glob("evaluation_trace_*.csv") if "_final" not in f.stem
        ]
        for f in sorted(candidates):
            frame = pd.read_csv(f)
            if frame.iteration.iloc[0] not in (0, final.iteration.max()):
                continue
            select = frame[
                (frame.scale == 1.0)
                & ~frame.perturbation
                & (frame.evaluation_seed == eval_seed)
            ].copy()
            representative.append(select)
    rep = pd.concat(representative)
    rep.to_csv(a.output / "representative_full_scale_trajectories.csv", index=False)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for ax, (key, label) in zip(
        axes.flat,
        [
            ("pitch", "Body pitch (rad)"),
            ("pitch_velocity", "Pitch velocity (rad/s)"),
            ("current_rear_clearance", "Rear-foot clearance (m)"),
            ("com_z", "COM height (m)"),
            ("com_speed", "COM speed (m/s)"),
            ("front_support", "Both front feet loaded"),
        ],
    ):
        for (method, iteration), f in rep.groupby(["method", "iteration"]):
            ax.plot(f.episode_seconds, f[key], label=f"{method}, iteration {iteration}")
        ax.set(xlabel="Episode time (s)", ylabel=label)
    axes[0, 0].legend(fontsize=7)
    export(fig, a.output, "09_full_scale_trajectories")
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for i, axis in enumerate("xyz"):
        for (method, iteration), f in rep.groupby(["method", "iteration"]):
            axes[i, 0].plot(
                f.episode_seconds, f["com_" + axis], label=f"{method}, {iteration}"
            )
            axes[i, 1].plot(
                f.episode_seconds,
                f["com_velocity_" + axis],
                label=f"{method}, {iteration}",
            )
        axes[i, 0].set(xlabel="Episode time (s)", ylabel=f"COM {axis} (m)")
        axes[i, 1].set(xlabel="Episode time (s)", ylabel=f"COM velocity {axis} (m/s)")
    axes[0, 0].legend(fontsize=7)
    export(fig, a.output, "10_full_scale_COM_trajectories")
    # Failure mode and qualification statistics diagnose a stalled curriculum.
    failures = (
        episodes.groupby(
            ["method", "training_seed", "termination", "illegal_bodies"], dropna=False
        )
        .size()
        .rename("episodes")
    )
    failures.to_csv(a.output / "termination_causes.csv")
    qualification = episodes.groupby(["method", "training_seed", "scale"]).agg(
        episodes=("takeoff", "size"),
        takeoff_rate=("takeoff", "mean"),
        supported_pitch_max=("max_supported_pitch", "max"),
        near_vertical_recovery_max=("recovery_seconds", "max"),
        five_second_success_rate=("success", "mean"),
        front_support=("front_support_fraction", "mean"),
    )
    qualification.to_csv(a.output / "training_qualification_summary.csv")
    outcomes = [json.loads((r / "outcome.json").read_text()) for r in runs]
    report = dict(
        common_final_training_steps=endpoint,
        training_seeds=sorted(set(c["training_seed"] for c in configs)),
        outcomes=outcomes,
        decision=json.loads((a.input / "diagnostic_decisions.json").read_text()),
        variability=(
            "Mean +/- one std across training seeds after trial averaging. "
            "Two seeds are diagnostic only."
        ),
        budgets=(
            "Training control steps matched per seed. Evaluation interactions "
            "are separately counted and never optimized on."
        ),
        physical_morphology=True,
        posture_curriculum=False,
        cpg=False,
        input_hashes={
            str(r / "training.csv"): hashlib.sha256(
                (r / "training.csv").read_bytes()
            ).hexdigest()
            for r in runs
        },
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copyfile(__file__, a.output / "report_source.py")


if __name__ == "__main__":
    main()
