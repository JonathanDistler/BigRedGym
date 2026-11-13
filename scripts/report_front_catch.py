"""Report committed history with separate standing/intermediate scores."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = a.directory
    out = a.output
    path = sorted(root.glob("checkpoint_*.pt"))[-1]
    saved = torch.load(path, weights_only=False, map_location="cpu")
    history = saved["history"]
    ev = pd.DataFrame(history["evaluation"])
    training = pd.DataFrame(history["training"])
    episodes = pd.DataFrame(history["episodes"])
    ev["condition"] = ev["mode"].str[0]
    ev["pitch_degrees"] = np.rad2deg(ev["peak_swing_pitch"])
    ev["safe_pitch_degrees"] = np.rad2deg(ev["peak_collision_free_pitch"])
    for name, frame in [
        ("evaluation", ev),
        ("training", training),
        ("episodes", episodes),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)
    standing = ev[ev.evaluation_type == "standing"]
    intermediate = ev[ev.evaluation_type == "intermediate"]
    table = standing.groupby(["condition", "iteration"]).agg(
        takeoff_rate=("certified_standing_takeoff", "mean"),
        peak_pitch_degrees=("pitch_degrees", "mean"),
        max_pitch_degrees=("pitch_degrees", "max"),
        standing_success=("standing_success", "mean"),
        recontacts=("front_recontacts", "mean"),
        backward_m=("backward_mean", "mean"),
        front_impulse=("front_impulse", "mean"),
        rear_impulse=("rear_impulse", "mean"),
        front_counter_impulse=("front_negative_impulse", "mean"),
        head_collision=("head_collision", "mean"),
        calf_collision=("calf_collision", "mean"),
        capture_error=("capture_error", "mean"),
        hold_seconds=("handstand_seconds", "max"),
    )
    table.to_csv(out / "standing_summary.csv")
    interm = intermediate.groupby(["condition", "start_kind", "iteration"]).agg(
        intermediate_success=("intermediate_success", "mean"),
        peak_pitch_degrees=("pitch_degrees", "mean"),
        hold_seconds=("handstand_seconds", "max"),
        recontacts=("front_recontacts", "mean"),
        head_collision=("head_collision", "mean"),
        calf_collision=("calf_collision", "mean"),
    )
    interm.to_csv(out / "intermediate_summary.csv")
    final = saved["iteration"]
    final_stand = standing[standing.iteration == final]
    milestones = final_stand.groupby("condition").apply(
        lambda d: pd.Series(
            {
                f"pass_{deg}": float((d.pitch_degrees >= deg).mean())
                for deg in [35, 45, 60, 80]
            }
        ),
        include_groups=False,
    )
    milestones.to_csv(out / "milestones.csv")

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=150)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for condition in "ABC":
        d = table.loc[condition]
        axes[0].plot(d.index, d.peak_pitch_degrees, "o-", label=condition)
        axes[1].plot(d.index, d.takeoff_rate, "o-", label=condition)
        axes[2].plot(d.index, d.backward_m, "o-", label=condition)
    axes[0].axhline(45, ls=":", color="gray")
    axes[0].set_ylabel("Standing-start certified peak pitch (deg)")
    axes[1].set_ylabel("Standing-start takeoff rate")
    axes[2].set_ylabel("World backward foot displacement (m)")
    for ax in axes:
        ax.set_xlabel("PPO update")
        ax.grid(alpha=0.2)
        ax.legend()
    save(fig, "standing_learning")
    # Select committed final traces. Uncommitted crash logs are deliberately excluded.
    traces = {}
    for condition in "ABC":
        suffix = f"{condition}_seed42_{final:06}.csv"
        matches = [p for p in history["trace_files"] if p.endswith(suffix)]
        if not matches:
            continue
        file = Path(matches[-1])
        if not file.exists():
            file = root / Path(*file.parts[file.parts.index("sessions") :])
        d = pd.read_csv(file)
        traces[condition] = (
            d[d.start_kind == "standing"].query("evaluation_seed==96000").copy()
        )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for condition, d in traces.items():
        t = d.elapsed_time
        axes[0, 0].plot(t, d.L_com, label=condition)
        axes[0, 1].plot(t, d.L_support.where(d.any_front_loaded), label=condition)
        axes[1, 0].plot(t, d.potential, label=condition)
        axes[1, 1].plot(t, d.K_system, label=condition)
        axes[2, 0].plot(t, d.front_moment_com, label=condition + " front")
        axes[2, 0].plot(t, d.rear_moment_com, ls="--", label=condition + " rear")
        axes[2, 1].plot(t, d.capture_error, label=condition)
    axes[1, 0].axhline(89.4166207975, ls=":", color="black")
    labels = [
        "COM momentum (kg m²/s)",
        "Active-support momentum (kg m²/s)",
        "COM potential (J)",
        "Full kinetic energy (J)",
        "Ground moments about COM (Nm)",
        "Capture-point support distance (m)",
    ]
    for ax, label in zip(axes.flat, labels):
        ax.set(xlabel="Episode time (s)", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    save(fig, "energy_momentum_contacts")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for condition, d in traces.items():
        t = d.elapsed_time
        axes[0, 0].plot(t, d.FL_backward, label=condition + " FL")
        axes[0, 0].plot(t, d.FR_backward, ls="--", label=condition + " FR")
        axes[0, 1].plot(t, d.FL_contact.astype(float), label=condition + " FL")
        axes[0, 1].plot(t, d.FR_contact.astype(float), ls="--", label=condition + " FR")
        axes[1, 0].plot(t, d.front_recontacts, label=condition)
        axes[1, 1].plot(t, d.front_force_weight_ratio, label=condition)
        axes[2, 0].plot(t, d.head_clearance, label=condition + " head")
        axes[2, 0].plot(t, d.calf_clearance, ls="--", label=condition + " front calf")
        axes[2, 1].plot(np.rad2deg(d.pitch), d.pitch_velocity, label=condition)
    axes[0, 0].set_ylabel("Backward displacement in world frame (m)")
    axes[0, 1].set_ylabel("Per-foot loaded contact")
    axes[1, 0].set_ylabel("Post-release recontact count")
    axes[1, 1].set_ylabel("Front normal force / robot weight")
    axes[2, 0].set_ylabel("Ground clearance (m)")
    axes[2, 0].axhline(0, ls=":", color="black")
    axes[2, 1].set(xlabel="Pitch (deg)", ylabel="Pitch speed (rad/s)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[:-1]:
        ax.set_xlabel("Episode time (s)")
    save(fig, "foot_placement_and_collisions")
    if not training.empty:
        training["condition"] = training["mode"].str[0]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        keys = [
            "post_update_kl",
            "clip_fraction",
            "bc_loss",
            "sum_reward_useful_placement",
            "sum_reward_stable_recatch",
            "steps_per_second",
        ]
        for condition in "ABC":
            d = (
                training[training.condition == condition]
                .groupby("iteration")
                .mean(numeric_only=True)
            )
            for ax, key in zip(axes.flat, keys):
                ax.plot(d.index, d[key], "o-", label=condition)
                ax.set(xlabel="PPO update", ylabel=key)
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        save(fig, "ppo_and_catch_rewards")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for condition in "ABC":
        d = intermediate[
            (intermediate.condition == condition) & (intermediate.iteration == final)
        ]
        means = d.groupby("start_kind").mean(numeric_only=True)
        axes[0].plot(means.index, means.handstand_seconds, "o-", label=condition)
        axes[1].plot(means.index, means.intermediate_success, "o-", label=condition)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Intermediate-start best hold (s)")
    axes[1].set_ylabel("Intermediate-start success rate")
    save(fig, "intermediate_only")
    if "C" in traces:
        d = traces["C"]
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), layout="constrained")
        for i, ax in enumerate(axes.flat):
            ax.plot(d.elapsed_time, d[f"target_{i}"], label="target")
            ax.plot(d.elapsed_time, d[f"joint_{i}"], ls="--", label="actual")
            ax.set(xlabel="Episode time (s)", ylabel=f"Joint {i} (rad)")
            ax.grid(alpha=0.2)
            for phase in [0.2, 0.4, 0.6, 0.8]:
                crossing = d[d.phase >= phase]
                if len(crossing):
                    ax.axvline(
                        crossing.elapsed_time.iloc[0], ls=":", color="gray", alpha=0.4
                    )
        axes[0, 0].legend()
        save(fig, "single_actor_joint_commands")
    if a.benchmark:
        data = pd.read_csv(a.benchmark / "benchmark.csv")
        data.to_csv(out / "benchmark.csv", index=False)
        process = data[data.backend != "serial"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for ax, key, label in zip(
            axes,
            ["steps_per_second", "aggregate_rss_mib", "ppo_update_seconds"],
            [
                "Controller transitions / s",
                "Aggregate RSS (MiB)",
                "PPO update time (s)",
            ],
        ):
            ax.plot(process.num_envs, process[key], "o-")
            ax.set(xlabel="Environment count", ylabel=label)
            ax.grid(alpha=0.2)
        save(fig, "cpu_benchmark")
    from experiments.front_catch.io import atomic_checkpoint

    policies = out / "policies"
    policies.mkdir()
    for mode, run in saved["runs"].items():
        actor = {
            key.removeprefix("actor."): value
            for key, value in run["network"].items()
            if key.startswith("actor.")
        }
        atomic_checkpoint(
            policies / f"{mode}.pt",
            dict(
                actor=actor,
                config=run["config"],
                actor_dim=73,
                actions=12,
                source_checkpoint=str(path),
                phase="CatchPhase",
                scope="Single CPU actor; hardware estimator not implemented",
            ),
        )
    summary = dict(
        checkpoint=str(path),
        iteration=final,
        completed=final == saved["protocol"]["updates"],
        total_training_interactions=sum(r["steps"] for r in saved["runs"].values()),
        evaluation_interactions=saved["evaluation_steps"],
        curriculum_stage=saved["curriculum"].stage,
        standing_probability=saved["curriculum"].mixture[0],
        standing_successes=int(final_stand.standing_success.sum()),
        intermediate_successes=int(
            intermediate[intermediate.iteration == final].intermediate_success.sum()
        ),
    )
    (out / "completion.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string())
    print(summary)


if __name__ == "__main__":
    main()
