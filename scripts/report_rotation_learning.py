"""Measured rotation diagnostics and factorial pilot plots, with CSV tables."""

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
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + suffix), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", default=[])
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    trajectories = []
    summaries = []
    prediction = []
    for path in sorted(a.diagnosis.glob("*_trajectory.csv")):
        d = pd.read_csv(path).copy()
        tag = path.stem.removesuffix("_trajectory")
        method = tag[0]
        seed = int(tag.split("_")[1].removeprefix("seed"))
        scale = float(d.scale.iloc[0])
        reset = int(tag.split("reset")[-1])
        d["method"] = method
        d["training_seed"] = seed
        d["evaluation_seed"] = reset
        d["label"] = f"{method} seed {seed} {scale:.0%}"
        trajectories.append(d)
        valid = d[~d.nonfoot_collision & ~d.self_collision]
        peak = valid.loc[valid.pitch.idxmax()]
        lift = d[d.takeoff].iloc[0]
        post = d[(d.time >= lift.time) & (d.time <= peak.time)]
        front_power = sum(
            post[f"torque_{j}"] * post[f"joint_velocity_{j}"] for j in range(6)
        )
        rear_power = sum(
            post[f"torque_{j}"] * post[f"joint_velocity_{j}"] for j in range(6, 12)
        )
        retraction = post[(post.joint_velocity_8 + post.joint_velocity_11) / 2 < -0.5]
        dt = float(np.median(np.diff(d.time)))
        eligible = post[post.model_eligible]
        summaries.append(
            dict(
                method=method,
                training_seed=seed,
                scale=scale,
                evaluation_seed=reset,
                liftoff_time=lift.time,
                peak_time=peak.time,
                peak_pitch_degrees=np.rad2deg(peak.pitch),
                liftoff_momentum=lift.momentum_y,
                peak_momentum=peak.momentum_y,
                mean_front_moment=post.front_moment_y.mean(),
                mean_rear_moment=post.rear_moment_y.mean(),
                front_contact_angular_impulse=post.front_moment_y.sum() * dt,
                momentum_change=peak.momentum_y - lift.momentum_y,
                barrier_at_liftoff=lift.potential_barrier,
                pendulum_energy_at_liftoff=lift.rotational_energy,
                centroidal_energy_at_liftoff=lift.centroidal_rotation_energy,
                front_positive_work=front_power.clip(lower=0).sum() * dt,
                front_negative_work=front_power.clip(upper=0).sum() * dt,
                rear_positive_work=rear_power.clip(lower=0).sum() * dt,
                rear_negative_work=rear_power.clip(upper=0).sum() * dt,
                retraction_delay=retraction.time.iloc[0] - lift.time
                if len(retraction)
                else float("nan"),
                tracking_rms=post.tracking_rms.mean(),
                saturation=post.saturation.mean(),
                bc_action_rms=post.bc_action_rms.mean(),
                model_eligible_samples=len(eligible),
                pendulum_residual_rms=np.sqrt(np.mean(eligible.pendulum_residual**2))
                if len(eligible)
                else float("nan"),
            )
        )
        # Conditional integration restarts at each eligible interval; not a
        # free rollout prediction from initial morphology or body pitch.
        theta = omega = None
        for _, r in d.iterrows():
            if not r.model_eligible:
                theta = omega = None
                continue
            if theta is None:
                theta = r.pendulum_theta
                omega = r.pendulum_omega
            prediction.append(
                dict(
                    method=method,
                    training_seed=seed,
                    scale=scale,
                    evaluation_seed=reset,
                    time=r.time,
                    theta_measured=r.pendulum_theta,
                    theta_predicted=theta,
                    omega_measured=r.pendulum_omega,
                    omega_predicted=omega,
                    acceleration_predicted=r.pendulum_acceleration,
                    acceleration_measured=r.measured_pendulum_acceleration,
                )
            )
            acceleration = (
                -9.81 / r.effective_length * np.sin(theta)
                - 2 * r.length_rate / r.effective_length * omega
            )
            theta += omega * dt
            omega += acceleration * dt
    pd.DataFrame(summaries).to_csv(a.output / "baseline_mechanics.csv", index=False)
    predictions = pd.DataFrame(prediction)
    predictions.to_csv(a.output / "pendulum_comparison.csv", index=False)
    all_traces = pd.concat(trajectories, ignore_index=True)
    full = all_traces[(all_traces.scale == 1) & (all_traces.evaluation_seed == 61000)]
    for name, y, label in [
        ("01_pitch", "pitch", "Body pitch (rad)"),
        ("02_omega", "pitch_velocity", "Body pitch velocity (rad/s)"),
        ("05_length", "effective_length", "Effective COM-support length (m)"),
        ("11_momentum", "momentum_y", "COM angular momentum y (kg m²/s)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for tag, d in full.groupby("label"):
            ax.plot(d.time, d[y], label=tag)
        ax.set(xlabel="Time from standing reset (s)", ylabel=label)
        ax.legend(fontsize=8)
        save(fig, a.output / name)
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, d in full.groupby("label"):
        ax.plot(np.rad2deg(d.pitch), d.pitch_velocity, label=tag)
    ax.set(xlabel="Body pitch (degrees)", ylabel="Pitch velocity (rad/s)")
    ax.legend(fontsize=8)
    save(fig, a.output / "03_phase_portrait")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, d) in zip(axes.flat, full.groupby("label")):
        ax.plot(
            d.time,
            d.rotational_energy.where(d.model_eligible),
            label="Pendulum kinetic estimate (eligible samples)",
        )
        ax.plot(
            d.time,
            d.centroidal_rotation_energy,
            label="COM rigid-axis kinetic estimate",
        )
        ax.plot(d.time, d.potential_barrier, label="Frozen COM-support barrier")
        ax.set(title=tag, xlabel="Time (s)", ylabel="Approximate energy (J)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    save(fig, a.output / "04_energy")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, d) in zip(axes.flat, full.groupby("label")):
        ax.plot(d.time, d.front_moment_y, label="Front ground contact")
        ax.plot(d.time, d.rear_moment_y, label="Rear ground contact")
        ax.set(title=tag, xlabel="Time (s)", ylabel="Moment about COM y (N m)")
        ax.legend(fontsize=7)
    save(fig, a.output / "06_contact_moments")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, d) in zip(axes.flat, full.groupby("label")):
        ax.plot(d.time, d.front_force_z, label="Front vertical reaction")
        ax.plot(d.time, d.rear_force_z, label="Rear vertical reaction")
        ax.set(title=tag, xlabel="Time (s)", ylabel="Ground reaction force (N)")
        ax.legend(fontsize=7)
    save(fig, a.output / "15_ground_reactions")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for tag, d in full.groupby("label"):
        axes[0].plot(d.time, d.com_support_x, label=tag)
        axes[1].plot(d.com_support_x, d.com_support_z, label=tag)
    axes[0].set(xlabel="Time (s)", ylabel="COM x relative to front support (m)")
    axes[1].set(xlabel="COM x from support (m)", ylabel="COM z above support (m)")
    axes[0].legend(fontsize=7)
    save(fig, a.output / "16_com_motion")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, d) in zip(axes.flat, full.groupby("label")):
        ax.plot(d.time, d.tracking_rms, label="Joint tracking RMS (rad)")
        ax.plot(
            d.time,
            (d.joint_velocity_8 + d.joint_velocity_11) / 2,
            label="Rear calf mean velocity (rad/s)",
        )
        ax.plot(d.time, d.saturation, label="Motor saturation fraction")
        ax.set(title=tag, xlabel="Time (s)")
        ax.legend(fontsize=7)
    save(fig, a.output / "12_tracking_retraction")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (method, seed), d in predictions[
        (predictions.scale == 1) & (predictions.evaluation_seed == 61000)
    ].groupby(["method", "training_seed"]):
        axes[0].plot(
            d.time, d.theta_measured - d.theta_predicted, label=f"{method} seed {seed}"
        )
        axes[1].scatter(
            d.acceleration_predicted,
            d.acceleration_measured,
            s=3,
            alpha=0.3,
            label=f"{method} seed {seed}",
        )
    axes[0].set(xlabel="Time (s)", ylabel="Conditional COM angle error (rad)")
    axes[1].set(
        xlabel="Pendulum predicted acceleration (rad/s²)",
        ylabel="Measured relative COM acceleration (rad/s²)",
    )
    axes[0].legend(fontsize=7)
    save(fig, a.output / "10_pendulum_comparison")
    p = pd.read_csv(a.pilot / "probes.csv")
    metrics = [
        "takeoff",
        "max_pitch",
        "genuine_flight_pitch",
        "max_pitch_velocity",
        "liftoff_momentum",
        "liftoff_energy",
        "liftoff_deficit",
        "mean_front_moment",
        "front_support_fraction",
        "nonfoot_collision",
        "success",
        "tracking_rms",
    ]
    per = (
        p.groupby(
            [
                "anchor",
                "method",
                "training_seed",
                "iteration",
                "actual_steps",
                "scale",
                "stochastic",
            ]
        )[metrics]
        .mean()
        .reset_index()
    )
    per.to_csv(a.output / "pilot_per_seed.csv", index=False)
    per.groupby(
        ["anchor", "method", "iteration", "actual_steps", "scale", "stochastic"]
    )[metrics].agg(["mean", "std"]).to_csv(a.output / "pilot_summary.csv")
    final = per[per.iteration == per.iteration.max()]
    final.to_csv(a.output / "pilot_final_per_seed.csv", index=False)
    training = []
    for f in a.pilot.glob("*/*_seed*/training.csv"):
        d = pd.read_csv(f)
        d = d[d.iteration > 30]
        d["anchor"] = f.parent.parent.name
        training.append(d)
    training = pd.concat(training, ignore_index=True)
    labels = np.load(a.diagnosis / "bc_phase_labels.npz")["weights"]
    fraction = np.clip((training.iteration - 31) / 4, 0, 1)
    training["bc_post_min_effective_coefficient"] = training.bc_weight * (
        1 - (training.anchor == "phase") * fraction * (1 - labels.min())
    )
    training.to_csv(a.output / "training.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for (anchor, method, seed), d in training.groupby(
        ["anchor", "method", "training_seed"]
    ):
        for ax, key in zip(
            axes.flat,
            [
                "bc_weight",
                "bc_post_min_effective_coefficient",
                "bc_demo_weight_mean",
                "demo_action_shift_rms",
                "bc_loss",
                "bc_objective_contribution",
                "post_update_kl",
                "clip_fraction",
            ],
        ):
            ax.plot(d.actual_steps, d[key], label=f"{anchor} {method} seed{seed}")
            ax.set(xlabel="Cumulative PPO transitions", ylabel=key)
    axes[0, 0].legend(fontsize=6)
    save(fig, a.output / "07_anchor_and_ppo")
    fig, ax = plt.subplots(figsize=(9, 4))
    for (anchor, method, seed), d in training.groupby(
        ["anchor", "method", "training_seed"]
    ):
        ax.step(
            d.actual_steps, d.scale, where="post", label=f"{anchor} {method} {seed}"
        )
    ax.set(
        xlabel="Cumulative PPO transitions",
        ylabel="Physical leg scale",
        yticks=[0.9, 0.95, 1.0],
    )
    ax.legend(fontsize=7)
    save(fig, a.output / "14_morphology")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, key in zip(
        axes.flat,
        [
            "takeoff",
            "max_pitch",
            "liftoff_momentum",
            "liftoff_energy",
            "liftoff_deficit",
            "mean_front_moment",
            "nonfoot_collision",
            "success",
        ],
    ):
        for (anchor, method, stochastic), d in per[per.scale == 1].groupby(
            ["anchor", "method", "stochastic"]
        ):
            g = d.groupby("actual_steps")[key].agg(["mean", "std"]).fillna(0)
            label = f"{anchor} {method} " + ("sample" if stochastic else "mean")
            (line,) = ax.plot(g.index, g["mean"], label=label, marker="o", markersize=3)
            ax.fill_between(
                g.index,
                g["mean"] - g["std"],
                g["mean"] + g["std"],
                alpha=0.10,
                color=line.get_color(),
            )
        ax.set(xlabel="Cumulative PPO transitions", ylabel=key)
    axes[0, 0].legend(fontsize=6)
    save(fig, a.output / "08_09_full_scale_learning")
    fig, ax = plt.subplots(figsize=(9, 5))
    s = pd.DataFrame(json.loads((a.search / "search_summary.json").read_text()))
    ax.bar(s.context, np.rad2deg(s.best_valid_pitch))
    ax.axhline(45, color="gray", linestyle="--")
    ax.axhline(60, color="gray", linestyle=":")
    ax.tick_params(axis="x", rotation=65, labelsize=7)
    ax.set(ylabel="Best collision-free prefix peak pitch (degrees)")
    save(fig, a.output / "13_search_feasibility")
    counts = []
    for (anchor, method, seed), d in training.groupby(
        ["anchor", "method", "training_seed"]
    ):
        for scale in (0.9, 0.95, 1.0):
            counts.append(
                dict(
                    anchor=anchor,
                    method=method,
                    training_seed=int(seed),
                    scale=scale,
                    interactions=int(d[f"steps_scale_{scale:.2f}"].sum()),
                )
            )
    pd.DataFrame(counts).to_csv(a.output / "morphology_interactions.csv", index=False)
    if a.heldout:
        fresh = pd.concat([pd.read_csv(f / "probes.csv") for f in a.heldout])
        fresh_per = (
            fresh.groupby(["anchor", "method", "training_seed", "scale", "stochastic"])[
                metrics
            ]
            .mean()
            .reset_index()
        )
        fresh_per.to_csv(a.output / "heldout_per_seed.csv", index=False)
        fresh_per.groupby(["anchor", "method", "scale", "stochastic"])[metrics].agg(
            ["mean", "std"]
        ).to_csv(a.output / "heldout_summary.csv")


if __name__ == "__main__":
    main()
