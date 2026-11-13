"""Measured momentum/energy/contact plots and diagnostic aggregates."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--validation", type=Path, required=True)
    a = p.parse_args()
    root = a.directory
    data = pd.read_csv(root / "comparison.csv")
    final = int(data.iteration.max())
    table = data.groupby(["condition", "iteration"]).agg(
        takeoff_rate=("takeoff", "mean"),
        pitch=("peak_certified_pitch", "mean"),
        collision_free_pitch=("peak_collision_free_pitch", "mean"),
        handstand=("success", "mean"),
        collision=("nonfoot_collision", "mean"),
        L_com=("peak_L_com", "mean"),
        L_support=("peak_L_support", "mean"),
        rear_work=("rear_work", "mean"),
        rear_positive_work=("rear_positive_work", "mean"),
        rear_push_impulse=("positive_rear_impulse", "mean"),
        front_push_impulse=("front_push_impulse", "mean"),
        net_push_impulse=("net_push_impulse", "mean"),
        apex_potential=("apex_potential", "mean"),
        apex_kinetic=("apex_kinetic", "mean"),
        apex_clearance=("apex_calf_clearance", "mean"),
        pass45=("milestone45", "mean"),
        pass60=("milestone60", "mean"),
        pass80=("milestone80", "mean"),
    )
    table["pitch_degrees"] = np.rad2deg(table.pitch)
    table["collision_free_degrees"] = np.rad2deg(table.collision_free_pitch)
    table.to_csv(root / "summary.csv")
    validation = pd.read_csv(a.validation / "schedule_summary.csv")
    schedules = validation.groupby("schedule").agg(
        pitch=("peak_certified_pitch", "mean"),
        apex_potential=("apex_potential", "mean"),
        apex_clearance=("apex_calf_clearance", "mean"),
        rear_work=("rear_positive_work", "mean"),
        takeoff=("takeoff", "mean"),
        collision=("nonfoot_collision", "mean"),
    )
    schedules["pitch_degrees"] = np.rad2deg(schedules.pitch)
    schedules.to_csv(root / "schedule_comparison.csv")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), layout="constrained")
    keys = [
        "L_support",
        "L_com",
        "com_height",
        "potential",
        "pitch",
        "pitch_velocity",
        "rear_power",
        "rear_positive_work",
        "calf_clearance",
    ]
    ref = json.loads((a.validation / "reference.json").read_text())
    for condition in "ABC":
        path = sorted(root.glob(f"{condition}_seed*/trace_{final:03}.csv"))[0]
        d = pd.read_csv(path)
        d = d[d.seed == 94000]
        for ax, key in zip(axes.flat, keys):
            ax.plot(d.time, d[key], label=condition)
            ax.set(xlabel="time (s)", ylabel=key)
            ax.grid(alpha=0.2)
        axes[0, 0].plot(
            d.time, d.L_target, ls="--", alpha=0.6, label=f"{condition} target"
        )
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].axhline(
        ref["potential"], ls=":", color="black", label="handstand target"
    )
    axes[1, 0].legend(fontsize=7)
    fig.savefig(root / "momentum_energy.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for condition in "ABC":
        d = pd.read_csv(sorted(root.glob(f"{condition}_seed*/trace_{final:03}.csv"))[0])
        d = d[d.seed == 94000]
        axes[0, 0].plot(d.phase, d.L_support, label=f"{condition} actual")
        axes[0, 0].plot(d.phase, d.L_target, ls="--", alpha=0.6)
        axes[0, 1].plot(d.time, d.front_force_z, label=f"{condition} front")
        axes[0, 1].plot(d.time, d.rear_force_z, ls="--", label=f"{condition} rear")
        axes[1, 0].plot(
            d.time, d.front_support.astype(float), label=f"{condition} support"
        )
        axes[1, 0].plot(
            d.time, d.collision.astype(float), ls="--", label=f"{condition} collision"
        )
        g = table.loc[condition]
        axes[1, 1].plot(g.index, g.pitch_degrees, "o-", label=condition)
    axes[0, 0].set(xlabel="maneuver progress", ylabel="support momentum (kg m²/s)")
    axes[0, 1].set(xlabel="time (s)", ylabel="vertical GRF (N)")
    axes[1, 0].set(xlabel="time (s)", ylabel="contact events")
    axes[1, 1].set(xlabel="PPO update", ylabel="certified peak pitch (degrees)")
    axes[1, 1].axhline(45, ls=":", color="gray")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(root / "phase_contacts_learning.png", dpi=150)
    plt.close(fig)
    # Twelve actor commands, actual motion and phase timing.
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), layout="constrained")
    path = sorted(root.glob(f"C_seed*/trace_{final:03}.csv"))[0]
    d = pd.read_csv(path)
    d = d[d.seed == 94000]
    for i, ax in enumerate(axes.flat):
        ax.plot(d.time, d[f"target_{i}"], label="target")
        ax.plot(d.time, d[f"joint_{i}"], ls="--", label="actual")
        ax.set(xlabel="time (s)", ylabel=f"joint {i} (rad)")
        ax.grid(alpha=0.2)
        for phase in (0.2, 0.4, 0.6, 0.8):
            crossing = d[d.phase >= phase]
            if len(crossing):
                ax.axvline(crossing.time.iloc[0], ls=":", color="gray", alpha=0.5)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(root / "joint_commands.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for condition in "ABC":
        d = pd.read_csv(sorted(root.glob(f"{condition}_seed*/trace_{final:03}.csv"))[0])
        d = d[d.seed == 94000]
        axes[0].plot(np.rad2deg(d.pitch), d.pitch_velocity, label=condition)
        axes[1].plot(d.time, d.rear_moment_com, label=f"{condition} rear")
        axes[1].plot(d.time, d.front_moment_com, ls="--", label=f"{condition} front")
        axes[2].plot(d.time, d.mechanical_energy, label=condition)
    axes[0].set(xlabel="body pitch (degrees)", ylabel="pitch rate (rad/s)")
    axes[1].set(xlabel="time (s)", ylabel="ground moment about COM (Nm)")
    axes[2].set(xlabel="time (s)", ylabel="mechanical energy (J)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(root / "phase_portrait_moments.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    stats_keys = [
        "post_update_kl",
        "clip_fraction",
        "bc_loss",
        "mean_r_momentum",
        "sum_reward_upward_energy",
        "sum_reward_terminal",
    ]
    for condition in "ABC":
        traces = [
            pd.read_csv(path)
            for path in sorted(root.glob(f"{condition}_seed*/training.csv"))
        ]
        if not traces:
            continue
        average = pd.concat(traces).groupby("iteration").mean(numeric_only=True)
        for ax, key in zip(axes.flat, stats_keys):
            ax.plot(average.index, average[key], "o-", label=condition)
            ax.set(xlabel="PPO update", ylabel=key)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    fig.savefig(root / "ppo_rewards.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for condition in "ABC":
        d = pd.read_csv(sorted(root.glob(f"{condition}_seed*/trace_{final:03}.csv"))[0])
        d = d[d.seed == 94000]
        axes[0].plot(d.time, d.pendulum_length, label=condition)
        axes[1].plot(d.time, d.I_support, label=condition)
        axes[2].plot(d.time, d.com_support_x, label=condition)
    for ax, key in zip(
        axes,
        [
            "COM-support length (m)",
            "locked support inertia (kg m²)",
            "COM behind/front of support (m)",
        ],
    ):
        ax.set(xlabel="time (s)", ylabel=key)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(root / "support_geometry.png", dpi=150)
    plt.close(fig)
    # Candidate comparison exposes pitch/COM-height tradeoffs.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for name in ("none", "rear_tuck", "front_fold", "front_open"):
        d = pd.read_csv(a.validation / f"schedule_{name}_92000.csv")
        axes[0].plot(d.time, np.rad2deg(d.pitch), label=name)
        axes[1].plot(d.time, d.potential, label=name)
        axes[2].plot(d.time, d.calf_clearance, label=name)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("pitch (degrees)")
    axes[1].set_ylabel("COM potential (J)")
    axes[2].set_ylabel("front calf clearance (m)")
    axes[2].axhline(0, color="black", ls=":")
    fig.savefig(root / "targeted_schedules.png", dpi=150)
    plt.close(fig)
    import torch
    from experiments.momentum_phase.tools import network

    torch.set_num_threads(1)
    balance = np.load(a.validation / "balance_demonstrations.npz")
    teacher = np.load(a.validation / "takeoff_demonstrations.npz")
    mask = teacher["seed"] >= 93004
    imitation = []
    for condition in "ABC":
        for path in sorted(root.glob(f"{condition}_seed*/checkpoint_*.pt")):
            net, saved = network(path)
            with torch.no_grad():
                balance_error = float(
                    (
                        net.actor(torch.tensor(balance["observations"])).tanh()
                        - torch.tensor(balance["actions"])
                    )
                    .square()
                    .mean()
                )
                takeoff_error = float(
                    (
                        net.actor(torch.tensor(teacher["observations"][mask])).tanh()
                        - torch.tensor(teacher["actions"][mask])
                    )
                    .square()
                    .mean()
                )
            imitation.append(
                dict(
                    condition=condition,
                    seed=path.parent.name,
                    iteration=saved["iteration"],
                    balance_action_mse=balance_error,
                    heldout_takeoff_action_mse=takeoff_error,
                )
            )
    pd.DataFrame(imitation).to_csv(root / "offline_imitation.csv", index=False)
    print(table.to_string())
    print(schedules.to_string())


if __name__ == "__main__":
    main()
