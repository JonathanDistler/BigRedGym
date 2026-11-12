"""Summarize bounded continuation with physical metrics and seed variation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run / "figures"
    out.mkdir(exist_ok=True)
    data = pd.read_csv(args.run / "rotation_diagnostics.csv")
    metrics = [
        "takeoff",
        "max_pitch",
        "genuine_flight_pitch",
        "pitch_velocity_liftoff",
        "pitch_velocity_peak",
        "angular_momentum_liftoff",
        "certified_impulse_min",
        "rear_clearance",
        "genuine_airborne_seconds",
        "com_support_x_peak",
        "front_support_at_peak",
        "joint_tracking_rms",
        "torque_saturation",
        "nonfoot_collision",
        "first_nonfoot_collision_time",
        "success",
    ]
    per = (
        data.groupby(
            ["method", "training_seed", "actual_steps", "iteration", "stochastic"]
        )[metrics]
        .mean()
        .reset_index()
    )
    per.to_csv(out / "rotation_per_seed.csv", index=False)
    per.groupby(["method", "actual_steps", "iteration", "stochastic"])[metrics].agg(
        ["mean", "std"]
    ).to_csv(out / "rotation_summary.csv")
    fig, axes = plt.subplots(4, 4, figsize=(17, 14))
    units = {
        "max_pitch": "Peak body pitch (rad)",
        "genuine_flight_pitch": "Certified flight pitch (rad)",
        "pitch_velocity_liftoff": "Liftoff pitch velocity (rad/s)",
        "pitch_velocity_peak": "Pitch velocity at peak (rad/s)",
        "angular_momentum_liftoff": "Liftoff COM angular momentum (kg m²/s)",
        "certified_impulse_min": "Certified minimum rear impulse (Ns)",
        "rear_clearance": "Rear clearance (m)",
        "genuine_airborne_seconds": "Certified flight duration (s)",
        "com_support_x_peak": "COM x from front support at peak (m)",
        "joint_tracking_rms": "Joint tracking RMS (rad)",
        "first_nonfoot_collision_time": "First nonfoot collision (s)",
    }
    for ax, metric in zip(axes.flat, metrics):
        for (method, stochastic), group in per.groupby(["method", "stochastic"]):
            g = group.groupby("actual_steps")[metric].agg(["mean", "std"]).fillna(0)
            label = ("Fixed " if method == "A" else "Curriculum ") + (
                "stochastic" if stochastic else "deterministic"
            )
            (line,) = ax.plot(g.index, g["mean"], marker="o", label=label)
            ax.fill_between(
                g.index,
                g["mean"] - g["std"],
                g["mean"] + g["std"],
                alpha=0.12,
                color=line.get_color(),
            )
            for seed, seed_group in group.groupby("training_seed"):
                ax.plot(
                    seed_group.actual_steps,
                    seed_group[metric],
                    alpha=0.3,
                    linewidth=0.6,
                    linestyle="--" if seed == 42 else ":",
                    color=line.get_color(),
                )
        ax.set(xlabel="Training transitions", ylabel=units.get(metric, metric))
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rotation_diagnostics.{ext}", dpi=180)
    plt.close(fig)
    training = pd.concat(
        [pd.read_csv(f) for f in args.run.glob("*_seed*/training.csv")]
    )
    counts = []
    for (method, seed), g in training.groupby(["method", "training_seed"]):
        for scale in (0.9, 0.95, 1.0):
            key = f"steps_scale_{scale:.2f}"
            counts.append(
                dict(
                    method=method,
                    training_seed=int(seed),
                    scale=scale,
                    total_interactions=int(g[key].sum()),
                    additional_interactions=int(g.loc[g.iteration > 10, key].sum()),
                )
            )
    pd.DataFrame(counts).to_csv(out / "morphology_interactions.csv", index=False)
    final = per[per.iteration == per.iteration.max()]
    final.to_csv(out / "final_rotation_per_seed.csv", index=False)
    summary = {}
    for (method, stochastic), group in final.groupby(["method", "stochastic"]):
        summary[f"{method}_{'stochastic' if stochastic else 'deterministic'}"] = {
            key: dict(mean=float(group[key].mean()), seed_std=float(group[key].std()))
            for key in metrics
        }
    summary["training"] = {
        key: dict(
            min=float(training.loc[training.iteration > 10, key].min()),
            max=float(training.loc[training.iteration > 10, key].max()),
        )
        for key in [
            "post_update_kl",
            "clip_fraction",
            "bc_weight",
            "bc_loss",
            "demo_action_shift_rms",
        ]
    }
    (args.run / "measured_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
