"""Nominal-dynamics learning curves and separately labelled reset evaluations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    ev = pd.DataFrame(saved["history"]["evaluation"])
    tr = pd.DataFrame(saved["history"]["training"])
    ev.to_csv(a.output / "evaluation.csv", index=False)
    tr.to_csv(a.output / "training.csv", index=False)
    rows = []
    for (mode, it, kind), g in ev.groupby(["mode", "iteration", "evaluation_type"]):
        rows.append(
            dict(
                mode=mode,
                iteration=it,
                evaluation_type=kind,
                actual_steps=g.actual_steps.iloc[0],
                takeoff_rate=g.certified_standing_takeoff.mean(),
                dynamic_45_rate=float((g.peak_swing_pitch >= np.deg2rad(45)).mean()),
                dynamic_60_rate=float((g.peak_swing_pitch >= np.deg2rad(60)).mean()),
                collision_free_pitch_deg=np.rad2deg(g.peak_collision_free_pitch.mean()),
                peak_swing_pitch_deg=np.rad2deg(g.peak_swing_pitch.mean()),
                handstand_success=g.success.mean(),
                intermediate_success=g.intermediate_success.mean(),
                handstand_seconds=g.handstand_seconds.mean(),
                episode_seconds=g.episode_seconds.mean(),
                calf_collision=g.calf_collision.mean(),
                head_collision=g.head_collision.mean(),
                front_opposing_impulse=g.front_negative_impulse.mean(),
                first_useful_front_time=g.first_useful_front_time.mean(),
                partial_inversion_seconds=g.partial_inversion_seconds.mean(),
            )
        )
    table = pd.DataFrame(rows)
    table.to_csv(a.output / "summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for mode, g in table[table.evaluation_type == "standing"].groupby("mode"):
        axes[0, 0].plot(g.actual_steps, g.collision_free_pitch_deg, "o-", label=mode)
        axes[0, 1].plot(g.actual_steps, g.takeoff_rate, "o-", label=mode)
    if len(tr):
        for mode, g in tr.groupby("mode"):
            axes[1, 0].plot(g.actual_steps, g.actual_kp, label=mode)
            axes[1, 1].plot(g.actual_steps, g.post_update_kl, label=mode)
    for ax, title in zip(
        axes.flat,
        [
            "Nominal standing collision-free pitch (deg)",
            "Certified takeoff rate",
            "Training stiffness",
            "PPO post-update KL",
        ],
    ):
        ax.set_title(title)
        ax.set_xlabel("Training interactions per policy")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.output / "learning_curves.png")
    fig.savefig(a.output / "learning_curves.pdf")
    plt.close(fig)
    final = table[
        (table.iteration == table.iteration.max())
        & (table.evaluation_type == "standing")
    ]
    dump(
        a.output / "assessment.json",
        dict(
            updates=saved["iteration"],
            total_training_interactions=sum(r["steps"] for r in saved["runs"].values()),
            evaluation_interactions=saved["evaluation_steps"],
            final=final.to_dict("records"),
            solved=bool((final.handstand_success > 0).any()),
            interpretation=(
                "Single-seed diagnostic unless multiple seeds present; "
                "pitch alone is not success"
            ),
        ),
    )


if __name__ == "__main__":
    main()
