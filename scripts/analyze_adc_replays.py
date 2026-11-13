"""Quantitative comparison from synchronized native-state replays, not new animation."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    records = []
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for path in sorted(args.directory.glob("*_standing_seed*.json")):
        info = json.loads(path.read_text())
        frame = pd.read_csv(path.with_suffix(".csv"))
        physics = pd.read_csv(path.with_name(path.stem + "_physics.csv"))
        safe = physics[physics.prefix_safe.astype(bool)]
        peak = safe.loc[safe.pitch.idxmax()]
        row = dict(
            mode=path.stem,
            takeoff=info["certified_standing_takeoff"],
            takeoff_time=info["takeoff_time"],
            collision_free_peak_pitch_deg=float(np.rad2deg(peak.pitch)),
            peak_pitch_time=float(peak.time),
            first_useful_front_time=info["first_useful_front_time"],
            useful_action_definition=(
                "Joint movement coinciding with increased geometry score; "
                "heuristic, not causal proof"
            ),
            front_opposing_angular_impulse=float(
                np.trapezoid(np.minimum(physics.front_moment_com, 0), physics.time)
            ),
            front_opposing_impulse_before_peak=float(
                np.trapezoid(
                    np.minimum(safe[safe.time <= peak.time].front_moment_com, 0),
                    safe[safe.time <= peak.time].time,
                )
            ),
            peak_com_x=float(peak.com_x),
            peak_com_z=float(peak.com_z),
            peak_com_velocity_x=float(peak.com_velocity_x),
            peak_L_com=float(peak.L_com),
            peak_L_support=float(peak.L_support),
            calf_collision=info["calf_collision"],
            head_collision=info["head_collision"],
            collision_bodies=info["collision_bodies"],
            partial_inversion_seconds=info["partial_inversion_seconds"],
            handstand_seconds=info["handstand_seconds"],
            five_second_success=info["standing_success"],
            survival_seconds=float(info["elapsed_episode_seconds"]),
            minimum_orientation_error_deg=float(
                np.rad2deg(safe.orientation_error.min())
            ),
        )
        for leg in ("FL", "FR"):
            for coordinate in ("world", "body", "com_relative"):
                row[f"{leg}_{coordinate}_dx"] = float(
                    frame[f"{leg}_{coordinate}_x"].iloc[-1]
                    - frame[f"{leg}_{coordinate}_x"].iloc[0]
                )
            row[leg + "_recontact_time"] = info.get(leg + "_first_recontact_time")
            row[leg + "_recontact_impulse_z"] = info.get(leg + "_recontact_impulse_z")
            row[leg + "_min_calf_clearance"] = float(
                frame[leg + "_calf_ground_clearance"].min()
            )
        records.append(row)
        label = path.stem.split("_standing")[0]
        axes[0].plot(frame.time, np.rad2deg(frame.pitch), label=label)
        axes[1].plot(frame.time, frame.FL_body_x, label=label + " body FL")
        axes[1].plot(frame.time, frame.FL_world_x, ls="--", label=label + " world FL")
        axes[2].plot(frame.time, frame.front_moment_com, label=label)
        axes[3].plot(
            frame.time, frame.FL_calf_ground_clearance, label=label + " FL calf"
        )
        axes[4].plot(frame.time, frame.L_com, label=label)
    pd.DataFrame(records).to_csv(
        args.directory / "mechanics_comparison.csv", index=False
    )
    for ax, title in zip(
        axes,
        [
            "Pitch (deg)",
            "Front foot x: body solid / world dashed (m)",
            "Front-contact pitch moment about COM (Nm)",
            "Front calf ground clearance (m)",
            "Whole-body COM angular momentum (kg m²/s)",
        ],
    ):
        ax.set_ylabel(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Simulation time (s); same standing seed, nominal actuators")
    fig.tight_layout()
    fig.savefig(args.directory / "mechanics_comparison.png")
    fig.savefig(args.directory / "mechanics_comparison.pdf")


if __name__ == "__main__":
    main()
