"""Audit foot placement and impulse before unsafe/post-impact samples."""

import argparse
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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checkpoint = sorted(a.directory.glob("checkpoint_*.pt"))[-1]
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    final = saved["iteration"]
    weight = saved["protocol"]["model_mass"] * 9.81
    rows = []
    for filename in saved["history"]["trace_files"]:
        if not filename.endswith(f"_{final:06}.csv"):
            continue
        file = Path(filename)
        if not file.exists():
            file = a.directory / Path(*file.parts[file.parts.index("sessions") :])
        mode = file.stem.rsplit("_", 1)[0]
        data = pd.read_csv(file)
        for seed, episode in data[data.start_kind == "standing"].groupby(
            "evaluation_seed"
        ):
            d = episode.reset_index(drop=True)
            safe = d[
                d.prefix_safe
                & d.historical_takeoff
                & (d.rear_clearance > 0.015)
                & (d.rear_force_z < 0.04 * weight)
            ]
            if safe.empty:
                continue
            peak = safe.loc[safe.pitch.idxmax()]
            release = d[d.historical_takeoff].iloc[0]
            recontacts = d[(d.front_recontacts > 0) & d.historical_takeoff]
            push = d[
                (d.rear_positive_power > 2)
                & d.rear_loaded
                & (d.pitch_velocity > 0.2)
                & (d.pitch > 0.03)
                & (d.phase < 0.4)
                & d.prefix_safe
            ]
            metrics = dict(
                condition=mode[0],
                mode=mode,
                seed=seed,
                peak_pitch_degrees=float(np.rad2deg(peak.pitch)),
                apex_potential=peak.potential,
                apex_kinetic=peak.K_system,
                apex_capture_error=peak.capture_error,
                apex_com_error=peak.current_com_error,
                apex_backward_m=peak.backward_mean,
                release_backward_m=release.backward_mean,
                terminal_backward_m=d.backward_mean.iloc[-1],
                release_time=release.elapsed_time,
                apex_time=peak.elapsed_time,
                first_recontact_time=recontacts.elapsed_time.iloc[0]
                if len(recontacts)
                else np.nan,
                front_postlift_impulse=peak.front_impulse - release.front_impulse,
                rear_postlift_impulse=peak.rear_impulse - release.rear_impulse,
                com_momentum_change_postlift=peak.L_com - release.L_com,
                front_moment_mean_swing=safe[
                    safe.elapsed_time <= peak.elapsed_time
                ].front_moment_com.mean(),
                apex_placement_geometry_gain=peak.static_capture_error
                - peak.capture_error,
                apex_head_clearance=peak.head_clearance,
                apex_calf_clearance=peak.calf_clearance,
                sampled_max_front_force_weight_ratio=safe.front_force_weight_ratio.max(),
            )
            if len(push):
                first = max(0, int(push.index.min()) - 1)
                last = int(push.index.max())
                before = d.loc[first]
                after = d.loc[last]
                metrics.update(
                    push_start_time=before.elapsed_time,
                    push_end_time=after.elapsed_time,
                    front_active_push_impulse=after.front_impulse
                    - before.front_impulse,
                    rear_active_push_impulse=after.rear_impulse - before.rear_impulse,
                    com_momentum_gain_push=after.L_com - before.L_com,
                )
            rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(a.output / "phase_metrics.csv", index=False)
    summary = frame.groupby("condition").mean(numeric_only=True)
    summary.to_csv(a.output / "mechanism_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for condition in "ABC":
        d = frame[frame.condition == condition]
        axes[0].scatter(d.peak_pitch_degrees, d.apex_potential, label=condition)
        axes[1].scatter(d.apex_backward_m, d.apex_capture_error, label=condition)
        axes[2].scatter(
            d.front_postlift_impulse, d.com_momentum_change_postlift, label=condition
        )
    axes[0].set(
        xlabel="Collision-free apex pitch (deg)",
        ylabel="Whole-robot potential at apex (J)",
    )
    axes[1].set(
        xlabel="Backward front-foot displacement at apex (m)",
        ylabel="Capture point support distance at apex (m)",
    )
    axes[2].set(
        xlabel="Front impulse: release to apex (kg m²/s)",
        ylabel="COM momentum change: release to apex (kg m²/s)",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(a.output / f"mechanism_audit.{suffix}", dpi=150)
    print(summary.to_string())


if __name__ == "__main__":
    main()
