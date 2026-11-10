"""Summarize saved physical traces and recovery trials without rerunning physics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    args = p.parse_args()
    root = args.evaluation
    summary = json.loads((root / "summary.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    metrics = [
        ("pitch_rad", "Body pitch (degrees)", 180 / np.pi),
        ("pitch_rate", "Pitch rate (rad/s)", 1),
        ("angular_momentum", "Sagittal angular momentum (N m s)", 1),
        ("com_support_x", "COM ahead of front support (m)", 1),
        ("joint_8", "Rear left knee (rad)", 1),
        ("rear_pitch_moment_Nm", "Rear ground moment about COM (N m)", 1),
    ]
    diagnosis = {}
    for name, results in summary["comparison"].items():
        traces = []
        for trial in range(len(results)):
            with (root / f"{name}_{trial}.csv").open() as f:
                traces.append(
                    [
                        {k: float(v) for k, v in row.items() if k != "body_contact"}
                        for row in csv.DictReader(f)
                    ]
                )
        t = np.array([r["time"] for r in traces[0]])
        for ax, (key, label, scale) in zip(axes.flat, metrics):
            values = np.array([[r[key] * scale for r in trace] for trace in traces])
            (line,) = ax.plot(t, values.mean(axis=0), label=name)
            ax.fill_between(
                t,
                values.min(axis=0),
                values.max(axis=0),
                alpha=0.10,
                color=line.get_color(),
            )
            ax.set(xlabel="Time (s)", ylabel=label)
            ax.grid(alpha=0.2)
        per_trial = []
        for trace in traces:
            peak = max(trace, key=lambda r: r["pitch_rad"])

            def onset(joint, sign, start=0.75):
                streak = 0
                for r in trace:
                    ok = (
                        r["time"] >= start
                        and sign * r[f"joint_velocity_{joint}"] >= 0.3
                    )
                    streak = streak + 1 if ok else 0
                    if streak >= 10:
                        return r["time"] - 0.018
                return None

            extension = onset(8, 1)

            per_trial.append(
                dict(
                    peak_time=peak["time"],
                    pitch_velocity_at_peak=peak["pitch_rate"],
                    momentum_at_peak=peak["angular_momentum"],
                    front_hip_at_peak=peak["joint_1"],
                    rear_hip_at_peak=peak["joint_7"],
                    com_support_x_at_peak=peak["com_support_x"],
                    rear_extension_onset_s=extension,
                    rear_retraction_onset_s=onset(8, -1, (extension or 0.75) + 0.06),
                    front_pitch_angular_impulse_Nms=sum(
                        r["front_pitch_moment_Nm"]
                        for r in trace
                        if 0.7 <= r["time"] <= 1.2
                    )
                    * 0.002,
                    rear_pitch_angular_impulse_Nms=sum(
                        r["rear_pitch_moment_Nm"]
                        for r in trace
                        if 0.7 <= r["time"] <= 1.2
                    )
                    * 0.002,
                )
            )
        diagnosis[name] = per_trial
    axes[0, 0].axhspan(80, 100, alpha=0.08, color="green")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(root / "trajectories.png", dpi=160)
    plt.close(fig)
    (root / "timing_diagnosis.json").write_text(json.dumps(diagnosis, indent=2))
    lines = [
        "# Frozen-policy handoff comparison",
        "",
        "16 matched starts per controller. Pitch velocity is measured at 1.6 s. "
        "Hold requires five continuous seconds; a gate crossing alone is not success.",
        "",
        "| Controller | Peak pitch | Terminal pitch | Terminal rate | "
        "Rear clearance | Full transition | Best forced hold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, rows in summary["comparison"].items():

        def mean(key):
            return np.mean([r[key] for r in rows])

        best = max(r["hold_s"] for a in summary["recovery"][name] for r in a["trials"])
        success = summary["supervisors"][name]["full_transition_successes"]
        lines.append(
            f"| {name} | {mean('maximum_pitch_deg'):.2f}° | "
            f"{mean('terminal_pitch_deg'):.2f}° | "
            f"{mean('terminal_pitch_velocity'):.3f} rad/s | "
            f"{mean('rear_clearance_m'):.3f} m | {success}/16 | {best:.2f} s |"
        )
    lines += [
        "",
        "![Physical trajectory traces](trajectories.png)",
        "",
        "Shading shows the range across 16 starts. Distance in summary.json is a "
        "scaled feature distance to empirical successful states, "
        "not a recovery probability.",
        "",
        "## Recovery probes",
        "",
        "| Region | Success | 95% Wilson interval |",
        "|---|---:|---:|",
    ]
    for a in summary["atlas_successes"]:
        lo, hi = a["wilson95"]
        lines.append(
            f"| {a['label']} | {a['successes']}/{a['count']} | {lo:.3f}–{hi:.3f} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
