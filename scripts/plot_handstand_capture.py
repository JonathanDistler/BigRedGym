"""Scientific multi-page capture traces; all contact impulses are solver impulses."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    traces = {
        n: json.loads((args.input / f"{n}_trace.json").read_text())
        for n in ["reference", "prior", "hybrid", "student"]
    }
    with PdfPages(args.input / "capture_diagnostics.pdf") as pdf:
        for field, label, columns in [
            ("q", "Joint position (rad)", 12),
            ("dq", "Joint speed (rad/s)", 12),
            ("q_target", "Filtered target (rad)", 12),
            ("torque_Nm", "Applied torque (Nm)", 12),
            ("com_position_m", "COM position (m)", 3),
            ("com_velocity_m_s", "COM velocity (m/s)", 3),
            ("com_minus_front_support_m", "COM minus front support (m)", 3),
            ("calf_clearance_m", "Calf clearance (m)", 4),
            ("foot_contacts", "Foot contact", 4),
        ]:
            rows = (columns + 2) // 3
            fig, axes = plt.subplots(
                rows, 3, figsize=(14, 3 * rows), sharex=True, squeeze=False
            )
            for n, t in traces.items():
                ts = [r["time_s"] for r in t["physics"]]
                v = np.array([r[field] for r in t["physics"]])
                for j in range(columns):
                    axes.flat[j].plot(ts, v[:, j], label=n)
            for j, ax in enumerate(axes.flat):
                if j >= columns:
                    ax.set_visible(False)
                    continue
                ax.set_ylabel(f"{label} [{j}]")
                ax.set_xlim(0.8, 2.3)
                ax.grid(alpha=0.3)
                ax.axvline(0.96, color="k", linestyle=":", alpha=0.4)
                if field == "torque_Nm":
                    limit = traces["reference"]["physics"][0]["effort_limit_Nm"][j]
                    ax.axhline(limit, color="k", linestyle="--")
                    ax.axhline(-limit, color="k", linestyle="--")
            axes.flat[0].legend()
            fig.suptitle(field)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        for field, label in [
            ("action", "Normalized action error vs reference"),
            ("foot_forces_N", "Foot force (N)"),
        ]:
            fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
            for n, t in traces.items():
                if field == "action":
                    rs = t["controls"]
                    ts = [r["time_s"] for r in rs]
                    vals = np.array(
                        [np.array(r["action"]) - r["reference_action"] for r in rs]
                    )
                else:
                    rs = t["physics"]
                    ts = [r["time_s"] for r in rs]
                    vals = np.array([r[field] for r in rs]).reshape(-1, 12)
                for j, ax in enumerate(axes.flat):
                    ax.plot(ts, vals[:, j], label=n)
            for j, ax in enumerate(axes.flat):
                ax.set_ylabel(f"{label} [{j}]")
                ax.set_xlim(0.8, 2.3)
                ax.grid(alpha=0.3)
            axes.flat[0].legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        for n, t in traces.items():
            rs = t["controls"]
            ts = np.array([r["time_s"] for r in rs])
            phase = [
                r.get(
                    "phase_error_s",
                    min(r["reference_index"], 47) * 0.02 - r["time_s"]
                    if n in ["prior", "student"]
                    else 0,
                )
                for r in rs
            ]
            axes[0].plot(ts, phase, label=n)
            contacts = t["contacts"]
            axes[1].scatter(
                [r["time_s"] for r in contacts],
                [np.linalg.norm(r["impulse_on_geom2_world_Ns"]) for r in contacts],
                s=3,
                label=n,
            )
            for r in contacts:
                if r["illegal"]:
                    axes[2].scatter(
                        r["time_s"],
                        np.linalg.norm(r["impulse_on_geom2_world_Ns"]),
                        s=20,
                        label=n + " " + str(r["bodies"]),
                    )
        for ax, label in zip(
            axes,
            [
                "Effective phase error (s)",
                "All contact impulse norm (Ns)",
                "Illegal contact impulse (Ns)",
            ],
        ):
            ax.set_ylabel(label)
            ax.set_xlim(0.8, 2.3)
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("Continuous simulation time (s)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for field, label in [
            ("position_world_m", "Contact point (m)"),
            ("normal_geom1_to_geom2", "Contact normal"),
            ("impulse_on_geom2_world_Ns", "Contact impulse (Ns)"),
        ]:
            fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
            for name, trace in traces.items():
                contacts = trace["contacts"]
                for j, ax in enumerate(axes):
                    ax.scatter(
                        [r["time_s"] for r in contacts],
                        [r[field][j] for r in contacts],
                        s=3,
                        label=name,
                    )
                    ax.set_ylabel(label + " " + ["x", "y", "z"][j])
                    ax.set_xlim(0.8, 2.3)
                    ax.grid(alpha=0.3)
            axes[0].legend()
            axes[-1].set_xlabel("Continuous simulation time (s)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(args.input / "capture_diagnostics.pdf")


if __name__ == "__main__":
    main()
