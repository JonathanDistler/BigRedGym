"""Measured dynamics figures and labeled pendulum predictions."""

import argparse
import json
import hashlib
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.morphology.dynamics_audit import pendulum_acceleration
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
KEYS = ["scale", "evaluation", "family", "mode", "seed", "duration", "knee"]


def save(fig, path, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def pendulum_audit(traces):
    rows = []
    summary = []
    for keys, f in traces.groupby(KEYS, sort=False):
        f = f.sort_values("time").copy()
        if len(f) < 5:
            continue
        t = f.time.to_numpy()
        # Fixed front support pivot from the start of each trajectory.
        dx = f.com_x.to_numpy() - f.support_x.iloc[0]
        z = f.com_z.to_numpy()
        length = np.hypot(dx, z)
        theta = np.unwrap(np.arctan2(-dx, -z))
        theta_dot = np.gradient(theta, t)
        theta_ddot = np.gradient(theta_dot, t)
        ldot = np.gradient(length, t)
        gravity = -9.81 / length * np.sin(theta)
        radial = -2 * ldot / length * theta_dot
        valid = (
            f.front_support
            & (f.load_RL + f.load_RR < 0.04 * f.mass * 9.81)
            & (f.illegal_contacts == 0)
            & (f.front_drift < 0.02)
        ).to_numpy()
        # Start a conditional integration at each eligible interval. Measured
        # L(t) is prescribed; this is not a controller or a forward morphology forecast.
        predicted = np.full(len(f), np.nan)
        running = False
        v = 0.0
        for i in range(len(f)):
            if not valid[i]:
                running = False
                continue
            if not running:
                predicted[i] = theta[i]
                v = theta_dot[i]
                running = True
            else:
                dt = t[i] - t[i - 1]
                acceleration = pendulum_acceleration(
                    predicted[i - 1], v, length[i - 1], ldot[i - 1]
                )
                v += dt * acceleration
                predicted[i] = predicted[i - 1] + dt * v
        f["pendulum_length_fixed_pivot"] = length
        f["pendulum_theta_downward"] = theta
        f["pendulum_theta_dot"] = theta_dot
        f["measured_theta_ddot"] = theta_ddot
        f["analytical_gravity_acceleration"] = gravity
        f["analytical_radial_acceleration"] = radial
        f["analytical_theta_conditional"] = predicted
        f["pendulum_model_eligible"] = valid
        f["pendulum_equation_residual"] = theta_ddot - gravity - radial
        # Derivatives around contact transitions are not suitable model evidence.
        inner = valid.copy()
        inner[:2] = False
        inner[-2:] = False
        for lag in (-2, -1, 1, 2):
            inner &= np.roll(valid, lag)
        summary.append(
            dict(
                zip(KEYS, keys),
                eligible_seconds=float(valid.sum() * 0.002),
                interior_samples=int(inner.sum()),
                residual_rms=float(
                    np.sqrt(np.mean((theta_ddot - gravity - radial)[inner] ** 2))
                )
                if inner.any()
                else "",
                conditional_theta_rmse=float(
                    np.sqrt(np.mean((predicted[inner] - theta[inner]) ** 2))
                )
                if inner.any()
                else "",
            )
        )
        rows.append(f)
    return pd.concat(rows), pd.DataFrame(summary)


def grouped_plot(ax, frame, y, label):
    for (family, mode), f in frame.groupby(["family", "mode"]):
        # Mean within seed over the same command grid, then seed variation.
        per = f.groupby(["seed", "scale"])[y].mean().reset_index()
        stats = per.groupby("scale")[y].agg(["mean", "std", "count"])
        x = stats.index.to_numpy()
        mean = stats["mean"].to_numpy()
        ax.plot(x, mean, "o-", label=f"{family}, {mode}")
        if (stats["count"] > 1).any():
            sd = stats["std"].fillna(0).to_numpy()
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.15)
    ax.set(xlabel="Physical leg scale", ylabel=label)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--posture", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    geometry = pd.read_csv(a.input / "geometry.csv")
    takeoff = pd.read_csv(a.input / "takeoff.csv")
    balance = pd.read_csv(a.input / "balance.csv")
    old = pd.read_csv(a.input / "legacy_contact_diagnosis.csv")
    paths = pd.read_csv(a.input / "path_checks.csv")
    traces = pd.read_csv(a.input / "trajectories.csv")
    for y, label, name in [
        (
            "pendulum_length",
            "Effective COM-to-front-support length (m)",
            "01_effective_length",
        ),
        ("pitch_inertia", "Whole-body COM pitch inertia (kg m²)", "02_pitch_inertia"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3))
        for family, f in geometry.groupby("family"):
            f = f.sort_values("scale")
            ax.plot(f.scale, f[y], "o-", label=family)
        ax.set(xlabel="Physical thigh/calf scale", ylabel=label)
        ax.legend()
        save(fig, a.output, name)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for family, f in geometry.groupby("family"):
        f = f.sort_values("scale")
        axes[0].plot(
            f.scale,
            f.measured_ik_path_barrier,
            "o-",
            label=f"{family}: articulated IK path",
        )
        axes[0].plot(
            f.scale,
            f.analytical_frozen_barrier,
            "s--",
            label=f"{family}: frozen pendulum prediction",
        )
        axes[1].plot(f.scale, f.measured_target_delta_U, "o-", label=family)
    axes[0].set(xlabel="Physical scale", ylabel="Potential-energy barrier (J)")
    axes[1].set(
        xlabel="Physical scale", ylabel="IK target minus standing potential (J)"
    )
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=7)
    save(fig, a.output, "03_energy_barrier")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    grouped_plot(axes[0], takeoff, "max_pitch", "Maximum body pitch (rad)")
    grouped_plot(
        axes[1],
        takeoff,
        "max_supported_pitch",
        "Maximum pitch with both front feet loaded (rad)",
    )
    axes[0].legend(fontsize=6)
    save(fig, a.output, "04_maximum_pitch")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    grouped_plot(
        axes[0],
        takeoff,
        "max_rear_clearance",
        "Maximum minimum rear-foot clearance (m)",
    )
    grouped_plot(axes[1], takeoff, "first_contact_time", "First nonfoot contact (s)")
    grouped_plot(
        axes[2],
        takeoff,
        "supported_airborne_seconds",
        "Longest valid supported takeoff (s)",
    )
    axes[0].legend(fontsize=6)
    save(fig, a.output, "05_clearance_and_contact_timing")
    summary = takeoff.groupby(["family", "mode", "scale"]).agg(
        attempts=("takeoff", "size"),
        takeoff_rate=("takeoff", "mean"),
        max_pitch=("max_pitch", "max"),
        mean_pitch=("max_pitch", "mean"),
        supported_pitch=("max_supported_pitch", "max"),
        rear_clearance=("max_rear_clearance", "max"),
        push_impulse=("push_impulse", "mean"),
        rear_angular_impulse=("rear_angular_impulse", "mean"),
        external_angular_impulse=("external_angular_impulse", "mean"),
        positive_work=("positive_work", "mean"),
        saturation=("saturation", "mean"),
        speed_ratio=("measured_speed_ratio", "max"),
        mean_contact_time=("first_contact_time", "mean"),
    )
    summary.to_csv(a.output / "takeoff_summary.csv")
    balance.groupby(["mode", "scale"]).agg(
        attempts=("success", "size"),
        success=("success", "mean"),
        longest_hold=("best_hold", "max"),
        mean_seconds=("seconds", "mean"),
        max_com_speed=("max_com_speed", "max"),
    ).to_csv(a.output / "balance_summary.csv")
    old.groupby("scale").agg(
        first_contact_min=("first_contact_time", "min"),
        first_contact_max=("first_contact_time", "max"),
        reset_clearance=("reset_rear_clearance", "min"),
        max_pitch=("max_pitch", "max"),
    ).to_csv(a.output / "legacy_summary.csv")
    repeated_join = paths.duplicated(["scale", "family", "fraction"])
    paths["path_kind"] = np.where(
        (paths.fraction > 1) | (paths.fraction.eq(1) & repeated_join),
        "front_pinned_rotation",
        "four_foot_crouch",
    )
    paths.to_csv(a.output / "path_checks_labeled.csv", index=False)
    paths.groupby(["path_kind", "family", "scale"]).agg(
        ik_error=("ik_error", "max"),
        min_clearance=("nonfoot_clearance", "min"),
        speed_ratio=("command_speed_ratio", "max"),
        static_torque_ratio=("static_torque_ratio", "max"),
        static_residual=("static_base_residual", "max"),
    ).to_csv(a.output / "ik_path_summary.csv")
    geometry.to_csv(a.output / "geometry.csv", index=False)
    # Explicit representatives: best supported takeoff and a failed attempt for
    # each scale, chosen using a predeclared lexicographic outcome rule.
    selected = []
    for scale, f in takeoff.groupby("scale"):
        best = f.sort_values(
            ["supported_airborne_seconds", "max_supported_pitch"], ascending=False
        ).iloc[0]
        failed = (
            f[~f.takeoff].sort_values("max_pitch").iloc[0]
            if (~f.takeoff).any()
            else None
        )
        for label, chosen in [("best", best), ("failed", failed)]:
            if chosen is None:
                continue
            match = np.ones(len(traces), dtype=bool)
            for key in KEYS:
                match &= traces[key] == chosen[key]
            t = traces[match].copy()
            t["selection"] = label
            selected.append(t)
    representatives = pd.concat(selected)
    representatives.to_csv(a.output / "representative_trajectories.csv", index=False)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for ax, (y, label) in zip(
        axes.flat,
        [
            ("pitch", "Body pitch (rad)"),
            ("pitch_velocity", "Pitch angular velocity (rad/s)"),
            ("com_z", "Whole-body COM height (m)"),
            ("rear_clearance", "Rear-foot clearance (m)"),
            ("rear_calf_clearance", "Rear-calf ground clearance (m)"),
            ("front_support", "Both front feet loaded"),
        ],
    ):
        for (scale, label2), f in representatives.groupby(["scale", "selection"]):
            ax.plot(
                f.time,
                f[y],
                linestyle="-" if label2 == "best" else "--",
                label=f"{scale:.2f} {label2}",
            )
        ax.set(xlabel="Time (s)", ylabel=label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Supported takeoff and failed takeoff; no completed handstand", y=1.02)
    save(fig, a.output, "06_representative_attempts")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (scale, f) in zip(
        axes, representatives[representatives.selection == "best"].groupby("scale")
    ):
        for leg in ("FL", "FR", "RL", "RR"):
            ax.plot(f.time, f["load_" + leg] / (f.mass * 9.81), label=leg)
        ax.set(
            xlabel="Time (s)",
            ylabel="Foot normal load / body weight",
            title=f"Scale {scale:.2f}",
        )
    axes[0].legend()
    save(fig, a.output, "09_foot_contacts")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for row_index, mode in enumerate(sorted(balance["mode"].unique())):
        for ax, scale in zip(axes[row_index], sorted(balance.scale.unique())):
            f = balance[(balance["mode"] == mode) & (balance.scale == scale)]
            table = f.pivot(index="knee", columns="duration", values="best_hold")
            im = ax.imshow(
                table.to_numpy(),
                origin="lower",
                vmin=0,
                vmax=balance.best_hold.max(),
                aspect="auto",
            )
            ax.set(
                xticks=range(3),
                xticklabels=table.columns,
                yticks=range(3),
                yticklabels=table.index,
                xlabel="Initial pitch velocity (rad/s)",
                ylabel="Initial body pitch (deg)",
                title=f"{scale:.2f}: {mode}",
            )
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        label="Longest valid handstand interval (s)",
        shrink=0.8,
        location="right",
    )
    # Manually reserve colorbar space; normal tight layout overlaps heatmaps.
    fig.subplots_adjust(right=0.84, hspace=0.65, wspace=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(a.output / f"10_balance_sensitivity.{ext}", bbox_inches="tight")
    plt.close(fig)
    contact_timings = []
    for keys, f in traces[traces.evaluation == "takeoff"].groupby(KEYS):
        record = dict(zip(KEYS, keys))
        for leg in ("RL", "RR"):
            loaded = (f["load_" + leg] > 0.03 * f.mass * 9.81).cummax()
            released = f[
                (f.time >= 0.45) & loaded & (f["load_" + leg] < 0.02 * f.mass * 9.81)
            ]
            record[leg + "_release_time"] = (
                released.time.iloc[0] if len(released) else ""
            )
        contact_timings.append(record)
    pd.DataFrame(contact_timings).to_csv(
        a.output / "rear_contact_timing.csv", index=False
    )
    computed, model_summary = pendulum_audit(traces)
    computed.to_csv(a.output / "pendulum_trajectories.csv", index=False)
    model_summary.to_csv(a.output / "pendulum_validation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for scale in sorted(takeoff.scale.unique()):
        f = computed[
            (computed.scale == scale)
            & (computed.evaluation == "takeoff")
            & (computed.family == "relative_extension")
            & (computed["mode"] == "pitch_feedback")
            & (computed.seed == 0)
            & np.isclose(computed.duration, 0.24)
            & np.isclose(computed.knee, -1.8)
        ]
        axes[0].plot(
            f.time, f.pendulum_theta_downward, label=f"{scale:.2f} measured COM angle"
        )
        axes[0].plot(
            f.time,
            f.analytical_theta_conditional,
            "--",
            label=f"{scale:.2f} conditional model",
        )
        axes[1].plot(f.time, f.pendulum_length_fixed_pivot, label=f"{scale:.2f}")
    axes[0].set(xlabel="Time (s)", ylabel="Pendulum angle from downward vertical (rad)")
    axes[1].set(xlabel="Time (s)", ylabel="Measured L(t), fixed front pivot (m)")
    axes[0].legend(fontsize=6)
    axes[1].legend()
    save(fig, a.output, "07_pendulum_validation")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, y in zip(axes, ("push_impulse", "positive_work", "saturation")):
        grouped_plot(ax, takeoff, y, y.replace("_", " "))
    axes[0].legend(fontsize=6)
    save(fig, a.output, "08_impulse_work_saturation")
    if a.posture:
        pg = pd.read_csv(a.posture / "geometry.csv")
        pt = pd.read_csv(a.posture / "takeoff.csv")
        pg.to_csv(a.output / "posture_geometry.csv", index=False)
        stats = pt.groupby("offset").agg(
            takeoff_rate=("takeoff", "mean"),
            mean_pitch=("max_pitch", "mean"),
            mean_clearance=("max_rear_clearance", "mean"),
            supported_airborne=("supported_airborne_seconds", "mean"),
            push_impulse=("push_impulse", "mean"),
            saturation=("saturation", "mean"),
        )
        stats.to_csv(a.output / "posture_summary.csv")
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        axes[0].plot(pg.offset, pg.pendulum_length, "o-")
        axes[0].set(ylabel="Effective pendulum length (m)")
        axes[1].plot(
            pg.offset, pg.frozen_barrier, "s--", label="Frozen pendulum prediction"
        )
        axes[1].plot(
            pg.offset,
            pg.target_delta_U,
            "o-",
            label="Articulated target energy difference",
        )
        axes[1].set(ylabel="Energy (J)")
        axes[1].legend(fontsize=6)
        axes[2].plot(
            stats.index,
            stats.mean_pitch * 180 / np.pi,
            "o-",
            label="Mean maximum pitch",
        )
        axes[2].set(ylabel="Mean maximum pitch (deg)")
        for ax in axes:
            ax.set_xlabel("Standing COM shift command x (m)")
        save(fig, a.output, "11_full_size_posture_ablation")
    manifest = dict(
        takeoff_attempts=len(takeoff),
        balance_attempts=len(balance),
        successful_takeoffs=int(takeoff.takeoff.sum()),
        five_second_balance=int(balance.success.sum()),
        representative_rule=(
            "best by supported airborne duration then supported pitch; failed "
            "by lowest maximum pitch"
        ),
        interpretation=(
            "Takeoff means >=20ms both rear feet unloaded and >15mm clear with "
            "both front feet loaded. It does not mean a complete handstand."
        ),
        analytical_model=(
            "theta_ddot=-(g/L)sin(theta)-2(Ldot/L)theta_dot; theta from "
            "downward vertical, fixed front pivot; no ankle torque. "
            "Conditional integration prescribes measured L(t)."
        ),
        uncertainty=(
            "Bands: std across paired initial-condition seeds after averaging "
            "command-grid conditions; these are not independent training "
            "seeds."
        ),
        missing_successful_complete_kickup=not bool((takeoff.success).any()),
    )
    manifest["input_sha256"] = {
        name: hashlib.sha256((a.input / name).read_bytes()).hexdigest()
        for name in (
            "geometry.csv",
            "takeoff.csv",
            "balance.csv",
            "trajectories.csv",
            "legacy_contact_diagnosis.csv",
            "path_checks.csv",
        )
    }
    manifest["actual_saved_physics_samples"] = {
        "takeoff": int((traces.evaluation == "takeoff").sum()),
        "balance": int((traces.evaluation == "balance").sum()),
    }
    manifest["posture_input"] = str(a.posture) if a.posture else None
    shutil.copyfile(__file__, a.output / "report_source.py")
    (a.output / "report.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
