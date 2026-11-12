"""Create detailed figures and causal-event summaries from completed diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    out = a.directory
    rows = json.loads((out / "comparison.json").read_text())
    summary = []
    events = []
    for label in sorted(set(r["label"] for r in rows)):
        for mode in (False, True):
            rs = [r for r in rows if r["label"] == label and r["stochastic"] == mode]
            summary.append(
                dict(
                    label=label,
                    stochastic=mode,
                    **{
                        k: float(np.mean([r[k] for r in rs]))
                        for k in (
                            "takeoff",
                            "peak",
                            "loaded",
                            "certified",
                            "max_gap",
                            "slip",
                            "recovered",
                            "catch",
                            "useful_feedback",
                            "collision",
                        )
                    },
                )
            )
    for r in rows:
        stem = f"{r['label']}_{r['seed']}_{int(r['stochastic'])}"
        tr = list(csv.DictReader((out / f"{stem}_physics.csv").open()))

        def values(k):
            return np.array([float(x.get(k) or "nan") for x in tr])

        time = values("time")
        eligible = np.array([x["model_eligible"] == "True" for x in tr])
        slip = values("loaded_slip")
        pitch = values("pitch")

        def index(mask):
            return float(time[np.flatnonzero(mask)[0]]) if np.any(mask) else None

        taken = np.array([x["takeoff"] == "True" for x in tr])
        support = np.array([x["support_valid"] == "True" for x in tr])
        invalid = r["first_invalid"]
        events.append(
            dict(
                label=r["label"],
                seed=r["seed"],
                stochastic=r["stochastic"],
                peak_time=r["peak_time"],
                slip_limit_time=index(slip > 0.02),
                support_loss_time=index(taken & ~support),
                collision_time=invalid["time"] if invalid else None,
                bodies=invalid["bodies"] if invalid else "",
                collision_after_peak=bool(invalid and invalid["time"] > r["peak_time"]),
                motor_violation_time=index(
                    (values("speed_ratio") > 1.001)
                    | (values("joint_violation") >= 0.001)
                ),
                model_valid_fraction=float(eligible.mean()),
                eligible_acceleration_rmse=float(
                    np.sqrt(np.mean(values("pendulum_residual")[eligible] ** 2))
                )
                if eligible.any()
                else None,
                energy_deficit_at_peak=float(
                    values("energy_deficit")[np.argmax(pitch)]
                ),
            )
        )
        if r["seed"] != 70400:
            continue
        fig, ax = plt.subplots(7, 2, figsize=(16, 23), sharex=True)
        groups = [
            (["pitch", "pitch_velocity"], "Body rotation [rad, rad/s]"),
            (
                ["rear_clearance", "front_drift", "loaded_slip"],
                "Clearance, drift, material slip [m]",
            ),
            (["load_FL", "load_FR"], "Front vertical contact force [N]"),
            (["gap", "max_gap"], "Current / historical support gap [s]"),
            (["com_x", "com_y", "com_z"], "COM [m]"),
            (
                ["com_velocity_x", "com_velocity_y", "com_velocity_z"],
                "COM velocity [m/s]",
            ),
            ([f"torque_{i}" for i in range(12)], "Applied joint torque [Nm]"),
            ([f"joint_velocity_{i}" for i in range(12)], "Joint velocity [rad/s]"),
            (
                ["support_momentum_y", "front_moment_y", "rear_moment_y"],
                "Angular momentum / contact moment",
            ),
            (
                ["measured_pendulum_acceleration", "pendulum_acceleration"],
                "COM acceleration: prediction shown only when eligible",
            ),
            (
                ["v2_progress", "v2_velocity", "v2_slip", "v2_clearance"],
                "V2 physics-rate rewards",
            ),
            (
                [
                    "joint_1",
                    "joint_2",
                    "joint_4",
                    "joint_5",
                    "joint_7",
                    "joint_8",
                    "joint_10",
                    "joint_11",
                ],
                "Front yielding / rear retraction [rad]",
            ),
        ]
        groups.extend(
            [
                (
                    ["pendulum_omega", "desired_com_omega", "predicted_com_omega_next"],
                    "Measured / desired / one-step predicted COM rate [rad/s]",
                ),
                (
                    ["speed_ratio", "joint_violation", "saturation"],
                    "Motor speed ratio, joint violation [rad], saturation fraction",
                ),
            ]
        )
        for axis, (keys, title) in zip(ax.flat, groups):
            for k in keys:
                v = values(k)
                if k == "pendulum_acceleration":
                    v = np.where(eligible, v, np.nan)
                axis.plot(time, v, label=k)
            axis.axvline(r["peak_time"], color="k", ls=":", label="peak")
            if invalid:
                axis.axvline(invalid["time"], color="r", ls=":", label="first invalid")
            axis.set_title(title)
            axis.legend(fontsize=6, ncol=3)
            axis.grid()
        fig.tight_layout()
        fig.savefig(out / f"{stem}_detailed.png")
        plt.close(fig)
        old = list(csv.DictReader((out / f"{stem}_legacy_rewards.csv").open()))
        for x in old:
            x["reward_legacy_curriculum_extra"] = float(x["reward"]) - sum(
                float(v) for k, v in x.items() if k.startswith("reward_")
            )
        new = list(csv.DictReader((out / f"{stem}_rewards.csv").open()))
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        for rr, axis, title in [
            (old, ax[0], "Legacy curriculum"),
            (new, ax[1], "V2: same initial policy"),
        ]:
            tt = [float(x["time"]) for x in rr]
            for k in rr[0]:
                if k.startswith("reward_"):
                    axis.plot(tt, np.cumsum([float(x[k]) for x in rr]), label=k)
            axis.set_title(title + " cumulative components")
            axis.legend(fontsize=6, ncol=3)
            axis.grid()
        fig.tight_layout()
        fig.savefig(out / f"{stem}_reward_comparison.png")
        plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "events.json").write_text(json.dumps(events, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
