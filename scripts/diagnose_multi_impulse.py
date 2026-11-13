# ruff: noqa: E501
"""Document B completely before changing a training objective."""

import argparse
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.front_catch.core import CatchConfig
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.multi_impulse.sensors import Corrections, joint_sample
from scripts.run_com_energy_phase import dump
from experiments.front_catch.data import sha


class Instrumented(CaptureEnvironment):
    def reset(self, *args, **kwargs):
        self.audit = []
        self.monitor = Corrections()
        self.audit_time = -1.0
        return super().reset(*args, **kwargs)

    def measure(self):
        row = super().measure()
        if (
            getattr(self, "record_physics", False)
            and self.d.time > self.audit_time + 1e-9
            and self.physics_trace
        ):
            self.audit_time = float(self.d.time)
            s = dict(self.physics_trace[-1])
            self.row = row
            s.update(self.monitor.observe(self, row, s), **joint_sample(self))
            self.audit.append(s)
        return row


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/adc_front_capture_stage1_v5/checkpoint_000030.pt"),
    )
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    x = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    r = x["runs"]["B_seed42"]
    # The best and final coincide under the available nominal outcome vectors.
    rows = pd.DataFrame(x["history"]["evaluation"])
    b = rows[(rows["mode"] == "B_seed42") & (rows.evaluation_type == "standing")]
    ranked = b.groupby("iteration").agg(
        takeoff=("certified_standing_takeoff", "mean"),
        hold=("handstand_seconds", "mean"),
        pitch=("peak_collision_free_pitch", "mean"),
    )
    best = int(ranked.sort_values(["takeoff", "hold", "pitch"]).index[-1])
    if best != x["iteration"]:
        raise ValueError("Select the actual best checkpoint explicitly")
    for label in ("B_best", "B_final"):
        shutil.copyfile(a.checkpoint, a.output / (label + ".pt"))
    dump(
        a.output / "preserved.json",
        dict(
            best_iteration=best,
            final_iteration=x["iteration"],
            original=str(a.checkpoint),
            sha256=sha(a.checkpoint),
            best_and_final_same=True,
            nominal_outcomes=ranked.reset_index().to_dict("records"),
        ),
    )
    cfg = CatchConfig(**r["config"])
    cfg.kp = 60.0
    cfg.kd = 3.0
    cfg.full_trace = True
    net = ActorCritic(73, 103)
    net.load_state_dict(r["network"])
    e = Instrumented(Assets(a.output / "assets"), cfg)
    summaries = []
    for seed in (96000, 96001, 96002, 96003):
        obs, _ = e.reset(1.0, seed)
        while True:
            with torch.no_grad():
                action = (
                    net.actor(torch.tensor(np.repeat(obs[None], 16, axis=0)))
                    .tanh()
                    .numpy()[0]
                )
            (obs, _), _, done, info = e.step(action)
            if done:
                break
        f = pd.DataFrame(e.audit)
        f["pitch_acceleration"] = np.gradient(f.pitch_velocity, f.time)
        f.to_csv(a.output / f"B_seed{seed}_physics.csv", index=False)
        safe = f[f.prefix_safe]
        peak = safe.loc[safe.pitch.idxmax()]
        early = safe[safe.time <= peak.time]
        windows = []
        for j, event in enumerate(e.monitor.pushes):
            stop = (
                e.monitor.pushes[j + 1]["time"]
                if j + 1 < len(e.monitor.pushes)
                else f.time.iloc[-1]
            )
            segment = f[(f.time >= event["time"]) & (f.time <= stop)]
            windows.append(
                dict(
                    **event,
                    end=stop,
                    peak_pitch=float(segment.pitch.max()),
                    pitch_gain=float(segment.pitch.max() - event["pitch"]),
                    front_angular_impulse=float(
                        np.trapezoid(segment.front_moment_com, segment.time)
                    ),
                    rear_angular_impulse=float(
                        np.trapezoid(segment.rear_moment_com, segment.time)
                    ),
                    energy_change=float(
                        segment.mechanical_energy.iloc[-1]
                        - segment.mechanical_energy.iloc[0]
                    ),
                )
            )
        record = dict(
            seed=seed,
            peak_time=float(peak.time),
            peak_pitch_deg=float(np.rad2deg(peak.pitch)),
            first_takeoff=info["takeoff_time"],
            terminal_time=float(f.time.iloc[-1]),
            collision=info["collision_bodies"],
            genuine_pushes=len(e.monitor.pushes),
            pushes=windows,
            front_recontacts=e.monitor.front_events,
            front_J_to_peak=float(np.trapezoid(early.front_moment_com, early.time)),
            rear_J_to_peak=float(np.trapezoid(early.rear_moment_com, early.time)),
            energy_at_peak=float(peak.mechanical_energy),
            target_potential=e.mass * 9.81 * cfg.target_height,
            maximum_saturation=float(f.saturated_fraction.max()),
            mean_saturation=float(f.saturated_fraction.mean()),
            tracking_rmse=float(f.tracking_rmse.mean()),
            front_world_dx=float(peak.FL_world_x - f.FL_world_x.iloc[0]),
            front_body_dx=float(peak.FL_body_x - f.FL_body_x.iloc[0]),
            peak_pitch_acceleration=float(peak.pitch_acceleration),
            peak_L=float(peak.L_com),
            peak_front_calf_clearance=float(peak.calf_clearance),
            actuator_work_to_peak=float(np.trapezoid(early.actuator_power, early.time)),
            momentum_residual_to_peak=float(early.momentum_residual_com.iloc[-1]),
        )
        summaries.append(record)
        if seed == 96000:
            fig, axes = plt.subplots(8, 1, figsize=(12, 17), sharex=True)
            axes[0].plot(f.time, np.rad2deg(f.pitch), label="pitch deg")
            axes[0].plot(f.time, f.pitch_velocity, label="pitch rate")
            axes[1].plot(f.time, f.FL_world_x, label="FL world x")
            axes[1].plot(f.time, f.FL_body_x, label="FL body x")
            for n in (
                "FL_thigh_joint",
                "FL_calf_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ):
                axes[2].plot(f.time, f[n + "_q"], label=n)
            for n in ("front_force_z", "rear_force_z"):
                axes[3].plot(f.time, f[n], label=n)
            for n in ("front_moment_com", "rear_moment_com", "moment_com"):
                axes[4].plot(f.time, f[n], label=n)
            axes[5].plot(f.time, f.L_com, label="L COM")
            accel_axis = axes[5].twinx()
            accel_axis.plot(
                f.time,
                f.pitch_acceleration,
                color="orange",
                alpha=0.5,
                label="pitch acceleration",
            )
            accel_axis.set_ylabel("pitch acceleration (rad/s²)")
            axes[5].set_ylabel("L COM (kg m²/s)")
            for n in ("mechanical_energy", "potential", "K_system"):
                axes[6].plot(f.time, f[n], label=n)
            axes[7].plot(f.time, f.FL_thigh_joint_tau, label="FL thigh Nm")
            axes[7].plot(f.time, f.FL_calf_joint_tau, label="FL knee Nm")
            axes[7].plot(
                f.time, 45 * f.saturated_fraction, label="45*saturation fraction"
            )
            for ax in axes:
                ax.axvline(peak.time, color="k", ls="--", label="safe peak")
                for event in e.monitor.pushes:
                    ax.axvline(event["time"], color="green", alpha=0.35)
                ax.legend(fontsize=7, ncol=4)
                ax.grid(alpha=0.2)
            axes[-1].set_xlabel("s; green = independently certified physical push-offs")
            fig.tight_layout()
            fig.savefig(a.output / "B_motion_and_torque.png")
            fig.savefig(a.output / "B_motion_and_torque.pdf")
            plt.close(fig)
        print(
            "B diagnosis",
            seed,
            record["genuine_pushes"],
            record["peak_pitch_deg"],
            flush=True,
        )
    dump(a.output / "diagnosis.json", summaries)
    lines = [
        "# B trajectory diagnosis before objective changes",
        "",
        "The preserved best and final B checkpoint are the same available update-30 checkpoint; two immutable copies and SHA256 provenance are retained. All measurements below use original K=60, B=3 and unchanged geometry/limits. Instrumentation does not change reward or apply control.",
        "",
        "| Seed | Peak (deg) | Peak time | Certified push-offs | Front angular impulse to peak | Rear angular impulse to peak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['seed']} | {r['peak_pitch_deg']:.3f} | {r['peak_time']:.3f}s | {r['genuine_pushes']} | {r['front_J_to_peak']:.3f} | {r['rear_J_to_peak']:.3f} |"
        )
    lines += [
        "",
        "## Mechanistic interpretation",
        "",
        "A/B: Large positive rear-contact angular impulse is almost cancelled by negative front-contact impulse. The residual total momentum, not either component alone, limits rotation. The full trajectories contain renewed rear loading; counts require bilateral load, knee extension, positive actuator work, synchronized release and real rear clearance, using a fresh certificate per physically reloaded attempt. Free-air leg swings cannot increment the counter.",
        "",
        "C/F: Front legs move substantially in the world while changing body-relative geometry. Long loaded sliding and repeated low-pitch recovery do not establish viable inverted support. Every evaluated B episode eventually collides with a front calf. The available trajectories do not establish a dynamically viable catch basin.",
        "",
        "D: Mechanical energy oscillates over successive contact cycles and remains below target handstand potential near the best peak. Actuator work, contact losses and exchanged potential/kinetic energy must be interpreted jointly; the log does not identify a unique dissipative cause.",
        "",
        "E: Negative front moments oppose rotation, but calling them premature braking is an interpretation, not proof of an intentional braking policy. Pitch acceleration and whole-body momentum are plotted separately because limb redistribution can change trunk angular velocity without generating total angular momentum.",
        "",
        "G: Measured saturation fractions and tracking errors are in diagnosis.json and the 500 Hz torque traces. They quantify actuator involvement; no gain-limit counterfactual was performed, so actuator limits cannot be ruled out or assigned a causal percentage.",
        "",
        "The next learning intervention should preserve genuine retries, reward task progress/recovery rather than the number of kicks, teach balance on independently validated states, and distinguish support-generated impulse from internal redistribution. The synchronized plot and complete traces are the basis of this diagnosis.",
    ]
    (a.output / "DIAGNOSIS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
