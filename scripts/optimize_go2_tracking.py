"""Fit/evaluate a model-derived handstand trajectory tracker; no PPO or SDK."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.trajectory_optimization.feedback import fit, TrajectoryLQR
from experiments.trajectory_optimization.staged import evaluate, Preparation
from gym.envs.go2.handstand_replay import load_trajectory, build_model, STATE_SPEC
from experiments.trajectory_optimization.model import REFERENCE_SHA, Robot
import mujoco


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--feedback", type=Path)
    p.add_argument("--gain-scale", type=float, default=0.4)
    p.add_argument("--start-step", type=int, default=110)
    p.add_argument("--spacing", type=int, default=10)
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--evaluation-seed", type=int, default=301000)
    p.add_argument("--gyro-noise", type=float, default=0.0)
    p.add_argument("--adapter-dry-run", action="store_true")
    p.add_argument("--direct-crouch", action="store_true")
    p.add_argument("--source-start-step", type=int, default=0)
    p.add_argument("--video", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--preparation", type=Path)
    p.add_argument("--preparation-duration", type=float, default=3.0)
    p.add_argument("--preparation-settle", type=float, default=0.5)
    a = p.parse_args()
    if a.source_start_step and not a.direct_crouch:
        p.error("Source phase search is available only for direct crouch probes")
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    if (
        a.trials < 0
        or not np.isfinite([a.gain_scale, a.gyro_noise]).all()
        or a.gyro_noise < 0
    ):
        raise ValueError("Invalid trial/noise/gain settings")
    a.output.mkdir(parents=True)
    if a.feedback:
        with np.load(a.feedback, allow_pickle=False) as f:
            base_gains = (
                f["base_gains"].copy() if "base_gains" in f else f["gains"].copy()
            )
            policy = TrajectoryLQR(
                f["actions"], f["centers"], base_gains * a.gain_scale
            )
        report = {
            "kind": "native_trajectory_lqr_v1",
            "feedback_sha256": hashlib.sha256(a.feedback.read_bytes()).hexdigest(),
        }
    else:
        policy, report = fit(a.reference, a.start_step, a.spacing, a.reduced)
        base_gains = policy.gains.numpy().copy()
        policy.gains.mul_(a.gain_scale)
    torch.jit.save(torch.jit.script(policy), str(a.output / "policy.pt"))
    np.savez_compressed(
        a.output / "feedback.npz",
        actions=policy.actions.numpy(),
        centers=policy.centers.numpy(),
        gains=policy.gains.numpy(),
        base_gains=base_gains,
    )
    report["arguments"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
    }
    report["source_hashes"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__),
            Path("experiments/trajectory_optimization/staged.py"),
            Path("experiments/trajectory_optimization/feedback.py"),
        ]
    }
    ref = load_trajectory(a.reference, REFERENCE_SHA, build_model())
    prep = None
    parameters = None
    if a.preparation or a.direct_crouch:
        prep = Preparation(
            a.reference, a.preparation_duration, settle=a.preparation_settle
        )
        if a.preparation:
            with np.load(a.preparation, allow_pickle=False) as f:
                parameters = f["parameters"].copy()
        else:
            parameters = np.zeros(prep.parameter_count)
    rows = []
    for i in range(a.trials + 1):
        factory = None
        if a.video or a.viewer:
            from experiments.trajectory_optimization.presentation import Presentation

            video = a.output / f"trial_{i:03d}.mp4" if a.video else None

            def factory(robot):
                return Presentation(robot, viewer=a.viewer, video=video)

        r = evaluate(
            ref,
            prep,
            parameters,
            a.evaluation_seed + i,
            i > 0,
            policy=policy,
            sensor_noise=a.gyro_noise,
            presentation=factory,
            adapter_dry_run=a.adapter_dry_run,
            source_start_step=a.source_start_step,
            skip_preparation=a.direct_crouch,
        )
        if a.video:
            label = (
                "cancelled"
                if r["cancelled"]
                else "success"
                if r["complete_success"]
                else "failure"
            )
            video.rename(video.with_stem(video.stem + "_" + label))
        rows.append(r)
        report["rows"] = rows
        report["heldout_successes"] = sum(x["complete_success"] for x in rows[1:])
        report["heldout_trials"] = len(rows) - 1
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if r["cancelled"]:
            break
        print(
            i,
            r["complete_success"],
            r["strict_hold_seconds"],
            r["first_illegal_contact_s"],
            flush=True,
        )
    if prep is None and rows[0]["complete_success"]:
        robot = Robot()
        d = mujoco.MjData(robot.m)
        mujoco.mj_setState(robot.m, d, ref["initial_state"], STATE_SPEC)
        from go2_deploy.custom_implementation.go2trajectory_lqr import SLICES

        meta = dict(
            kind="native_trajectory_lqr_v1",
            task_name="go2trajectory_lqr",
            custom_implementation="go2trajectory_lqr.py",
            obs_vector=list(SLICES),
            obs_sizes={k: b - a for k, (a, b) in SLICES.items()},
            scaling={k: 1.0 for k in SLICES} | {"dof_pos_target": 2.0},
            default_dof_pos=ref["nominal"].tolist(),
            phase_frequency=0.0,
            kp=60.0,
            kd=3.0,
            ctrl_freq=50,
            exp_moving_avg=False,
            num_obs=43,
            num_actions=12,
            command_limits={"lin_vel_x": 0.0, "lin_vel_y": 0.0, "yaw_vel": 0.0},
            joint_bounds=robot.bounds.tolist(),
            initial_q=d.qpos.tolist(),
            initial_v=d.qvel.tolist(),
            policy_sha256=hashlib.sha256(
                (a.output / "policy.pt").read_bytes()
            ).hexdigest(),
            reference_sha256=REFERENCE_SHA,
            nominal_validation_pass=True,
            simulated_actuator_model="original_URDF_effort_clip_and_overspeed",
            constrained_mesh_certificate=False,
            provisional_torque_curve_calibrated=False,
            max_provisional_torque_curve_excess_Nm=rows[0][
                "max_provisional_torque_curve_excess_Nm"
            ],
            runner_contract="Cornell PR26 optional callbacks",
        )
        (a.output / "deploy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        calibration = {
            "source": "model_and_ta_gyro_estimate",
            "calibrated": False,
            "imu": {
                "body_from_imu": np.eye(3).tolist(),
                "world_alignment": np.eye(3).tolist(),
                "gyro_bias_imu": [0.0, -0.009, 0.0],
            },
            "observation_note": "No COM input; confirm IMU frame and gyro bias",
        }
        (a.output / "calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
