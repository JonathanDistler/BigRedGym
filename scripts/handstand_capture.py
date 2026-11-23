"""Staff simulation entry point for the released CapturePolicy (full evaluation only).

Training commands remain in the research release; this staff branch does not
retrain or replace the policy. This module never imports the Unitree SDK.
"""

import argparse
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import os
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_deploy.custom_implementation.go2handstand import (
    load_assets,
    Session,
    IMUCalibration,
    calibrate_stationary,
)
from gym.envs.go2.handstand_replay import (
    Replay,
    build_model,
    KP,
    KD,
    PHYSICS_DT,
    CONTROL_DT,
)


def presentation_style(model):
    """Visual fields only: brighter lighting and distinct front/rear leg colors."""
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.1, 0.1, 0.1]
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name
        color = (
            [0.82, 0.84, 0.87, 1.0]
            if not model.geom_bodyid[g]
            else [0.48, 0.53, 0.62, 1.0]
        )
        if name.startswith(("FL_", "FR_")):
            color = [0.1, 0.5, 0.95, 1.0]
        if name.startswith(("RL_", "RR_")):
            color = [0.95, 0.42, 0.08, 1.0]
        model.geom_rgba[g] = color
        model.geom_matid[g] = -1


def ground_grid(scene):
    for axis in (0, 1):
        for value in np.arange(-1.5, 1.51, 0.25):
            a = np.array([-1.5, value, 0.001])
            b = np.array([1.5, value, 0.001])
            if axis:
                a[[0, 1]] = a[[1, 0]]
                b[[0, 1]] = b[[1, 0]]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.array([0.48, 0.51, 0.56, 1.0]),
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, a, b)
            scene.ngeom += 1


class Presentation:
    def __init__(self, robot, viewer=False, video=None, fps=30, width=640, height=480):
        self.robot = robot
        presentation_style(robot.m)
        self.stack = ExitStack()
        self.frames = 0
        self.next_frame = 0.0
        self.fps = fps
        self.renderer = self.writer = self.viewer = None
        try:
            if viewer:
                if sys.platform == "linux" and not os.environ.get("DISPLAY"):
                    raise ValueError(
                        "No DISPLAY for --viewer; use MUJOCO_GL=egl with --video"
                    )
                from mujoco import viewer as mjviewer

                self.viewer = self.stack.enter_context(
                    mjviewer.launch_passive(robot.m, robot.d)
                )
                with self.viewer.lock():
                    ground_grid(self.viewer.user_scn)
                self.viewer.cam.distance = 1.6
                self.viewer.cam.azimuth = 130
                self.viewer.cam.elevation = -15
            if video is not None:
                try:
                    import imageio.v2 as imageio
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Install scripts/handstand_video_requirements.txt for --video"
                    ) from exc

                robot.m.vis.global_.offwidth = max(robot.m.vis.global_.offwidth, width)
                robot.m.vis.global_.offheight = max(
                    robot.m.vis.global_.offheight, height
                )
                self.renderer = mujoco.Renderer(robot.m, height=height, width=width)
                self.stack.callback(self.renderer.close)
                self.writer = imageio.get_writer(
                    str(video), fps=fps, codec="libx264", macro_block_size=1
                )
                self.stack.callback(self.writer.close)
                self.camera = mujoco.MjvCamera()
                self.camera.distance = 1.6
                self.camera.azimuth = 130
                self.camera.elevation = -15
            self.capture(force=True)
        except BaseException:
            self.stack.close()
            raise

    def capture(self, force=False):
        r = self.robot
        if self.viewer and self.viewer.is_running():
            self.viewer.cam.lookat[:] = r.d.subtree_com[r.base]
        if self.renderer and (force or r.d.time + 1e-10 >= self.next_frame):
            self.camera.lookat[:] = r.d.subtree_com[r.base]
            self.renderer.update_scene(r.d, camera=self.camera)
            ground_grid(self.renderer.scene)
            self.writer.append_data(self.renderer.render().copy())
            self.frames += 1
            self.next_frame += 1 / self.fps

    def sync(self):
        if self.viewer:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        self.stack.close()


def trial(policy, trajectory, seed, perturbed, presentation=None, sensor_options=None):
    r = Replay(build_model(), trajectory)
    rng = np.random.default_rng(seed)
    r.d.qpos[r.qids] += rng.uniform(
        -0.003 if perturbed else 0, 0.003 if perturbed else 0, 12
    )
    r.d.qvel[r.vids] += rng.uniform(
        -0.03 if perturbed else 0, 0.03 if perturbed else 0, 12
    )
    mujoco.mj_forward(r.m, r.d)
    r.anchor = r.d.geom_xpos[r.foot_geoms[:2], :2].copy()
    opts = dict(sensor_options or {})
    observer = opts.get("observer", "inertial")
    observer_options = (
        {"kind": observer, "max_interval": 0.012, "max_age": 0.015} if opts else None
    )
    session = Session(policy, trajectory, build_model(), observer_options)
    sensor_rng = np.random.default_rng(seed + 8000000)
    calibration = IMUCalibration(
        np.eye(3),
        np.eye(3),
        opts.get("lever", [0, 0, 0]),
        [0, 0, 0],
        [0, -0.009, 0] if opts.get("ta") else [0, 0, 0],
        accel_scale=9.81 / 9.55 if opts.get("scale_correction") else 1.0,
        max_interval=0.012,
        alpha_tau=0.02,
        stationary_gyro_limit=0.04,
    )
    if opts.get("stationary_calibration"):
        # Synthetic stationary telemetry; biases are estimated from samples.
        readings = []
        calibration_rng = np.random.default_rng(seed + 9000000)
        for n in range(125):
            force = np.array([0, 0, 9.81])
            gyro = np.zeros(3)
            if opts.get("ta"):
                force = force * opts.get("scale", 9.55 / 9.81) + opts.get(
                    "bias", [0.05, -0.05, 0.1]
                )
                force += calibration_rng.normal(0, opts.get("accel_noise", 0.03), 3)
                gyro = np.array([0, -0.009, 0]) + calibration_rng.normal(
                    0, opts.get("gyro_noise", 0.007), 3
                )
            readings.append(
                (
                    n * 0.002,
                    (
                        2 * n,
                        np.array([1.0, 0, 0, 0]),
                        gyro,
                        force,
                        trajectory["initial_target"].copy(),
                        np.zeros(12),
                        np.full(4, r.weight / 4),
                    ),
                )
            )
        parameters = dict(
            body_from_imu=np.eye(3),
            world_alignment=np.eye(3),
            lever_arm_body=opts.get("lever", [0, 0, 0]),
            accel_bias_imu=[0, 0, 0],
            gyro_bias_imu=[0, -0.009, 0] if opts.get("ta") else [0, 0, 0],
            accel_scale=9.81 / 9.55 if opts.get("scale_correction") else 1.0,
            max_interval=0.012,
            alpha_tau=0.02,
            stationary_gyro_limit=0.04,
        )
        calibration = IMUCalibration(**calibrate_stationary(parameters, readings))
    drift = []
    dropped = 0
    physics_step = 0

    def measured_loads():
        # Synthetic generator only; deployed actors cannot query contacts.
        loads = np.zeros(4)
        for j in range(r.d.ncon):
            contact = r.d.contact[j]
            for i, g in enumerate(r.foot_geoms):
                if (contact.geom1 == 0 and contact.geom2 == g) or (
                    contact.geom2 == 0 and contact.geom1 == g
                ):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(r.m, r.d, j, wrench)
                    loads[i] += max(0, wrench[0])
        return loads

    def ingest(force, alpha, drop=False):
        omega = r.d.qvel[3:6].copy()
        lever = np.array(opts.get("lever", [0, 0, 0]))
        raw = force + np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
        if opts.get("ta"):
            raw = raw * opts.get("scale", 9.55 / 9.81) + opts.get(
                "bias", [0.05, -0.05, 0.1]
            )
            raw += sensor_rng.normal(0, opts.get("accel_noise", 0.03), 3)
            omega += [0, -0.009, 0] + sensor_rng.normal(
                0, opts.get("gyro_noise", 0.007), 3
            )
        # Generate noise at every physical sample, even if transport drops it,
        # so paired loss/no-loss runs share the same timestamp-indexed noise.
        if drop:
            return
        quat = r.d.qpos[3:7].copy()
        if opts:
            quat, omega, raw = calibration.correct(float(r.d.time), quat, omega, raw)
        session.ingest(
            float(r.d.time),
            raw,
            quat,
            omega,
            r.d.qpos[r.qids],
            r.d.qvel[r.vids],
            measured_loads() if observer == "contact" else None,
        )

    # Known nominal standing calibration, not simulator COM/velocity input.
    nominal_data = mujoco.MjData(r.m)
    mujoco.mj_setState(
        r.m,
        nominal_data,
        trajectory["initial_state"],
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    session.activate(
        0.0,
        float(nominal_data.qpos[2]) + opts.get("height_error", 0.0),
        calibrated=True,
    )
    if opts:
        ingest(np.array([0, 0, 9.81]), np.zeros(3))
    else:
        session.ingest(
            0.0,
            np.zeros(3),
            r.d.qpos[3:7],
            r.d.qvel[3:6],
            r.d.qpos[r.qids],
            r.d.qvel[r.vids],
        )
    view = presentation(r) if presentation else None
    cancelled = False
    steps = 0
    try:
        for k in range(400):
            wall = time.monotonic()
            if view and not view.sync():
                cancelled = True
                break
            mujoco.mj_subtreeVel(r.m, r.d)
            observed = session.snapshot(float(r.d.time))
            drift.append(
                np.r_[
                    observed[43] * 0.6 - r.d.subtree_com[r.base, 2],
                    observed[44:47] - r.d.subtree_linvel[r.base],
                ]
            )
            _, target = session.propose(k * CONTROL_DT)
            # Explicit software sink, never a hardware acknowledgement.
            r.target = target.copy()
            session.acknowledge(target)
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                old_omega = r.d.qvel[3:6].copy()
                velocity = r.d.qvel[r.vids].copy()
                torque = np.clip(
                    KP * (target - r.d.qpos[r.qids]) - KD * velocity,
                    -r.effort,
                    r.effort,
                )
                torque[(abs(velocity) >= r.speed) & (torque * velocity > 0)] = 0
                r.d.qfrc_applied[:] = 0
                r.d.qfrc_applied[r.vids] = torque
                mujoco.mj_step(r.m, r.d)
                acceleration = r.d.qacc[:3].copy()
                mujoco.mj_forward(r.m, r.d)
                rotation = r.d.xmat[r.base].reshape(3, 3)
                force = rotation.T @ (acceleration - r.m.opt.gravity)
                physics_step += 1
                drop = (
                    opts.get("drop_count", 0)
                    and 1 <= physics_step % 100 <= opts["drop_count"]
                )
                if drop:
                    dropped += 1
                if opts:
                    ingest(force, (r.d.qvel[3:6] - old_omega) / PHYSICS_DT, drop=drop)
                elif not drop:
                    session.ingest(
                        float(r.d.time),
                        force,
                        r.d.qpos[3:7],
                        r.d.qvel[3:6],
                        r.d.qpos[r.qids],
                        r.d.qvel[r.vids],
                    )
                r.measure(velocity)
                if view:
                    view.capture()
            steps = k + 1
            if view and view.viewer:
                time.sleep(max(0, CONTROL_DT - (time.monotonic() - wall)))
            if not r.safe or (r.best_hold >= 5 and r.takeoff):
                break
        if view:
            view.capture(force=True)
    finally:
        if view:
            view.close()
    success = bool(
        r.safe and r.takeoff and r.captured and r.best_hold >= 5 and not cancelled
    )
    return dict(
        seed=seed,
        observer=observer,
        sensor_options=opts,
        dropped_measurements=dropped,
        gap_intervals=session.estimator.gap_intervals,
        contact_corrections=getattr(session.estimator, "corrections", 0),
        com_error_rms=np.sqrt(np.mean(np.array(drift) ** 2, axis=0)).tolist()
        if drift
        else None,
        com_error_max=np.max(abs(np.array(drift)), axis=0).tolist() if drift else None,
        condition={"joint_position": 0.003, "joint_velocity": 0.03}
        if perturbed
        else {},
        complete_success=success,
        strict_hold_seconds=r.best_hold,
        prefix_safe=r.safe,
        certified_takeoff=r.takeoff,
        front_capture=r.captured,
        illegal_contacts=r.illegal_contacts,
        failure="viewer_closed"
        if cancelled
        else None
        if success
        else "safety_gate"
        if not r.safe
        else "insufficient_strict_hold",
        cancelled=cancelled,
        steps=steps,
        simulated_seconds=float(r.d.time),
        video_frames=view.frames if view else 0,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["full"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--candidate",
        action="store_true",
        help="Evaluate a retrained candidate; release pinning remains the default",
    )
    p.add_argument(
        "--trajectory", "--reference", dest="trajectory", type=Path, required=True
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Held-out trials, plus one nominal trial (research convention)",
    )
    p.add_argument("--evaluation-seed", type=int, default=301000)
    p.add_argument("--controllers", nargs="+", choices=["student"], default=["student"])
    p.add_argument(
        "--trial-seeds",
        type=int,
        nargs="+",
        help="Explicit held-out seeds, plus the usual nominal trial",
    )
    p.add_argument("--observer", choices=["inertial", "contact"], default="inertial")
    p.add_argument("--sensor-profile", choices=["ideal", "ta"], default="ideal")
    p.add_argument("--accel-scale", type=float, default=9.55 / 9.81)
    p.add_argument(
        "--accel-bias",
        nargs=3,
        type=float,
        default=[0.05, -0.05, 0.1],
        help="Assumed residual bias stress, not a TA measurement (m/s²)",
    )
    p.add_argument("--gyro-noise", type=float, default=0.007)
    p.add_argument("--accel-noise", type=float, default=0.03)
    p.add_argument("--scale-correction", action="store_true")
    p.add_argument("--stationary-calibration", action="store_true")
    p.add_argument("--initial-height-error", type=float, default=0.0)
    p.add_argument(
        "--imu-drop-count",
        type=int,
        choices=range(6),
        default=0,
        help="Drop N consecutive samples per 200 ms; N=5 gives 12 ms intervals",
    )
    p.add_argument(
        "--model-imu",
        action="store_true",
        help="Generate IMU lever-arm acceleration using the URDF mounting",
    )
    p.add_argument("--viewer", action="store_true")
    p.add_argument(
        "--video",
        action="store_true",
        help="Save a labelled MP4 for every trial, including failures",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()
    if a.trials < 0 or a.fps < 1 or a.fps > 500 or a.width < 16 or a.height < 16:
        p.error("Invalid trial count or rendering settings")
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    kwargs = (
        {"checkpoint_sha": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()}
        if a.candidate
        else {}
    )
    policy, trajectory = load_assets(a.checkpoint, a.trajectory, **kwargs)
    a.output.mkdir(parents=True)
    rows = []
    seeds = [a.evaluation_seed] + (
        a.trial_seeds
        if a.trial_seeds is not None
        else [a.evaluation_seed + i for i in range(1, a.trials + 1)]
    )
    for i, seed in enumerate(seeds):
        video = a.output / f"trial_{i:03d}_seed_{seed}.mp4" if a.video else None
        factory = (
            (lambda r: Presentation(r, a.viewer, video, a.fps, a.width, a.height))
            if a.video or a.viewer
            else None
        )
        opts = None
        if (
            a.sensor_profile != "ideal"
            or a.observer != "inertial"
            or a.imu_drop_count
            or a.initial_height_error
            or a.model_imu
            or a.stationary_calibration
        ):
            opts = dict(
                observer=a.observer,
                ta=a.sensor_profile == "ta",
                scale=a.accel_scale,
                bias=a.accel_bias,
                gyro_noise=a.gyro_noise,
                accel_noise=a.accel_noise,
                scale_correction=a.scale_correction,
                stationary_calibration=a.stationary_calibration,
                height_error=a.initial_height_error,
                drop_count=a.imu_drop_count,
                lever=[-0.02557, 0, 0.04232] if a.model_imu else [0, 0, 0],
            )
        row = trial(policy, trajectory, seed, i > 0, factory, opts)
        if video:
            labelled = video.with_stem(
                video.stem
                + (
                    "_cancelled"
                    if row["cancelled"]
                    else "_success"
                    if row["complete_success"]
                    else "_failure"
                )
            )
            video.rename(labelled)
            row["video"] = labelled.name
        rows.append(row)
        summary = {
            "student": {
                "nominal": rows[0],
                "successes": sum(r["complete_success"] for r in rows),
                "trials": sum(not r["cancelled"] for r in rows),
                "heldout_successes": sum(r["complete_success"] for r in rows[1:]),
                "heldout_trials": sum(not r["cancelled"] for r in rows[1:]),
                "rows": rows,
            },
            "software_only": True,
            "policy_kind": "capture_residual_v1",
            "viewer": a.viewer,
            "video": a.video,
        }
        (a.output / "full.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            "full student",
            i,
            row["complete_success"],
            row["strict_hold_seconds"],
            flush=True,
        )
        if row["cancelled"]:
            break


if __name__ == "__main__":
    main()
