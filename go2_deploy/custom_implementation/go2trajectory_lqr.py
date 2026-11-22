"""43-input model-derived trajectory tracker for Cornell's PR #26 callbacks.

Orientation, gyro and encoders only: no accelerometer integration or COM input.
This module creates no SDK clients and sends no robot commands. Load the exported
complete TorchScript actor through the approved bundle loader, not trot export.
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import torch

ORDER = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
SPEED = np.tile([30.1, 30.1, 15.7], 4)
SLICES = {
    "projected_gravity": (0, 3),
    "base_ang_vel": (3, 6),
    "dof_pos_obs": (6, 18),
    "dof_vel": (18, 30),
    "trajectory_previous_target": (30, 42),
    "trajectory_elapsed": (42, 43),
}


class CustomDeployConfig:
    obs_vector = list(SLICES)
    custom_obs_sizes = {name: b - a for name, (a, b) in SLICES.items()}
    ctrl_freq, kp, kd, exp_moving_avg = 50, 60.0, 3.0, False

    class DeployScaling:
        projected_gravity = base_ang_vel = dof_pos_obs = dof_vel = 1.0
        trajectory_previous_target = trajectory_elapsed = 1.0
        dof_pos_target = 2.0


def finite(x, shape):
    x = np.asarray(x, dtype=float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError("Invalid trajectory sensor/asset")
    return x


def rotation(q):
    q = finite(q, (4,))
    if abs(np.linalg.norm(q) - 1) > 0.001:
        raise ValueError("Invalid IMU quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def prepare_runtime(cfg):
    if (cfg.ctrl_freq, cfg.kp, cfg.kd, cfg.exp_moving_avg) != (50, 60, 3, False):
        raise ValueError("Trajectory requires 50 Hz, PD 60/3 and EMA off")
    import os

    folder = Path(cfg.bundle_dir)
    meta = cfg.bundle_meta
    if meta.get("kind") != "native_trajectory_lqr_v1" or meta["num_obs"] != 43:
        raise ValueError("Wrong trajectory bundle kind/dimensions")
    if (
        hashlib.sha256((folder / "policy.pt").read_bytes()).hexdigest()
        != meta["policy_sha256"]
    ):
        raise ValueError("Trajectory actor hash mismatch")
    if meta.get("nominal_validation_pass") is not True:
        raise ValueError("Independent nominal replay must pass before bundling")
    if (
        list(cfg.obs_vector) != list(SLICES)
        or any(meta["scaling"].get(k) != 1 for k in SLICES)
        or meta["scaling"].get("dof_pos_target") != 2
    ):
        raise ValueError("Observation ordering/action scaling mismatch")
    actor = cfg.loaded_policy
    if set(actor.state_dict()) != {"actions", "centers", "gains"}:
        raise ValueError("Complete model-derived actor required")
    n = len(actor.actions)
    if (
        tuple(actor.actions.shape) != (n, 12)
        or tuple(actor.centers.shape) != (n, 30)
        or tuple(actor.gains.shape) != (n, 12, 30)
    ):
        raise ValueError("Complete trajectory/feedback actor required")
    if n < 2 or not 0 < float(actor.limit) <= 0.2:
        raise ValueError("Invalid trajectory length/feedback limit")
    if any(
        not torch.isfinite(x).all() for x in [actor.actions, actor.centers, actor.gains]
    ):
        raise ValueError("Nonfinite actor buffers")
    calibration = json.loads(
        Path(
            os.environ.get("GO2_TRAJECTORY_CALIBRATION", folder / "calibration.json")
        ).read_text()
    )
    for key in ["body_from_imu", "world_alignment"]:
        value = finite(calibration["imu"][key], (3, 3))
        if not np.allclose(value.T @ value, np.eye(3), atol=1e-8) or not np.isclose(
            np.linalg.det(value), 1
        ):
            raise ValueError("IMU calibration requires proper rotations")
    finite(calibration["imu"]["gyro_bias_imu"], (3,))
    nominal = finite(meta["default_dof_pos"], (12,))
    bounds = finite(meta["joint_bounds"], (12, 2))
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid joint limits")
    if not np.allclose(nominal, np.asarray(cfg.default_dof_pos), rtol=0, atol=1e-7):
        raise ValueError("Wrong nominal posture")
    with torch.no_grad():
        for _ in range(5):
            actor(torch.zeros(43))
    return dict(
        calibration=calibration,
        nominal=nominal,
        bounds=bounds,
        initial_q=finite(meta["initial_q"], (19,)),
        initial_v=finite(meta["initial_v"], (18,)),
    )


def get_custom_buffers_dict():
    return dict(
        trajectory_step=0,
        trajectory_pending=None,
        trajectory_applied=None,
        trajectory_observation=None,
        trajectory_mode=None,
        trajectory_last_control=None,
        trajectory_active=False,
        trajectory_cancel_requested=False,
        trajectory_unique_tick=None,
        trajectory_unique_receipt=None,
        trajectory_rx_fault=None,
        trajectory_snapshot_receipt=None,
        trajectory_fault_latched=False,
    )


def initialize_controller(c):
    c.trajectory_runtime = c.cfg.custom_runtime
    c.trajectory_observation = np.zeros(43, np.float32)


def fault(c, reason):
    c.trajectory_fault_latched = True
    c.trajectory_active = False
    c.trajectory_pending = None
    c.trajectory_observation = None
    raise ValueError(reason)


def on_lowstate_msg(c, msg, received):
    # No full-rate integration: only unique receipt freshness is retained.
    try:
        tick = int(msg.tick)
        if not 0 <= tick < 2**32:
            raise ValueError("Invalid LowState tick")
        previous = c.trajectory_unique_tick
        if previous is None or 0 < (tick - previous) % (2**32) < 2**31:
            c.trajectory_unique_tick = tick
            c.trajectory_unique_receipt = float(received)
        elif tick != previous:
            raise ValueError("Out-of-order LowState")
    except (TypeError, ValueError, AttributeError) as e:
        c.trajectory_rx_fault = str(e)


def update_custom_buffers(c, t):
    try:
        _update_custom_buffers(c, t)
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        fault(c, str(exc))


def _update_custom_buffers(c, t):
    mode = getattr(c._state, "name", "")
    if mode != "CUSTOM_CTRL":
        c.trajectory_active = False
        c.trajectory_step = 0
        c.trajectory_pending = None
        c.trajectory_observation = np.zeros(43, np.float32)
        c.trajectory_mode = mode
        c.trajectory_last_control = None
        c.trajectory_rx_fault = None
        c.trajectory_fault_latched = False
        c.trajectory_cancel_requested = False
        return
    if c.trajectory_fault_latched:
        fault(c, "Trajectory fault requires mode reset")
    if c.trajectory_cancel_requested:
        fault(c, "Trajectory cancelled")
    with c.last_lowstate_msg_lock:
        msg = c.last_lowstate_msg
        if msg is None:
            fault(c, "Missing LowState")
        quat = finite(msg.imu_state.quaternion, (4,)).copy()
        gyro = finite(msg.imu_state.gyroscope, (3,)).copy()
        q = finite([m.q for m in msg.motor_state[:12]], (12,))[ORDER]
        dq = finite([m.dq for m in msg.motor_state[:12]], (12,))[ORDER]
        if int(msg.tick) != c.trajectory_unique_tick:
            fault(c, "Missing callback synchronization")
        receipt = c.trajectory_unique_receipt
        fault_message = c.trajectory_rx_fault
    now = getattr(c, "lowstate_clock", lambda: t)()
    if fault_message:
        fault(c, fault_message)
    if (
        receipt is None
        or not np.isfinite([now, t, receipt]).all()
        or not -1e-9 <= now - receipt <= 0.015 + 1e-9
    ):
        fault(c, "Stale/frozen LowState or missing callback")
    runtime = c.trajectory_runtime
    imu = runtime["calibration"]["imu"]
    mount = np.asarray(imu["body_from_imu"])
    world = np.asarray(imu["world_alignment"])
    r = world @ rotation(quat) @ mount.T
    omega = mount @ (gyro - np.asarray(imu["gyro_bias_imu"]))
    if not c.trajectory_active:
        if runtime["calibration"].get("calibrated") is not True:
            fault(c, "Staff-confirmed IMU calibration required for activation")
        if c.trajectory_applied is None:
            fault(c, "Enter INTERMEDIATE before custom control")
        if np.max(abs(c.trajectory_applied - q)) > 0.15:
            fault(c, "Initial published target differs from measured posture")
        if (
            np.max(abs(q - runtime["initial_q"][7:])) > 0.04
            or np.max(abs(dq - runtime["initial_v"][6:])) > 0.2
        ):
            fault(c, "Initial posture/velocity does not match trajectory")
        expected_gravity = rotation(runtime["initial_q"][3:7]).T @ [0, 0, -1]
        if (
            np.linalg.norm(r.T @ [0, 0, -1] - expected_gravity) > 0.05
            or np.max(abs(omega - runtime["initial_v"][3:6])) > 0.1
        ):
            fault(c, "Initial IMU state does not match trajectory")
        c.trajectory_step = 0
        c.trajectory_active = True
    elif abs(t - c.trajectory_last_control - 0.02) > 0.002 + 1e-9:
        fault(c, "Control cadence differs from 50 Hz")
    if c.trajectory_pending is not None:
        fault(c, "Previous target was not published")
    if (
        (q < runtime["bounds"][:, 0] - 0.001).any()
        or (q > runtime["bounds"][:, 1] + 0.001).any()
        or (abs(dq) > SPEED * 1.001).any()
    ):
        fault(c, "Encoder joint/speed limit exceeded")
    c.trajectory_observation = np.r_[
        r.T @ [0, 0, -1],
        omega,
        q - runtime["nominal"],
        dq / SPEED,
        (c.trajectory_applied - runtime["nominal"]) / 2,
        c.trajectory_step / 400,
    ].astype(np.float32)
    c.trajectory_snapshot_receipt = receipt
    c.trajectory_mode = mode
    c.trajectory_last_control = t


def bound_command(c, offset, kp_mult=1.0, kd_mult=1.0):
    try:
        return _bound_command(c, offset, kp_mult, kd_mult)
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        fault(c, str(exc))


def _bound_command(c, offset, kp_mult=1.0, kd_mult=1.0):
    if getattr(c._state, "name", "") != "CUSTOM_CTRL":
        return offset + c.cfg.default_dof_pos
    if not c.trajectory_active or c.trajectory_observation is None:
        fault(c, "Trajectory inactive")
    if c.kp_mult != 1 or c.kd_mult != 1 or kp_mult != 1 or kd_mult != 1:
        fault(c, "Trajectory gains must remain 60/3")
    if c.trajectory_pending is not None:
        fault(c, "Unpublished command")
    if (
        hasattr(c, "lowstate_clock")
        and not -1e-9
        <= c.lowstate_clock() - c.trajectory_snapshot_receipt
        <= 0.020 + 1e-9
    ):
        fault(c, "Command snapshot is stale")
    runtime = c.trajectory_runtime
    previous = c.trajectory_applied
    offset = finite(np.asarray(offset), (12,))
    if np.any(abs(offset) > 2 + 1e-6):
        fault(c, "Normalized action exceeds bounds")
    desired = runtime["nominal"] + offset
    lo = np.maximum(runtime["bounds"][:, 0], previous - 0.7 * SPEED * 0.02)
    hi = np.minimum(runtime["bounds"][:, 1], previous + 0.7 * SPEED * 0.02)
    wire = np.clip(desired, lo, hi).astype(np.float32)
    wire = np.where(
        wire.astype(float) < lo,
        np.nextafter(lo.astype(np.float32), np.float32(np.inf)),
        wire,
    )
    wire = np.where(
        wire.astype(float) > hi,
        np.nextafter(hi.astype(np.float32), np.float32(-np.inf)),
        wire,
    )
    c.trajectory_pending = wire.astype(float)
    c.last_action[0] = torch.tensor(
        c.trajectory_pending - runtime["nominal"], dtype=c.last_action.dtype
    )
    return torch.from_numpy(c.trajectory_pending.copy())


def on_command_sent(c, command, accepted, t):
    if accepted is not True:
        fault(c, "Command publication failed")
    actual = finite([m.q for m in command.motor_cmd[:12]], (12,))[ORDER]
    if getattr(c._state, "name", "") == "CUSTOM_CTRL":
        if c.trajectory_pending is None or not np.array_equal(
            actual, c.trajectory_pending
        ):
            fault(c, "Published target mismatch")
        c.trajectory_step += 1
        c.trajectory_pending = None
    c.trajectory_applied = actual.copy()


def get_custom_obs_pieces():
    def getter(a, b):
        def piece(c, msg):
            if c.trajectory_observation is None:
                raise ValueError("Missing trajectory snapshot")
            return torch.from_numpy(c.trajectory_observation[a:b].copy())

        return piece

    return {name: getter(a, b) for name, (a, b) in SLICES.items()}
