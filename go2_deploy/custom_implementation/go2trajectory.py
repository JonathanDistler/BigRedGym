"""Cornell four-hook interface for NEW constrained position trajectories.

Provisional group filename. Software-only; not the successful CapturePolicy.
No COM estimate is invented: this open-loop reference uses 43 sensor/history/time
channels and low-level PD, not the old 47-input capture/balance feedback law.
Never launch MainController/deploy.py for this dry run: their initialization can
contact the robot before any hook. Required runner changes remain isolated and
unimplemented here: explicit 50 Hz, EMA off, precise nominal posture/gains,
bounded target application and actual acknowledgement, staff-qualified motor
protections, watchdog, calibration and recovery. No SDK import or publisher.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch import nn

JOINT_ORDER = [
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
SPEED = np.tile([30.1, 30.1, 15.7], 4)
LOWER = np.tile([-1.0472, -1.5708, -2.7227], 4)
LOWER[[7, 10]] = -0.5236
UPPER = np.tile([1.0472, 3.4907, -0.83776], 4)
UPPER[[7, 10]] = 4.5379
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
    custom_obs_sizes = {name: end - start for name, (start, end) in SLICES.items()}
    ctrl_freq, kp, kd, exp_moving_avg = (
        50,
        60.0,
        3.0,
        False,
    )  # requirements, not inherited by stock runner

    class DeployScaling:
        projected_gravity = base_ang_vel = dof_pos_obs = dof_vel = 1.0
        trajectory_previous_target = trajectory_elapsed = 1.0
        dof_pos_target = 2.0


class ReferenceActor(nn.Module):
    def __init__(self, targets, nominal):
        super().__init__()
        self.register_buffer(
            "actions", torch.tensor((targets - nominal) / 2, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor):
        k = (obs[..., 42] * 400).round().long().clamp(0, self.actions.size(0) - 1)
        return self.actions[k]


def get_custom_buffers_dict():
    return {
        "trajectory_mode": "disabled",
        "trajectory_targets": None,
        "trajectory_nominal": None,
        "trajectory_step": 0,
        "trajectory_applied": None,
        "trajectory_acknowledgement": None,
        "trajectory_sensor_snapshot": None,
        "trajectory_observation": None,
        "trajectory_cancel_requested": False,
        "trajectory_last_time": None,
    }


def _software_only(c):
    if hasattr(c, "lowcmd_publisher"):
        c.trajectory_mode = "disabled"
        c.trajectory_observation = None
        raise RuntimeError(
            "Physical activation disabled; software-only trajectory review"
        )


def configure_software_validation(c, directory):
    _software_only(c)
    directory = Path(directory)
    meta = json.loads((directory / "deploy_meta.json").read_text())
    expected = {
        "kind": "constrained_position_trajectory_v1",
        "control_hz": 50,
        "kp": 60,
        "kd": 3,
        "ema": False,
        "action_scale": 2.0,
        "hardware_ready": False,
    }
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Inconsistent trajectory metadata")
    path = directory / "position_reference.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Trajectory hash mismatch")
    with np.load(path, allow_pickle=False) as f:
        targets = f["targets"].copy()
        nominal = f["nominal"].copy()
        times = f["time"].copy()
        if f["joint_order"].tolist() != JOINT_ORDER:
            raise ValueError("Incorrect hardware joint order")
    if (
        targets.ndim != 2
        or targets.shape[1] != 12
        or len(targets) < 2
        or nominal.shape != (12,)
    ):
        raise ValueError("Expected position trajectory Nx12 and nominal12")
    if (
        not np.isfinite(targets).all()
        or not np.isfinite(nominal).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("Nonfinite trajectory")
    if (
        times.shape != (len(targets),)
        or abs(times[0]) > 1e-9
        or not np.allclose(np.diff(times), 0.02, rtol=0, atol=1e-9)
    ):
        raise ValueError("Expected 50 Hz trajectory beginning at zero")
    if (
        (targets < LOWER).any()
        or (targets > UPPER).any()
        or (abs((targets - nominal) / 2) > 1).any()
    ):
        raise ValueError("Position/action limits violated")
    if (abs(np.diff(targets, axis=0)) > 0.7 * SPEED * 0.02 + 1e-9).any():
        raise ValueError("Applied-target slew violated")
    c.trajectory_targets = targets
    c.trajectory_nominal = nominal
    c.trajectory_step = 0
    c.trajectory_applied = targets[0].copy()
    c.trajectory_last_time = None
    c.trajectory_acknowledgement = None
    c.trajectory_cancel_requested = False
    c.trajectory_sensor_snapshot = None
    c.trajectory_observation = None
    c.trajectory_mode = "software_validation"
    policy = torch.jit.script(ReferenceActor(targets, nominal))
    with torch.no_grad():
        for _ in range(5):
            policy(torch.zeros(43))
    return policy


def update_custom_buffers(main_controller, t):
    c = main_controller
    _software_only(c)
    c.trajectory_observation = None
    if c.trajectory_mode != "software_validation" or c.trajectory_targets is None:
        raise RuntimeError("NOT READY: independently validated trajectory required")
    try:
        if c.trajectory_cancel_requested:
            raise RuntimeError("Explicit trajectory cancellation")
        if (
            c.trajectory_last_time is not None
            and abs(t - c.trajectory_last_time - 0.02) > 0.002
        ):
            raise ValueError("Control cadence differs from 50 Hz")
        if c.trajectory_acknowledgement is not None:
            step, applied = c.trajectory_acknowledgement
            expected = c.trajectory_targets[
                min(c.trajectory_step, len(c.trajectory_targets) - 1)
            ]
            if step != c.trajectory_step or not np.array_equal(applied, expected):
                raise ValueError("Missing or incorrect applied-target acknowledgement")
            c.trajectory_applied = np.asarray(applied).copy()
            c.trajectory_step += 1
            c.trajectory_acknowledgement = None
        elif c.trajectory_last_time is not None:
            raise ValueError("Prior command was not acknowledged")
        sample = c.trajectory_sensor_snapshot
        if sample is None or len(sample) != 5 or not np.isfinite(t):
            raise ValueError("Missing synchronized sensor snapshot")
        stamp, quat, gyro, q, dq = sample
        quat = np.asarray(quat, dtype=float)
        gyro = np.asarray(gyro, dtype=float)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        if (
            quat.shape != (4,)
            or gyro.shape != (3,)
            or q.shape != (12,)
            or dq.shape != (12,)
        ):
            raise ValueError("Invalid telemetry dimensions")
        if not np.isfinite(np.r_[stamp, quat, gyro, q, dq]).all() or not (
            0 <= t - stamp <= 0.006
        ):
            raise ValueError("Nonfinite, stale or future sensor snapshot")
        if abs(np.linalg.norm(quat) - 1) > 0.001:
            raise ValueError("Uncalibrated orientation")
        # Calibrated wxyz rotation body->world; projected gravity is its third row.
        w, x, y, z = quat / np.linalg.norm(quat)
        gravity = -np.array(
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        )
        c.trajectory_observation = np.r_[
            gravity,
            gyro,
            q - c.trajectory_nominal,
            dq / SPEED,
            (c.trajectory_applied - c.trajectory_nominal) / 2,
            c.trajectory_step * 0.02 / 8,
        ].astype(np.float32)
        if not np.isfinite(c.trajectory_observation).all():
            raise ValueError("Nonfinite float32 observation")
        c.trajectory_sensor_snapshot = None
        c.trajectory_last_time = t
    except (ValueError, TypeError, RuntimeError):
        c.trajectory_mode = "disabled"
        c.trajectory_observation = None
        raise


def get_custom_obs_pieces():
    def getter(start, end):
        def piece(c, lowstate_msg):
            _software_only(c)
            if (
                c.trajectory_mode != "software_validation"
                or c.trajectory_observation is None
            ):
                raise RuntimeError("Missing active trajectory observation")
            return torch.from_numpy(c.trajectory_observation[start:end].copy())

        return piece

    return {name: getter(start, end) for name, (start, end) in SLICES.items()}
