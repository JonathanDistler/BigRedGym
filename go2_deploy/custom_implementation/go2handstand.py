"""Cornell MAE 4110 handstand: four observation hooks plus optional runner bridges.

The runner reads LowState directly. Optional callbacks preserve 500 Hz IMU data
and acknowledge successful bounded-command publication (not motor tracking).
Use --export-bundle to package the COMPLETE hybrid law; --recording retains the
software-only recorded-sensor parity check. Live simulation/video entry point:
  python scripts/handstand_capture.py full --help

No SDK imports or robot connection are made by this module. Physical readiness
requires staff validation of calibration, motor limits, contact loads and recovery.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import os
import threading
import shutil
import mujoco
import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from go2_deploy.custom_implementation.custom_config import CustomConfig

CHECKPOINT_SHA = "90117fc2ff80033663578e12ba664e4b1ff4c15c4c04f19619409a800458ba6c"
REFERENCE_SHA = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
JOINTS = [
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
PERMUTATION = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
SPEED = np.tile([30.1, 30.1, 15.7], 4)


def checked(path, digest):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


class InferencePolicy(nn.Module):
    num_obs = 47
    num_actions = 12

    def __init__(self, state):
        super().__init__()
        for key in (
            "actions",
            "centers",
            "gains",
            "scale",
            "bound",
            "active_steps",
            "balance_weight",
        ):
            self.register_buffer(key, state[key].detach().clone())

    def components(self, obs: torch.Tensor):
        phase = (obs[..., 42] * 400).round().long()
        prefix_k = phase.clamp(0, int(self.active_steps) - 1)
        k = phase.clamp(0, self.actions.size(0) - 1)
        delta = torch.cat(
            (
                obs[..., :42] - self.centers[prefix_k, :42],
                obs[..., 43:47] - self.centers[prefix_k, 43:47],
            ),
            -1,
        )
        delta = (delta / self.scale).clamp(-20, 20)
        capture = (
            self.bound
            * torch.einsum("...ij,...j->...i", self.gains[prefix_k], delta).tanh()
        )
        features = torch.cat((obs[..., :30], obs[..., 43:47]), -1)
        centers = torch.cat((self.centers[k, :30], self.centers[k, 43:47]), -1)
        balance = ((features - centers) @ self.balance_weight.T).clamp(
            -self.bound, self.bound
        )
        prefix = (self.actions[prefix_k] + capture).clamp(-1, 1)
        late = (self.actions[k] + balance).clamp(-1, 1)
        action = torch.where((phase >= self.active_steps)[..., None], late, prefix)
        return action, capture, balance, k

    def forward(self, obs: torch.Tensor):
        return self.components(obs)[0]


def load_assets(checkpoint, reference, checkpoint_sha=CHECKPOINT_SHA):
    if sys.version_info[:2] != (3, 11) or mujoco.__version__ != "3.11.0":
        raise ValueError(
            "Saved handstand reference requires Python 3.11 and MuJoCo 3.11.0"
        )
    checked(checkpoint, checkpoint_sha)
    checked(reference, REFERENCE_SHA)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("kind") != "capture_residual_v1"
        or saved.get("reference_sha256") != REFERENCE_SHA
    ):
        raise ValueError("Inconsistent checkpoint metadata")
    state = saved["policy"]
    shapes = {
        "actions": (320, 12),
        "centers": (320, 47),
        "gains": (320, 12, 46),
        "scale": (46,),
        "balance_weight": (12, 34),
    }
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
            raise ValueError(f"Invalid checkpoint tensor: {key}")
    if int(state["active_steps"]) != 110 or not np.isclose(float(state["bound"]), 0.05):
        raise ValueError("Inconsistent phase/bound metadata")
    with np.load(reference, allow_pickle=False) as data:
        trajectory = {key: data[key].copy() for key in data.files}
    if not np.array_equal(
        state["actions"].numpy(), trajectory["actions"].astype(np.float32)
    ):
        raise ValueError("Checkpoint/reference action mismatch")
    return InferencePolicy(state).eval(), trajectory


class TargetPath:
    def __init__(self, trajectory, bounds):
        self.nominal = trajectory["nominal"].copy()
        self.initial = trajectory["initial_target"].copy()
        self.bounds = np.asarray(bounds).copy()
        self.reset()

    def reset(self):
        self.target = self.initial.copy()
        self.step = 0

    def apply(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 12-action vector")
        desired = np.clip(self.nominal + 2 * action.clip(-1, 1), *self.bounds.T)
        self.target += np.clip(
            desired - self.target, -0.7 * SPEED * 0.02, 0.7 * SPEED * 0.02
        )
        self.step += 1
        return self.target.copy()


def finite(value, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Expected finite sensor shape {shape}")
    return value


def rotation(quaternion):
    q = finite(quaternion, (4,))
    if abs(np.linalg.norm(q) - 1) > 0.001:
        raise ValueError("Uncalibrated quaternion")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q / np.linalg.norm(q))
    return matrix.reshape(3, 3)


class SensorEstimator:
    def __init__(
        self,
        model,
        initial_height,
        epoch,
        *,
        calibrated=False,
        joint_order=JOINTS,
        max_interval=0.003,
        max_age=0.006,
    ):
        if not calibrated or list(joint_order) != JOINTS:
            raise ValueError(
                "Explicit frame/height calibration and canonical joint order required"
            )
        if not np.isfinite([initial_height, epoch]).all():
            raise ValueError("Invalid initialization")
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(name).id for name in JOINTS]
        self.qids = model.jnt_qposadr[joints]
        self.vids = model.jnt_dofadr[joints]
        self.base = model.body("base").id
        self.height = float(initial_height)
        self.velocity = np.zeros(3)
        self.timestamp = float(epoch)
        self.samples = 0
        self.max_interval, self.max_age = max_interval, max_age
        if not 0 < max_interval <= 0.012 or not 0 < max_age <= 0.015:
            raise ValueError("Observer timing bounds exceed reviewed limits")
        self.gap_intervals = 0

    def update(self, timestamp, force, quaternion):
        force = finite(force, (3,))
        r = rotation(quaternion)
        dt = float(timestamp) - self.timestamp
        if not np.isfinite(dt) or dt <= 0 or dt > self.max_interval + 1e-9:
            raise ValueError("Duplicate, out-of-order or missing IMU measurement")
        self.gap_intervals += int(dt > 0.003 + 1e-9)
        # Semi-implicit Euler, exactly as research InertialObserver at dt=.002.
        self.velocity += dt * (r @ force + [0, 0, -9.81])
        self.height += dt * self.velocity[2]
        self.timestamp = float(timestamp)
        self.samples += 1

    def com(self, quaternion, gyro, q, dq, now):
        if (
            not np.isfinite(now)
            or now - self.timestamp < -1e-9
            or now - self.timestamp > self.max_age + 1e-9
        ):
            raise ValueError("Stale or future IMU estimate")
        rotation(quaternion)
        d, m = self.data, self.model
        d.qpos[:3] = [0, 0, 1]
        d.qpos[3:7] = quaternion
        d.qpos[self.qids] = finite(q, (12,))
        d.qvel[:3] = 0
        d.qvel[3:6] = finite(gyro, (3,))
        d.qvel[self.vids] = finite(dq, (12,))
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)  # FK/Jacobian only; no collision/dynamics queries.
        jac = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac, self.base)
        return np.r_[
            (self.height + d.subtree_com[self.base, 2] - d.qpos[2]) / 0.6,
            self.velocity + jac @ d.qvel,
        ].astype(np.float32)

    def observation(self, quaternion, gyro, q, dq, target, nominal, step, now):
        r = rotation(quaternion)
        q, dq = finite(q, (12,)), finite(dq, (12,))
        target, nominal = finite(target, (12,)), finite(nominal, (12,))
        base = np.r_[
            r.T @ [0, 0, -1],
            finite(gyro, (3,)),
            q - nominal,
            dq / SPEED,
            (target - nominal) / 2,
            step * 0.02 / 8,
        ].astype(np.float32)
        # Research FK receives rounded float32 observation channels.
        com = self.com(
            quaternion, base[3:6], base[6:18] + nominal, base[18:30] * SPEED, now
        )
        result = np.r_[base, com].astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite observation after sensor normalization")
        return result


class ContactEstimator(SensorEstimator):
    """Complementary base-state observer, anchored only by measured loaded feet.

    On level ground, h_base = radius - (R p_foot)_z and
    v_base = -J_foot(q, R) [0, omega_body, dq] for a non-slipping contact.
    IMU propagation continues in flight; no simulator contact/COM state is read.
    Force threshold units and ordering MUST be verified on the physical robot.
    """

    def __init__(self, *args, contact_threshold=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(contact_threshold) or contact_threshold <= 0:
            raise ValueError("Invalid measured contact-force threshold")
        self.contact_threshold = float(contact_threshold)
        self.feet = [
            self.model.body(f"{leg}_foot").id for leg in ("FL", "FR", "RL", "RR")
        ]
        self.geoms = [
            int(np.flatnonzero(self.model.geom_bodyid == b)[0]) for b in self.feet
        ]
        if any(
            self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in self.geoms
        ):
            raise ValueError("Contact observer requires spherical model feet")
        self.contact_seconds = np.zeros(4)
        self.corrections = 0
        self.last_contact_time = self.timestamp

    def correct_contacts(self, quaternion, gyro, q, dq, loads):
        loads = finite(loads, (4,))
        if np.any(loads < 0):
            raise ValueError("Negative foot load")
        dt = self.timestamp - self.last_contact_time
        if dt <= 0:
            return
        self.last_contact_time = self.timestamp
        self.contact_seconds = np.where(
            loads >= self.contact_threshold, self.contact_seconds + dt, 0
        )
        support = np.flatnonzero(self.contact_seconds >= 0.012 - 1e-9)
        if not len(support):
            return
        m, d = self.model, self.data
        d.qpos[:3] = [0, 0, 0]
        d.qpos[3:7] = quaternion
        d.qpos[self.qids] = finite(q, (12,))
        d.qvel[:3] = 0
        d.qvel[3:6] = finite(gyro, (3,))
        d.qvel[self.vids] = finite(dq, (12,))
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)  # No collision/dynamics or contact queries.
        heights, velocities = [], []
        jac = np.zeros((3, m.nv))
        for i in support:
            g = self.geoms[i]
            heights.append(m.geom_size[g, 0] - d.geom_xpos[g, 2])
            point = d.geom_xpos[g].copy()
            point[2] -= m.geom_size[g, 0]
            mujoco.mj_jac(m, d, jac, None, point, self.feet[i])
            velocities.append(-jac @ d.qvel)
        heights, velocities = np.array(heights), np.array(velocities)
        # Mutually inconsistent loaded feet indicate slip/contact sensor error.
        if (
            np.ptp(heights) > 0.04
            or np.max(np.linalg.norm(velocities - velocities.mean(0), axis=1)) > 0.4
        ):
            return
        # First-order complementary correction; SI state, no policy changes.
        self.height += (1 - np.exp(-dt / 0.03)) * (heights.mean() - self.height)
        self.velocity += (1 - np.exp(-dt / 0.04)) * (velocities.mean(0) - self.velocity)
        self.corrections += 1


class IMUCalibration:
    def __init__(
        self,
        body_from_imu,
        world_alignment,
        lever_arm_body,
        accel_bias_imu,
        gyro_bias_imu,
        accel_scale=1.0,
        max_interval=0.003,
        alpha_tau=0.0,
        stationary_gyro_limit=0.02,
    ):
        self.mount = finite(body_from_imu, (3, 3))
        self.world = finite(world_alignment, (3, 3))
        for matrix in (self.mount, self.world):
            if not np.allclose(
                matrix.T @ matrix, np.eye(3), atol=1e-8
            ) or not np.isclose(np.linalg.det(matrix), 1):
                raise ValueError("Calibration requires proper rotations")
        self.lever = finite(lever_arm_body, (3,))
        self.accel_bias = finite(accel_bias_imu, (3,))
        self.gyro_bias = finite(gyro_bias_imu, (3,))
        self.accel_scale = float(accel_scale)
        self.max_interval, self.alpha_tau = float(max_interval), float(alpha_tau)
        if (
            not 0.8 <= self.accel_scale <= 1.2
            or not 0 < self.max_interval <= 0.012
            or not 0 <= self.alpha_tau <= 0.1
        ):
            raise ValueError("Invalid IMU scale or timing")
        self.stationary_gyro_limit = float(stationary_gyro_limit)
        if not 0 < self.stationary_gyro_limit <= 0.04:
            raise ValueError("Invalid stationary gyro threshold")
        self.alpha = np.zeros(3)
        self.previous_time = None
        self.previous_gyro = None

    def correct(self, timestamp, quaternion, gyro, specific_force):
        if not np.isfinite(timestamp):
            raise ValueError("Invalid calibration timestamp")
        omega = self.mount @ (finite(gyro, (3,)) - self.gyro_bias)
        alpha = np.zeros(3)
        if self.previous_time is not None:
            dt = timestamp - self.previous_time
            if dt <= 0 or dt > self.max_interval + 1e-9:
                raise ValueError("Calibration sample gap")
            measured_alpha = (omega - self.previous_gyro) / dt
            gain = 1.0 if not self.alpha_tau else 1 - np.exp(-dt / self.alpha_tau)
            self.alpha += gain * (measured_alpha - self.alpha)
            alpha = self.alpha
        # First sample must be stationary; derivative noise is not filtered.
        elif np.linalg.norm(omega) > self.stationary_gyro_limit:
            raise ValueError("Calibration initialization must be stationary")
        r = self.world @ rotation(quaternion) @ self.mount.T
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, r.ravel())
        force = self.mount @ (
            (finite(specific_force, (3,)) - self.accel_bias) * self.accel_scale
        )
        force -= np.cross(alpha, self.lever) + np.cross(
            omega, np.cross(omega, self.lever)
        )
        self.previous_time = float(timestamp)
        self.previous_gyro = omega.copy()
        return q, omega, force


def calibrate_stationary(imu_config, readings, *, samples=100, contact_threshold=10.0):
    """Estimate biases from a recent, stationary, loaded four-foot stance.

    A single pose does NOT identify both scale and bias: the configured scale is
    held fixed. Expected specific force is R.T [0,0,g]; no COM truth is used.
    readings: (receipt_time, raw LowState tuple) in canonical encoder order.
    """
    if len(readings) < samples:
        raise ValueError("Need at least 100 recent stationary LowState samples")
    readings = readings[-samples:]
    times = np.array([t for t, _ in readings])
    if (
        np.any(np.diff(times) <= 0)
        or times[-1] - times[0] < 0.15
        or np.max(np.diff(times)) > 0.012
    ):
        raise ValueError("Stationary calibration stream is incomplete")
    calibration = IMUCalibration(**imu_config)
    velocities = np.array([finite(packet[5], (12,)) for _, packet in readings])
    # Real resting logs have isolated encoder-velocity spikes above 0.1 rad/s,
    # while positions stay within 0.0003 rad. Require low window RMS as well as
    # the existing activation peak bound; position/gyro checks still apply.
    if (
        np.max(abs(velocities)) > 0.2
        or np.max(np.sqrt(np.mean(velocities**2, axis=0))) > 0.05
    ):
        raise ValueError("Robot must be stationary for calibration")
    gyro, bias, positions, rotations = [], [], [], []
    for _, packet in readings:
        _, quat, omega, force, q, dq, loads = packet
        if loads is None or np.any(finite(loads, (4,)) < contact_threshold):
            raise ValueError(
                "Stationary calibration requires all four measured foot loads"
            )
        if np.linalg.norm(omega - calibration.gyro_bias) > 0.05:
            raise ValueError("Robot must be stationary for calibration")
        r = calibration.world @ rotation(quat) @ calibration.mount.T
        expected = calibration.mount.T @ (r.T @ [0, 0, 9.81]) / calibration.accel_scale
        bias.append(finite(force, (3,)) - expected)
        gyro.append(finite(omega, (3,)))
        positions.append(finite(q, (12,)))
        rotations.append(r)
    if (
        np.max(np.ptp(positions, axis=0)) > 0.005
        or np.max(abs(np.array(rotations) - rotations[0])) > 0.01
    ):
        raise ValueError("Pose moved during stationary calibration")
    if np.max(np.std(gyro, axis=0)) > 0.015 or np.max(np.std(bias, axis=0)) > 0.3:
        raise ValueError("Stationary calibration sensor noise exceeds reviewed range")
    result = dict(imu_config)
    result["gyro_bias_imu"] = np.mean(gyro, axis=0).tolist()
    result["accel_bias_imu"] = np.mean(bias, axis=0).tolist()
    if (
        np.linalg.norm(result["accel_bias_imu"]) > 1
        or np.linalg.norm(result["gyro_bias_imu"]) > 0.1
    ):
        raise ValueError("Stationary bias correction is implausibly large")
    return result


class Session:
    def __init__(self, policy, trajectory, model, observer_options=None):
        self.observer_options = dict(observer_options or {})
        self.policy, self.trajectory, self.model = policy, trajectory, model
        ids = [model.joint(name).id for name in JOINTS]
        self.path = TargetPath(trajectory, model.jnt_range[ids])
        self.active = False
        self.pending = None

    def activate(self, epoch, initial_height, *, calibrated=False):
        self.cancel()
        options = self.observer_options.copy()
        kind = options.pop("kind", "inertial")
        if kind not in ("inertial", "contact"):
            raise ValueError("Unknown observer kind")
        estimator = ContactEstimator if kind == "contact" else SensorEstimator
        self.estimator = estimator(
            self.model, initial_height, epoch, calibrated=calibrated, **options
        )
        self.path.reset()
        self.telemetry = None
        self.active = True
        self.last_proposal_time = None

    def cancel(self):
        self.active = False
        self.pending = None

    def ingest(self, timestamp, force, quaternion, gyro, q, dq, loads=None):
        if not self.active:
            raise RuntimeError("Explicit activation required")
        try:
            if self.telemetry is None and timestamp == self.estimator.timestamp:
                # Initial standing sample defines epoch; integrate no interval twice.
                finite(force, (3,))
                rotation(quaternion)
            else:
                self.estimator.update(timestamp, force, quaternion)
            finite(gyro, (3,))
            finite(q, (12,))
            finite(dq, (12,))
            if isinstance(self.estimator, ContactEstimator):
                if loads is None:
                    raise ValueError("Contact observer requires measured foot loads")
                self.estimator.correct_contacts(quaternion, gyro, q, dq, loads)
            self.telemetry = (
                np.array(quaternion),
                np.array(gyro),
                np.array(q),
                np.array(dq),
            )
        except (ValueError, TypeError):
            self.cancel()
            raise

    def snapshot(self, now):
        if not self.active or self.telemetry is None or self.pending is not None:
            raise RuntimeError("Inactive, missing telemetry, or unacknowledged command")
        try:
            return self.estimator.observation(
                *self.telemetry,
                self.path.target,
                self.path.nominal,
                self.path.step,
                now,
            )
        except ValueError:
            self.cancel()
            raise

    def propose(self, now):
        obs = self.snapshot(now)
        if (
            self.last_proposal_time is not None
            and abs(now - self.last_proposal_time - 0.02) > 0.002
        ):
            self.cancel()
            raise ValueError("Control cadence differs from 50 Hz")
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs)).numpy()
        finite(action, (12,))
        # Apply only to a temporary path: state advances on acknowledgement.
        desired = np.clip(
            self.path.nominal + 2 * action.clip(-1, 1), *self.path.bounds.T
        )
        self.last_proposal_time = now
        self.pending = self.path.target + np.clip(
            desired - self.path.target, -0.7 * SPEED * 0.02, 0.7 * SPEED * 0.02
        )
        return action.copy(), self.pending.copy()

    def acknowledge(self, target):
        if (
            not self.active
            or self.pending is None
            or not np.array_equal(target, self.pending)
        ):
            self.cancel()
            raise ValueError("Applied target differs from proposed bounded target")
        self.path.target = self.pending.copy()
        self.path.step += 1
        self.pending = None


SLICES = {
    "projected_gravity": (0, 3),
    "base_ang_vel": (3, 6),
    "dof_pos_obs": (6, 18),
    "dof_vel": (18, 30),
    "handstand_previous_target": (30, 42),
    "handstand_elapsed": (42, 43),
    "handstand_com": (43, 47),
}


class CustomDeployConfig(CustomConfig):
    fixed_gains = require_intermediate = True
    task_name = "go2handstand"
    default_dof_pos = [
        -2.2248883690812155e-09,
        0.7334159659826462,
        -1.4654903159477315,
        2.2248883690812155e-09,
        0.7334159659826462,
        -1.4654903159477315,
        -2.2248883253671876e-09,
        0.7334159659826462,
        -1.4654903159477313,
        2.2248883253671876e-09,
        0.7334159659826462,
        -1.4654903159477313,
    ]

    def get_custom_buffers_dict(self):
        return get_custom_buffers_dict()

    def update_custom_buffers(self, main_controller, t):
        return update_custom_buffers(main_controller, t)

    def get_custom_obs_pieces(self):
        return get_custom_obs_pieces()

    # Optional runner bridges preserve every IMU sample and actual publications.
    def prepare_runtime(self, cfg):
        return prepare_runtime(cfg)

    def initialize_controller(self, main_controller):
        return initialize_controller(main_controller)

    def on_lowstate_msg(self, main_controller, msg, received):
        return on_lowstate_msg(main_controller, msg, received)

    def bound_command(self, main_controller, offset, kp_mult=1.0, kd_mult=1.0):
        return bound_command(main_controller, offset, kp_mult, kd_mult)

    def on_command_sent(self, main_controller, lowcmd, accepted, t):
        return on_command_sent(main_controller, lowcmd, accepted, t)

    def runtime_status(self, c):
        with c.handstand_rx_lock:
            samples = len(c.handstand_stationary_rx)
        result = {
            "task": self.task_name,
            "calibration_confirmed": c.handstand_calibration.get("calibrated") is True,
            "stationary_samples": samples,
            "required_samples": c.handstand_calibration.get("stationary_samples", 0),
            "step": c.handstand_session.path.step,
        }
        with c.last_lowstate_msg_lock:
            msg = c.last_lowstate_msg
            if msg is not None:
                q = read_lowstate(msg)[4]
                error = abs(q - c.handstand_session.path.initial)
                result["largest_initial_joint_error_rad"] = float(error.max())
                result["limiting_joint"] = JOINTS[int(error.argmax())]
        return result

    obs_vector = list(SLICES)
    custom_obs_sizes = {name: end - start for name, (start, end) in SLICES.items()}
    # Review requirements only: Cornell DeployConfig does not inherit these!
    ctrl_freq, kp, kd, exp_moving_avg = 50, 60.0, 3.0, False

    class DeployScaling:
        projected_gravity = base_ang_vel = dof_pos_obs = dof_vel = 1.0
        handstand_previous_target = handstand_elapsed = handstand_com = 1.0
        dof_pos_target = 2.0  # RLController action multiplier; NOT an obs divisor.


class CandidateDeployConfig(CustomDeployConfig):
    """Explicitly exported staff retrain; never changes the pinned release task."""

    task_name = "go2handstand_candidate"

    def prepare_runtime(self, cfg):
        digest = cfg.bundle_meta.get("candidate_checkpoint_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Candidate requires an explicit checkpoint SHA256")
        return prepare_runtime(cfg, checkpoint_sha=digest)


def get_custom_buffers_dict():
    return {
        "handstand_rx_lock": threading.Lock(),
        "handstand_stationary_rx": deque(maxlen=256),
        "handstand_rx": deque(maxlen=256),
        "handstand_rx_fault": None,
        "handstand_rx_seen": False,
        "handstand_last_mode": None,
        "handstand_tick": None,
        "handstand_gap_intervals": 0,
        "handstand_sensor_time": None,
        "handstand_last_receive": None,
        "handstand_last_sent": None,
        "handstand_last_control": None,
        "handstand_activation_wait": None,
        "handstand_session": None,
        "handstand_snapshot": None,
        "handstand_mode": "runner",
        "handstand_imu_queue": deque(maxlen=100),
        "handstand_control_snapshot": None,
        "handstand_activation": None,
        "handstand_command_ack": None,
        "handstand_cancel_requested": False,
    }


def _update_software_buffers(main_controller, t):
    """Consume queued sensor intervals once; phase advances only on explicit ack.

    External dry-run driver appends calibrated tuples
    (timestamp, force, wxyz, body_gyro, canonical_q, canonical_dq).
    The control snapshot is (timestamp, wxyz, gyro, q, dq), from the same clock.
    No latest-LowState sample is fabricated, repeated or used as an ack.
    """
    c = main_controller
    c.handstand_snapshot = None
    session = c.handstand_session
    if c.handstand_mode != "software_validation" or session is None:
        raise RuntimeError("NOT READY: configure software-only telemetry validation")
    try:
        if c.handstand_cancel_requested:
            session.cancel()
            c.handstand_imu_queue.clear()
            c.handstand_activation = c.handstand_command_ack = None
            c.handstand_control_snapshot = None
            c.handstand_cancel_requested = False
            raise RuntimeError("Handstand cancelled; explicit reset required")
        if c.handstand_activation is not None:
            epoch, initial_height, calibrated = c.handstand_activation
            session.activate(epoch, initial_height, calibrated=calibrated)
            c.handstand_activation = None
        if c.handstand_command_ack is not None:
            session.acknowledge(c.handstand_command_ack)
            c.handstand_command_ack = None
        if len(c.handstand_imu_queue) == c.handstand_imu_queue.maxlen:
            raise ValueError("IMU queue capacity reached; possible lost measurements")
        while c.handstand_imu_queue:
            session.ingest(*c.handstand_imu_queue.popleft())
        snapshot = c.handstand_control_snapshot
        if snapshot is None or abs(float(snapshot[0]) - t) > 1e-6:
            raise ValueError("Missing synchronized control snapshot")
        # Validate encoders/orientation even when an upstream driver supplies a
        # distinct current FK snapshot after the last integration interval.
        rotation(snapshot[1])
        for value, shape in zip(snapshot[2:], [(3,), (12,), (12,)]):
            finite(value, shape)
        session.telemetry = tuple(np.array(value) for value in snapshot[1:])
        c.handstand_snapshot = session.snapshot(t)
        c.handstand_control_snapshot = None
    except (ValueError, TypeError, RuntimeError):
        session.cancel()
        c.handstand_snapshot = None
        raise


def read_lowstate(msg):
    """Copy actual Unitree LowState fields into canonical FL/FR/RL/RR order."""
    if msg is None:
        raise ValueError("Missing runner LowState")
    tick = int(msg.tick)
    if not 0 <= tick < 2**32:
        raise ValueError("Invalid LowState tick")
    imu = msg.imu_state
    return (
        tick,
        finite(imu.quaternion, (4,)).copy(),
        finite(imu.gyroscope, (3,)).copy(),
        finite(imu.accelerometer, (3,)).copy(),
        finite([m.q for m in msg.motor_state[:12]], (12,))[PERMUTATION],
        finite([m.dq for m in msg.motor_state[:12]], (12,))[PERMUTATION],
        finite(msg.foot_force, (4,))[[1, 0, 3, 2]]
        if hasattr(msg, "foot_force")
        else None,
    )


def prepare_runtime(cfg, checkpoint_sha=CHECKPOINT_SHA):
    """Load and warm before DDS initialization; calibration is never invented."""
    if cfg.ctrl_freq != 50 or cfg.exp_moving_avg or cfg.kp != 60 or cfg.kd != 3:
        raise ValueError("Handstand requires 50 Hz, EMA off, Kp=60, Kd=3")
    if cfg.obs_vector != list(SLICES) or any(
        cfg.obs_sizes.get(k) != end - start for k, (start, end) in SLICES.items()
    ):
        raise ValueError("Handstand observation order/sizes differ from training")
    if any(
        not np.array_equal(np.asarray(getattr(cfg.DeployScaling, k)), 1.0)
        for k in SLICES
    ) or not np.array_equal(np.asarray(cfg.DeployScaling.dof_pos_target), 2.0):
        raise ValueError("Handstand observation/action scaling differs from training")
    folder = Path(cfg.bundle_dir)
    policy, trajectory = load_assets(
        folder / "capture_checkpoint.pt", folder / "trajectory.npz", checkpoint_sha
    )
    candidate = cfg.loaded_policy
    expected = policy.state_dict()
    actual = candidate.state_dict()
    if set(expected) != set(actual) or any(
        not torch.equal(expected[k], actual[k]) for k in expected
    ):
        raise ValueError("Bundle policy is not the complete pinned CapturePolicy")
    with torch.no_grad():
        for step in [0, 1, 47, 109, 110, 111, 319, 320, 400]:
            obs = policy.centers[min(step, 319)].clone()
            obs[:42] += torch.linspace(-0.003, 0.003, 42)
            obs[42] = step / 400
            if not torch.allclose(candidate(obs), policy(obs), atol=1e-7, rtol=0):
                raise ValueError("Bundle hybrid inference differs at a phase boundary")
    if not np.allclose(
        np.asarray(cfg.default_dof_pos), trajectory["nominal"], atol=1e-7, rtol=0
    ):
        raise ValueError("Bundle nominal posture mismatch")
    path = os.environ.get(
        "GO2_HANDSTAND_CALIBRATION", str(folder / "calibration.model.json")
    )
    if not Path(path).exists():
        raise ValueError(
            "Set GO2_HANDSTAND_CALIBRATION or export a new model calibration"
        )
    calibration = json.loads(Path(path).read_text())
    if calibration.get("calibrated") is not True and calibration.get("source") not in {
        "model_and_ta_estimates",
        "staff_log_candidate",
    }:
        raise ValueError(
            "IMU/frame/height calibration or explicit model preflight is required"
        )
    for key in ["initial_height", "tick_period_s"]:
        if not np.isfinite(calibration[key]) or calibration[key] <= 0:
            raise ValueError("Invalid height or tick period")
    age = calibration.get("max_lowstate_age_s", 0.006)
    if not np.isfinite(age) or not 0 < age <= 0.015:
        raise ValueError("LowState age must remain bounded at 15 ms or less")
    command_age = calibration.get("max_command_state_age_s", age)
    if not np.isfinite(command_age) or not 0 < command_age <= 0.020:
        raise ValueError("Command snapshot age must remain bounded at 20 ms")
    count = calibration.get("stationary_samples", 0)
    if count != 0 and (not isinstance(count, int) or not 100 <= count <= 256):
        raise ValueError("Stationary calibration requires 100 to 256 samples")
    IMUCalibration(**calibration["imu"])
    model_path = checked(folder / "go2_fk.mjb", cfg.bundle_meta["handstand_fk_sha256"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with torch.no_grad():
        for _ in range(5):
            policy(torch.zeros(47))
    return policy, trajectory, model, calibration


def initialize_controller(c):
    resources = getattr(c.cfg, "custom_runtime", None)
    if resources is None:
        raise ValueError("Load the handstand bundle before initializing the runner")
    policy, trajectory, model, calibration = resources
    c.handstand_session = Session(
        policy, trajectory, model, calibration.get("observer")
    )
    c.handstand_calibration = calibration
    c.handstand_snapshot = np.zeros(47, np.float32)
    log_event = getattr(getattr(c, "csv_logger", None), "log_event", None)
    if log_event is not None:
        log_event("handstand_runtime_config", {"calibration": calibration})


def on_lowstate_msg(c, msg, received):
    """Optional runner callback bridge: preserve EVERY IMU frame, no SDK access."""
    with c.handstand_rx_lock:
        c.handstand_rx_seen = True
        if getattr(c._state, "name", "") != "CUSTOM_CTRL":
            c.handstand_rx.clear()
            c.handstand_rx_fault = None
            c.handstand_last_mode = getattr(c._state, "name", "")
            if c.handstand_last_mode == "INTERMEDIATE":
                try:
                    c.handstand_stationary_rx.append(
                        (float(received), read_lowstate(msg))
                    )
                except (ValueError, TypeError, AttributeError):
                    c.handstand_stationary_rx.clear()
            else:
                c.handstand_stationary_rx.clear()
            return
        try:
            packet = read_lowstate(msg)
            if len(c.handstand_rx) == c.handstand_rx.maxlen:
                raise ValueError("LowState buffer overflow: lost IMU samples")
            c.handstand_rx.append((float(received), packet))
        except (ValueError, TypeError, AttributeError) as exc:
            c.handstand_rx_fault = str(exc)


def _runner_fault(c, message):
    c.handstand_session.cancel()
    c.handstand_snapshot = None
    raise ValueError(message)


def update_custom_buffers(c, t):
    """Read runner state directly; external per-cycle preparation is unnecessary."""
    if c.handstand_mode == "software_validation":
        return _update_software_buffers(c, t)
    if c.handstand_session is None:
        initialize_controller(c)
    mode = getattr(c._state, "name", "")
    if mode != "CUSTOM_CTRL":
        c.handstand_session.cancel()
        c.handstand_snapshot = np.zeros(47, np.float32)
        c.handstand_last_mode = mode
        c.handstand_activation_wait = None
        with c.handstand_rx_lock:
            c.handstand_rx.clear()
            c.handstand_rx_fault = None
        return
    try:
        with c.last_lowstate_msg_lock:
            # Actual runner source, not a driver-injected control snapshot.
            current = read_lowstate(c.last_lowstate_msg)
            received = float(c.last_lowstate_received_time)
            with c.handstand_rx_lock:
                packets = list(c.handstand_rx)
                c.handstand_rx.clear()
                fault = c.handstand_rx_fault
        # A DDS callback may arrive after the cycle-entry timestamp t.
        # Check receipt age using the runner clock AFTER copying its snapshot.
        receipt_now = getattr(c, "lowstate_clock", lambda: t)()
        if fault:
            raise ValueError(fault)
        if not c.handstand_rx_seen:
            raise ValueError("Runner callback bridge is missing")
        if (
            not np.isfinite([t, receipt_now]).all()
            or not -1e-9
            <= receipt_now - received
            <= c.handstand_calibration.get("max_lowstate_age_s", 0.006) + 1e-9
        ):
            raise ValueError("Stale or future LowState")
        starting = c.handstand_last_mode != "CUSTOM_CTRL"
        if starting:
            if c.handstand_calibration.get("calibrated") is not True:
                raise ValueError(
                    "Model preflight only: staff calibration confirmation "
                    "required before activation"
                )
            if c.handstand_last_sent is None:
                raise ValueError(
                    "Enter INTERMEDIATE before CUSTOM_CTRL "
                    "to establish the dispatched target"
                )
            if not packets:
                if c.handstand_activation_wait is None:
                    c.handstand_activation_wait = t
                if t - c.handstand_activation_wait > 0.02:
                    raise ValueError("No activation IMU stream")
                c.handstand_snapshot = None
                return False
            imu_config = c.handstand_calibration["imu"]
            if c.handstand_calibration.get("stationary_samples", 0):
                with c.handstand_rx_lock:
                    readings = list(c.handstand_stationary_rx)
                    c.handstand_stationary_rx.clear()
                if not readings or not -1e-9 <= receipt_now - readings[-1][0] <= 0.04:
                    raise ValueError(
                        "Need a fresh stationary calibration before activation"
                    )
                imu_config = calibrate_stationary(
                    imu_config,
                    readings,
                    samples=c.handstand_calibration["stationary_samples"],
                    contact_threshold=c.handstand_calibration["observer"][
                        "contact_threshold"
                    ],
                )
            c.handstand_effective_imu = imu_config
            log_event = getattr(getattr(c, "csv_logger", None), "log_event", None)
            if log_event is not None:
                log_event(
                    "handstand_calibration",
                    {
                        "configured": c.handstand_calibration,
                        "effective_imu": imu_config,
                    },
                )
            c.handstand_imu_calibration = IMUCalibration(**imu_config)
            c.handstand_tick = None
            c.handstand_gap_intervals = 0
            c.handstand_sensor_time = 0.0
            c.handstand_last_raw = None
            c.handstand_activation_wait = None
            c.handstand_last_receive = None
            c.handstand_last_control = None
        elif (
            c.handstand_last_control is not None
            and abs(t - c.handstand_last_control - 0.02) > 0.002 + 1e-9
        ):
            raise ValueError("Runner cadence differs from 50 Hz")
        session = c.handstand_session
        for arrival, packet in packets:
            tick, quaternion, gyro, force, q, dq, loads = packet
            if c.handstand_tick is not None:
                delta = (tick - c.handstand_tick) % (2**32)
                if delta == 0:
                    if any(
                        not np.array_equal(a, b)
                        for a, b in zip(packet[1:], c.handstand_last_raw[1:])
                    ):
                        raise ValueError("Conflicting duplicate IMU timestamp")
                    continue  # Duplicate never advances integration/freshness.
                dt = delta * c.handstand_calibration["tick_period_s"]
                if (
                    not 0
                    < dt
                    <= session.observer_options.get("max_interval", 0.003) + 1e-9
                ):
                    raise ValueError("Missing or out-of-order IMU frame")
                c.handstand_gap_intervals += int(dt > 0.003 + 1e-9)
                c.handstand_sensor_time += dt
            else:
                session.activate(
                    0.0, c.handstand_calibration["initial_height"], calibrated=True
                )
                session.path.target = finite(c.handstand_last_sent, (12,)).copy()
                if np.any(session.path.target < session.path.bounds[:, 0]) or np.any(
                    session.path.target > session.path.bounds[:, 1]
                ):
                    raise ValueError(
                        "Initial dispatched target is outside joint limits"
                    )
            quaternion, gyro, force = c.handstand_imu_calibration.correct(
                c.handstand_sensor_time, quaternion, gyro, force
            )
            session.ingest(
                c.handstand_sensor_time, force, quaternion, gyro, q, dq, loads
            )
            c.handstand_last_raw = packet
            c.handstand_tick = tick
            c.handstand_last_receive = arrival
        if (
            c.handstand_last_receive is None
            or not -1e-9
            <= receipt_now - c.handstand_last_receive
            <= c.handstand_calibration.get("max_lowstate_age_s", 0.006) + 1e-9
        ):
            raise ValueError("No fresh distinct IMU sample")
        if c.handstand_tick != current[0]:
            raise ValueError("Unsynchronized encoder/IMU snapshot")
        if starting and (
            np.max(abs(session.telemetry[2] - session.path.initial)) > 0.15
            or np.max(abs(session.telemetry[3])) > 0.2
        ):
            raise ValueError(
                "Activation requires calibrated stationary reference posture"
            )
        q, dq = session.telemetry[2:]
        if (
            np.any(q < session.path.bounds[:, 0] - 0.001)
            or np.any(q > session.path.bounds[:, 1] + 0.001)
            or np.any(abs(dq) > SPEED * 1.001)
        ):
            raise ValueError("Encoder joint range or velocity protection tripped")
        if session.pending is not None:
            raise ValueError("Previous command was not published")
        c.handstand_snapshot = session.snapshot(c.handstand_sensor_time)
        c.handstand_last_mode = mode
        c.handstand_last_control = t
    except (ValueError, TypeError, RuntimeError, AttributeError) as exc:
        _runner_fault(c, str(exc))


def bound_command(c, offset, kp_mult=1.0, kd_mult=1.0):
    if getattr(c._state, "name", "") != "CUSTOM_CTRL":
        return offset + c.cfg.default_dof_pos
    session = c.handstand_session
    if not session.active or c.handstand_snapshot is None:
        raise ValueError("Handstand is inactive or faulted")
    if hasattr(c, "lowstate_clock"):
        # Bound age of the snapshot actually used by the actor, not a newer
        # callback which arrived while this batch was being processed.
        age = c.lowstate_clock() - c.handstand_last_receive
        limit = c.handstand_calibration.get("max_command_state_age_s", 0.006)
        if not np.isfinite(age) or not -1e-9 <= age <= limit + 1e-9:
            _runner_fault(c, "Command sensor snapshot exceeded age budget")
    if c.kp_mult != 1 or c.kd_mult != 1 or kp_mult != 1 or kd_mult != 1:
        raise ValueError("Handstand gains must stay 60/3")
    if session.pending is not None:
        raise ValueError("Unpublished handstand command")
    desired = np.clip(
        session.path.nominal + finite(np.asarray(offset), (12,)), *session.path.bounds.T
    )
    target = session.path.target + np.clip(
        desired - session.path.target, -0.7 * SPEED * 0.02, 0.7 * SPEED * 0.02
    )
    # Unitree MotorCmd.q is float32 on the wire. Quantize inward so neither
    # the joint bounds nor the applied-target slew limit is rounded outward.
    lo = np.maximum(session.path.bounds[:, 0], session.path.target - 0.7 * SPEED * 0.02)
    hi = np.minimum(session.path.bounds[:, 1], session.path.target + 0.7 * SPEED * 0.02)
    wire = target.astype(np.float32)
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
    target = wire.astype(float)
    session.pending = target.copy()
    # No second action scale or EMA. Ack tracks the actual float32 command.
    c.last_action[0] = torch.tensor(
        target - session.path.nominal, dtype=c.last_action.dtype
    )
    return torch.from_numpy(target.copy())


def on_command_sent(c, lowcmd, accepted, t):
    """Track successful DDS publication, never claim physical motor tracking."""
    if accepted is not True:
        _runner_fault(c, "LowCmd publication failed or was not confirmed")
    target = finite([m.q for m in lowcmd.motor_cmd[:12]], (12,))[PERMUTATION]
    if getattr(c._state, "name", "") == "CUSTOM_CTRL":
        c.handstand_session.acknowledge(target)
    c.handstand_last_sent = target.copy()


def get_custom_obs_pieces():
    # Standard Cornell gyro/gravity/q/dq getters match units/order only for an
    # uncalibrated latest message. Override them to keep ONE calibrated snapshot.
    def getter(start, end):
        def piece(main_controller, lowstate_msg):
            if main_controller.handstand_snapshot is None:
                raise RuntimeError("Missing handstand sensor estimate")
            return torch.from_numpy(
                main_controller.handstand_snapshot[start:end].copy()
            )

        return piece

    return {name: getter(start, end) for name, (start, end) in SLICES.items()}


def build_fk_model():
    """Reuse Cornell's existing nominal Go2 model/URDF for FK only, never step it."""
    from gym.envs.go2.go2_config import Go2Cfg
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.helpers import class_to_dict

    def namespace(value):
        return (
            SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
            if isinstance(value, dict)
            else value
        )

    cfg = namespace(class_to_dict(Go2Cfg()))
    cfg.sim_dt = 0.002
    backend = MuJocoCPUBackend()
    model = backend._load_model(cfg, discard_visual=True)
    backend._configure_model(model, cfg, "cpu")
    return model


def configure_software_validation(controller, checkpoint, reference):
    if sys.version_info[:2] != (3, 11) or mujoco.__version__ != "3.11.0":
        raise RuntimeError("Reference validation requires Python 3.11/MuJoCo 3.11.0")
    policy, trajectory = load_assets(checkpoint, reference)
    # Complete law is scripted, not zero-vector traced; both phase branches live.
    policy = torch.jit.script(policy)
    with torch.no_grad():
        for _ in range(5):
            policy(torch.zeros(47))
    controller.handstand_session = Session(policy, trajectory, build_fk_model())
    controller.handstand_mode = "software_validation"
    return policy, trajectory


def dry_run(checkpoint, reference, recording):
    """Recorded SIMULATED sensors only; acknowledgements belong to dry-run sink."""
    torch.set_num_threads(1)
    c = SimpleNamespace(**get_custom_buffers_dict())
    policy, trajectory = configure_software_validation(c, checkpoint, reference)
    with np.load(recording, allow_pickle=False) as data:
        samples, controls = data["samples"], data["controls"]
    if (
        samples.ndim != 2
        or samples.shape[1] != 8
        or controls.ndim != 2
        or controls.shape[1] != 92
    ):
        raise ValueError("Expected recorded synthetic IMU8/control92 schema")
    c.handstand_activation = (0.0, 0.3413013268255194, True)
    j = 0
    error = dict(observation=0.0, target=0.0, previous_applied_target=0.0)
    for i, row in enumerate(controls):
        now = row[0]
        if i == 0:
            c.handstand_imu_queue.append(
                (now, [0, 0, 9.81], row[1:5], row[5:8], row[8:20], row[20:32])
            )
        while j < len(samples) and samples[j, 0] <= now + 1e-9:
            sample = samples[j]
            c.handstand_imu_queue.append(
                (sample[0], sample[5:8], sample[1:5], row[5:8], row[8:20], row[20:32])
            )
            j += 1
        c.handstand_control_snapshot = (now, row[1:5], row[5:8], row[8:20], row[20:32])
        update_custom_buffers(c, now)
        obs = torch.cat(
            [
                get_custom_obs_pieces()[name](c, None)
                for name in CustomDeployConfig.obs_vector
            ]
        ).numpy()
        error["observation"] = max(
            error["observation"], float(abs(obs - row[45:92]).max())
        )
        error["previous_applied_target"] = max(
            error["previous_applied_target"],
            float(abs(c.handstand_session.path.target - row[32:44]).max()),
        )
        action, target = c.handstand_session.propose(now)
        if not np.isfinite(action).all() or (abs(action) > 1).any():
            raise ValueError("Invalid action")
        if i + 1 < len(controls):
            error["target"] = max(
                error["target"], float(abs(target - controls[i + 1, 32:44]).max())
            )
        # This is an explicit SOFTWARE sink acknowledgment, never a hardware ack.
        c.handstand_command_ack = target.copy()
    c.handstand_cancel_requested = True
    try:
        update_custom_buffers(c, float(controls[-1, 0]))
    except RuntimeError as exc:
        if "cancelled" not in str(exc):
            raise
    if max(error.values()) > 1e-7:
        raise AssertionError(error)
    result = {
        "software_only": True,
        "commands": len(controls),
        "imu_samples": j,
        "maximum_absolute_errors": error,
        "physical_activation_enabled": False,
    }
    print(json.dumps(result, indent=2))
    return result


def model_calibration(trajectory):
    """URDF mounting and saved initial base height, NOT hardware measurements."""
    import xml.etree.ElementTree as ET

    path = Path(__file__).resolve().parents[2] / "resources/robots/go2/urdf/go2.urdf"
    origin = ET.parse(path).getroot().find("joint[@name='imu_joint']/origin")
    if origin is None or origin.attrib.get("rpy") != "0 0 0":
        raise ValueError("Review changed model IMU mounting")
    model = build_fk_model()
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model, data, trajectory["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION
    )
    return {
        "source": "model_and_ta_estimates",
        "calibrated": False,
        "initial_height": float(data.qpos[2]),
        "tick_period_s": 0.001,
        "max_lowstate_age_s": 0.015,
        "max_command_state_age_s": 0.020,
        "stationary_samples": 100,
        "observer": {
            "kind": "contact",
            "max_interval": 0.012,
            "max_age": 0.015,
            "contact_threshold": 10.0,
        },
        "imu": {
            "body_from_imu": np.eye(3).tolist(),
            "world_alignment": np.eye(3).tolist(),
            "lever_arm_body": [float(x) for x in origin.attrib["xyz"].split()],
            "accel_bias_imu": [0.0, 0.0, 0.0],
            "gyro_bias_imu": [0.0, -0.009, 0.0],
            "accel_scale": 9.81 / 9.55,
            "max_interval": 0.012,
            "alpha_tau": 0.02,
            "stationary_gyro_limit": 0.04,
        },
        "requires_staff_confirmation": [
            "Confirm model vs physical mounting and world alignment/yaw",
            "Initial height is the model base origin, not COM or body surface",
            "Confirm provisional 0.001 s tick units; receipt rate is insufficient",
            "Confirm foot-force validity, units, 10 raw-unit threshold, "
            "FR/FL/RR/RL order and non-slip level ground",
            "Rest norm 9.55 is a scale hypothesis; biases need stationary samples",
            "Verify timing tails, motor limits and recovery",
        ],
    }


def export_runtime_bundle(checkpoint, reference, destination, *, candidate=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a fresh bundle directory")
    checkpoint_sha = (
        hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if candidate
        else CHECKPOINT_SHA
    )
    policy, trajectory = load_assets(checkpoint, reference, checkpoint_sha)
    model = build_fk_model()
    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / "capture_checkpoint.pt")
    shutil.copy2(reference, destination / "trajectory.npz")
    torch.jit.save(torch.jit.script(policy), str(destination / "policy.pt"))
    mujoco.mj_saveModel(model, str(destination / "go2_fk.mjb"))

    def digest(name):
        return hashlib.sha256((destination / name).read_bytes()).hexdigest()

    meta = {
        "task_name": "go2handstand_candidate" if candidate else "go2handstand",
        "custom_implementation": "go2handstand.py",
        "obs_vector": list(SLICES),
        "obs_sizes": CustomDeployConfig.custom_obs_sizes,
        "scaling": {name: 1.0 for name in SLICES} | {"dof_pos_target": 2.0},
        "default_dof_pos": trajectory["nominal"].tolist(),
        "phase_frequency": 0.0,
        "kp": 60.0,
        "kd": 3.0,
        "ctrl_freq": 50,
        "exp_moving_avg": False,
        "command_limits": {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "yaw_vel": 0.0},
        "num_obs": 47,
        "num_actions": 12,
        "policy_sha256": digest("policy.pt"),
        "handstand_fk_sha256": digest("go2_fk.mjb"),
        "checkpoint_sha256": checkpoint_sha,
        "reference_sha256": REFERENCE_SHA,
        "physical_readiness": "UNVALIDATED",
    }
    if candidate:
        meta["candidate_checkpoint_sha256"] = checkpoint_sha
        meta["candidate_status"] = (
            "staff retrain; performance must be evaluated separately"
        )
    (destination / "deploy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Keep the old example name as an alias, now with complete model values.
    example = json.dumps(model_calibration(trajectory), indent=2) + "\n"
    for name in ("calibration.model.json", "calibration.example.json"):
        (destination / name).write_text(example)
    return meta


if __name__ == "__main__":
    # File-path execution needs repository root for existing Cornell model code.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--export-bundle", type=Path)
    parser.add_argument(
        "--candidate",
        action="store_true",
        help="Explicit separate candidate task for a retrained checkpoint",
    )
    arguments = parser.parse_args()
    if arguments.export_bundle:
        export_runtime_bundle(
            arguments.checkpoint,
            arguments.reference,
            arguments.export_bundle,
            candidate=arguments.candidate,
        )
    elif arguments.recording:
        dry_run(arguments.checkpoint, arguments.reference, arguments.recording)
    else:
        parser.error("Choose --recording or --export-bundle")
