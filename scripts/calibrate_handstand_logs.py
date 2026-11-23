"""Estimate identifiable calibration values from Cornell lowstate.csv offline.

Never connects to a robot. Writes a review candidate with calibrated=false;
mounting/yaw/lever arm/contact units cannot be identified from one resting pose.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_deploy.custom_implementation.go2handstand import (
    CustomDeployConfig,
    JOINTS,
    PERMUTATION,
    calibrate_stationary,
    build_fk_model,
    rotation,
)


def read_log(path):
    rows = []
    with Path(path).open() as file:
        for row in csv.DictReader(file):

            def values(names):
                x = np.array([float(row[n]) for n in names])
                if not np.isfinite(x).all():
                    raise ValueError("Nonfinite calibration log")
                return x

            t = float(row["t"])
            tick = int(row["tick"])
            if not np.isfinite(t) or not 0 <= tick < 2**32:
                raise ValueError("Invalid calibration log timestamp")
            quat = values([f"imu_quat_{a}" for a in "wxyz"])
            rotation(quat)  # Validate quaternion before computing any estimates.
            gyro = values([f"imu_gyro_{a}" for a in "xyz"])
            accel = values([f"imu_accel_{a}" for a in "xyz"])
            q = values([f"motor{i}_q" for i in range(12)])[PERMUTATION]
            dq = values([f"motor{i}_dq" for i in range(12)])[PERMUTATION]
            loads = values([f"foot_force_{i}" for i in range(4)])[[1, 0, 3, 2]]
            rows.append((t, (tick, quat, gyro, accel, q, dq, loads), row["state"]))
    if len(rows) < 100:
        raise ValueError("Need at least 100 LowState rows")
    return rows


def timing(rows):
    times = np.array([r[0] for r in rows])
    ticks = np.array([r[1][0] for r in rows], dtype=np.int64)
    intervals = np.diff(times)
    deltas = np.diff(ticks) % 2**32
    if np.any(intervals <= 0) or np.any(deltas >= 2**31):
        raise ValueError("Nonmonotonic calibration timestamps")
    distinct = np.r_[True, deltas > 0]
    elapsed_ticks = np.r_[0, np.cumsum(deltas)][distinct]
    elapsed = (times - times[0])[distinct]
    if len(elapsed) < 100 or elapsed[-1] < 0.15:
        raise ValueError("Insufficient distinct sensor timestamps")
    slope, intercept = np.polyfit(elapsed_ticks, elapsed, 1)
    if not 0 < slope < 1:
        raise ValueError("Invalid estimated tick period")
    source_gaps = deltas[deltas > 0] * slope
    return {
        "estimated_tick_period_s": float(slope),
        "distinct_rate_hz": float((len(elapsed) - 1) / elapsed[-1]),
        "duplicate_ticks": int(np.sum(deltas == 0)),
        "receipt_gap_max_s": float(intervals.max()),
        "source_gap_max_s": float(source_gaps.max()),
        "source_gaps_over_12ms": int(np.sum(source_gaps > 0.012 + 1e-6)),
        "receipt_jitter_rms_s": float(
            np.std(elapsed - (slope * elapsed_ticks + intercept))
        ),
        "absolute_transport_latency": "not identifiable without clock synchronization",
    }


def fit_accelerometer(means):
    """Fit a = bias + radius * direction from distinct stationary pose means."""
    a = np.asarray(means, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 4 or not np.isfinite(a).all():
        raise ValueError(
            "Supply >=4 diverse stationary acceleration means, shape (N,3)"
        )
    design = np.c_[2 * a, np.ones(len(a))]
    if np.linalg.matrix_rank(design) < 4 or np.linalg.cond(design) > 200:
        raise ValueError(
            "Stationary poses do not identify accelerometer scale and bias"
        )
    solution = np.linalg.lstsq(design, np.sum(a * a, axis=1), rcond=None)[0]
    bias = solution[:3]
    radius = np.sqrt(solution[3] + bias @ bias)
    scale = 9.81 / radius
    if not np.isfinite(scale) or not 0.8 <= scale <= 1.2 or np.linalg.norm(bias) > 1:
        raise ValueError("Accelerometer fit outside reviewed calibration bounds")
    rms = float(
        np.sqrt(np.mean((np.linalg.norm(a - bias, axis=1) * scale - 9.81) ** 2))
    )
    if rms > 0.1:
        raise ValueError("Accelerometer fit residual exceeds 0.1 m/s^2")
    return float(scale), bias.tolist(), rms


def command_evidence(rows, path):
    """Compare targets to telemetry 20 ms later; evidence, not a motor-limit test."""
    modes = dict(Counter(r[2] for r in rows))
    if not path.exists():
        return {"states": modes, "control_rows": 0, "reason": "No control.csv"}
    with path.open() as file:
        commands = list(csv.DictReader(file))
    if not commands:
        return {
            "states": modes,
            "control_rows": 0,
            "reason": "No policy commands were logged; activation did not run",
        }
    times = np.array([r[0] for r in rows])
    targets, measured = [], []
    for command in commands:
        t = float(command["t"]) + 0.02
        i = int(np.searchsorted(times, t))
        if i == len(times) or times[i] - t > 0.02:
            continue
        target = np.array([float(command[f"lowcmd_q_{j}"]) for j in range(12)])[
            PERMUTATION
        ]
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite command log")
        targets.append(target)
        measured.append(rows[i][1][4])
    if not targets:
        return {
            "states": modes,
            "control_rows": len(commands),
            "reason": "No command/telemetry pairs within comparison window",
        }
    targets, measured = np.array(targets), np.array(measured)
    return {
        "states": modes,
        "control_rows": len(commands),
        "matched_rows": len(targets),
        "comparison_delay_s": 0.02,
        "command_range_rad": np.ptp(targets, axis=0).tolist(),
        "encoder_range_rad": np.ptp(measured, axis=0).tolist(),
        "tracking_rms_rad": np.sqrt(
            np.mean((targets - measured) ** 2, axis=0)
        ).tolist(),
        "interpretation": (
            "measured tracking evidence only; not hardware protection verification"
        ),
    }


def estimate(
    rows,
    preset,
    acceleration_means=None,
    *,
    allow_recovery=False,
    align_initial_heading=False,
):
    config = json.loads(json.dumps(preset))
    config["calibrated"] = False
    config["source"] = "staff_log_candidate"
    stats = timing(rows)
    config["tick_period_s"] = stats["estimated_tick_period_s"]
    fit = None
    if acceleration_means is not None:
        scale, bias, rms = fit_accelerometer(acceleration_means)
        config["imu"].update(accel_scale=scale, accel_bias_imu=bias)
        fit = {"scale": scale, "bias": bias, "rms_m_s2": rms}
    stationary = None
    # Require a contiguous hold, not a mixture of poses across trials.
    for end in range(len(rows), 99, -1):
        window = rows[end - 100 : end]
        allowed = {"INTERMEDIATE", "RECOVERY"} if allow_recovery else {"INTERMEDIATE"}
        if len({r[2] for r in window}) != 1 or any(r[2] not in allowed for r in window):
            continue
        try:
            effective = calibrate_stationary(
                config["imu"],
                [(r[0], r[1]) for r in window],
                contact_threshold=config["observer"]["contact_threshold"],
            )
        except ValueError:
            continue
        stationary = window
        config["imu"] = effective
        if fit is not None:
            config["imu"]["accel_bias_imu"] = fit["bias"]
        break
    if stationary is None:
        raise ValueError(
            "No valid 100-sample stationary hold window; "
            "inspect foot loads, pose and timing"
        )
    heading = None
    if align_initial_heading:
        body_rotation = (
            np.array(config["imu"]["world_alignment"])
            @ rotation(stationary[-1][1][1])
            @ np.array(config["imu"]["body_from_imu"]).T
        )
        yaw = float(np.arctan2(body_rotation[1, 0], body_rotation[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        correction = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        config["imu"]["world_alignment"] = (
            correction @ np.array(config["imu"]["world_alignment"])
        ).tolist()
        heading = {
            "removed_initial_yaw_rad": yaw,
            "meaning": (
                "choose initial body-forward as policy world X; "
                "assumes mounting is correct"
            ),
        }
    model = build_fk_model()
    data = mujoco.MjData(model)
    qids = [model.jnt_qposadr[model.joint(n).id] for n in JOINTS]
    feet = [model.body(f"{leg}_foot").id for leg in ("FL", "FR", "RL", "RR")]
    geoms = [int(np.flatnonzero(model.geom_bodyid == b)[0]) for b in feet]
    heights = []
    mount = np.array(config["imu"]["body_from_imu"])
    world = np.array(config["imu"]["world_alignment"])
    for _, packet, _ in stationary[-10:]:
        data.qpos[:3] = 0
        r = world @ rotation(packet[1]) @ mount.T
        mujoco.mju_mat2Quat(data.qpos[3:7], r.ravel())
        data.qpos[qids] = packet[4]
        mujoco.mj_kinematics(model, data)
        h = np.array([model.geom_size[g, 0] - data.geom_xpos[g, 2] for g in geoms])
        if np.ptp(h) > 0.04 or np.any(h <= 0):
            raise ValueError(
                "Loaded feet disagree on base height; verify geometry/frame/contact"
            )
        heights.append(float(h.mean()))
    config["initial_height"] = float(np.median(heights))
    measured_q = np.median([r[1][4] for r in stationary], axis=0)
    posture_error = measured_q - np.array(CustomDeployConfig.default_dof_pos)
    report = {
        "timing": stats,
        "stationary_samples": 100,
        "stationary_window_state": stationary[0][2],
        "gyro_bias_rad_s": config["imu"]["gyro_bias_imu"],
        "gyro_noise_std_rad_s": np.std([r[1][2] for r in stationary], axis=0).tolist(),
        "raw_accel_noise_std_m_s2": np.std(
            [r[1][3] for r in stationary], axis=0
        ).tolist(),
        "initial_heading_alignment": heading,
        "accel_bias_m_s2": config["imu"]["accel_bias_imu"],
        "accel_scale_fit": fit,
        "initial_base_height_m": config["initial_height"],
        "reference_posture": {
            "measured_q_rad": measured_q.tolist(),
            "error_rad": posture_error.tolist(),
            "max_error_rad": float(abs(posture_error).max()),
            "limiting_joint": JOINTS[int(abs(posture_error).argmax())],
            "within_existing_activation_bound": bool(abs(posture_error).max() <= 0.15),
        },
        "height_assumptions": (
            "configured frames/geometry, level non-slip ground, all four loaded feet"
        ),
        "requires_independent_confirmation": [
            "body_from_imu mounting rotation and world/yaw alignment",
            "IMU lever arm from physical geometry or independently measured motion",
            "foot-force units, mapping, threshold, and model "
            "joint/torque/velocity semantics",
            "absolute transport latency and actual real-time control cadence",
        ],
        "single_pose_limitation": (
            "fixed-scale bias estimate only; one pose cannot identify "
            "scale and bias separately"
        ),
        "activation_enabled": False,
    }
    return config, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument(
        "--stationary-accel-means",
        type=Path,
        help="Optional (N,3) .npy of >=4 diverse, measured stationary pose means",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--align-initial-heading",
        action="store_true",
        help="Choose the initial body-forward direction as policy world X",
    )
    parser.add_argument(
        "--control",
        type=Path,
        help="Optional command CSV (defaults to control.csv beside LowState)",
    )
    parser.add_argument(
        "--allow-recovery",
        action="store_true",
        help="Offline only: also inspect stationary RECOVERY windows",
    )
    args = parser.parse_args()
    rows = read_log(args.lowstate)
    candidate, report = estimate(
        rows,
        json.loads(args.preset.read_text()),
        np.load(args.stationary_accel_means, allow_pickle=False)
        if args.stationary_accel_means
        else None,
        allow_recovery=args.allow_recovery,
        align_initial_heading=args.align_initial_heading,
    )
    report["command_evidence"] = command_evidence(
        rows, args.control or args.lowstate.with_name("control.csv")
    )
    with args.lowstate.open() as file:
        counts = Counter()
        for row in csv.DictReader(file):
            for button in ("A", "B", "X", "Y", "L2", "R2"):
                if button in row and json.loads(row[button]) == [1, 0]:
                    counts[button] += 1
    report["remote_press_events"] = {
        b: counts[b] for b in ("A", "B", "X", "Y", "L2", "R2")
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "calibration.candidate.json").write_text(
        json.dumps(candidate, indent=2) + "\n"
    )
    (args.output / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
