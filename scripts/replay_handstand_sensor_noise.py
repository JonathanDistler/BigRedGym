"""Replay recorded nominal sensor streams with declared perturbations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from go2_deploy.handstand.policy import load_assets
from go2_deploy.handstand.estimator import SensorEstimator
from gym.envs.go2.handstand_replay import build_model


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/capture_ppo_conservative/latest.pt"),
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    policy, t = load_assets(args.checkpoint, args.reference)
    recording = np.load(args.recording)
    imu, controls = recording["samples"], recording["controls"]
    results = {}
    torch.set_num_threads(1)
    for case in [
        "ideal",
        "noise",
        "bias",
        "latency",
        "jitter",
        "initialization",
        "combined",
    ]:
        rng = np.random.default_rng(4110)
        b = SensorEstimator(
            build_model(),
            0.3413013268255194
            + (0.02 if case in ["initialization", "combined"] else 0),
            0,
            calibrated=True,
        )
        theta = np.deg2rad(0.5) if case in ["initialization", "combined"] else 0
        tilt = np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0])
        j = 0
        maximum = np.zeros(47)
        action_error = height_error = velocity_error = 0.0
        for control in controls:
            now = control[0]
            while j < len(imu) and imu[j, 0] <= now + 1e-9:
                row = imu[max(0, j - 5)] if case in ["latency", "combined"] else imu[j]
                quat, force = row[1:5].copy(), row[5:8].copy()
                if case in ["noise", "combined"]:
                    force += rng.normal(0, 0.08, 3)
                if case in ["bias", "combined"]:
                    force += [0.02, -0.01, 0.03]
                rotated = np.empty(4)
                mujoco.mju_mulQuat(rotated, tilt, quat)
                stamp = imu[j, 0]
                # ±0.2ms sample timestamp error, order preserved; final sample
                # in each batch remains at snapshot time to avoid future data.
                if case in ["jitter", "combined"] and abs(stamp - now) > 1e-9:
                    stamp += rng.uniform(-0.0002, 0.0002)
                b.update(stamp, force, rotated)
                j += 1
            quat = np.empty(4)
            mujoco.mju_mulQuat(quat, tilt, control[1:5])
            gyro, q, dq = (
                control[5:8].copy(),
                control[8:20].copy(),
                control[20:32].copy(),
            )
            if case in ["noise", "combined"]:
                gyro += rng.normal(0, 0.005, 3)
                q += rng.normal(0, 0.0005, 12)
                dq += rng.normal(0, 0.01, 12)
            obs = b.observation(
                quat, gyro, q, dq, control[32:44], t["nominal"], int(control[44]), now
            )
            original = control[45:92].astype(np.float32)
            delta = abs(obs - original)
            maximum = np.maximum(maximum, delta)
            height_error = max(height_error, float(delta[43] * 0.6))
            velocity_error = max(
                velocity_error, float(np.linalg.norm(obs[44:47] - original[44:47]))
            )
            with torch.no_grad():
                action_error = max(
                    action_error,
                    float(
                        (
                            policy(torch.from_numpy(obs))
                            - policy(torch.from_numpy(original))
                        )
                        .abs()
                        .max()
                    ),
                )
        results[case] = {
            "max_com_height_difference_m": height_error,
            "max_com_velocity_difference_m_s": velocity_error,
            "max_normalized_action_difference": action_error,
            "max_observation_difference_by_channel": maximum.tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": (
                    "Open-loop sensor replay; "
                    "targets held to recorded applied history. "
                    "Measures input sensitivity, not closed-loop success probability. "
                    "Synthetic distributions are illustrative, "
                    "not measured Go2 specifications."
                ),
                "seed": 4110,
                "conditions": {
                    "accel_noise_std_m_s2": 0.08,
                    "accel_bias_m_s2": [0.02, -0.01, 0.03],
                    "signal_delay_s": 0.01,
                    "timestamp_jitter_s": 0.0002,
                    "gyro_noise_std_rad_s": 0.005,
                    "encoder_noise_std_rad": 0.0005,
                    "encoder_velocity_noise_std_rad_s": 0.01,
                    "initial_height_error_m": 0.02,
                    "initial_frame_tilt_deg": 0.5,
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                k: {key: value for key, value in v.items() if "channel" not in key}
                for k, v in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
