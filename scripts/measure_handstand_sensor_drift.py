"""Reproducible synthetic stationary sensor sensitivity; not hardware evidence."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from go2_deploy.handstand.estimator import SensorEstimator
from gym.envs.go2.handstand_replay import build_model


def main():
    results = {}
    for name in [
        "ideal",
        "noise",
        "bias",
        "latency",
        "jitter",
        "initialization",
        "combined",
    ]:
        rng = np.random.default_rng(4110)
        height = 0.34 + (0.02 if name in ["initialization", "combined"] else 0)
        b = SensorEstimator(build_model(), height, 0, calibrated=True)
        # Synthetic acceleration varies in time to expose 10 ms latency.
        true_velocity = np.zeros(3)
        true_height = 0.34
        maxv = maxh = 0.0
        t = 0.0
        for i in range(4000):
            dt = 0.002 + (
                rng.uniform(-0.0002, 0.0002) if name in ["jitter", "combined"] else 0
            )
            t += dt
            a = np.array([0.2 * np.sin(2 * np.pi * t), 0, 0.1 * np.cos(2 * np.pi * t)])
            true_velocity += dt * a
            true_height += dt * true_velocity[2]
            sensor_t = t - (0.01 if name in ["latency", "combined"] else 0)
            measured = np.array(
                [
                    0.2 * np.sin(2 * np.pi * sensor_t),
                    0,
                    0.1 * np.cos(2 * np.pi * sensor_t),
                ]
            ) + [0, 0, 9.81]
            if name in ["noise", "combined"]:
                measured += rng.normal(0, 0.08, 3)
            if name in ["bias", "combined"]:
                measured += [0.02, -0.01, 0.03]
            # Initialization case includes 0.5 degree frame error.
            theta = np.deg2rad(0.5) if name in ["initialization", "combined"] else 0
            quat = [np.cos(theta / 2), 0, np.sin(theta / 2), 0]
            b.update(t, measured, quat)
            maxv = max(maxv, float(np.linalg.norm(b.velocity - true_velocity)))
            maxh = max(maxh, abs(b.height - true_height))
        results[name] = {
            "duration_s": t,
            "max_root_height_error_m": maxh,
            "max_root_velocity_error_m_s": maxv,
            "final_root_height_error_m": b.height - true_height,
            "final_root_velocity_error_m_s": (b.velocity - true_velocity).tolist(),
        }
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "seed": 4110,
                "scope": (
                    "Synthetic root observer sensitivity, not a calibrated hardware "
                    "error distribution. 0.08 m/s2 noise, [0.02,-0.01,0.03] "
                    "m/s2 bias, 10ms signal latency, ±0.2ms timestamp jitter, "
                    "2cm height/0.5deg orientation initialization error. "
                    "Exact reported timestamps used."
                ),
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
