"""Replay simulated robot telemetry through hooks/Session. No SDK or transport."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from go2_deploy.handstand.bundle import load_review_bundle
from go2_deploy.handstand.session import Session
from go2_deploy.custom_implementation import handstand_provisional as hooks


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument(
        "--bundle", type=Path, default=Path("artifacts/handstand_deployment")
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy, model, meta = load_review_bundle(args.bundle)
    with np.load(args.bundle / "trajectory.npz") as data:
        trajectory = {key: data[key].copy() for key in data.files}
    records = np.load(args.recording)
    imu, controls = records["samples"], records["controls"]
    session = Session(policy, trajectory, model)
    session.activate(0, 0.3413013268255194, calibrated=True)
    controller = SimpleNamespace(**hooks.get_custom_buffers_dict())
    controller.handstand_session = session
    durations = []
    errors = {"observation": 0.0, "target": 0.0, "previous_applied_target": 0.0}
    j = 0
    for index, row in enumerate(controls):
        now = row[0]
        start = time.perf_counter_ns()
        if index == 0:
            session.ingest(0, [0, 0, 9.81], row[1:5], row[5:8], row[8:20], row[20:32])
        while j < len(imu) and imu[j, 0] <= now + 1e-9:
            sample = imu[j]
            # Encoder samples only matter at the 50 Hz synchronized snapshot.
            session.ingest(
                sample[0], sample[5:8], sample[1:5], row[5:8], row[8:20], row[20:32]
            )
            j += 1
        # Use current synchronized control orientation after the last physics
        # sample, matching research forward-kinematics snapshot timing.
        session.telemetry = tuple(
            np.array(part) for part in (row[1:5], row[5:8], row[8:20], row[20:32])
        )
        hooks.update_custom_buffers(controller, now)
        pieces = hooks.get_custom_obs_pieces()
        obs = torch.cat(
            [
                pieces[name](controller, None)
                for name in hooks.CustomDeployConfig.obs_vector
            ]
        ).numpy()
        errors["observation"] = max(
            errors["observation"], float(abs(obs - row[45:92]).max())
        )
        errors["previous_applied_target"] = max(
            errors["previous_applied_target"],
            float(abs(session.path.target - row[32:44]).max()),
        )
        action, target = session.propose(now)
        assert np.isfinite(action).all() and (abs(action) <= 1).all()
        session.acknowledge(target)
        if index + 1 < len(controls):
            errors["target"] = max(
                errors["target"], float(abs(target - controls[index + 1, 32:44]).max())
            )
        durations.append((time.perf_counter_ns() - start) / 1e6)
    session.cancel()
    report = {
        "status": "SOFTWARE DRY RUN ONLY; NOT HARDWARE AUTHORIZATION",
        "commands": len(controls),
        "imu_samples_consumed": session.estimator.samples,
        "expected_imu_samples": j,
        "trailing_samples_after_last_snapshot": len(imu)-j,
        "maximum_absolute_errors": errors,
        "full_software_cycle_ms": {
            "median": float(np.median(durations)),
            "p99": float(np.percentile(durations, 99)),
            "max": float(max(durations)),
        },
        "cycles_over_20ms": sum(value > 20 for value in durations),
        "cancelled": not session.active,
        "limitations": (
            "Replay scheduling is not a real-time deadline test. "
            "Physical actuator protections, transport, hardware clock "
            "synchronization and recovery are not evaluated."
        ),
    }
    assert report["imu_samples_consumed"] == j
    assert max(errors.values()) < 1e-7
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
