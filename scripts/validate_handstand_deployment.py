"""Software-only matched regression. Never imports the Unitree SDK."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from experiments.handstand_robustness.capture import CaptureTrainingEnvironment
from gym.envs.go2.handstand_replay import build_model
from scripts.handstand_takeoff import load
from go2_deploy.handstand.policy import load_assets, TargetPath
from go2_deploy.handstand.estimator import SensorEstimator


class TelemetryEnvironment(CaptureTrainingEnvironment):
    def __init__(self, t, seed, condition):
        super().__init__(build_model(), t, seed, condition, horizon=400)
        # Calibrated standing height, not perturbed/root ground truth.
        self.sensor = SensorEstimator(
            build_model(), self.capture_kinematics.observer.height, 0, calibrated=True
        )
        self.sensor_rows = []

    def after_physics_step(self):
        super().after_physics_step()
        r = self.d.xmat[self.base].reshape(3, 3)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, r.ravel())
        force = r.T @ (self.d.qacc[:3] - self.m.opt.gravity)
        self.sensor.update(self.d.time, force, quat)
        self.sensor_rows.append(np.r_[self.d.time, quat, force])

    def deploy_observation(self):
        return self.sensor.observation(
            self.d.qpos[3:7].copy(),
            self.d.qvel[3:6].copy(),
            self.d.qpos[self.qids].copy(),
            self.d.qvel[self.vids].copy(),
            self.target,
            self.trajectory["nominal"],
            self.index,
            self.d.time,
        )


def main():
    ap = argparse.ArgumentParser(__doc__)
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
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument(
        "--original-only",
        action="store_true",
        help="Run the unchanged research full_eval without unused teacher assets",
    )
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy, t = load_assets(args.checkpoint, args.reference)
    if args.original_only:
        from scripts.handstand_capture import full_eval
        import hashlib

        args.controllers = ["student"]
        args.evaluation_seed = 301000
        args.output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = full_eval(args, t)
        report = {
            "result": result,
            "seconds": time.perf_counter() - started,
            "provenance": {
                "checkpoint_sha256": hashlib.sha256(
                    args.checkpoint.read_bytes()
                ).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    args.reference.read_bytes()
                ).hexdigest(),
                "note": (
                    "Unmodified research full_eval; "
                    "unused teacher assets are not required."
                ),
            },
        }
        (args.output / "full.json").write_text(json.dumps(report, indent=2) + "\n")
        return
    original = load(args.checkpoint)[0].eval()
    scripted = torch.jit.script(policy)
    errors = dict(
        observation=0.0, observer=0.0, action=0.0, target=0.0, capture=0.0, balance=0.0
    )
    channels = np.zeros(47)
    timings = []
    rows = []
    for i in range(args.trials + 1):
        e = TelemetryEnvironment(
            t,
            301000 + i,
            {} if i == 0 else {"joint_position": 0.003, "joint_velocity": 0.03},
        )
        path = TargetPath(t, e.bounds)
        control_rows = []
        for k in range(400):
            obs, deploy = e.observation(), e.deploy_observation()
            if i == 0:
                control_rows.append(
                    np.r_[
                        e.d.time,
                        e.d.qpos[3:7],
                        e.d.qvel[3:6],
                        e.d.qpos[e.qids],
                        e.d.qvel[e.vids],
                        e.target,
                        k,
                        obs,
                    ]
                )
            channels = np.maximum(channels, np.abs(obs - deploy))
            errors["observation"] = max(
                errors["observation"], float(abs(obs - deploy).max())
            )
            errors["observer"] = max(
                errors["observer"],
                float(
                    abs(
                        np.r_[
                            e.imu_observer.height - e.sensor.height,
                            e.imu_observer.velocity - e.sensor.velocity,
                        ]
                    ).max()
                ),
            )
            with torch.no_grad():
                x = torch.from_numpy(obs)
                expected = original(x).numpy()
                start = time.perf_counter_ns()
                actual = scripted(torch.from_numpy(deploy)).numpy()
                timings.append((time.perf_counter_ns() - start) / 1e6)
                # Equal inputs isolate inference parity from observer differences.
                a, c, b, index = policy.components(x)
                pk, delta = original.inputs(x)
                ec = original.bound * original.latent(x).tanh()
                features = torch.cat((x[:30], x[43:47]))
                center = torch.cat(
                    (original.centers[index, :30], original.centers[index, 43:47])
                )
                eb = ((features - center) @ original.balance_weight.T).clamp(
                    -original.bound, original.bound
                )
                errors["capture"] = max(errors["capture"], float((c - ec).abs().max()))
                errors["balance"] = max(errors["balance"], float((b - eb).abs().max()))
            errors["action"] = max(
                errors["action"], float(abs(expected - actual).max())
            )
            target = path.apply(actual)
            _, _, done = e.step(actual)
            errors["target"] = max(
                errors["target"], float(abs(target - e.target).max())
            )
            if done:
                break
        rows.append(e.report() | {"seed": 301000 + i, "nominal": i == 0})
        print(i, rows[-1]["complete_success"], flush=True)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "partial.json").write_text(json.dumps(rows, indent=2))
        if i == 0:
            np.savez_compressed(
                args.output / "nominal_imu.npz",
                samples=np.array(e.sensor_rows),
                controls=np.array(control_rows),
            )
    report = {
        "heldout_successes": sum(r["complete_success"] for r in rows[1:]),
        "heldout_trials": args.trials,
        "nominal_success": rows[0]["complete_success"],
        "maximum_absolute_errors": errors,
        "observation_errors_by_channel": channels.tolist(),
        "inference_ms": {
            "median": float(np.median(timings)),
            "p99": float(np.percentile(timings, 99)),
            "max": float(max(timings)),
        },
        "rows": rows,
        "python": sys.version,
        "mujoco": mujoco.__version__,
    }
    (args.output / "regression.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("rows", "observation_errors_by_channel")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
