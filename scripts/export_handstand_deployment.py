"""Export the complete inference bundle for software-only staff review."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import torch
from go2_deploy.handstand.policy import (
    load_assets,
    CHECKPOINT_SHA,
    REFERENCE_SHA,
    JOINTS,
)
from go2_deploy.custom_implementation.handstand_provisional import CustomDeployConfig
from gym.envs.go2.handstand_replay import build_model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/capture_ppo_conservative/latest.pt"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "Choose a fresh output directory; bundles are not overwritten"
        )
    p, t = load_assets(args.checkpoint, args.reference)
    args.output.mkdir(parents=True)
    torch.jit.save(torch.jit.script(p), str(args.output / "policy.pt"))
    shutil.copy2(args.checkpoint, args.output / "capture_checkpoint.pt")
    shutil.copy2(args.reference, args.output / "trajectory.npz")
    # Generated nominal FK model has collision primitives and no external meshes.
    model = build_model()
    mujoco.mj_saveModel(model, str(args.output / "go2_fk.mjb"))
    meta = {
        "task_name": "go2_handstand",
        "phase_frequency": 0.0,
        "command_limits": {},
        "schema": 1,
        "status": "NOT READY FOR HARDWARE",
        "kind": "capture_residual_v1",
        "num_obs": 47,
        "num_actions": 12,
        "ctrl_freq": 50,
        "exp_moving_avg": False,
        "kp": 60.0,
        "kd": 3.0,
        "action_scale_rad": 2.0,
        "default_dof_pos": t["nominal"].tolist(),
        "initial_target": t["initial_target"].tolist(),
        "joint_order": JOINTS,
        "obs_vector": CustomDeployConfig.obs_vector,
        "obs_sizes": CustomDeployConfig.custom_obs_sizes,
        "scaling": {
            **{key: 1.0 for key in CustomDeployConfig.obs_vector},
            "dof_pos_target": 2.0,
        },
        "active_steps": 110,
        "reference_steps": 320,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "reference_sha256": REFERENCE_SHA,
        "cornell_commit": "8ca9bf98373a234de662f9fb8242a228b7efde21",
        "required_runner_changes": [
            "lossless timestamped calibrated 500 Hz IMU queue",
            "atomic 50 Hz telemetry snapshot",
            "explicit activation/reset/cancel",
            "target slew and acknowledgement before phase advancement",
            "validated actuator effort and overspeed protections",
            "50 Hz and EMA off applied explicitly",
        ],
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.output.iterdir()
        },
    }
    meta["policy_sha256"] = meta["files"]["policy.pt"]
    (args.output / "deploy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
