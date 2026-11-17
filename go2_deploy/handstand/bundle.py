"""Load a hash-checked offline review bundle; never opens a transport."""

import json
from pathlib import Path
import mujoco
import torch
from .policy import CHECKPOINT_SHA, REFERENCE_SHA, JOINTS, checked


def load_review_bundle(directory):
    directory = Path(directory)
    meta = json.loads((directory / "deploy_meta.json").read_text())
    required = {
        "num_obs": 47,
        "num_actions": 12,
        "ctrl_freq": 50,
        "exp_moving_avg": False,
        "kp": 60.0,
        "kd": 3.0,
        "action_scale_rad": 2.0,
        "active_steps": 110,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "reference_sha256": REFERENCE_SHA,
        "joint_order": JOINTS,
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError(f"Inconsistent bundle metadata: {key}")
    for name, digest in meta["files"].items():
        if Path(name).name != name:
            raise ValueError("Bundle file must be a basename")
        checked(directory / name, digest)
    checked(directory / "capture_checkpoint.pt", CHECKPOINT_SHA)
    checked(directory / "trajectory.npz", REFERENCE_SHA)
    policy = torch.jit.load(str(directory / "policy.pt"), map_location="cpu").eval()
    # Warm up TorchScript before activation; no zero-input action is issued.
    with torch.no_grad():
        for _ in range(5):
            result = policy(torch.zeros(47))
    if result.shape != (12,) or not torch.isfinite(result).all():
        raise ValueError("Invalid policy output")
    if mujoco.__version__ != "3.11.0":
        raise ValueError("FK binary requires pinned MuJoCo 3.11.0")
    model = mujoco.MjModel.from_binary_path(str(directory / "go2_fk.mjb"))
    return policy, model, meta
