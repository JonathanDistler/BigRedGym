"""Complete phase dependent inference, independent of training code."""

from pathlib import Path
import hashlib
import numpy as np
import torch
from torch import nn

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


def load_assets(checkpoint, reference):
    checked(checkpoint, CHECKPOINT_SHA)
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
