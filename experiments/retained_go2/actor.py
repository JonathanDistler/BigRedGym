"""One checkpoint/forward pass; learned competence gate, no pitch/time switch."""

import numpy as np
import torch
from torch import nn
from experiments.morphology.learning.ppo import ActorCritic
from experiments.balance_feedback.actor import CHANNELS

# Contact centroids jump during legitimate unloading; they are not a smooth
# competence cue. Route on gravity, joint geometry and COM height instead.
# Balance feedback and the shared correction still receive all93 inputs.
FEATURES = np.r_[np.arange(3), np.arange(6, 18), 55]
FLOOR = np.r_[[0.02] * 3, [0.02] * 12, 0.03]


class Specialization(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.affine = nn.Linear(93, 12, dtype=torch.float64)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        self.router = nn.Linear(len(FEATURES), 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.correction = nn.Sequential(nn.Linear(93, 64), nn.ELU(), nn.Linear(64, 12))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer("enabled", torch.tensor(0.0))
        self.register_buffer("indices", torch.tensor(FEATURES, dtype=torch.long))
        for name, value in [
            ("center", torch.zeros(len(FEATURES))),
            ("scale", torch.ones(len(FEATURES))),
            ("inner_low", torch.full((len(FEATURES),), -float("inf"))),
            ("inner_high", torch.full((len(FEATURES),), float("inf"))),
            ("outer_width", torch.ones(len(FEATURES))),
        ]:
            self.register_buffer(name, value)
        self.register_buffer("lower", torch.full((12,), -0.999999, dtype=torch.float64))
        self.register_buffer("upper", torch.full((12,), 0.999999, dtype=torch.float64))
        # Stage B protects competent specialist coefficients; the shared residual,
        # router and critic remain trainable. No external controller is deployed.
        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.affine.parameters():
            p.requires_grad_(False)

    def gate(self, x):
        feature = x[..., self.indices]
        score = self.router((feature - self.center) / self.scale).squeeze(-1)
        learned = ((score + 1) / 2).clamp(0, 1)
        outside = (
            torch.maximum(self.inner_low - feature, feature - self.inner_high).clamp(
                min=0
            )
            / self.outer_width
        )
        trust = (1 - outside.amax(-1)).clamp(0, 1)
        return self.enabled * learned * trust

    def forward(self, x):
        base = self.base(x)
        balance = torch.atanh(self.affine(x.double()).clamp(self.lower, self.upper)).to(
            x.dtype
        )
        g = self.gate(x).unsqueeze(-1)
        # Exact competent endpoints; no cancellation error in the zero/one cases.
        latent = torch.where(
            g == 0, base, torch.where(g == 1, balance, base + g * (balance - base))
        )
        return latent + self.correction(x)

    def position_targets(self, x):
        return self(x).tanh()


class Student(ActorCritic):
    def __init__(self):
        super().__init__(93, 123, log_std=-6.0, log_std_min=-6.0)
        self.actor = Specialization(self.actor)

    @classmethod
    def from_adc(cls, run, balance=None):
        net = cls()
        for label, dim in [("actor", 93), ("critic", 123)]:
            state = {
                k[len(label) + 1 :]: v.clone()
                for k, v in run["network"].items()
                if k.startswith(label + ".")
            }
            state["0.weight"] = torch.nn.functional.pad(
                state["0.weight"], (0, dim - state["0.weight"].shape[1])
            )
            (net.actor.base if label == "actor" else net.critic).load_state_dict(state)
        if balance:
            state = balance["network"]
            with torch.no_grad():
                net.actor.affine.weight[:, CHANNELS] = state["feedback.weight"]
                net.actor.affine.bias.copy_(state["feedback.bias"])
                net.actor.lower.copy_(state["lower"].clamp(-0.999999, 0.999999))
                net.actor.upper.copy_(state["upper"].clamp(-0.999999, 0.999999))
        return net
