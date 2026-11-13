"""One deployable actor with a zero-initialized state-conditioned residual."""

import copy
import torch
from torch import nn
from torch.distributions import Normal
from experiments.morphology.learning.ppo import ActorCritic


class ContextNet(nn.Module):
    def __init__(self, base, inputs, output, enabled=True):
        super().__init__()
        self.base = base
        self.inputs = inputs
        self.context = nn.Sequential(
            nn.Linear(inputs + 20, 64), nn.ELU(), nn.Linear(64, output)
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        for p in self.context.parameters():
            p.requires_grad_(enabled)

    def forward(self, x):
        gate = (
            ((x[..., 0] - 0.5) / 0.3).clamp(0, 1).unsqueeze(-1)
            if self.inputs == 73
            else 1.0
        )
        return self.base(x[..., : self.inputs].contiguous()) + gate * self.context(x)


class MultiActorCritic(ActorCritic):
    def __init__(self, parent=None, enabled=True):
        super().__init__(73, 103)
        self.phase_noise = enabled
        if parent is not None:
            self.load_state_dict(parent)
        self.actor = ContextNet(self.actor, 73, 12, enabled)
        self.critic = ContextNet(self.critic, 103, 1, enabled)

    def distribution(self, obs):
        gate = (
            ((obs[..., 0] - 0.5) / 0.3).clamp(0, 1).unsqueeze(-1)
            if self.phase_noise
            else 0.0
        )
        scale = self.log_std.clamp(self.log_std_min, 1).exp() * (1 - 0.95 * gate)
        return Normal(self.actor(obs), scale)


def initialize(parent, enabled=True):
    # Identical added parameters and base initialization in every matched lane.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261113)
        net = MultiActorCritic(parent["network"], enabled)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    old = ActorCritic(73, 103)
    old.load_state_dict(parent["network"])
    if "optimizer" in parent:
        names = list(old.named_parameters())
        ids = parent["optimizer"]["param_groups"][0]["params"]
        current = dict(net.named_parameters())
        for (name, _), pid in zip(names, ids):
            renamed = name.replace("actor.", "actor.base.").replace(
                "critic.", "critic.base."
            )
            if pid in parent["optimizer"]["state"]:
                optimizer.state[current[renamed]] = copy.deepcopy(
                    parent["optimizer"]["state"][pid]
                )
    return net, optimizer


def apply_warm(net, opt, warm):
    net.load_state_dict(warm["network"])
    opt.load_state_dict(copy.deepcopy(warm["optimizer"]))
