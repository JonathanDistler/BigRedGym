"""Shared actor/privileged critic PPO and measurable physical-morphology schedule."""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor_dim,
        critic_dim,
        hidden=(256, 256, 128),
        log_std=-0.7,
        log_std_min=-3.0,
    ):
        super().__init__()
        self.log_std_min = float(log_std_min)

        def mlp(inputs, output, gain):
            layers = []
            for size in hidden:
                layer = nn.Linear(inputs, size)
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers += [layer, nn.ELU()]
                inputs = size
            layer = nn.Linear(inputs, output)
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
            return nn.Sequential(*layers, layer)

        self.actor = mlp(actor_dim, 12, 0.01)
        self.critic = mlp(critic_dim, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((12,), float(log_std)))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(self.log_std_min, 1).exp())

    @staticmethod
    def log_probability(distribution, raw):
        # Stable tanh Jacobian; likelihood is for the bounded joint-target action.
        correction = 2 * (np.log(2) - raw - torch.nn.functional.softplus(-2 * raw))
        return (distribution.log_prob(raw) - correction).sum(-1)

    def sample(self, actor, critic):
        distribution = self.distribution(actor)
        raw = distribution.sample()
        return (
            raw,
            self.log_probability(distribution, raw),
            self.critic(critic).squeeze(-1),
        )


def advantages(rewards, values, next_values, dones, gamma, lam):
    """Time-major arrays [T,N]; every finite-horizon ending stops bootstrapping."""
    result = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        carry = delta + gamma * lam * (1 - dones[t]) * carry
        result[t] = carry
    return result, result + values


def update(net, optimizer, batch, hp, demonstrations=None):
    observations, privileged, raw, old_log, returns, adv = batch
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp(min=1e-8)
    stats = []
    stop = False
    for _ in range(hp["epochs"]):
        for ids in torch.randperm(len(adv)).split(hp["batch_size"]):
            distribution = net.distribution(observations[ids])
            log = net.log_probability(distribution, raw[ids])
            logratio = log - old_log[ids]
            ratio = logratio.exp()
            kl = ((ratio - 1) - logratio).mean()
            if kl.item() > 1.5 * hp["target_kl"]:
                stop = True
                break
            policy = -torch.minimum(
                ratio * adv[ids], ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * adv[ids]
            ).mean()
            value = (
                (net.critic(privileged[ids]).squeeze(-1) - returns[ids]).square().mean()
            )
            # Base Normal entropy is logged/regularized; transformed entropy is
            # state-dependent and the exact policy likelihood remains corrected.
            entropy = distribution.entropy().sum(-1).mean()
            bc_loss = torch.zeros(())
            bc_unweighted = torch.zeros(())
            demo_weight = torch.ones(())
            if demonstrations is not None and hp.get("bc_weight", 0.0) > 0:
                demo_obs, demo_actions = demonstrations[:2]
                demo_ids = torch.randint(len(demo_obs), (hp.get("bc_batch_size", 128),))
                # Fixed normalization keeps the anchor independent of learned std.
                demo_latent = torch.atanh(demo_actions[demo_ids].clamp(-0.999, 0.999))
                joint_loss = (
                    (net.actor(demo_obs[demo_ids]) - demo_latent)
                    / hp.get("bc_latent_scale", 0.05)
                ).square()
                frame_loss = joint_loss.mean(-1)
                bc_unweighted = frame_loss.mean()
                weights = torch.ones_like(frame_loss)
                if len(demonstrations) == 3:
                    phase = float(hp.get("bc_phase_fraction", 0.0))
                    demo_weights = demonstrations[2][demo_ids]
                    if demo_weights.ndim == 2:
                        frame_loss = (
                            joint_loss * (1 - phase * (1 - demo_weights))
                        ).mean(-1)
                    else:
                        weights = 1 - phase * (1 - demo_weights)
                demo_weight = (
                    demo_weights.mean()
                    if len(demonstrations) == 3 and demo_weights.ndim == 2
                    else weights.mean()
                )
                bc_loss = (frame_loss * weights).mean()
            loss = (
                policy
                + hp["value_weight"] * value
                - hp["entropy_weight"] * entropy
                + hp.get("bc_weight", 0.0) * bc_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PPO objective")
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), hp["max_grad_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite PPO gradient")
            optimizer.step()
            stats.append(
                [
                    policy.item(),
                    value.item(),
                    kl.item(),
                    (abs(ratio - 1) > hp["clip"]).float().mean().item(),
                    norm.item(),
                    entropy.item(),
                    bc_loss.item(),
                    bc_unweighted.item(),
                    demo_weight.item(),
                ]
            )
        if stop:
            break
    if not stats:
        raise RuntimeError("PPO performed no optimizer update")
    result = dict(
        zip(
            (
                "policy_loss",
                "value_loss",
                "approx_kl",
                "clip_fraction",
                "gradient_norm",
                "entropy",
                "bc_loss",
                "bc_unweighted_loss",
                "bc_demo_weight_mean",
            ),
            np.mean(stats, axis=0),
        )
    )
    result["kl_early_stop"] = stop
    result["bc_weight"] = float(hp.get("bc_weight", 0.0))
    result["bc_objective_contribution"] = result["bc_weight"] * result["bc_loss"]
    with torch.no_grad():
        distribution = net.distribution(observations)
        delta = net.log_probability(distribution, raw) - old_log
        result["post_update_kl"] = float(((delta.exp() - 1) - delta).mean())
        result["action_mean_abs"] = float(net.actor(observations).tanh().abs().mean())
        result["action_std"] = float(net.log_std.exp().mean())
    return result


@dataclass
class PerformanceCurriculum:
    method: str
    protocol: dict
    stage: int = 0
    stage_entry_step: int = 0
    outcomes: list = field(default_factory=list)
    passing_windows: int = 0
    events: list = field(default_factory=list)
    windows: list = field(default_factory=list)

    @property
    def scale(self):
        return 1.0 if self.method == "A" else self.protocol["scales"][self.stage]

    def completed(self, episode, actual_steps):
        p = self.protocol
        if (
            self.method == "A"
            or self.stage == len(p["scales"]) - 1
            or episode["scale"] != self.scale
        ):
            return
        pitch = np.deg2rad(p["progress_pitch_degrees"][self.stage])
        recovered = episode["recovery_seconds"] >= 0.1 or episode["success"]
        eligible = (
            episode["takeoff"]
            and episode["max_supported_pitch"] >= pitch
            and episode["front_support_fraction"] >= 0.6
            and episode["max_speed_ratio"] <= 1.001
            and episode["joint_limit_violation"] < 0.001
            and (recovered or not p["recovery_required"][self.stage])
        )
        self.outcomes.append(bool(eligible))
        if len(self.outcomes) < p["stage_window_episodes"]:
            return
        rate = float(np.mean(self.outcomes))
        self.outcomes.clear()
        passed = rate >= p["stage_required_fraction"]
        self.passing_windows = self.passing_windows + 1 if passed else 0
        self.windows.append(
            dict(
                actual_steps=actual_steps,
                scale=self.scale,
                stage=self.stage,
                eligible_fraction=rate,
                passed=passed,
                pitch_threshold=float(pitch),
                recovery_required=p["recovery_required"][self.stage],
            )
        )
        if (
            self.passing_windows >= p["stage_passing_windows"]
            and actual_steps - self.stage_entry_step >= p["minimum_stage_steps"]
        ):
            old = self.scale
            self.stage += 1
            self.stage_entry_step = actual_steps
            self.passing_windows = 0
            self.events.append(
                dict(
                    actual_steps=actual_steps,
                    previous_scale=old,
                    scale=self.scale,
                    qualifying_fraction=rate,
                    condition=copy.deepcopy(self.windows[-1]),
                )
            )
