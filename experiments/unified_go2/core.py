"""One trainable MLP, competence-certified offline labels, serial native worlds."""

import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from experiments.morphology.learning.ppo import ActorCritic
from experiments.multi_impulse.core import MultiEnvironment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def save_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        )
    )


class FeedbackResidual(torch.nn.Module):
    """Single actor's affine feedback plus learned residual, no expert dispatch.

    A quadratic gravity-error feature keeps the residual's value and first
    derivative zero at upright. It is not an angular threshold or interpolation
    of two expert actions. All coefficients are trainable during PPO.
    """

    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.affine = torch.nn.Linear(93, 12, dtype=torch.float64)
        torch.nn.init.zeros_(self.affine.weight)
        torch.nn.init.zeros_(self.affine.bias)
        self.register_buffer("lower", torch.full((12,), -0.999999, dtype=torch.float64))
        self.register_buffer("upper", torch.full((12,), 0.999999, dtype=torch.float64))

    def position_targets(self, x):
        gravity_error = (1 - x[..., 0:1]).square()
        target = self.affine(x.double()) + gravity_error * self.residual(x)
        return target.to(x.dtype)

    def forward(self, x):
        target = self.position_targets(x)
        # Same final position action after the existing PPO tanh convention.
        return torch.atanh(target.clamp(self.lower, self.upper)).to(x.dtype)


class Student(ActorCritic):
    def __init__(self):
        super().__init__(93, 123, log_std=-4.6, log_std_min=-6.0)
        self.actor = FeedbackResidual(self.actor)

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
            (net.actor.residual if label == "actor" else net.critic).load_state_dict(
                state
            )
        if balance is not None:
            from experiments.balance_feedback.actor import CHANNELS

            state = balance["network"]
            with torch.no_grad():
                net.actor.affine.weight[:, CHANNELS] = state["feedback.weight"]
                net.actor.affine.bias.copy_(state["feedback.bias"])
                net.actor.lower.copy_(state["lower"].clamp(-0.999999, 0.999999))
                net.actor.upper.copy_(state["upper"].clamp(-0.999999, 0.999999))
        return net


def adc_network(run):
    net = ActorCritic(93, 123)
    state = copy.deepcopy(run["network"])
    state["actor.0.weight"] = torch.nn.functional.pad(state["actor.0.weight"], (0, 20))
    state["critic.0.weight"] = torch.nn.functional.pad(
        state["critic.0.weight"], (0, 20)
    )
    net.load_state_dict(state)
    return net


def descriptor(obs):
    """Measured capture-state coordinates: orientation, speed, support, COM, load."""
    return np.r_[obs[:3], obs[3:6], obs[53:59], obs[87:91]]


SCALES = np.r_[[0.12] * 3, [0.5] * 3, [0.12, 0.12, 0.1, 0.3, 0.3, 0.3], [0.3] * 4]


def capture_distance(obs, centers):
    if not len(centers):
        return float("inf")
    return float(
        np.min(np.sqrt(np.mean(((descriptor(obs) - centers) / SCALES) ** 2, axis=-1)))
    )


class Environment(MultiEnvironment):
    def step(self, action):
        previous_hold = self.multi_hold
        previous_takeoff = self.tracker.confirmed
        obs, reward, done, info = super().step(action)
        distance = capture_distance(obs[0], getattr(self, "capture_centers", []))
        # Diagnostic local neighborhood, not a certified viability oracle.
        term = 0.0
        if (
            self._momentum_prefix_safe
            and self.row["front_support"]
            and np.isfinite(distance)
        ):
            credit = float(np.exp(-(distance**2)))
            old = getattr(self, "capture_credit", 0.0)
            term = 0.1 * max(0.0, credit - old)
            self.capture_credit = max(old, credit)
        self.capture_distance = distance
        sample = self.momentum_sample
        counter = 0.0
        if (
            self._momentum_prefix_safe
            and self.row["front_support"]
            and sample["pitch_velocity"] > 0
            and sample["pitch"] + 0.12 * sample["pitch_velocity"] < np.deg2rad(75)
        ):
            counter = (
                -0.01
                * self.cfg.dt
                * float(
                    np.tanh(max(0.0, -sample["front_moment_com"]) / (self.weight * 0.2))
                )
            )
        loss = -0.2 * float(previous_hold >= 1 and self.multi_hold == 0)
        takeoff = float(
            not previous_takeoff
            and self.tracker.confirmed
            and self.start_kind == "standing"
        )
        extra = term + counter + loss + takeoff
        self.reward_sum += extra
        reward += extra
        info.update(
            reward_capture_region=term,
            reward_harmful_countermoment=counter,
            reward_balance_loss=loss,
            reward_certified_takeoff=takeoff,
            capture_distance=distance,
            capture_neighborhood=bool(distance < 1),
        )
        self.trace[-1].update(
            reward_capture_region=term,
            reward_harmful_countermoment=counter,
            reward_balance_loss=loss,
            reward_certified_takeoff=takeoff,
            capture_distance=distance,
        )
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        self.capture_credit = 0.0
        self.capture_distance = float("inf")
        return super().reset(*args, **kwargs)

    def restore_start(self, *args, **kwargs):
        result = super().restore_start(*args, **kwargs)
        self.capture_credit = 0.0
        self.capture_distance = float("inf")
        return result


class Vector:
    """Memory-bounded serial vector; exact lane/RNG restoration; no hidden resets."""

    def __init__(self, assets, cfg, bank, n, centers, seed=42):
        self.n = n
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.stage = 0
        self.envs = [Environment(assets, copy.deepcopy(cfg)) for _ in range(n)]
        for e in self.envs:
            e.capture_centers = np.array(centers)
        self.kinds = ["standing"] * n

    def start(self, i, kind=None):
        kind = kind or self.rng.choice(
            ["standing", "bridge", "balance"],
            p=[
                (0.1, 0.2, 0.35, 0.6)[self.stage],
                (0.15, 0.3, 0.4, 0.3)[self.stage],
                (0.75, 0.5, 0.25, 0.1)[self.stage],
            ],
        )
        self.kinds[i] = kind
        e = self.envs[i]
        if kind == "standing":
            return e.reset(1.0, int(self.rng.integers(2**31)))
        states = self.bank["train"][kind]
        if not states:
            raise ValueError("No validated " + kind + " states")
        return e.restore_start(
            states[int(self.rng.integers(len(states)))],
            "near_vertical" if kind == "balance" else "partial",
        )

    def reset(self, mode=None, mixture=None):
        x = [self.start(i) for i in range(self.n)]
        return np.stack([r[0] for r in x]), np.stack([r[1] for r in x])

    def step(self, mode, actions, mixture):
        results = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            (obs, crit), reward, done, info = e.step(a)
            terminal = crit.copy()
            row = e.trace[-1].copy()
            episode = None
            if done:
                episode = dict(info, start_distribution=self.kinds[i])
                obs, crit = self.start(i)
            results.append((obs, crit, reward, done, terminal, episode, row))
        return results

    def state(self):
        return dict(
            worlds=[e.state() for e in self.envs],
            rng=copy.deepcopy(self.rng.bit_generator.state),
            stage=self.stage,
            kinds=self.kinds.copy(),
        )

    def restore(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.stage = state["stage"]
        self.kinds = state["kinds"].copy()
        for e, s in zip(self.envs, state["worlds"]):
            e.restore(s)
        # Parent restore returns observations; explicit read keeps the contract clear.
        return np.stack([e.observations()[0] for e in self.envs]), np.stack(
            [e.observations()[1] for e in self.envs]
        )


def promote(stage, rows, reference):
    balance = [r for r in rows if r["distribution"] == "balance"]
    bridge = [r for r in rows if r["distribution"] == "bridge"]
    stand = [r for r in rows if r["distribution"] == "standing"]
    success = np.mean([r["success"] for r in balance]) if balance else 0.0
    bridge_hold = max((r["handstand_seconds"] for r in bridge), default=0)
    takeoff = (
        np.mean([r["certified_standing_takeoff"] for r in stand]) if stand else 0.0
    )
    if success < reference:
        return 0  # Explicit balance regression reduces standing exposure.
    if stage == 0 and success >= 0.75:
        return 1
    if stage == 1 and success >= 0.75 and bridge_hold >= 1:
        return 2
    if stage == 2 and success >= 0.75 and bridge_hold >= 5 and takeoff >= 0.75:
        return 3
    return stage


def source_contract():
    paths = []
    for root in [
        "experiments/unified_go2",
        "experiments/multi_impulse",
        "experiments/front_catch",
        "experiments/adc_front_capture",
        "experiments/momentum_phase",
        "experiments/com_energy_phase",
        "experiments/morphology/learning",
    ]:
        paths.extend(Path(root).glob("*.py"))
    paths.append(Path("scripts/unified_go2.py"))
    return {str(p): digest(p) for p in sorted(paths)}
