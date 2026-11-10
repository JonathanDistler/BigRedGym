"""Opt-in short reference bridge; no training environment or policy is changed."""

from dataclasses import asdict, dataclass, replace

import numpy as np

from gym.envs.go2.kickup_reference import reference


@dataclass(frozen=True)
class PostureTransition:
    takeover_s: float = 0.74
    front_hip: float = 0.3526922970
    front_knee: float = -2.0802952421
    front_gain: float = 0.9
    front_damping: float = 0.73349801206
    rear_hip: float = 1.02129456199
    rear_knee: float = -1.5986139901
    tuck_delay: float = 0.0
    tuck_duration: float = 0.12
    switch_s: float = 1.2
    blend_s: float = 0.6
    hip_delay: float = -0.1317054813
    pitch_gain: float = 0.0
    pitch_rate_gain: float = 0.0
    front_hip_delta: float = 0.0
    front_knee_delta: float = 0.0
    rear_hip_delta: float = 0.0
    rear_knee_delta: float = 0.0
    reference_s: float = 0.5
    com_gain: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Transition parameters must be finite")
        if not 0.3 <= self.takeover_s <= self.switch_s or self.blend_s <= 0:
            raise ValueError("Invalid transition timing")
        if (
            not 0 < self.front_gain <= 1
            or self.tuck_duration <= 0
            or self.reference_s <= 0
        ):
            raise ValueError("Invalid transition gains/durations")

    def to_dict(self):
        return asdict(self)

    def reference(self, t, source, standing):
        p = replace(
            source,
            front_push_hip=self.front_hip,
            front_push_knee=self.front_knee,
            front_gain=self.front_gain,
            rear_tuck_hip=self.rear_hip,
            rear_tuck_knee=self.rear_knee,
            tuck_delay=self.tuck_delay,
            tuck_duration=self.tuck_duration,
            hip_delay=self.hip_delay,
        )
        return reference(t, p, standing)


def smooth(t, start, duration):
    u = np.clip((t - start) / duration, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def rate_limited_target(desired, previous, velocity_limits, physics_dt):
    """Keep physical joint commands within one native velocity-limited step."""
    step = velocity_limits * physics_dt
    return previous + (desired - previous).clamp(-step, step)


# Coordinates searched by the bounded optimizer. No momentum/clearance reward.
BOUNDS = dict(
    takeover_s=(0.45, 0.84),
    front_hip=(0.10, 0.65),
    front_knee=(-2.4, -1.65),
    front_gain=(0.55, 1.0),
    front_damping=(0.5, 1.2),
    rear_hip=(0.8, 1.8),
    rear_knee=(-2.0, -0.95),
    tuck_delay=(0.0, 0.12),
    tuck_duration=(0.08, 0.30),
    switch_s=(1.02, 1.38),
    blend_s=(0.3, 0.95),
    hip_delay=(-0.2, 0.08),
    pitch_gain=(-0.3, 0.3),
    pitch_rate_gain=(-0.15, 0.15),
)


class BoundedCMA:
    """Small full-covariance CMA-ES, reflected box constraints, NumPy only.

    Search is performed in unit-box coordinates; ask/tell minimizes a cost.
    No physical simulator states are altered to satisfy constraints.
    """

    def __init__(self, mean, sigma=0.14, population=10, seed=873):
        self.mean = np.asarray(mean, dtype=float).copy()
        self.n = len(self.mean)
        self.sigma = sigma
        self.population = population
        self.rng = np.random.default_rng(seed)
        mu = population // 2
        self.weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1 / (self.weights @ self.weights)
        n, me = self.n, self.mueff
        self.cc = (4 + me / n) / (n + 4 + 2 * me / n)
        self.cs = (me + 2) / (n + me + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + me)
        self.cmu = min(1 - self.c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        self.damps = 1 + 2 * max(0, np.sqrt((me - 1) / (n + 1)) - 1) + self.cs
        self.chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.cov = np.eye(n)
        self.generation = 0

    def ask(self):
        vals, vecs = np.linalg.eigh(self.cov)
        self.invsqrt = (vecs / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
        root = vecs * np.sqrt(np.maximum(vals, 1e-12))
        raw = self.mean + self.sigma * (
            self.rng.normal(size=(self.population, self.n)) @ root.T
        )
        # Reflection preserves finite bounds without piling samples at endpoints.
        return 1 - np.abs(np.mod(raw, 2) - 1)

    def tell(self, candidates, costs):
        selected = np.asarray(candidates)[np.argsort(costs)[: len(self.weights)]]
        old = self.mean.copy()
        self.mean = self.weights @ selected
        y = (self.mean - old) / self.sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (self.invsqrt @ y)
        self.generation += 1
        norm = np.linalg.norm(self.ps)
        h = norm / np.sqrt(
            1 - (1 - self.cs) ** (2 * self.generation)
        ) / self.chi < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + h * np.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y
        ys = (selected - old) / self.sigma
        self.cov = (
            (1 - self.c1 - self.cmu + self.c1 * (1 - h) * self.cc * (2 - self.cc))
            * self.cov
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * (ys.T * self.weights) @ ys
        )
        self.cov = (self.cov + self.cov.T) / 2
        self.sigma *= np.exp(self.cs / self.damps * (norm / self.chi - 1))
