"""Go2 schedules. These do not modify a shared MjModel or reset a learner."""

from dataclasses import dataclass, field
import numpy as np
import mujoco


def effective_inertia(env):
    """Free-coordinate effective inertia 1/(M^-1)jj, including rotor armature.

    Local linear approximation; contacts/coupling and saturation invalidate an
    exact critical damping interpretation. Passive damping is subtracted below.
    """
    mass = np.zeros((env.m.nv, env.m.nv))
    mujoco.mj_fullM(env.m, env.d, mass)
    inverse = np.linalg.inv(mass)
    return 1 / np.diag(inverse)[env.vids]


def damping(env, stiffness):
    return np.maximum(
        0, 2 * np.sqrt(stiffness * effective_inertia(env)) - env.m.dof_damping[env.vids]
    )


@dataclass
class ActuatorCurriculum:
    nominal_k: float = 60.0
    nominal_b: float = 3.0
    initial_k: float = 60.0
    rule: str = "episode_ema"
    alpha: float = 0.05
    min_seconds: float = 0.0
    max_seconds: float = 8.0
    ema: float = 0.0
    progress: float = 0.0
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not (self.nominal_k <= self.initial_k <= 1.5 * self.nominal_k):
            raise ValueError("Initial gain must be within [nominal, 1.5 nominal]")
        if self.rule not in ("episode_ema", "recovery"):
            raise ValueError(self.rule)
        if not 0 < self.alpha <= 1 or self.max_seconds <= self.min_seconds:
            raise ValueError("Invalid EMA parameters")

    @property
    def stiffness(self):
        return self.initial_k + self.progress * (self.nominal_k - self.initial_k)

    def update(self, episodes, iteration):
        if not episodes:
            return
        if self.rule == "episode_ema":
            sample = np.mean(
                [
                    r.get("elapsed_episode_seconds", r["episode_seconds"])
                    for r in episodes
                ]
            )
            self.ema = (1 - self.alpha) * self.ema + self.alpha * sample
            candidate = np.clip(
                (self.ema - self.min_seconds) / (self.max_seconds - self.min_seconds),
                0,
                1,
            )
        else:
            # Real recovery required to complete; pitch alone cannot anneal fully.
            quality = []
            for r in episodes:
                pitch = np.clip(
                    (r["peak_collision_free_pitch"] - np.deg2rad(30)) / np.deg2rad(55),
                    0,
                    1,
                )
                contact = float(r.get("valid_support_transition", False))
                hold = np.clip(r.get("handstand_seconds", 0) / 5, 0, 1)
                quality.append(0.35 * pitch + 0.25 * contact + 0.4 * hold)
            self.ema = (1 - self.alpha) * self.ema + self.alpha * np.mean(quality)
            candidate = self.ema
        previous = self.progress
        self.progress = max(previous, float(candidate))
        if self.progress > previous:
            self.events.append(
                dict(
                    iteration=iteration,
                    progress=self.progress,
                    stiffness=self.stiffness,
                    ema=self.ema,
                )
            )


@dataclass
class ContactCurriculum:
    stage: int = 0
    events: list = field(default_factory=list)

    def mixture(self, enabled, bank):
        if not enabled:
            # Same starting mixture as legacy anticipatory experiment, fixed.
            standing = 0.35
        else:
            standing = (0.35, 0.5, 0.65, 0.8, 1.0)[self.stage]
        weights = np.array([0.24, 0.23, 0.23, 0.15, 0.15])
        kinds = ("loaded", "pre_liftoff", "post_liftoff", "partial", "near_vertical")
        weights *= [bool(bank["train"][k]) for k in kinds]
        if not weights.sum():
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return [standing, *((1 - standing) * weights / weights.sum()).tolist()]

    def update(self, rows, iteration):
        if self.stage == 4 or not rows:
            return
        intermediate = [r for r in rows if r["evaluation_type"] == "intermediate"]
        standing = [r for r in rows if r["evaluation_type"] == "standing"]
        threshold = (45, 60, 75, 85)[self.stage]
        if (
            intermediate
            and standing
            and np.mean([r["certified_standing_takeoff"] for r in standing]) >= 0.75
            and np.mean(
                [
                    r["peak_collision_free_pitch"] >= np.deg2rad(threshold)
                    and r.get("handstand_seconds", 0) >= 0.1
                    for r in intermediate
                ]
            )
            >= 0.5
        ):
            self.stage += 1
            self.events.append(dict(iteration=iteration, stage=self.stage))
