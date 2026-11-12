"""Episode-boundary morphology selection; schedule state is checkpointable."""

from dataclasses import dataclass, field
from gym.envs.go2.leg_morphology import SCALES


@dataclass
class Curriculum:
    method: str = "A"
    start: float = 0.85
    threshold: float = 0.7
    min_episodes: int = 32
    required_windows: int = 2
    fixed_interval: int = 250
    stage: int = 0
    passed_windows: int = 0
    outcomes: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.method not in ("A", "B", "C", "D") or self.start not in SCALES:
            raise ValueError("invalid method or start scale")
        self.scales = [s for s in SCALES if s >= self.start]
        if (
            not 0 < self.threshold <= 1
            or self.min_episodes < 1
            or self.required_windows < 1
        ):
            raise ValueError("invalid progression thresholds")
        if self.fixed_interval < 1:
            raise ValueError("fixed_interval must be positive")

    @property
    def scale(self):
        return (
            1.0
            if self.method == "A"
            else self.start
            if self.method == "B"
            else self.scales[self.stage]
        )

    def completed(self, episode_scale, success, iteration):
        if self.method != "C" or episode_scale != self.scale:
            return
        self.outcomes.append(bool(success))
        if len(self.outcomes) < self.min_episodes:
            return
        rate = sum(self.outcomes) / len(self.outcomes)
        self.outcomes.clear()
        self.passed_windows = self.passed_windows + 1 if rate >= self.threshold else 0
        if self.passed_windows >= self.required_windows:
            self.advance(
                iteration,
                (
                    f"{self.required_windows} windows of {self.min_episodes} episodes; "
                    f"success >= {self.threshold}"
                ),
                rate,
            )

    def iteration(self, iteration):
        if self.method == "D":
            desired = min(iteration // self.fixed_interval, len(self.scales) - 1)
            while self.stage < desired:
                self.advance(iteration, "fixed iteration interval", None)

    def advance(self, iteration, condition, success_rate):
        if self.stage >= len(self.scales) - 1:
            return
        old = self.scale
        self.stage += 1
        self.passed_windows = 0
        self.events.append(
            dict(
                iteration=iteration,
                previous_scale=old,
                scale=self.scale,
                condition=condition,
                success_rate=success_rate,
            )
        )
