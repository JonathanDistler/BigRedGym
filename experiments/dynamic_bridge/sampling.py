"""Prioritize real pre-stall snapshots; checkpoint all sampling state."""

import copy
import numpy as np
from experiments.retained_go2.learning import Vector as RetainedVector


def prior(states):
    """Smooth reset weights, never an action/controller switch or viability claim."""
    values = [s["values"]["momentum_sample"] for s in states]
    pitch = np.array([s["pitch"] for s in values])
    omega = np.array([s["pitch_velocity"] for s in values])
    # Include the earlier preparation window, not only the highest reached pose.
    return 0.25 + np.exp(-(((pitch - np.deg2rad(38)) / np.deg2rad(12)) ** 2)) * (
        0.5 + 0.5 * np.exp(-((omega / 1.5) ** 2))
    )


class Vector(RetainedVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge_prior = prior(self.bank["train"]["bridge"])
        self.bridge_failure = np.ones(len(self.bridge_prior))
        self.bridge_trials = np.zeros(len(self.bridge_prior), dtype=np.int64)
        self.bridge_ids = [-1] * self.n

    def bridge_probabilities(self):
        # At least 20% uniform coverage prevents permanent starvation.
        w = self.bridge_prior * (0.25 + self.bridge_failure)
        return 0.2 / len(w) + 0.8 * w / w.sum()

    def start(self, i, kind=None):
        if kind is None:
            kind = self.rng.choice(
                ["standing", "bridge", "balance"], p=self.probabilities()
            )
        self.bridge_ids[i] = -1
        if kind != "bridge":
            return super().start(i, kind)
        index = int(
            self.rng.choice(len(self.bridge_prior), p=self.bridge_probabilities())
        )
        self.bridge_ids[i] = index
        self.kinds[i] = kind
        return self.envs[i].restore_start(self.bank["train"][kind][index], "partial")

    def step(self, *args, **kwargs):
        ids = self.bridge_ids.copy()
        result = super().step(*args, **kwargs)
        for index, row in zip(ids, result):
            episode = row[5]
            if index >= 0 and episode:
                # Recovery, not arbitrary high pitch or repeated kicks, defines improvement.
                recovery = min(1.0, max(0.0, episode["handstand_seconds"]) / 5.0)
                self.bridge_failure[index] = 0.9 * self.bridge_failure[index] + 0.1 * (
                    1 - recovery
                )
                self.bridge_trials[index] += 1
        return result

    def state(self):
        s = super().state()
        s["dynamic_bridge"] = copy.deepcopy(
            dict(
                prior=self.bridge_prior,
                failure=self.bridge_failure,
                trials=self.bridge_trials,
                ids=self.bridge_ids,
            )
        )
        return s

    def restore(self, state):
        extra = state["dynamic_bridge"]
        if not np.array_equal(extra["prior"], self.bridge_prior):
            raise ValueError("Bridge sampling bank changed")
        result = super().restore(state)
        self.bridge_failure = extra["failure"].copy()
        self.bridge_trials = extra["trials"].copy()
        self.bridge_ids = extra["ids"].copy()
        return result
