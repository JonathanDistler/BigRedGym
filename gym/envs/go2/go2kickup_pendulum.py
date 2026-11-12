"""Opt-in CPU task registration; compatible with the 44/74/12 research runner.

This registry is deliberately separate from the GPU LeggedRobot registry: the
archived checkpoint is a CPU MuJoCo research actor, not a LeggedRobot policy.
"""

from dataclasses import dataclass
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.morphology.learning.pendulum_reference import (
    PendulumInstrument,
    score_trajectory,
)
from experiments.morphology.learning.rotation import RotationConfig


@dataclass
class Go2KickupPendulumConfig(RotationConfig):
    rotation_enabled: bool = False
    reference_file: str = ""
    reference_stage: int = 0
    warm_states: str = ""


class Go2KickupPendulum(PendulumInstrument):
    task_name = "go2kickup_pendulum"

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if getattr(self, "_restoring", False):
            return result
        self.active_stage = getattr(self.cfg, "reference_stage", 3)
        path = getattr(self.cfg, "warm_states", "")
        if path and self.active_stage < 3:
            stages = json.loads(Path(path).read_text())
            paths = stages[str(self.active_stage)]
            seed = int(args[1] if len(args) > 1 else kwargs["seed"])
            state = torch.load(
                paths[seed % len(paths)], map_location="cpu", weights_only=False
            )
            self._restoring = True
            self.restore(copy.deepcopy(state))
            self._restoring = False
            result = self.observations()
        self.reference_clock = None
        self.reference_phi = 0.0
        self.reference_seen_target = False
        self.reference_success = False
        self.record_rotation_trace = True
        self.rotation_trace = []
        self._reference_table = (
            pd.read_csv(self.cfg.reference_file)
            if getattr(self.cfg, "reference_file", "")
            else None
        )
        if self._reference_table is not None:
            self._reference_table["time"] -= self._reference_table.time.iloc[0]
        self.reward_sum = 0.0
        self.terms_sum = {}
        return result

    def step(self, action):
        previous = {
            k: v for k, v in self.terms_sum.items() if k.startswith("reference_")
        }
        obs, reward, done, info = super().step(action)
        ref = self._reference_table
        if ref is None:
            return obs, reward, done, info
        s = self.rotation_sample
        if (
            self.reference_clock is None
            and self.tracker.confirmed
            and s["pitch"] >= np.deg2rad(30)
        ):
            self.reference_clock = float(self.d.time)
            self.rotation_trace = []
        terms = dict(
            reference_tracking=0.0,
            reference_hold=0.0,
            reference_slip=0.0,
            reference_contact=0.0,
            reference_failure=0.0,
        )
        if self.reference_clock is not None:
            time = float(self.d.time) - self.reference_clock
            dx = s["com_x"] - (s["FL_position_x"] + s["FR_position_x"]) / 2
            dz = s["com_z"]
            theta = np.arctan2(dx, dz)
            length = np.hypot(dx, dz)
            errors = [
                (s["pitch"] - np.interp(time, ref.time, ref.pitch)) / 0.1,
                (s["pitch_velocity"] - np.interp(time, ref.time, ref.pitch_velocity)),
                (theta - np.interp(time, ref.time, ref.theta)) / 0.1,
                (length - np.interp(time, ref.time, ref.length)) / 0.05,
            ]
            supported = (
                s["FL_load"] > 0.05 * self.weight and s["FR_load"] > 0.05 * self.weight
            )
            phi = (
                float(np.exp(-np.mean(np.square(errors))))
                if supported and not self.nonfoot_collision
                else 0.0
            )
            if time >= 0.7:
                done = True
            valid = score_trajectory(self.rotation_trace, self.weight)
            self.reference_success = valid["success"]
            terms.update(
                reference_tracking=2
                * (self.cfg.gamma * (0.0 if done else phi) - self.reference_phi),
                reference_hold=2
                * self.cfg.dt
                * float(
                    supported
                    and abs(s["pitch"] - np.pi / 4) <= np.deg2rad(5)
                    and abs(s["pitch_velocity"]) <= 0.5
                ),
                reference_slip=-self.cfg.dt
                * (
                    np.hypot(s["FL_velocity_x"], s["FL_velocity_y"])
                    + np.hypot(s["FR_velocity_x"], s["FR_velocity_y"])
                ),
                reference_contact=-0.1 * self.cfg.dt * float(not supported),
                reference_failure=-float(self.nonfoot_collision or self.self_collision),
            )
            self.reference_phi = 0.0 if done else phi
        extra = sum(terms.values())
        self.reward_sum += extra
        for k, v in terms.items():
            self.terms_sum[k] = previous.get(k, 0.0) + v
        info.update(self.metrics())
        info.update({f"reward_{k}": v for k, v in terms.items()})
        info.update(
            reward=reward + extra,
            reference_success=self.reference_success,
            reference_stage=self.active_stage,
        )
        return obs, reward + extra, done, info


CPU_TASKS = {"go2kickup_pendulum": (Go2KickupPendulum, Go2KickupPendulumConfig)}


def register_cpu_task():
    return CPU_TASKS.copy()
