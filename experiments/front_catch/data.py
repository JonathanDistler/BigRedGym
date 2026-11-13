"""Validated reset banks and preserved-actor adaptation; no trajectory search."""

import copy
import hashlib
from pathlib import Path
import mujoco
import numpy as np
import torch
from experiments.front_catch.core import CatchConfig, CatchPhase
from experiments.momentum_phase.physics import sample
from experiments.morphology.learning.environment import Environment
from experiments.morphology.learning.ppo import ActorCritic


INITIAL = Path(
    "output/momentum_phase_diagnostic_20261112_v1/C_seed42/checkpoint_002.pt"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def network(path=INITIAL):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    net = ActorCritic(73, 103)
    state = copy.deepcopy(saved["network"])
    for label, size in (("actor", 73), ("critic", 103)):
        key = f"{label}.0.weight"
        w = state[key]
        if w.shape[1] > size:
            raise ValueError("Incompatible checkpoint architecture")
        state[key] = torch.cat([w, torch.zeros((len(w), size - w.shape[1]))], 1)
    net.load_state_dict(state)
    return net, saved


def config(saved, mode="A", full_trace=False):
    parent = {
        k: v
        for k, v in saved["config"].items()
        if k in CatchConfig.__dataclass_fields__
    }
    return CatchConfig(
        **dict(
            parent,
            condition="A",
            catch_mode=mode,
            full_trace=full_trace,
            phase_filter=False,
            action_tau=0.0,
        )
    )


def validate_state(env):
    row = Environment.measure(env)
    contacts = [c for c in env.d.contact if c.dist < 0]
    max_pen = max([-float(c.dist) for c in contacts], default=0.0)
    q = env.d.qpos[env.qids]
    velocity = env.d.qvel[env.vids]
    valid = bool(
        np.isfinite(env.d.qpos).all()
        and np.isfinite(env.d.qvel).all()
        and not row["illegal_contacts"]
        and not row["self_contacts"]
        and (q >= env.bounds[:, 0] - 0.001).all()
        and (q <= env.bounds[:, 1] + 0.001).all()
        and np.max(abs(velocity) / env.speed) <= 1.001
        and max_pen <= 0.0025
    )
    return valid, dict(
        pitch=row["pitch"],
        front_support=row["front_support"],
        rear_clearance=row["rear_clearance"],
        max_penetration=max_pen,
        illegal_bodies=row["illegal_bodies"],
        valid=valid,
    )


def raw_state(
    env,
    kind,
    seed,
    integration=None,
    qpos=None,
    qvel=None,
    target=None,
    time=0.0,
    torque=None,
):
    env.reset(1.0, seed)
    env.record_physics = False
    if integration is not None:
        mujoco.mj_setState(
            env.m, env.d, integration, mujoco.mjtState.mjSTATE_INTEGRATION
        )
    else:
        env.d.qpos[:] = qpos
        env.d.qvel[:] = qvel
        env.d.time = time
        env.d.qfrc_applied[:] = 0
        if torque is not None:
            env.d.qfrc_applied[env.vids] = np.clip(torque, -env.effort, env.effort)
    mujoco.mj_forward(env.m, env.d)
    env.row = Environment.measure(env)
    if target is not None:
        env.target = np.clip(target, *env.bounds.T)
    else:
        env.target = env.d.qpos[env.qids].copy()
    env.anchor = env.d.geom_xpos[env.foot_geoms[:2], :2].copy()
    env.fixed_support = np.r_[env.anchor.mean(0), 0.0]
    env._previous_q = env.d.qpos[env.qids].copy()
    env._previous_v = env.d.qvel[env.vids].copy()
    env._last_tracker_time = env._work_time = float(env.d.time)
    env.momentum_sample = sample(env, env.row)
    env.previous_physics = env.momentum_sample.copy()
    env.phase = CatchPhase(env.cfg.phase_times)
    env.phase.progress = 0.8 if kind == "near_vertical" else 0.4
    env.phase.takeoff_time = float(env.d.time)
    env.phase.front_loaded = bool(env.row["front_support"])
    env.record_physics = True
    state = env.state()
    env.restore_start(state, kind)
    return validate_state(env)


def state_id(env):
    return hashlib.sha256(
        np.round(np.r_[env.d.qpos, env.d.qvel], 9).tobytes()
    ).hexdigest()


class Curriculum:
    """Shared performance gates keep A/B/C reset distributions matched."""

    def __init__(self):
        self.stage = 0
        self.events = []

    @property
    def mixture(self):
        standing = (0.35, 0.5, 0.65, 0.8, 1.0)[self.stage]
        return [standing] + [(1 - standing) * x for x in (0.24, 0.23, 0.23, 0.15, 0.15)]

    def update(self, rows, iteration):
        if self.stage >= 4:
            return
        thresholds = (45, 60, 80, 85)
        passed = all(
            len(r) > 0
            and np.mean([x["certified_standing_takeoff"] for x in r]) >= 0.75
            and np.mean(
                [x["peak_swing_pitch"] >= np.deg2rad(thresholds[self.stage]) for x in r]
            )
            >= 0.5
            for r in rows
        )
        if passed:
            self.stage += 1
            self.events.append(
                dict(
                    iteration=iteration,
                    stage=self.stage,
                    standing_probability=self.mixture[0],
                )
            )
