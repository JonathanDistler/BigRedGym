"""Checkpoint adaptation and deterministic, event-aligned teacher playback."""

import copy
from pathlib import Path
import numpy as np
import torch
from experiments.momentum_phase.core import MomentumConfig, MomentumEnvironment
from experiments.morphology.learning.ppo import ActorCritic
from scripts.morphology_feasibility import save_csv


DEFAULT_INITIAL = Path(
    "output/com_energy_phase_diagnostic_20261112_v2/C_seed42/checkpoint_000.pt"
)


def network(checkpoint, seed=42):
    torch.manual_seed(seed)
    net = ActorCritic(61, 91)
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    state = copy.deepcopy(saved["network"])
    for label, size in [("actor", 61), ("critic", 91)]:
        key = f"{label}.0.weight"
        w = state[key]
        if w.shape[1] > size:
            raise ValueError("Incompatible checkpoint inputs")
        state[key] = torch.cat([w, torch.zeros((len(w), size - w.shape[1]))], 1)
    net.load_state_dict(state)
    return net, saved


def configuration(ref, times, condition="A", envelope=None):
    cfg = MomentumConfig(
        condition=condition,
        phase_times=tuple(times),
        target_height=ref["height"],
        target_rotation=tuple(ref["rotation"]),
        inertia_reference=ref["measurements"]["inertia_support_y"],
    )
    if envelope:
        cfg.momentum_com_cap = envelope["com_cap"]
        cfg.momentum_support_cap = envelope["support_cap"]
        cfg.sigma_momentum = envelope["sigma_support"]
    return cfg


def overlay(env, action, kind):
    """Ablation-only bounded targets after certified release."""
    if not env.tracker.confirmed or kind in ("none", "phase_filter"):
        return action
    age = env.d.time - env.tracker.first_takeoff_time
    ramp = np.clip(age / 0.08, 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    target = np.clip(env.nominal + env.cfg.action_scale * action, *env.bounds.T)
    if kind == "rear_tuck":
        target[[7, 10]] += 0.15 * ramp
        target[[8, 11]] -= 0.3 * ramp
    elif kind == "rear_extend":
        target[[7, 10]] -= 0.15 * ramp
        target[[8, 11]] += 0.25 * ramp
    elif kind == "front_fold":
        target[[1, 4]] += 0.12 * ramp
        target[[2, 5]] -= 0.2 * ramp
    elif kind == "front_open":
        target[[1, 4]] -= 0.12 * ramp
        target[[2, 5]] += 0.2 * ramp
    elif kind == "front_compliance":
        target[:6] = (1 - 0.5 * ramp) * target[:6] + 0.5 * ramp * env.d.qpos[
            env.qids[:6]
        ]
    elif kind == "combined":
        target[[7, 10]] += 0.15 * ramp
        target[[8, 11]] -= 0.3 * ramp
        target[[1, 4]] -= 0.1 * ramp
        target[[2, 5]] += 0.15 * ramp
    else:
        raise ValueError(kind)
    return np.clip((target - env.nominal) / env.cfg.action_scale, -1, 1)


def rollout(net, env, seed, kind="none", capture=False):
    obs, priv = env.reset(1.0, seed)
    frames = []
    while True:
        with torch.no_grad():
            action = net.actor(torch.tensor(obs)).tanh().numpy()
        action = overlay(env, action, kind)
        before = (obs.copy(), priv.copy(), action.copy(), env.phase.progress)
        (obs, priv), reward, done, info = env.step(action)
        # Only actual collision-free state-action pairs, no invented completion labels.
        if capture and not env.nonfoot_collision and not env.self_collision:
            frames.append(before)
        if done:
            break
    valid = [s for s in env.physics_trace if s["prefix_safe"]]
    flight = [s for s in valid if s["flight_valid"]]
    push = [
        s for s in valid if s["rear_loaded"] and s["phase"] >= 0.2 and s["phase"] < 0.4
    ]
    peak = (
        max(flight, key=lambda r: r["pitch"])
        if flight
        else max(valid, key=lambda r: r["pitch"])
        if valid
        else env.physics_trace[0]
    )
    info.update(
        seed=seed,
        schedule=kind,
        peak_collision_free_pitch=max((s["pitch"] for s in valid), default=0),
        peak_certified_pitch=max((s["pitch"] for s in flight), default=0),
        peak_L_com=max((s["L_com"] for s in valid), default=0),
        peak_L_support=max(
            (s["L_support"] for s in valid if s["support_valid"]), default=0
        ),
        positive_rear_impulse=sum(
            max(s["rear_moment_com"], 0) * env.m.opt.timestep for s in push
        ),
        net_push_impulse=sum(s["moment_com"] * env.m.opt.timestep for s in push),
        front_push_impulse=sum(
            s["front_moment_com"] * env.m.opt.timestep for s in push
        ),
        max_momentum_residual_com=max(
            (abs(s["momentum_residual_com"]) for s in valid), default=0
        ),
        max_momentum_residual_fixed=max(
            (abs(s["momentum_residual_fixed"]) for s in valid), default=0
        ),
        apex_potential=peak["potential"],
        apex_kinetic=peak["K_system"],
        apex_L=peak["L_support"],
        apex_omega=peak["pitch_velocity"],
        apex_calf_clearance=peak["calf_clearance"],
        max_potential=max((s["potential"] for s in valid), default=0),
        max_mechanical_energy=max((s["mechanical_energy"] for s in valid), default=0),
        milestone45=any(s["pitch"] >= np.deg2rad(45) for s in flight),
        milestone60=any(s["pitch"] >= np.deg2rad(60) for s in flight),
        milestone80=any(s["pitch"] >= np.deg2rad(80) for s in flight),
    )
    # Demonstrations exclude the impact frame and everything after it.
    return info, frames


def evaluate(net, env, seeds, directory, label):
    rows = []
    traces = []
    physics = []
    for seed in seeds:
        info, _ = rollout(net, env, seed)
        rows.append(dict(**info, checkpoint=label))
        traces.extend(dict(**s, seed=seed, checkpoint=label) for s in env.trace)
        physics.extend(
            dict(**s, seed=seed, checkpoint=label) for s in env.physics_trace
        )
    save_csv(directory / f"evaluation_{label}.csv", rows)
    save_csv(directory / f"trace_{label}.csv", traces)
    save_csv(directory / f"physics_{label}.csv", physics)
    return rows


def balance_demonstrations(assets, ref, times, archive):
    """State/action labels from archived successful balance continuations only."""
    from experiments.com_energy_phase.core import EnergyEnvironment
    from experiments.morphology.learning.takeoff import inverse_target_action

    bank = np.load(Path(archive) / "recoverable_states.npz")
    env = EnergyEnvironment(
        assets,
        configuration(ref, times, "C"),
    )
    # Momentum reset intentionally cannot fabricate a takeoff certificate. Use the
    # original validated intermediate-reset facility to form offline observations.
    frames = []
    metadata = []
    for index in ref["checked_indices"]:
        env.reset_intermediate(
            bank[f"{index}_integration"], index, dict(source=str(archive))
        )
        action, projection = inverse_target_action(env, bank[f"{index}_target"])
        if (
            projection["action_projection_fraction"] > 0
            or projection["target_error"] > 0.02
        ):
            continue
        s = sample_for_features(env)
        a, c = env.observations()
        # Use the exact actor feature codec without a controller step.
        feature_env = object.__new__(MomentumEnvironment)
        feature_env.momentum_sample = s
        feature_env.m = env.m
        feature_env.cfg = env.cfg
        extra = feature_env.features()
        frames.append(
            (
                np.r_[a, extra].astype(np.float32),
                np.r_[c, extra].astype(np.float32),
                action,
                0.8,
            )
        )
        metadata.append(dict(index=index, **projection))
    return frames, metadata


def sample_for_features(env):
    from experiments.momentum_phase.physics import sample

    env.pitch_axis = np.array([0.0, 1.0, 0.0])
    env.fixed_support = np.r_[env.anchor.mean(0), 0.0]
    return sample(env, env.row)


def save_frames(path, frames, seeds):
    if not frames:
        raise ValueError("No valid offline frames")
    np.savez_compressed(
        path,
        observations=np.stack([r[0] for r in frames]),
        privileged=np.stack([r[1] for r in frames]),
        actions=np.stack([r[2] for r in frames]),
        phase=np.array([r[3] for r in frames]),
        seed=np.array(seeds),
    )
