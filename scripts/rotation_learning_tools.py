"""Trusted checkpoint replay, BC phase labels, and bounded action search."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.rotation import RotationConfig, RotationEnvironment
from experiments.morphology.learning.takeoff import inverse_target_action
from experiments.morphology.task import write_json
from scripts.morphology_feasibility import save_csv


def load_actor(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    p = state["curriculum"]["protocol"]
    net = ActorCritic(
        44, 74, p["ppo"]["hidden_dims"], log_std_min=p["ppo"]["log_std_min"]
    )
    net.load_state_dict(state["network"])
    return net, state, p


def action(net, obs):
    with torch.no_grad():
        return net.actor(torch.tensor(obs)).tanh().numpy()


def environment(assets, protocol, enabled=False):
    return RotationEnvironment(
        assets,
        RotationConfig(
            **dict(
                protocol["environment"],
                terminate_on_goal=False,
                rotation_enabled=enabled,
            )
        ),
    )


def prepare_phases(output, dataset, assets):
    data = np.load(dataset)
    phase = np.ones(len(data["time"]), dtype=np.float32)
    certificates = []
    env = RotationEnvironment(assets, RotationConfig(rotation_enabled=False))
    for scale, seed in sorted(set(zip(data["scale"], data["seed"]))):
        indices = np.flatnonzero((data["scale"] == scale) & (data["seed"] == seed))
        env.reset(float(scale), int(seed))
        for index in indices:
            # Label the observation BEFORE its action by the replay certificate.
            if env.tracker.confirmed:
                age = max(0.0, data["time"][index] - env.tracker.first_takeoff_time)
                phase[index] = 0.1 + 0.9 * np.exp(-age / 0.10)
            _, _, done, _ = env.step(data["actions"][index])
            if done:
                break
        certificates.append(
            dict(
                scale=float(scale),
                seed=int(seed),
                takeoff=env.tracker.confirmed,
                certified_time=env.tracker.first_takeoff_time,
            )
        )
    np.savez_compressed(output / "bc_phase_labels.npz", weights=phase)
    write_json(
        output / "bc_phase_labels.json",
        dict(
            dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            frames=len(phase),
            post_certificate_frames=int(np.sum(phase < 1)),
            minimum_frame_weight=float(phase.min()),
            certificates=certificates,
            floor=0.1,
            seconds_time_constant=0.10,
            note="Replay certificate required; no-takeoff frames stay fully anchored.",
        ),
    )


def diagnose(args, assets):
    summaries, contexts, counterfactuals = [], [], []
    for method in ("A", "B"):
        for training_seed in (42, 43):
            checkpoint = (
                args.source / f"{method}_seed{training_seed}" / "checkpoint_000030.pt"
            )
            net, state, p = load_actor(checkpoint)
            bc = copy.deepcopy(net)
            bc_saved = torch.load(
                "output/multimorph_bc_20261111_v2/bc_actor.pt",
                weights_only=False,
                map_location="cpu",
            )
            bc.actor.load_state_dict(bc_saved["actor"])
            for scale in (1.0,) if method == "A" else (0.95, 1.0):
                for seed in (61000, 61001):
                    env = environment(assets, p)
                    obs, _ = env.reset(scale, seed)
                    env.record_rotation_trace = True
                    snapshot = None
                    control = []
                    while True:
                        act = action(net, obs)
                        anchor = action(bc, obs)
                        begin = len(env.rotation_trace)
                        (obs, _), _, done, info = env.step(act)
                        for row in env.rotation_trace[begin:]:
                            row["bc_action_rms"] = float(
                                np.sqrt(np.mean((act - anchor) ** 2))
                            )
                            row["bc_coefficient"] = 2.0
                            row["bc_onpolicy_latent_penalty"] = float(
                                2
                                * np.mean(
                                    (
                                        (
                                            np.arctanh(np.clip(act, -0.999, 0.999))
                                            - np.arctanh(np.clip(anchor, -0.999, 0.999))
                                        )
                                        / 0.05
                                    )
                                    ** 2
                                )
                            )
                        control.append(
                            dict(
                                time=float(env.d.time),
                                qpos=env.d.qpos.copy(),
                                qvel=env.d.qvel.copy(),
                                action=act.copy(),
                            )
                        )
                        if env.tracker.confirmed and snapshot is None:
                            snapshot = copy.deepcopy(env.state())
                            snapshot["values"]["rotation_trace"] = []
                            snapshot["values"]["record_rotation_trace"] = False
                        if done:
                            break
                    tag = f"{method}_seed{training_seed}_scale{scale:.2f}_reset{seed}"
                    save_csv(args.output / f"{tag}_trajectory.csv", env.rotation_trace)
                    np.savez_compressed(
                        args.output / f"{tag}_states.npz",
                        **{k: np.array([r[k] for r in control]) for k in control[0]},
                    )
                    summaries.append(
                        dict(
                            **info,
                            method=method,
                            training_seed=training_seed,
                            evaluation_seed=seed,
                            trace=tag,
                        )
                    )
                    if seed == 61000 and snapshot is not None:
                        torch.save(
                            dict(
                                world=snapshot,
                                checkpoint=str(checkpoint),
                                scale=scale,
                                seed=seed,
                            ),
                            args.output / f"{tag}_liftoff.pt",
                        )
                        contexts.append(
                            dict(tag=tag, path=str(args.output / f"{tag}_liftoff.pt"))
                        )
                        for mode in ("hold_all", "hold_rear"):
                            cf = environment(assets, p)
                            cf.restore(copy.deepcopy(snapshot))
                            target = cf.target.copy()
                            obs, _ = cf.observations()
                            while True:
                                act = action(net, obs)
                                if mode == "hold_all":
                                    act, _ = inverse_target_action(cf, target)
                                else:
                                    held, _ = inverse_target_action(cf, target)
                                    for j, name in enumerate(cf.names):
                                        if name.startswith(("RL", "RR")):
                                            act[j] = held[j]
                                (obs, _), _, done, cf_info = cf.step(act)
                                if done:
                                    break
                            counterfactuals.append(
                                dict(
                                    **cf_info,
                                    method=method,
                                    training_seed=training_seed,
                                    mode=mode,
                                    evaluation_seed=seed,
                                )
                            )
    save_csv(args.output / "baseline_episodes.csv", summaries)
    save_csv(args.output / "counterfactuals.csv", counterfactuals)
    write_json(args.output / "contexts.json", contexts)
    dataset = Path("output/multimorph_bc_20261111_v2/bc_dataset.npz")
    prepare_phases(args.output, dataset, assets)


def search(args, assets):
    contexts = json.loads((args.source / "contexts.json").read_text())
    rows, results = [], []
    rng = np.random.default_rng(71000)
    for context in contexts:
        saved = torch.load(context["path"], map_location="cpu", weights_only=False)
        net, _, p = load_actor(Path(saved["checkpoint"]))
        env = environment(assets, p)
        # Three action knots, grouped by front/rear and joint type, 18 variables.
        mean = np.zeros((3, 6))
        std = np.full_like(mean, 0.08)
        best_score = -np.inf
        best = None
        best_peak = 0.0
        interactions = 0
        for generation in range(3):
            candidates = rng.normal(mean, std, size=(24, 3, 6))
            candidates[0] = mean
            ranked = []
            for candidate_index, residual in enumerate(candidates):
                env.restore(copy.deepcopy(saved["world"]))
                env.rotation_trace = []
                env.record_rotation_trace = True
                obs, _ = env.observations()
                start = float(env.d.time)
                actions = []
                states = []
                valid = True
                info = None
                for step in range(25):
                    knot = min(2, step * 3 // 25)
                    offsets = (
                        np.tile(residual[knot, :3], 2).tolist()
                        + np.tile(residual[knot, 3:], 2).tolist()
                    )
                    act = np.clip(action(net, obs) + np.array(offsets), -1, 1)
                    states.append(np.r_[env.d.qpos.copy(), env.d.qvel.copy()])
                    actions.append(act.copy())
                    (obs, _), _, done, info = env.step(act)
                    interactions += 1
                    valid &= (
                        not info["nonfoot_collision"]
                        and not info["self_collision"]
                        and info["max_speed_ratio"] <= 1.001
                        and info["joint_limit_violation"] < 0.001
                    )
                    if done:
                        break
                peak = float(info["max_pitch"])
                valid_peak = max(
                    [
                        r["pitch"]
                        for r in env.rotation_trace
                        if not r["nonfoot_collision"] and not r["self_collision"]
                    ],
                    default=0.0,
                )
                score = (
                    valid_peak
                    - 0.8 * float(not valid)
                    - 0.05 * abs(info["pitch_velocity"])
                    - 0.02 * np.mean(residual**2)
                )
                ranked.append((score, residual.copy()))
                rows.append(
                    dict(
                        context=context["tag"],
                        generation=generation,
                        candidate=candidate_index,
                        valid=bool(valid),
                        peak_pitch=peak,
                        valid_prefix_peak=valid_peak,
                        final_pitch=info["pitch"],
                        final_omega=info["pitch_velocity"],
                        duration=float(env.d.time) - start,
                        steps=len(actions),
                        max_speed_ratio=info["max_speed_ratio"],
                        joint_limit_violation=info["joint_limit_violation"],
                        nonfoot_collision=info["nonfoot_collision"],
                        score=score,
                    )
                )
                if score > best_score:
                    best_score = score
                    best = dict(
                        actions=np.array(actions),
                        states=np.array(states),
                        trace=copy.deepcopy(env.rotation_trace),
                        valid=bool(valid),
                        peak=valid_peak,
                        residual=residual.copy(),
                    )
                if valid:
                    best_peak = max(best_peak, valid_peak)
            elites = sorted(ranked, key=lambda x: x[0], reverse=True)[:6]
            samples = np.stack([v for _, v in elites])
            mean = 0.3 * mean + 0.7 * samples.mean(0)
            std = np.maximum(0.015, 0.3 * std + 0.7 * samples.std(0))
        tag = context["tag"]
        save_csv(args.output / f"{tag}_search_best_trajectory.csv", best["trace"])
        np.savez_compressed(
            args.output / f"{tag}_search_best.npz",
            actions=best["actions"],
            states=best["states"],
            residual=best["residual"],
        )
        qualifies = bool(best["valid"] and best["peak"] >= np.deg2rad(45))
        results.append(
            dict(
                context=tag,
                training_interactions=0,
                search_interactions=interactions,
                candidates=72,
                best_valid_pitch=best_peak,
                best_saved_valid=best["valid"],
                best_saved_pitch=best["peak"],
                reached_45=bool(best_peak >= np.deg2rad(45)),
                reached_60=bool(best_peak >= np.deg2rad(60)),
                optional_demo_valid=qualifies,
            )
        )
        save_csv(args.output / "search_candidates.csv", rows)
        write_json(args.output / "search_summary.json", results)
        print(results[-1], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=["diagnose", "search"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(args.output / "assets")
    (diagnose if args.mode == "diagnose" else search)(args, assets)


if __name__ == "__main__":
    main()
