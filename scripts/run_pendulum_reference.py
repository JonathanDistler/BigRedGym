"""Bundle-backed, contact-aware OFFLINE feasibility experiment. No PPO here."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.rotation import RotationConfig
from experiments.morphology.learning.pendulum_reference import (
    PendulumInstrument,
    ReferenceGate,
    score_trajectory,
)
from experiments.morphology.learning.pendulum_models import (
    enrich,
    identify,
    validate,
    reference,
)
from experiments.morphology.learning.takeoff import inverse_target_action
from experiments.morphology.task import write_json
from scripts.rotation_learning_tools import action, load_actor
from scripts.morphology_feasibility import save_csv


BUNDLE_SHA = "1779e4669007a481aeba9973a037c3907739d77fde2c76876997f0f153091b42"
TRAIN_RESETS = [65000, 65001, 65002]
MODEL_TEST_RESETS = [65010, 65011, 65012]
GATE_RESETS = list(range(65100, 65110))


def verify_bundle(bundle, output):
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != BUNDLE_SHA:
        raise ValueError("Original verified bundle SHA256 mismatch")
    copied = []
    preserved = 0
    with zipfile.ZipFile(bundle) as z:
        manifest = json.loads(z.read("molab_manifest.json"))
        for name, expected in manifest["files"].items():
            content = z.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(name)
            original = Path(name)
            if original.exists():
                if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
                    raise ValueError(
                        "Archived original differs from working copy: " + name
                    )
                preserved += 1
            selected = name.startswith(
                "output/rotation_pilot_20261111_v1/global/A_seed"
            ) and (
                name.endswith("checkpoint_000036.pt")
                or name.endswith("configuration.json")
            )
            selected |= name == (
                "output/rotation_search_20261111_v2/"
                "A_seed42_scale1.00_reset61000_optional_rotation_demo.npz"
            )
            selected |= name == "output/multimorph_bc_20261111_v2/bc_actor.pt"
            selected |= name == "output/multimorph_bc_20261111_v2/bc_dataset.npz"
            if selected:
                target = output / "inputs" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                copied.append(name)
    write_json(
        output / "bundle_verification.json",
        dict(
            bundle_sha256=digest,
            archive_files_verified=len(manifest["files"]),
            existing_files_preserved=preserved,
            inputs_copied=copied,
            base_commit=manifest["base_commit"],
        ),
    )


def new_env(assets, p):
    return PendulumInstrument(
        assets,
        RotationConfig(
            **dict(p["environment"], terminate_on_goal=False, rotation_enabled=False)
        ),
    )


def compatible_actor(output, seed, assets):
    root = output / "inputs/output/rotation_pilot_20261111_v1/global" / f"A_seed{seed}"
    checkpoint = root / "checkpoint_000036.pt"
    net, state, p = load_actor(checkpoint)
    config = json.loads((root / "configuration.json").read_text())
    env = new_env(assets, p)
    obs, critic = env.reset(1.0, 65000)
    assert obs.shape == (44,) and critic.shape == (74,)
    assert net.actor[0].weight.shape[1] == 44
    assert net.actor[-1].weight.shape[0] == 12
    assert tuple(p["ppo"]["hidden_dims"]) == (256, 256, 128)
    assert (
        env.cfg.action_scale == 2
        and env.cfg.kp == 60
        and env.cfg.kd == 3
        and env.cfg.target_filter == 1
    )
    assert (
        config["actor_observation_dim"] == 44 and config["critic_observation_dim"] == 74
    )
    return net, p, checkpoint


def collect_case(net, p, assets, seed, output, tag):
    env = new_env(assets, p)
    obs, _ = env.reset(1.0, seed)
    env.record_rotation_trace = True
    snapshot = None
    prefix = []
    while True:
        act = action(net, obs)
        prefix.append(
            dict(
                observations=obs.copy(),
                actions=act.copy(),
                qpos=env.d.qpos.copy(),
                qvel=env.d.qvel.copy(),
                target=env.target.copy(),
                time=float(env.d.time),
            )
        )
        (obs, _), _, done, info = env.step(act)
        s = env.rotation_sample
        if (
            snapshot is None
            and env.tracker.confirmed
            and s["pitch"] >= np.deg2rad(30)
            and s["pitch"] <= np.deg2rad(35)
            and s["FL_load"] > 0.05 * env.weight
            and s["FR_load"] > 0.05 * env.weight
            and not info["nonfoot_collision"]
            and not info["self_collision"]
            and info["max_speed_ratio"] <= 1.001
            and info["joint_limit_violation"] < 0.001
        ):
            snapshot = copy.deepcopy(env.state())
            snapshot["values"]["rotation_trace"] = []
            snapshot["values"]["record_rotation_trace"] = False
        if done:
            break
    frame = pd.DataFrame(env.rotation_trace)
    frame.to_csv(output / f"{tag}_trajectory.csv", index=False)
    if snapshot is not None:
        torch.save(snapshot, output / f"{tag}_start.pt")
    np.savez_compressed(
        output / f"{tag}_prefix.npz",
        **{k: np.array([r[k] for r in prefix]) for k in prefix[0]},
    )
    return frame, snapshot, info, env.mass


def rollout(env, net, world, candidate, ref, record=False):
    env.restore(copy.deepcopy(world))
    env.rotation_trace = []
    env.record_rotation_trace = True
    obs, _ = env.observations()
    initial_target = env.target.copy()
    start = float(env.d.time)
    controls = []
    action_history = []
    for step in range(35):
        mode = candidate.get("mode", "residual")
        if mode == "hold":
            act, _ = inverse_target_action(env, initial_target)
        elif mode == "legacy":
            act = candidate["actions"][min(step, len(candidate["actions"]) - 1)]
        else:
            knots = np.asarray(candidate["knots"])
            u = step * (len(knots) - 1) / 34
            left = int(u)
            right = min(left + 1, len(knots) - 1)
            delta = (1 - (u - left)) * knots[left] + (u - left) * knots[right]
            act = np.clip(action(net, obs) + delta, -1, 1)
        action_history.append(act.copy())
        if record:
            controls.append(
                dict(
                    observations=obs.copy(),
                    actions=act.copy(),
                    qpos=env.d.qpos.copy(),
                    qvel=env.d.qvel.copy(),
                    target=env.target.copy(),
                    time=float(env.d.time) - start,
                )
            )
        (obs, _), _, done, info = env.step(act)
        if done:
            break
    stats = score_trajectory(env.rotation_trace, env.weight)
    trace = env.rotation_trace
    time = np.array([r["time"] - start for r in trace])
    body = np.array([r["pitch"] for r in trace])
    speed = np.array([r["pitch_velocity"] for r in trace])
    goal = np.interp(time, ref.time, ref.pitch)
    goal_v = np.interp(time, ref.time, ref.pitch_velocity)
    error = float(np.mean((body - goal) ** 2))
    v_error = float(np.mean((speed - goal_v) ** 2))
    changing = (
        0.0
        if len(action_history) < 2
        else float(np.mean(np.diff(np.array(action_history), axis=0) ** 2))
    )
    # Opposing moments below 40 degrees resist useful progress; braking near
    # the target is allowed. No unconditional reward for reaction magnitude.
    wrong_moment = float(
        np.mean(
            [
                max(0.0, -r["front_moment_y"])
                if r["pitch"] < np.deg2rad(40)
                else max(0.0, r["front_moment_y"])
                if r["pitch"] > np.deg2rad(50)
                else 0.0
                for r in trace
            ]
        )
    )
    material_x = np.array(
        [r["com_x"] - (r["FL_position_x"] + r["FR_position_x"]) / 2 for r in trace]
    )
    material_z = np.array([r["com_z"] for r in trace])
    theta_error = float(
        np.mean(
            (np.arctan2(material_x, material_z) - np.interp(time, ref.time, ref.theta))
            ** 2
        )
    )
    length_error = float(
        np.mean(
            (
                (
                    np.hypot(material_x, material_z)
                    - np.interp(time, ref.time, ref.length)
                )
                / 0.1
            )
            ** 2
        )
    )
    score = (
        4 * min(stats["hold_seconds"] / 0.3, 1)
        + 2 * float(stats["reached_target_low_speed"])
        - 5 * error
        - 0.15 * v_error
        - 3 * float(stats["unsafe"])
        - 20 * stats["slip_distance"]
        - 3 * stats["contact_gap_seconds"]
        - 0.02 * wrong_moment
        - 0.05 * changing
        - 0.5 * theta_error
        - 0.5 * length_error
    )
    stats.update(
        score=score,
        tracking_error=error,
        velocity_error=v_error,
        com_angle_error=theta_error,
        length_error=length_error,
        steps=env.steps - world["values"]["steps"],
        duration=float(env.d.time) - start,
    )
    return stats, copy.deepcopy(trace) if record else None, controls


def model_seed(env, net, world, ref):
    """Finite full-simulator response samples initialize JOINT action residuals."""
    values = []
    used = 0
    for j in range(-1, 12):
        env.restore(copy.deepcopy(world))
        obs, _ = env.observations()
        for _ in range(3):
            act = action(net, obs)
            if j >= 0:
                act[j] = np.clip(act[j] + 0.03, -1, 1)
            (obs, _), _, done, _ = env.step(act)
            used += 1
            if done:
                break
        r = env.rotation_sample
        dx = r["com_x"] - (r["FL_position_x"] + r["FR_position_x"]) / 2
        dz = r["com_z"]
        vx = r["com_velocity_x"] - (r["FL_velocity_x"] + r["FR_velocity_x"]) / 2
        vz = r["com_velocity_z"]
        omega = (dz * vx - dx * vz) / (dx**2 + dz**2)
        radial = (dx * vx + dz * vz) / np.hypot(dx, dz)
        values.append(np.array([r["pitch_velocity"], omega, radial]))
    response = np.stack([(v - values[0]) / 0.03 for v in values[1:]], axis=1)
    desired = ref.iloc[3][["pitch_velocity", "omega", "length_rate"]].to_numpy(
        dtype=float
    )
    delta = response.T @ np.linalg.solve(
        response @ response.T + np.eye(3) * 0.02, desired - values[0]
    )
    delta = np.clip(delta, -0.10, 0.10)
    knots = np.stack([delta, delta * 0.6, np.zeros(12), -delta * 0.25, -delta * 0.25])
    return knots, used, response


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path("output/rotation_learning_molab_bundle.zip")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-cases", type=Path)
    parser.add_argument("--population", type=int, default=20, choices=range(8, 33))
    parser.add_argument("--generations", type=int, default=4, choices=range(1, 7))
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    verify_bundle(a.bundle, a.output)
    assets = Assets(a.output / "assets")
    gate = ReferenceGate()
    models = {}
    all_errors = []
    all_forecasts = []
    all_one_step = []
    case_rows = []
    actors = {}
    cases = {}
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    write_json(
        a.output / "protocol.json",
        dict(
            experiment="pendulum_guided_rotation_v1",
            branch="experiment/pendulum_guided_rotation",
            source_commit=source_commit,
            source_is_uncommitted_snapshot=True,
            backend="cpu_mujoco",
            training_seeds=[42, 43],
            identification_resets=TRAIN_RESETS,
            model_test_resets=MODEL_TEST_RESETS,
            robustness_resets=GATE_RESETS,
            gate=gate.dictionary(),
            population=a.population,
            generations=a.generations,
            horizon_seconds=0.7,
            action_knots=5,
            action_dimensions=12,
            ppo_interactions=0,
            wandb="disabled",
            angle="COM theta from upward vertical; +y torque; body pitch distinct",
            source_hashes={
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in [
                    __file__,
                    "experiments/morphology/learning/pendulum_reference.py",
                    "experiments/morphology/learning/pendulum_models.py",
                ]
            },
        ),
    )
    for training_seed in (42, 43):
        net, p, checkpoint = compatible_actor(a.output, training_seed, assets)
        actors[training_seed] = (net, p, checkpoint)
        training = []
        held = []
        cases[training_seed] = {}
        for reset in TRAIN_RESETS + MODEL_TEST_RESETS + GATE_RESETS:
            tag = f"seed{training_seed}_reset{reset}"
            prior = a.reuse_cases / f"{tag}_trajectory.csv" if a.reuse_cases else None
            if prior and prior.exists():
                old_meta = json.loads(
                    (a.reuse_cases / "bundle_verification.json").read_text()
                )
                assert old_meta["bundle_sha256"] == BUNDLE_SHA
                frame = pd.read_csv(prior)
                world = torch.load(
                    a.reuse_cases / f"{tag}_start.pt",
                    weights_only=False,
                    map_location="cpu",
                )
                assert (
                    world["values"]["seed"] == reset and world["values"]["scale"] == 1.0
                )
                prefix = np.load(a.reuse_cases / f"{tag}_prefix.npz")
                info = dict(
                    takeoff=bool(frame.takeoff.any()),
                    episode_steps=len(prefix["time"]),
                    max_pitch=float(frame[~frame.nonfoot_collision].pitch.max()),
                )
                mass = float(assets.models[1.0].body_mass.sum())
                for suffix in ("trajectory.csv", "start.pt", "prefix.npz"):
                    shutil.copy2(
                        a.reuse_cases / f"{tag}_{suffix}", a.output / f"{tag}_{suffix}"
                    )
            else:
                frame, world, info, mass = collect_case(
                    net, p, assets, reset, a.output, tag
                )
            enriched = enrich(frame, mass)
            enriched.to_csv(a.output / f"{tag}_model.csv", index=False)
            case_rows.append(
                dict(
                    training_seed=training_seed,
                    reset=reset,
                    takeoff=info["takeoff"],
                    start_valid=world is not None,
                    steps=info["episode_steps"],
                    peak_pitch=info["max_pitch"],
                )
            )
            if reset in TRAIN_RESETS:
                training.append(enriched)
            elif reset in MODEL_TEST_RESETS:
                held.append((tag, enriched))
            cases[training_seed][reset] = world
        model = identify(training, mass)
        models[training_seed] = model
        errors, forecasts, one_step = validate(held, model)
        all_errors.extend(errors)
        all_forecasts.extend(forecasts)
        all_one_step.extend(one_step)
        write_json(a.output / f"model_seed{training_seed}.json", model)
    save_csv(a.output / "reset_cases.csv", case_rows)
    save_csv(a.output / "model_errors.csv", all_errors)
    save_csv(a.output / "model_forecasts.csv", all_forecasts)
    save_csv(a.output / "model_one_step.csv", all_one_step)
    history = []
    robust = []
    screen = []
    selected = []
    search_steps = 0
    mapping_steps = 0
    validation_steps = 0
    for training_seed in (42, 43):
        net, p, _ = actors[training_seed]
        env = new_env(assets, p)
        model = models[training_seed]
        if any(cases[training_seed][r] is None for r in TRAIN_RESETS):
            raise ValueError("No valid start in training reset family")
        world = cases[training_seed][TRAIN_RESETS[0]]
        env.restore(copy.deepcopy(world))
        # Use actual start state and material-pivot kinematics from full trace.
        d = pd.read_csv(
            a.output / f"seed{training_seed}_reset{TRAIN_RESETS[0]}_model.csv"
        )
        initial = d.iloc[
            np.argmin(abs(d.time - world["values"]["_last_model_time"]))
        ].to_dict()
        ref = reference(initial, model)
        ref.to_csv(a.output / f"reference_seed{training_seed}.csv", index=False)
        knots, used, response = model_seed(env, net, world, ref)
        mapping_steps += used
        np.savez_compressed(
            a.output / f"initial_action_mapping_seed{training_seed}.npz",
            response=response,
            knots=knots,
        )
        rng = np.random.default_rng(training_seed + 72000)
        mean = knots
        std = np.full_like(mean, 0.07)
        ranked_all = []
        for generation in range(a.generations):
            population = [
                dict(mode="residual", knots=k)
                for k in rng.normal(mean, std, size=(a.population, 5, 12))
            ]
            population[0] = dict(mode="residual", knots=mean.copy())
            if generation == 0:
                population[1] = dict(mode="residual", knots=np.zeros((5, 12)))
                population[2] = dict(mode="hold")
            ranked = []
            for candidate_id, candidate in enumerate(population):
                outcomes = []
                for reset in TRAIN_RESETS:
                    result, _, _ = rollout(
                        env,
                        net,
                        cases[training_seed][reset],
                        candidate,
                        ref,
                        record=False,
                    )
                    search_steps += result["steps"]
                    outcomes.append(result["score"])
                    history.append(
                        dict(
                            training_seed=training_seed,
                            generation=generation,
                            candidate=candidate_id,
                            reset=reset,
                            mode=candidate["mode"],
                            **result,
                        )
                    )
                score = float(np.mean(outcomes) - 0.5 * np.std(outcomes))
                ranked_all.append(
                    (score, copy.deepcopy(candidate), generation, candidate_id)
                )
                if candidate["mode"] == "residual":
                    ranked.append((score, candidate["knots"]))
            elite = np.stack(
                [
                    k
                    for _, k in sorted(ranked, key=lambda r: r[0], reverse=True)[
                        : max(3, a.population // 4)
                    ]
                ]
            )
            mean = 0.3 * mean + 0.7 * elite.mean(0)
            std = np.maximum(0.015, 0.3 * std + 0.7 * elite.std(0))
            save_csv(a.output / "search_cases.csv", history)
            print(
                f"seed={training_seed} generation={generation} "
                f"best score={max(r[0] for r in ranked_all):.4f}",
                flush=True,
            )
        top = sorted(ranked_all, key=lambda r: r[0], reverse=True)[:3]
        # Validation cases were not used for identification, model selection,
        # finite-response initialization, or CEM ranking. No refitting follows.
        for rank, (score, candidate, generation, candidate_id) in enumerate(top):
            tag = f"seed{training_seed}_candidate{rank}"
            serial = dict(
                mode=candidate["mode"],
                knots=candidate.get("knots", np.zeros((5, 12))).tolist(),
            )
            write_json(a.output / f"{tag}.json", serial)
            for reset in GATE_RESETS:
                world = cases[training_seed][reset]
                if world is None:
                    robust.append(
                        dict(
                            training_seed=training_seed,
                            rank=rank,
                            reset=reset,
                            success=False,
                            reason="invalid_start",
                        )
                    )
                    continue
                result, trace, controls = rollout(env, net, world, candidate, ref, True)
                validation_steps += result["steps"]
                robust.append(
                    dict(training_seed=training_seed, rank=rank, reset=reset, **result)
                )
                if rank == 0 or result["success"]:
                    save_csv(a.output / f"{tag}_reset{reset}_trajectory.csv", trace)
                    np.savez_compressed(
                        a.output / f"{tag}_reset{reset}_states.npz",
                        **{k: np.array([r[k] for r in controls]) for k in controls[0]},
                    )
            # The original actor receives exactly the same start/reset family.
        for reset in GATE_RESETS:
            world = cases[training_seed][reset]
            if world is None:
                continue
            result, trace, _ = rollout(
                env,
                net,
                world,
                dict(mode="residual", knots=np.zeros((5, 12))),
                ref,
                True,
            )
            validation_steps += result["steps"]
            screen.append(dict(training_seed=training_seed, reset=reset, **result))
            save_csv(
                a.output / f"baseline_seed{training_seed}_reset{reset}_trajectory.csv",
                trace,
            )
        # Selection uses TRAINING score only; rank 0 is the sole confirmatory
        # robustness test. Ranks 1/2 are disclosed exploratory diagnostics.
        selected.append(dict(training_seed=training_seed, rank=0))
    save_csv(a.output / "robustness_cases.csv", robust)
    save_csv(a.output / "baseline_cases.csv", screen)
    passed_by_seed = {
        str(seed): sum(
            r["success"]
            for r in robust
            if r["training_seed"] == seed and r["rank"] == 0
        )
        / len(GATE_RESETS)
        for seed in (42, 43)
    }
    passed = all(v >= gate.required_fraction for v in passed_by_seed.values())
    provenance = {
        str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.output.glob("seed*_candidate0*.npz")
    }
    write_json(
        a.output / "trajectory_gate.json",
        dict(
            passed=passed,
            required=gate.dictionary(),
            validation_resets=GATE_RESETS,
            rank=0,
            success_by_seed=passed_by_seed,
            selected=selected,
            demonstration_hashes=provenance,
            ppo_allowed=passed,
            ppo_interactions=0,
            note="Both seeds must pass rank 0; runners-up are exploratory only.",
        ),
    )
    write_json(
        a.output / "completion.json",
        dict(
            phase="offline_complete",
            go=passed,
            ppo_interactions=0,
            search_interactions=search_steps,
            action_mapping_interactions=mapping_steps,
            validation_interactions=validation_steps,
            case_collection_interactions=sum(r["steps"] for r in case_rows),
            stop_reason="reference_gate_failed"
            if not passed
            else "reference_gate_passed; BC/PPO requires gated runner",
        ),
    )
    print(json.dumps(passed_by_seed), flush=True)


if __name__ == "__main__":
    main()
