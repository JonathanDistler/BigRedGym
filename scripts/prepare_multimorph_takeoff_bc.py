"""Adapt a preserved kick reference offline, certify prefixes, fit one shared actor."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import (
    TakeoffConfig,
    TakeoffEnvironment,
    inverse_target_action,
)
from experiments.morphology.task import write_json
from scripts.validate_takeoff_learning import crouch_pose
from scripts.morphology_feasibility import save_csv
from gym.envs.go2.kickup_reference import blend

SCALES = (0.9, 0.95, 1.0)
REFERENCE = Path("output/handoff_reference_search/candidate_reference.json")


def adapted_action(env, crouch, p, source):
    """Geometry IK and separately optimized thrust, with physical pitch feedback."""
    t = env.steps * env.cfg.dt
    q0 = env.nominal
    qc = crouch[env.qids]
    a, _ = blend(t, source["settle"], p["crouch_seconds"])
    start = source["settle"] + p["crouch_seconds"]
    b, _ = blend(t, start, p["duration"])
    thrust = qc.copy()
    angle = np.deg2rad(p["pitch_degrees"])
    for j, name in enumerate(env.names):
        if name.startswith(("FL", "FR")):
            thrust[j] = (
                0
                if "hip" in name
                else -angle - p["knee"] / 2
                if "thigh" in name
                else p["knee"]
            )
        else:
            thrust[j] = (
                0
                if "hip" in name
                else p["rear_hip"]
                if "thigh" in name
                else p["rear_knee"]
            )
    desired = q0 + a * (qc - q0) + b * (thrust - qc)
    if t >= start:
        c, _ = blend(t, start + p["duration"], 0.5)
        final = thrust.copy()
        for j, name in enumerate(env.names):
            if name.startswith(("FL", "FR")):
                final[j] = (
                    0
                    if "hip" in name
                    else -np.deg2rad(80) + 0.8
                    if "thigh" in name
                    else -1.6
                )
            else:
                final[j] = (
                    0
                    if "hip" in name
                    else source["rear_tuck_hip"]
                    if "thigh" in name
                    else source["rear_tuck_knee"]
                )
        desired += c * (final - thrust)
        pitch = angle * b + c * np.deg2rad(55)
        correction = (
            -0.6 * (pitch - env.row["pitch"]) + 0.07 * env.row["pitch_velocity"]
        )
        for j, name in enumerate(env.names):
            if name.startswith(("FL", "FR")) and "thigh" in name:
                desired[j] += correction
    desired = np.clip(desired, *env.bounds.T)
    desired = env.target + np.clip(
        desired - env.target,
        -0.7 * env.speed * env.cfg.dt,
        0.7 * env.speed * env.cfg.dt,
    )
    return inverse_target_action(env, desired)


def rollout(assets, scale, seed, p, source, capture=False):
    env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
    obs, _ = env.reset(scale, seed)
    crouch = crouch_pose(assets, scale)
    env.trace_enabled = capture
    frames = []
    errors = []
    for _ in range(70):
        action, error = adapted_action(env, crouch, p, source)
        errors.append(error)
        before = dict(
            observations=obs.copy(),
            actions=action.copy(),
            qpos=env.d.qpos.copy(),
            qvel=env.d.qvel.copy(),
            target=env.target.copy(),
            contacts=env.row["loads"].copy(),
            com=env.row["com"].copy(),
            time=env.steps * env.cfg.dt,
        )
        (obs, _), _, done, info = env.step(action)
        if capture:
            frames.append(
                dict(
                    **before,
                    torque=env.d.qfrc_applied[env.vids].copy(),
                    commanded_target=env.target.copy(),
                )
            )
        # End the demonstration at a useful, certified kick, before later falls.
        if (
            info["takeoff"]
            and info["genuine_flight_pitch"] >= np.deg2rad(25)
            and info["genuine_airborne_seconds"] >= 0.06
        ):
            break
        if done:
            break
    accepted = bool(
        info["takeoff"]
        and info["genuine_flight_pitch"] >= np.deg2rad(20)
        and info["genuine_airborne_seconds"] >= 0.04
        and not info["nonfoot_collision"]
        and not info["self_collision"]
        and info["max_speed_ratio"] <= 1.001
        and info["joint_limit_violation"] < 0.001
        and max(e["action_projection_fraction"] for e in errors) == 0
        and max(e["target_error"] for e in errors) < 1e-6
    )
    return dict(**info, seed=seed, accepted=accepted), frames, env.physics_trace


def demonstrations(assets, out):
    source = json.loads(REFERENCE.read_text())
    search = []
    chosen = {}
    all_frames = []
    physics = []
    episodes = []
    for scale in SCALES:
        candidates = []
        # Reference thrust is the first warm start. A second reference-derived
        # family retunes thrust authority; IK/contact constraints are unchanged.
        for duration, knee, hip, rear_knee in itertools.product(
            (source["extend"], 0.12, 0.24),
            (-2.2, -2.1),
            (source["rear_push_hip"], 0.30),
            (source["rear_push_knee"],),
        ):
            candidates.append(
                dict(
                    duration=duration,
                    knee=knee,
                    rear_hip=hip,
                    rear_knee=rear_knee,
                    crouch_seconds=0.25,
                    pitch_degrees=25,
                )
            )
        best = None
        for index, p in enumerate(candidates):
            results = [
                rollout(assets, scale, seed, p, source)[0]
                for seed in range(50000, 50004)
            ]
            rate = float(np.mean([r["accepted"] for r in results]))
            score = (rate, float(np.mean([r["genuine_flight_pitch"] for r in results])))
            search.append(
                dict(scale=scale, candidate=index, rate=rate, rotation=score[1], **p)
            )
            if best is None or score > best[0]:
                best = (score, p)
            if rate == 1.0:
                break
        chosen[str(scale)] = best[1]
        print("TEACHER SEARCH", scale, best, flush=True)
        for seed in range(51000, 51016):
            result, frames, traces = rollout(assets, scale, seed, best[1], source, True)
            episodes.append(result)
            if result["accepted"]:
                for frame in frames:
                    all_frames.append(dict(**frame, seed=seed, scale=scale))
                physics.extend(dict(**r, seed=seed) for r in traces)
        save_csv(out / "teacher_search.csv", search)
        save_csv(out / "teacher_episodes.csv", episodes)
        rate = np.mean([r["accepted"] for r in episodes if r["scale"] == scale])
        print(f"TEACHER heldout scale={scale} accepted={rate:.3f}", flush=True)
        if rate < 0.875:
            write_json(
                out / "teacher_gate.json",
                dict(
                    passed=False,
                    failed_scale=scale,
                    parameters=chosen,
                    reason=(
                        "Fewer than 14/16 independently certified collision-free "
                        "kick prefixes; BC not attempted"
                    ),
                ),
            )
            return False
    data = {k: np.stack([f[k] for f in all_frames]) for k in all_frames[0]}
    np.savez_compressed(out / "bc_dataset.npz", **data)
    save_csv(out / "teacher_physics.csv", physics)
    write_json(
        out / "teacher_gate.json",
        dict(
            passed=True,
            parameters=chosen,
            reference_path=str(REFERENCE),
            reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            source_reference=source,
            adaptation=(
                "Per-morphology standing/crouch IK with unchanged physical "
                "mass/COM/contact model; independent thrust/timing search and "
                "front pitch feedback. Targets differ across geometry. "
                "Collision-free certified partial-maneuver prefixes only."
            ),
            generation_seeds=list(range(51000, 51016)),
            search_seeds=list(range(50000, 50004)),
        ),
    )
    return True


def evaluate_actor(net, assets, seeds, std=0.05):
    results = []
    for scale in SCALES:
        for stochastic in (False, True):
            for seed in seeds:
                rng = torch.Generator().manual_seed(seed + 1000000)
                env = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
                obs, _ = env.reset(scale, seed)
                for _ in range(100):
                    with torch.no_grad():
                        mean = net.actor(torch.tensor(obs))
                        action = (
                            (
                                mean
                                + (
                                    std * torch.randn(12, generator=rng)
                                    if stochastic
                                    else 0
                                )
                            )
                            .tanh()
                            .numpy()
                        )
                    (obs, _), _, done, info = env.step(action)
                    if done:
                        break
                results.append(
                    dict(
                        **info,
                        evaluation_seed=seed,
                        stochastic=stochastic,
                        latent_std=std,
                    )
                )
    return results


def fit(
    assets,
    out,
    updates,
    initial_actor=None,
    latent_std=0.05,
    selection_start=52000,
    final_start=53000,
):
    data = np.load(out / "bc_dataset.npz")
    write_json(
        out / "morphology_metadata.json",
        dict(
            actor_dim=44,
            actions=12,
            frame_alignment=(
                "Observations/qpos/qvel/contacts/com/target are pre-command; "
                "commanded_target and torque are post-step. Actions are teacher "
                "labels; perturbed data retain actual commanded_target separately."
            ),
            morphologies=[
                dict(
                    scale=s,
                    mass=float(assets.models[s].body_mass.sum()),
                    nominal=assets.nominals[s].tolist(),
                    initial_com=data["com"][
                        np.flatnonzero(data["scale"] == s)[0]
                    ].tolist(),
                    frames=int((data["scale"] == s).sum()),
                )
                for s in SCALES
            ],
        ),
    )
    train = data["seed"] < 51012
    x = torch.tensor(data["observations"], dtype=torch.float32)
    y = torch.tensor(data["actions"], dtype=torch.float32)
    z = torch.atanh(y.clamp(-0.999, 0.999))
    ids = torch.tensor(np.flatnonzero(train))
    torch.manual_seed(42)
    net = ActorCritic(44, 74)
    # The original full-scale fit is a common warm start, never condition-specific.
    original = Path(initial_actor or "output/takeoff_bc_investigation_v1/bc_actor.pt")
    net.actor.load_state_dict(
        torch.load(original, weights_only=False, map_location="cpu")["actor"]
    )
    optimizer = torch.optim.Adam(net.actor.parameters(), lr=3e-4)
    losses = []
    evaluations = []
    best = None
    stale = 0
    for step in range(updates + 1):
        if step % 250 == 0:
            with torch.no_grad():
                pred = net.actor(x).tanh()
                loss = float((pred[~train] - y[~train]).square().mean())
            measured = evaluate_actor(
                net, assets, range(selection_start, selection_start + 4), std=latent_std
            )
            evaluations.extend(dict(**r, update=step) for r in measured)
            rates = [
                np.mean(
                    [
                        r["takeoff"]
                        for r in measured
                        if r["scale"] == s and not r["stochastic"]
                    ]
                )
                for s in SCALES
            ]
            stochastic_rates = [
                np.mean(
                    [
                        r["takeoff"]
                        for r in measured
                        if r["scale"] == s and r["stochastic"]
                    ]
                )
                for s in SCALES
            ]
            score = (min(rates), np.mean(rates), np.mean(stochastic_rates), -loss)
            losses.append(
                dict(
                    update=step,
                    validation_mse=loss,
                    train_mse=float((pred[train] - y[train]).square().mean()),
                    minimum_deterministic_rate=min(rates),
                    mean_stochastic_rate=np.mean(stochastic_rates),
                )
            )
            print(
                "BC",
                step,
                "rates",
                rates,
                "stochastic",
                stochastic_rates,
                "mse",
                loss,
                flush=True,
            )
            if best is None or score > best[0]:
                best = (score, copy.deepcopy(net.actor.state_dict()), step)
                stale = 0
            else:
                stale += 1
            save_csv(out / "bc_loss.csv", losses)
            save_csv(out / "bc_validation.csv", evaluations)
            if step >= 1000 and (
                stale >= 4
                or (min(rates) == 1 and min(stochastic_rates) >= 0.75 and loss < 1e-5)
            ):
                break
        if step == updates:
            break
        batch = ids[torch.randint(len(ids), (256,))]
        loss = (net.actor(x[batch]) - z[batch]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.actor.parameters(), 1)
        optimizer.step()
    net.actor.load_state_dict(best[1])
    torch.save(
        dict(
            actor=best[1],
            actor_dim=44,
            critic_dim=74,
            action_configuration=asdict(TakeoffConfig()),
            dataset_sha256=hashlib.sha256(
                (out / "bc_dataset.npz").read_bytes()
            ).hexdigest(),
            source="multimorph_adapted_reference",
            selected_update=best[2],
            warm_start_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        ),
        out / "bc_actor.pt",
    )
    final = evaluate_actor(
        net, assets, range(final_start, final_start + 16), std=latent_std
    )
    save_csv(out / "bc_evaluation.csv", final)
    rates = {
        str(s): dict(
            deterministic=float(
                np.mean(
                    [
                        r["takeoff"]
                        for r in final
                        if r["scale"] == s and not r["stochastic"]
                    ]
                )
            ),
            stochastic=float(
                np.mean(
                    [r["takeoff"] for r in final if r["scale"] == s and r["stochastic"]]
                )
            ),
        )
        for s in SCALES
    }
    passed = all(
        r["deterministic"] >= 0.875 and r["stochastic"] >= 0.75 for r in rates.values()
    )
    write_json(
        out / "bc_gate.json",
        dict(
            passed=passed,
            rates=rates,
            selected_update=best[2],
            completed_updates=step,
            evaluation_seeds=list(range(final_start, final_start + 16)),
            latent_std=latent_std,
            actor_sha256=hashlib.sha256((out / "bc_actor.pt").read_bytes()).hexdigest(),
            criterion=(
                "At least 14/16 deterministic and 12/16 sampled takeoffs at "
                "every morphology at recorded latent std; independent of "
                "selection seeds. Partial kick endpoint, not sustained handstand."
            ),
        ),
    )
    print("BC FINAL", rates, "passed", passed, flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--fit-existing", action="store_true")
    a = p.parse_args()
    if not a.fit_existing:
        a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    if not a.fit_existing:
        shutil.copyfile(__file__, a.output / Path(__file__).name)
        if not demonstrations(assets, a.output):
            return
    elif not json.loads((a.output / "teacher_gate.json").read_text())["passed"]:
        raise ValueError("Teacher gate failed")
    fit(assets, a.output, a.updates)


if __name__ == "__main__":
    main()
