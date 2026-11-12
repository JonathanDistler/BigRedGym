"""Focused dynamic pass-through search; preserves all prior environments/results."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.takeoff import inverse_target_action
from experiments.morphology.task import write_json
from scripts.run_pendulum_reference import new_env
from scripts.rotation_learning_tools import action, load_actor
from scripts.morphology_feasibility import save_csv


TRAIN = [61000, 61002, 61003]
TEST = list(range(67200, 67210))


def energy(row, mass):
    dx = row["com_x"] - (row["FL_position_x"] + row["FR_position_x"]) / 2
    z = row["com_z"]
    vx = row["com_velocity_x"] - (row["FL_velocity_x"] + row["FR_velocity_x"]) / 2
    vz = row["com_velocity_z"]
    length = np.hypot(dx, z)
    omega = (z * vx - dx * vz) / length**2
    radial = (dx * vx + z * vz) / length
    inertia = row["inertia_com_y"] + mass * length**2
    barrier = mass * 9.81 * max(0.0, length - z)
    kinetic = 0.5 * inertia * max(0.0, omega) ** 2
    return dict(
        com_angle=float(np.arctan2(dx, z)),
        com_omega=float(omega),
        length=float(length),
        length_rate=float(radial),
        rotational_energy=float(kinetic),
        remaining_coast_barrier=float(barrier),
        energy_deficit=float(max(0.0, barrier - kinetic)),
        required_com_omega=float(np.sqrt(2 * barrier / inertia)),
        shortening_acceleration=float(-2 * radial / length * omega),
    )


def assess(trace, mass):
    if not trace:
        raise ValueError("Empty trajectory")
    t = np.array([r["time"] for r in trace])
    pitch = np.array([r["pitch"] for r in trace])
    omega = np.array([r["pitch_velocity"] for r in trace])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.002
    native = np.array(
        [
            not (r["nonfoot_collision"] or r["self_collision"])
            and r["max_speed_ratio"] <= 1.001
            and r["joint_violation"] < 0.001
            and r["torque_ratio"] <= 1.00001
            for r in trace
        ]
    )
    both = np.array(
        [
            r["FL_load"] > 0.05 * mass * 9.81 and r["FR_load"] > 0.05 * mass * 9.81
            for r in trace
        ]
    )
    rear = np.array([r["rear_force_z"] < 0.04 * mass * 9.81 for r in trace])
    prefix = np.logical_and.accumulate(native)
    gap = longest = hold = recovery = 0.0
    for r, supported, p, w, valid in zip(trace, both, pitch, omega, native):
        gap = 0.0 if supported else gap + dt
        longest = max(longest, gap)
        stable = (
            valid
            and supported
            and abs(p - np.deg2rad(60)) <= np.deg2rad(5)
            and abs(w) <= 0.5
            and np.linalg.norm(
                [r["com_velocity_x"], r["com_velocity_y"], r["com_velocity_z"]]
            )
            < 0.75
        )
        hold = hold + dt if stable else 0.0
        recovery = max(recovery, hold)
    slips = np.array(
        [
            max(
                np.hypot(r["FL_velocity_x"], r["FL_velocity_y"]),
                np.hypot(r["FR_velocity_x"], r["FR_velocity_y"]),
            )
            for r in trace
        ]
    )
    slip = float(np.sum(slips * both) * dt)
    drift = max(r["front_drift"] for r in trace)
    crossings = {}
    for degrees in (45, 60):
        hit = np.flatnonzero(
            (pitch >= np.deg2rad(degrees)) & (omega > 0.0) & both & rear & prefix
        )
        crossings[degrees] = int(hit[0]) if len(hit) else None
    k = crossings[45]
    cross = k is not None
    after = 0.0 if k is None else float(t[-1] - t[k])
    progress = (
        0.0
        if k is None
        else float(np.rad2deg(pitch[prefix & (np.arange(len(t)) >= k)].max()) - 45)
    )
    full_valid = bool(
        native.all() and longest <= 0.02000001 and slip <= 0.02 and drift <= 0.04
    )
    improved = bool(cross and full_valid and after >= 0.15 and progress >= 5.0)
    result = dict(
        cross45=bool(cross),
        cross60=bool(crossings[60] is not None),
        improved=improved,
        full_valid=full_valid,
        peak_pitch=float(pitch.max()),
        valid_prefix_peak=float(pitch[prefix & both & rear].max())
        if (prefix & both & rear).any()
        else 0.0,
        omega45=float(omega[k]) if k is not None else None,
        post_cross_survival=after,
        progress_after45_degrees=progress,
        useful_velocity45=bool(k is not None and omega[k] >= 0.25),
        terminal_recovery_60=bool(full_valid and recovery >= 0.15),
        terminal_recovery_seconds=recovery,
        contact_gap=longest,
        slip_distance=slip,
        front_drift=float(drift),
        nonfoot_collision=any(r["nonfoot_collision"] for r in trace),
        self_collision=any(r["self_collision"] for r in trace),
        max_speed_ratio=max(r["max_speed_ratio"] for r in trace),
        max_torque_ratio=max(r["torque_ratio"] for r in trace),
        max_joint_violation=max(r["joint_violation"] for r in trace),
        terminal_pitch=float(pitch[-1]),
        terminal_omega=float(omega[-1]),
    )
    if k is not None:
        result.update(
            {f"at45_{key}": value for key, value in energy(trace[k], mass).items()}
        )
    return result


def execute(env, net, world, candidate, horizon=40, record=False):
    env.restore(copy.deepcopy(world))
    env.rotation_trace = []
    env.record_rotation_trace = True
    obs, _ = env.observations()
    target = env.target.copy()
    controls = []
    for step in range(horizon):
        if candidate["mode"] == "hold":
            act, _ = inverse_target_action(env, target)
        elif candidate["mode"] == "actions":
            if step < len(candidate["actions"]):
                act = np.array(candidate["actions"][step])
            else:
                act, _ = inverse_target_action(env, target)
        else:
            knot = 0 if step < 9 else 1 if step < 17 else 2 if step < 25 else 3
            groups = np.asarray(candidate["knots"])[knot]
            delta = np.r_[np.tile(groups[:3], 2), np.tile(groups[3:], 2)]
            act = np.clip(action(net, obs) + delta, -1, 1)
        controls.append(
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
        if candidate["mode"] == "actions":
            target = env.target.copy()
        if done:
            break
    stats = assess(env.rotation_trace, env.mass)
    valid_peak = stats["valid_prefix_peak"]
    p45 = stats["cross45"]
    p60 = stats["cross60"]
    progress = 4 * min(valid_peak / np.deg2rad(60), 1.2) + float(p45) + 2 * float(p60)
    penalty = (
        3 * float(not stats["full_valid"])
        + 20 * stats["slip_distance"]
        + 2 * stats["contact_gap"]
    )
    recover = 4 * float(stats["terminal_recovery_60"]) + 0.5 * float(
        stats["full_valid"]
    )
    movement = (
        float(np.mean(np.diff(np.array([r["actions"] for r in controls]), axis=0) ** 2))
        if len(controls) > 1
        else 0.0
    )
    stats.update(
        final_score=progress + recover - penalty - 0.03 * movement,
        exploration_score=progress
        + recover
        - 0.5 * float(not stats["full_valid"])
        - 5 * stats["slip_distance"]
        - 0.03 * movement,
        steps=len(controls),
        illegal_bodies=env.row["illegal_bodies"],
    )
    return stats, copy.deepcopy(env.rotation_trace) if record else None, controls


def start(net, p, assets, reset):
    env = new_env(assets, p)
    obs, _ = env.reset(1.0, reset)
    steps = 0
    while True:
        (obs, _), _, done, info = env.step(action(net, obs))
        steps += 1
        if env.tracker.confirmed:
            world = copy.deepcopy(env.state())
            world["values"]["rotation_trace"] = []
            return world, steps
        if done:
            raise ValueError(f"No valid takeoff for reset {reset}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population", type=int, default=24, choices=range(12, 33))
    parser.add_argument("--generations", type=int, default=4, choices=range(1, 7))
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    prior = Path("output/rotation_search_20261111_v2")
    diagnosis = Path("output/rotation_diagnosis_20261111_v1")
    historical = []
    prior_steps = 0
    # Re-evaluate six actual old optimized action sequences at their own states.
    for context in json.loads((diagnosis / "contexts.json").read_text()):
        saved = torch.load(context["path"], weights_only=False, map_location="cpu")
        net, _, p = load_actor(Path(saved["checkpoint"]))
        env = new_env(assets, p)
        planned = np.load(prior / f"{context['tag']}_search_best.npz")["actions"]
        stats, trace, _ = execute(
            env,
            net,
            saved["world"],
            dict(mode="actions", actions=planned),
            len(planned),
            True,
        )
        prior_steps += stats["steps"]
        historical.append(
            dict(source=context["tag"], kind="replayed_old_optimized_actions", **stats)
        )
        save_csv(a.output / f"old_{context['tag']}_trajectory.csv", trace)
    # Re-score saved full native-dynamics traces; no diagnostic replay needed.
    old_root = Path("output/pendulum_guided_rotation_v3")
    for path in sorted(
        list(old_root.glob("seed*_candidate0_reset*_trajectory.csv"))
        + list(old_root.glob("challenge_seed*_rank*_reset*_trajectory.csv"))
    ):
        mass = json.loads((old_root / "model_seed42.json").read_text())["mass"]
        stats = assess(pd.read_csv(path).to_dict("records"), mass)
        historical.append(dict(source=str(path), kind="rescored_saved_trace", **stats))
    save_csv(a.output / "historical_reassessment.csv", historical)
    write_json(
        a.output / "protocol.json",
        dict(
            name="dynamic_pass_through_v1",
            train_resets=TRAIN,
            validation_resets=TEST,
            policy_seeds=[42, 43],
            horizon_seconds=0.8,
            population=a.population,
            generations=a.generations,
            action_knots=4,
            grouped_dimensions=24,
            pass45=(
                "Supported upward crossing; rear unloaded; native valid prefix. "
                "No intermediate hold."
            ),
            improved=(
                "Full valid horizon; >=50 degree peak; "
                ">=.15 s safe survival after crossing 45."
            ),
            recovery=(
                "Separate 60 +/-5 degrees, abs(omega)<=.5, "
                "COM speed<.75, front supported .15 s."
            ),
            physical_limits=(
                "Native efforts, speed ratio<=1.001, joint violation<.001; "
                "no nonfoot/self collisions."
            ),
            retained_motion_limits=(
                "Gaps<=.02 s; foot-center motion<=.02 m; front drift<=.04 m."
            ),
            early_search=(
                "Early generations explore with smaller failure penalty; "
                "acceptance always uses full physical limits."
            ),
            ppo_interactions=0,
        ),
    )
    rows = []
    validation = []
    account = dict(
        prior_replay_steps=prior_steps,
        collection_steps=0,
        search_steps=0,
        validation_steps=0,
        ppo_interactions=0,
    )
    input_hashes = {}
    for seed in (42, 43):
        checkpoint = Path(
            f"output/multimorph_bc_ppo_continuation_20261111_v2/A_seed{seed}/checkpoint_000030.pt"
        )
        input_hashes[str(checkpoint)] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        net, _, p = load_actor(checkpoint)
        env = new_env(assets, p)
        assert (
            net.actor[0].weight.shape[1] == 44 and net.actor[-1].weight.shape[0] == 12
        )
        assert (
            env.cfg.kp,
            env.cfg.kd,
            env.cfg.action_scale,
            env.cfg.target_filter,
        ) == (60.0, 3.0, 2.0, 1.0)
        worlds = {}
        for reset in TRAIN + TEST:
            world, count = start(net, p, assets, reset)
            worlds[reset] = world
            account["collection_steps"] += count
            torch.save(world, a.output / f"seed{seed}_reset{reset}_start.pt")
        warm = np.load(prior / f"A_seed{seed}_scale1.00_reset61000_search_best.npz")[
            "residual"
        ]
        mean = np.r_[warm, warm[-1:]]
        std = np.full_like(mean, 0.07)
        rng = np.random.default_rng(74000 + seed)
        bank = []
        for generation in range(a.generations):
            candidates = [
                dict(mode="residual", knots=k)
                for k in rng.normal(mean, std, (a.population, 4, 6))
            ]
            candidates[0] = dict(mode="residual", knots=mean.copy())
            candidates[1] = dict(mode="residual", knots=np.zeros((4, 6)))
            candidates[2] = dict(mode="hold")
            ranked = []
            for index, candidate in enumerate(candidates):
                outcomes = []
                final = []
                for reset in TRAIN:
                    result, _, _ = execute(env, net, worlds[reset], candidate)
                    account["search_steps"] += result["steps"]
                    rows.append(
                        dict(
                            seed=seed,
                            generation=generation,
                            candidate=index,
                            reset=reset,
                            **result,
                        )
                    )
                    outcomes.append(
                        result["exploration_score"]
                        if generation < 2
                        else result["final_score"]
                    )
                    final.append(result["final_score"])
                score = float(np.mean(outcomes) - 0.5 * np.std(outcomes))
                bank.append(
                    (
                        float(np.mean(final) - 0.5 * np.std(final)),
                        copy.deepcopy(candidate),
                    )
                )
                if candidate["mode"] == "residual":
                    ranked.append((score, candidate["knots"]))
            elite = np.stack(
                [c for _, c in sorted(ranked, key=lambda r: r[0], reverse=True)[:6]]
            )
            mean = 0.3 * mean + 0.7 * elite.mean(0)
            std = np.maximum(0.015, 0.3 * std + 0.7 * elite.std(0))
            save_csv(a.output / "search_cases.csv", rows)
            print(
                f"seed={seed} generation={generation} "
                f"best_final_score={max(b[0] for b in bank):.3f}",
                flush=True,
            )
        chosen = sorted(bank, key=lambda r: r[0], reverse=True)[:3]
        for rank, (_, candidate) in enumerate(chosen):
            write_json(
                a.output / f"seed{seed}_candidate{rank}.json",
                dict(
                    mode=candidate["mode"],
                    knots=candidate.get("knots", np.zeros((4, 6))).tolist(),
                ),
            )
            for reset in TEST:
                result, trace, controls = execute(
                    env, net, worlds[reset], candidate, record=True
                )
                account["validation_steps"] += result["steps"]
                validation.append(dict(seed=seed, rank=rank, reset=reset, **result))
                if rank == 0 or result["improved"]:
                    tag = f"seed{seed}_rank{rank}_reset{reset}"
                    save_csv(a.output / f"{tag}_trajectory.csv", trace)
                    np.savez_compressed(
                        a.output / f"{tag}_states.npz",
                        **{
                            key: np.array([r[key] for r in controls])
                            for key in controls[0]
                        },
                    )
        save_csv(a.output / "validation_cases.csv", validation)
    rates = {
        str(seed): float(
            np.mean(
                [
                    r["improved"]
                    for r in validation
                    if r["seed"] == seed and r["rank"] == 0
                ]
            )
        )
        for seed in (42, 43)
    }
    any_valid = any(r["improved"] for r in validation)
    robust = all(v >= 0.8 for v in rates.values())
    write_json(
        a.output / "decision.json",
        dict(
            any_valid_improvement=any_valid,
            robust_improvement=robust,
            rank0_success_by_seed=rates,
            ppo_pilot_prepared=any_valid,
            ppo_launched=False,
            molab_go=robust,
            reason="robust_dynamic_progression"
            if robust
            else "no_robust_collision_free_dynamic_progression",
        ),
    )
    if any_valid:
        write_json(
            a.output / "pilot_preparation.json",
            dict(
                updates=10,
                maximum_updates=20,
                training_seeds=[42, 43],
                source_checkpoints=input_hashes,
                reuse="Existing RotationRun/migrate; no environment rebuild",
                bc_prefix_weight=2.0,
                reset_optimizer=False,
                launch_enabled=False,
                required_before_launch="Robust reset and terminal recovery audit",
                demonstrations=[
                    f"seed{r['seed']}_rank{r['rank']}_reset{r['reset']}_states.npz"
                    for r in validation
                    if r["improved"]
                ],
            ),
        )
    write_json(a.output / "interaction_accounting.json", account)
    write_json(a.output / "source_checkpoint_hashes.json", input_hashes)


if __name__ == "__main__":
    main()
