"""Bounded serial capture CEM, followed by unchanged neural balance feedback."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions, render_episode
from scripts.search_centroidal_bridge import rollout as seed_rollout
from experiments.unified_go2.core import Environment, load, digest
from experiments.retained_go2.actor import Student
from experiments.balance_feedback.actor import BalanceActor
from experiments.centroidal_capture.core import (
    metrics,
    capture_quality,
    correction,
    modify,
)
import numpy as np
import torch


def prepare(e, student, starts, seed_params, onset):
    _, (_, _, commands, _, _) = seed_rollout(
        e, student, starts["start"], seed_params, 32, False
    )
    all_commands = np.array(list(starts["prefix_actions"]) + list(commands))
    e.restore(copy.deepcopy(starts["standing"]))
    n = round(onset / 0.02)
    for u in all_commands[:n]:
        _, _, done, _ = e.step(u)
        if done:
            raise RuntimeError("Invalid seed prefix")
    return copy.deepcopy(e.state()), all_commands[:n], all_commands[n:]


def rollout(e, balance, start, reference, params, steps=60, detailed=False):
    e.restore(copy.deepcopy(start))
    obs, _ = e.observations()
    rows = []
    states = [start["integration"]]
    commands = []
    best_quality = -100.0
    hold = best_hold = 0.0
    peak = 0.0
    contact_time = 0.0
    for k in range(steps):
        u = (
            reference[min(k, len(reference) - 1)].copy()
            if k < e.capture_plan_steps
            else actions(balance, obs[None], "balance")[0]
        )
        if k < e.capture_plan_steps:
            u = modify(u, e.names, correction(params, k * 0.02))
        commands.append(u)
        (obs, _), _, done, info = e.step(u)
        x = metrics(e, detailed)
        if x["safe"]:
            peak = max(peak, np.rad2deg(x["pitch"]))
            best_quality = max(best_quality, capture_quality(x))
            contact_time += 0.02 * float(
                x["front_loaded"] and x["pitch"] > np.deg2rad(60)
            )
        stable = (
            x["safe"]
            and x["both_front"]
            and x["orientation_error"] < 0.20
            and abs(x["pitch_velocity"]) < 1
            and np.linalg.norm(x["com_velocity"]) < 0.6
            and x["rear_clearance"] > 0.08
        )
        hold = hold + 0.02 if stable else 0.0
        best_hold = max(best_hold, hold)
        rows.append(x)
        if detailed:
            states.append(e.state()["integration"])
        if done:
            break
    # Do not reward bigger pitch. A safe capture/hold outranks an aerial near miss.
    score = best_quality + 15 * best_hold + min(contact_time, 1) * 2
    score += 100 * e.multi_best_hold
    score -= 2 * float(not x["safe"])
    score += min(len(rows) * 0.02, 1) * 0.4
    result = dict(
        score=float(score),
        peak=float(peak),
        best_quality=best_quality,
        stable_capture_seconds=best_hold,
        front_contact_above60_seconds=contact_time,
        legal=bool(x["safe"]),
        collision=x["collision"],
        terminal_pitch=float(np.rad2deg(x["pitch"])),
        terminal_pitch_rate=x["pitch_velocity"],
        terminal_gap=x["gaps"][:2],
        terminal_com_velocity=x["com_velocity"],
        balance_hold_seconds=float(e.multi_best_hold),
        five_second_success=bool(e.ever_hold_success),
        certified_takeoff=bool(e.tracker.confirmed),
        termination=info["termination"],
        seconds=len(rows) * 0.02,
    )
    return result, dict(
        rows=rows,
        states=np.array(states),
        actions=np.array(commands),
        final=copy.deepcopy(e.state()),
        initial=start,
        params=params,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-trajectory", choices=["61", "83"], required=True)
    p.add_argument("--onset", type=float, default=0.62)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--render", action="store_true")
    p.add_argument("--handoff", type=float, default=1.32)
    p.add_argument("--warm-start", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source = Path("output/centroidal_bridge_capture_v2")
    starts = load(source / "starts.pt")
    cfg, data, bank, assets = setup(starts["args"]["preparation"], a.output)
    cfg.seconds = 12.0
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    student = Student()
    student.load_state_dict(load(starts["args"]["checkpoint"])["network"])
    balance = BalanceActor()
    balance.load_state_dict(
        load(starts["args"]["preparation"] / "balance.pt")["network"]
    )
    seed_params = load(
        source
        / ("incumbent_2_8.pt" if a.seed_trajectory == "61" else "incumbent_4_10.pt")
    )["params"]
    start, prefix, reference = prepare(e, student, starts, seed_params, a.onset)
    protocol = dict(
        seed=a.seed_trajectory,
        onset=a.onset,
        steps=a.steps,
        population=a.population,
        checkpoint_sha=digest(starts["args"]["checkpoint"]),
        balance_sha=digest(starts["args"]["preparation"] / "balance.pt"),
        version=4,
        handoff=a.handoff,
    )
    torch.save(
        dict(
            standing=starts["standing"],
            initial=start,
            prefix=prefix,
            reference=reference,
            protocol=protocol,
        ),
        a.output / "inputs.pt",
    )
    mean = np.zeros((5, 6))
    mean[:, [0, 2]] = np.array([0.0, 0.0, -0.65, -1.25, -1.25])[:, None]
    mean[:, [1, 3]] = np.array([0.0, 0.0, 0.35, 0.65, 0.65])[:, None]
    std = np.ones_like(mean) * 0.18
    if a.warm_start:
        mean = load(a.warm_start)["params"].copy()
    mean[:, 2:4] = mean[:, :2] = 0.5 * (mean[:, :2] + mean[:, 2:4])
    e.capture_plan_steps = round((a.handoff - a.onset) / 0.02)
    std *= 0.6
    best_params = mean.copy()
    rng = np.random.default_rng(109)
    baseline, _ = rollout(e, balance, start, reference, best_params, a.steps)
    best = baseline
    history = []
    begin = 0
    if (a.output / "progress.pt").exists():
        v = load(a.output / "progress.pt")
        assert v["protocol"] == protocol
        mean, std, best_params, best, history, begin = [
            v[k] for k in ["mean", "std", "best_params", "best", "history", "iteration"]
        ]
        rng.bit_generator.state = v["rng"]
    print("seed", a.seed_trajectory, "baseline", baseline, flush=True)
    for iteration in range(begin, a.iterations):
        population = np.clip(rng.normal(mean, std, (a.population, 5, 6)), -1.5, 1.5)
        population[:, :, 2:4] = population[:, :, :2] + rng.normal(
            0, 0.015, (a.population, 5, 2)
        )
        population[0] = best_params
        population[1] = mean
        results = []
        for i, params in enumerate(population):
            result, _ = rollout(e, balance, start, reference, params, a.steps)
            results.append(result)
            history.append(dict(iteration=iteration, index=i, **result))
            if result["score"] > best["score"]:
                best, best_params = result, params.copy()
                torch.save(
                    dict(params=params, result=result),
                    a.output / f"incumbent_{iteration}_{i}.pt",
                )
        elite = np.argsort([r["score"] for r in results])[-max(3, a.population // 5) :]
        mean = 0.25 * mean + 0.75 * population[elite].mean(0)
        std = np.maximum(0.045, 0.25 * std + 0.75 * population[elite].std(0))
        torch.save(
            dict(
                protocol=protocol,
                mean=mean,
                std=std,
                best_params=best_params,
                best=best,
                history=history,
                iteration=iteration + 1,
                rng=rng.bit_generator.state,
            ),
            a.output / "progress.pt",
        )
        print("iteration", iteration, best, flush=True)
    comparison = {}
    for label, params in [("baseline", np.zeros((5, 6))), ("best", best_params)]:
        result, saved = rollout(e, balance, start, reference, params, a.steps, True)
        # Cold model reload, no resets during the episode, from saved normal standing.
        fresh_assets = type(assets)(a.output / ("replay_assets_" + label))
        check = Environment(fresh_assets, copy.deepcopy(cfg))
        check.capture_centers = data["centers"]
        check.restore(copy.deepcopy(starts["standing"]))
        all_commands = np.array(list(prefix) + list(saved["actions"]))
        standing_states = [check.state()["integration"]]
        for u in all_commands:
            check.step(u)
            standing_states.append(check.state()["integration"])
        result["standing_replay_error"] = float(
            np.max(abs(check.state()["integration"] - saved["final"]["integration"]))
        )
        saved.update(
            standing=starts["standing"],
            standing_actions=all_commands,
            standing_states=np.array(standing_states),
            result=result,
            protocol=protocol,
        )
        torch.save(saved, a.output / (label + ".pt"))
        comparison[label] = result
        if a.render:
            render_episode(
                check,
                standing_states,
                a.output / (label + ".gif"),
                f"{a.seed_trajectory} {label} standing capture",
            )
    (a.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print("comparison", comparison, flush=True)


if __name__ == "__main__":
    main()
