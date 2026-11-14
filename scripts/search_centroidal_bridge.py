"""Serial resumable CEM over smooth policy residuals, using full native states."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions, render_episode
from experiments.unified_go2.core import Environment, load, digest
from experiments.retained_go2.actor import Student
from experiments.centroidal_bridge.physics import audit
import numpy as np
import torch


def rollout(e, net, start, params, steps, detailed=False):
    e.restore(copy.deepcopy(start))
    obs, _ = e.observations()
    records, states, commands, audits = (
        [],
        [e.dstate() if hasattr(e, "dstate") else start["integration"]],
        [],
        [],
    )
    peak = -180.0
    for k in range(steps):
        u = actions(net, obs[None])[0]
        # Symmetric joint groups with temporal knots for yield and recontact.
        offset = np.array(
            [
                np.interp(k * 0.02, [0, 0.12, 0.26, 0.44, 0.64], params[:, j])
                for j in range(4)
            ]
        )
        for i, name in enumerate(e.names):
            if "thigh" in name or "calf" in name:
                group = (0 if name.startswith(("FL_", "FR_")) else 2) + int(
                    "calf" in name
                )
                u[i] += offset[group]
        u = u.clip(-1, 1)
        commands.append(u.copy())
        (obs, _), _, done, info = e.step(u)
        s = e.momentum_sample.copy()
        if e._momentum_prefix_safe:
            peak = max(peak, float(np.rad2deg(s["pitch"])))
        records.append(dict(s, capture_distance=float(e.capture_distance)))
        if detailed:
            states.append(e.state()["integration"])
            audits.append(audit(e))
        if done:
            break
    safe = bool(e._momentum_prefix_safe)
    score = (
        min(peak, 80)
        + 0.8 * min(max(s["pitch_velocity"], -4), 2)
        - 0.15 * min(e.capture_distance, 20)
    )
    score -= 10 * (not safe)
    score -= max(0, peak - 85)
    score -= 2 * max(0, abs(s["pitch_velocity"]) - 2)
    return dict(
        score=float(score),
        peak=peak,
        safe=safe,
        terminal_pitch=float(np.rad2deg(s["pitch"])),
        terminal_velocity=s["pitch_velocity"],
        capture_distance=float(e.capture_distance),
        collision=bool(s["collision"]),
        takeoff=bool(e.tracker.confirmed),
    ), (records, states, commands, audits, e.state())


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/retained_go2_distillation_v3/student.pt"),
    )
    p.add_argument(
        "--preparation", type=Path, default=Path("output/retained_go2_preparation_v3")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--seed", type=int, default=96000)
    p.add_argument("--onset", type=float, default=0.50)
    p.add_argument("--steps", type=int, default=42)
    p.add_argument("--render", action="store_true")
    p.add_argument("--parameters", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(a.preparation, a.output)
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    net = Student()
    net.load_state_dict(load(a.checkpoint)["network"])
    obs, _ = e.reset(1.0, a.seed)
    standing = copy.deepcopy(e.state())
    prefix_actions = []
    for _ in range(round(a.onset / 0.02)):
        u = actions(net, obs[None])[0]
        prefix_actions.append(u)
        (obs, _), _, done, _ = e.step(u)
        if done:
            raise RuntimeError("Prefix terminated")
    start = copy.deepcopy(e.state())
    torch.save(
        dict(
            standing=standing,
            start=start,
            prefix_actions=prefix_actions,
            checkpoint_sha=digest(a.checkpoint),
            args=vars(a),
        ),
        a.output / "starts.pt",
    )
    rng = np.random.default_rng(734)
    mean, std = np.zeros((5, 4)), np.ones((5, 4)) * 0.25
    best_params = mean.copy()
    baseline, _ = rollout(e, net, start, mean, a.steps)
    best = baseline
    history = []
    resume = a.output / "progress.pt"
    begin = 0
    if resume.exists():
        saved = load(resume)
        mean, std, best_params, best, history, begin = [
            saved[k]
            for k in ["mean", "std", "best_params", "best", "history", "iteration"]
        ]
        rng.bit_generator.state = saved["rng"]
    if a.parameters:
        best_params = load(a.parameters)["params"]
        best, _ = rollout(e, net, start, best_params, a.steps)
    print("baseline", baseline, flush=True)
    for iteration in range(begin, a.iterations):
        population = np.clip(rng.normal(mean, std, (a.population, 5, 4)), -0.8, 0.8)
        population[0] = best_params
        results = []
        for index, params in enumerate(population):
            result, _ = rollout(e, net, start, params, a.steps)
            results.append(result)
            history.append(dict(iteration=iteration, candidate=index, **result))
            if result["score"] > best["score"]:
                best, best_params = result, params.copy()
                torch.save(
                    dict(result=best, params=best_params),
                    a.output / f"incumbent_{iteration}_{index}.pt",
                )
        elite = np.argsort([r["score"] for r in results])[-max(3, a.population // 5) :]
        mean = 0.25 * mean + 0.75 * population[elite].mean(0)
        std = np.maximum(0.06, 0.25 * std + 0.75 * population[elite].std(0))
        torch.save(
            dict(
                mean=mean,
                std=std,
                best_params=best_params,
                best=best,
                history=history,
                iteration=iteration + 1,
                rng=rng.bit_generator.state,
            ),
            resume,
        )
        print("iteration", iteration, "best", best, flush=True)
    for label, params in [("baseline", np.zeros((5, 4))), ("best", best_params)]:
        result, (records, states, commands, audits, final) = rollout(
            e, net, start, params, a.steps, True
        )
        torch.save(
            dict(
                initial=start,
                final=final,
                actions=np.array(commands),
                parameters=params,
                records=records,
                audits=audits,
            ),
            a.output / (label + ".pt"),
        )
        np.savez_compressed(
            a.output / (label + "_states.npz"), integration=np.array(states)
        )
        (a.output / (label + ".json")).write_text(
            json.dumps(
                dict(result=result, records=records, audits=audits),
                indent=2,
                default=lambda x: x.item() if hasattr(x, "item") else str(x),
            )
        )
        # Independent fresh environment replay of exact saved commands.
        check = Environment(assets, copy.deepcopy(cfg))
        check.capture_centers = data["centers"]
        check.restore(copy.deepcopy(start))
        for u in commands:
            check.step(u)
        result["replay_max_state_error"] = float(
            np.max(np.abs(check.state()["integration"] - final["integration"]))
        )
        if a.render:
            render_episode(e, states, a.output / (label + ".gif"), label)
        if label == "baseline":
            baseline = result
        else:
            best = result
    (a.output / "comparison.json").write_text(
        json.dumps(
            dict(
                baseline=baseline,
                best=best,
                checkpoint=str(a.checkpoint),
                checkpoint_sha=digest(a.checkpoint),
                history=history,
            ),
            indent=2,
        )
    )
    print("comparison", baseline, best, flush=True)


if __name__ == "__main__":
    main()
