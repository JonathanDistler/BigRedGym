"""One bounded front-support intervention; captured prefix stays unchanged."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions, render_episode
from experiments.unified_go2.core import Environment, load
from experiments.balance_feedback.actor import BalanceActor
import numpy as np
import torch


def envelope(t, duration):
    return max(0.0, min(1.0, t / 0.10, (duration - t) / 0.15))


def gates(e):
    s = e.physics_trace[-1]
    r = e.row
    f = e.d.geom_xpos[e.foot_geoms[:2], :2]
    d = f[1] - f[0]
    com = e.d.subtree_com[e.base]
    u = np.clip((com[:2] - f[0]) @ d / max(d @ d, 1e-10), 0, 1)
    return dict(
        safe=bool(
            e._momentum_prefix_safe
            and not r["illegal_contacts"]
            and not r["self_contacts"]
        ),
        rear_air=bool(
            r["rear_clearance"] > 0.1 and (r["loads"][2:] < 0.02 * e.weight).all()
        ),
        front_support=bool(r["front_support"]),
        orientation=bool(s["orientation_error"] < np.deg2rad(10)),
        body_height=bool(abs(e.d.qpos[2] - e.cfg.target_base_height) < 0.06),
        com_height=bool(abs(s["com_height"] - e.cfg.target_height) < 0.06),
        com_speed=bool(s["com_speed"] < 0.25),
        angular_speed=bool(s["body_angular_speed"] < 0.5),
        support_error=bool(np.linalg.norm(com[:2] - f[0] - u * d) < 0.04),
        slip=bool(s["front_slip_speed"] < 0.05),
        calf=bool(s["calf_clearance"] > 0),
        head=bool(s["head_clearance"] > 0),
        force=bool(s["front_force_weight_ratio"] < 4),
    )


def run(e, balance, start, reference, params, detailed=False):
    e.restore(copy.deepcopy(start))
    obs, _ = e.observations()
    frames = [start["integration"]]
    cmds = []
    logs = []
    peak = 0.0
    quality = []
    original = e.measure
    if detailed:

        def measure():
            row = original()
            if e.physics_trace:
                s = e.physics_trace[-1]
                g = gates(e)
                logs.append(
                    dict(
                        s,
                        time=float(e.d.time),
                        base_height=float(e.d.qpos[2]),
                        base_vz=float(e.d.qvel[2]),
                        gate=g,
                        strict_current=float(e.multi_hold),
                        front_q=e.d.qpos[e.qids[:6]].tolist(),
                        front_target=e.target[:6].tolist(),
                        front_tau=e.d.qfrc_applied[e.vids[:6]].tolist(),
                        front_load=e.row["loads"][:2].tolist(),
                    )
                )
            return row

        e.measure = measure
    try:
        for k in range(350):
            t = k * 0.02
            u = (
                reference[k].copy()
                if t < params[3] and k < len(reference)
                else actions(balance, obs[None], "balance")[0]
            )
            amp = envelope(t, params[2])
            u[[1, 4]] += params[0] * amp
            u[[2, 5]] += params[1] * amp
            u = u.clip(-1, 1)
            cmds.append(u)
            (obs, _), _, done, info = e.step(u)
            s = e.momentum_sample
            if e._momentum_prefix_safe:
                peak = max(peak, np.rad2deg(s["pitch"]))
            quality.append(
                -20 * abs(e.d.qpos[2] - 0.57)
                - 20 * abs(s["com_height"] - e.cfg.target_height)
                - 2 * abs(s["com_velocity_z"])
                - s["com_speed"]
                - s["body_angular_speed"]
            )
            if detailed:
                frames.append(e.state()["integration"])
            if done or e.ever_hold_success:
                break
    finally:
        if detailed:
            del e.measure
    score = (
        1000 * e.multi_best_hold
        + np.mean(quality[-min(25, len(quality)) :])
        + 0.15 * len(cmds) * 0.02
    )
    result = dict(
        score=float(score),
        strict_seconds=float(e.multi_best_hold),
        success=bool(e.ever_hold_success),
        legal=bool(e._momentum_prefix_safe),
        termination=info["termination"],
        time=float(e.d.time),
        peak_pitch=float(peak),
        min_body_height=min((r["base_height"] for r in logs), default=None),
        min_head_clearance=min((r["head_clearance"] for r in logs), default=None),
        min_calf_clearance=min((r["calf_clearance"] for r in logs), default=None),
        terminal_com_vz=s["com_velocity_z"],
    )
    return result, dict(
        actions=np.array(cmds),
        states=np.array(frames),
        logs=logs,
        final=copy.deepcopy(e.state()),
        params=params,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    seed = load("output/centroidal_capture_reference_20261114/capture.pt")
    e.restore(copy.deepcopy(seed["standing"]))
    n = 66
    for u in seed["standing_actions"][:n]:
        e.step(u)
    start = copy.deepcopy(e.state())
    reference = seed["standing_actions"][n:]
    balance = BalanceActor()
    balance.load_state_dict(
        load("output/retained_go2_preparation_v3/balance.pt")["network"]
    )
    zero = np.array([0.0, 0.0, 0.5, 0.0])
    base, _ = run(e, balance, start, reference, zero)
    mean = np.array([-0.025, 0.08, 0.55, 0.04])
    std = np.array([0.045, 0.06, 0.2, 0.08])
    low = np.array([-0.15, -0.03, 0.2, 0])
    high = np.array([0.15, 0.30, 1.2, 0.4])
    rng = np.random.default_rng(1114)
    best = base
    bp = zero.copy()
    history = []
    for generation in range(2):
        pop = np.clip(rng.normal(mean, std, (20, 4)), low, high)
        pop[0] = bp
        pop[1] = mean
        results = []
        for i, params in enumerate(pop):
            r, _ = run(e, balance, start, reference, params)
            results.append(r)
            history.append(
                dict(generation=generation, index=i, params=params.tolist(), **r)
            )
            if r["score"] > best["score"]:
                best, bp = r, params.copy()
            if r["success"]:
                break
        elites = np.argsort([r["score"] for r in results])[-4:]
        mean = pop[elites].mean(0)
        std = np.maximum([0.01, 0.01, 0.04, 0.02], pop[elites].std(0))
        torch.save(
            dict(
                mean=mean,
                std=std,
                best=best,
                params=bp,
                history=history,
                rng=rng.bit_generator.state,
                standing=seed["standing"],
                initial=start,
            ),
            a.output / "progress.pt",
        )
        print("generation", generation, best, flush=True)
        if best["success"]:
            break
    comparison = {}
    for label, params in [("baseline", zero), ("best", bp)]:
        r, saved = run(e, balance, start, reference, params, True)
        commands = np.array(list(seed["standing_actions"][:n]) + list(saved["actions"]))
        # Independent model and uninterrupted standing replay.
        from experiments.morphology.learning.environment import Assets

        check = Environment(Assets(a.output / ("cold_" + label)), copy.deepcopy(cfg))
        check.capture_centers = data["centers"]
        check.restore(copy.deepcopy(seed["standing"]))
        frames = [check.state()["integration"]]
        for u in commands:
            check.step(u)
            frames.append(check.state()["integration"])
        r["cold_error"] = float(
            np.max(abs(check.state()["integration"] - saved["final"]["integration"]))
        )
        r["max_pitch_from_standing"] = float(np.rad2deg(check.peak_safe_pitch))
        saved.update(
            standing=seed["standing"],
            standing_actions=commands,
            standing_states=np.array(frames),
            result=r,
        )
        torch.save(saved, a.output / (label + ".pt"))
        comparison[label] = r
        if label == "best" and r["strict_seconds"] > base["strict_seconds"]:
            render_episode(
                check, frames, a.output / "best.gif", "Front support stabilization"
            )
    (a.output / "comparison.json").write_text(
        json.dumps(dict(candidates=len(history), comparison=comparison), indent=2)
    )
    print(comparison, flush=True)


if __name__ == "__main__":
    main()
