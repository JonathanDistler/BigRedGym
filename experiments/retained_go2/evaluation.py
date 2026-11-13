"""Matched endpoint/bridge suites and explicit retention gate, nominal dynamics."""

import copy
import numpy as np
import torch, mujoco
from experiments.unified_go2.core import Environment
from scripts.unified_go2 import actions, render_episode


def evaluate(
    net, cfg, bank, assets, centers, controller="student", render_out=None, quick=False
):
    e = Environment(assets, copy.deepcopy(cfg))
    e.capture_centers = centers
    specs = [("standing", s) for s in [96000, 96001, 96002, 96003]]
    specs += [("balance", s) for s in bank["evaluation"]["balance"][:4]]
    specs += [("bridge", s) for s in bank["evaluation"]["bridge"][:2]]
    rows = []
    for index, (kind, start) in enumerate(specs):
        e.cfg.seconds = 12.0 if kind == "standing" else 6.2
        obs, _ = (
            e.reset(1.0, start)
            if kind == "standing"
            else e.restore_start(
                start, "near_vertical" if kind == "balance" else "partial"
            )
        )
        gates = []
        states = []

        def record():
            state = np.empty(
                mujoco.mj_stateSize(e.m, mujoco.mjtState.mjSTATE_INTEGRATION)
            )
            mujoco.mj_getState(e.m, e.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            states.append(state)

        want = render_out is not None and index in [0, 4]
        if want:
            record()
        while True:
            with torch.no_grad():
                gates.append(
                    float(net.actor.gate(torch.tensor(obs[None]))[0])
                    if hasattr(getattr(net, "actor", None), "gate")
                    else 0.0
                )
            (obs, _), _, done, info = e.step(actions(net, obs[None], controller)[0])
            if want:
                record()
            if done:
                break
        row = {
            k: info[k]
            for k in [
                "certified_standing_takeoff",
                "peak_collision_free_pitch",
                "handstand_seconds",
                "episode_seconds",
                "termination",
                "front_support_fraction",
                "nonfoot_collision",
                "self_collision",
                "genuine_pushes",
                "counted_front_recontacts",
                "seconds_above45",
                "seconds_above60",
                "seconds_above75",
                "max_torque_ratio",
                "max_speed_ratio",
            ]
        }
        row.update(
            distribution=kind,
            index=index,
            seed=start if kind == "standing" else None,
            controller=controller,
            success=bool(e.ever_hold_success),
            standing_success=bool(kind == "standing" and e.ever_hold_success),
            gate_min=min(gates),
            gate_max=max(gates),
            seconds_above85=sum(
                0.002
                for r in e.physics_trace
                if r["prefix_safe"]
                and r["pitch"] >= np.deg2rad(85)
                and r["rear_clearance"] > 0.1
            ),
            capture_distance_min=min(
                (r.get("capture_distance", float("inf")) for r in e.trace),
                default=float("inf"),
            ),
        )
        row["entered_empirical_capture_neighborhood"] = row["capture_distance_min"] < 1
        rows.append(row)
        print(
            controller,
            kind,
            index,
            "takeoff",
            row["certified_standing_takeoff"],
            "pitch",
            round(np.rad2deg(row["peak_collision_free_pitch"]), 2),
            "hold",
            round(row["handstand_seconds"], 3),
            "gate",
            round(row["gate_max"], 3),
            flush=True,
        )
        if want:
            from scripts.morphology_feasibility import save_csv

            render_out.mkdir(parents=True, exist_ok=True)
            name = f"{controller}_{kind}_{index}"
            save_csv(render_out / (name + "_trace.csv"), e.trace)
            save_csv(render_out / (name + "_physics.csv"), e.physics_trace)
            np.savez_compressed(render_out / (name + "_states.npz"), integration=states)
            render_episode(
                e, np.array(states), render_out / (name + ".gif"), controller
            )
    return rows


def retention(rows, baseline):
    stand = [r for r in rows if r["distribution"] == "standing"]
    bal = [r for r in rows if r["distribution"] == "balance"]
    ref = [r for r in baseline if r["distribution"] == "standing"]
    certified = sum(r["certified_standing_takeoff"] for r in stand)
    success = sum(r["success"] for r in bal)
    pitch = np.mean([r["peak_collision_free_pitch"] for r in stand])
    pitch_ref = np.mean([r["peak_collision_free_pitch"] for r in ref])
    passed = (
        certified == len(stand)
        and success >= 3
        and pitch >= pitch_ref - np.deg2rad(1.0)
    )
    return dict(
        passed=bool(passed),
        standing_takeoffs=int(certified),
        standing_n=len(stand),
        balance_successes=int(success),
        balance_n=len(bal),
        mean_standing_peak_degrees=float(np.rad2deg(pitch)),
        baseline_mean_peak_degrees=float(np.rad2deg(pitch_ref)),
        balance_holds=[r["handstand_seconds"] for r in bal],
        standing_failure_causes=[r["termination"] for r in stand],
    )
