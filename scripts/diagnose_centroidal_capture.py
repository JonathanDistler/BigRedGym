"""500 Hz capture diagnosis and actual-integration-step wrench audit."""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions
from scripts.search_centroidal_bridge import rollout as seed_rollout
from experiments.unified_go2.core import Environment, load
from experiments.retained_go2.actor import Student
from experiments.balance_feedback.actor import BalanceActor
from experiments.centroidal_capture.core import metrics
from experiments.centroidal_bridge.physics import audit
import mujoco
import numpy as np
import torch


def used_wrench(e):
    # Called AFTER mj_step but BEFORE mj_forward: contacts/frames still describe
    # the configuration at which this step's constraint forces were solved.
    m, d = e.m, e.d
    com = d.subtree_com[e.base].copy()
    moment = np.zeros(3)
    force = np.zeros(3)
    for i, c in enumerate(d.contact):
        b = m.geom_bodyid[[c.geom1, c.geom2]]
        inside = (m.body_rootid[b] == e.base) & (b != 0)
        if inside[0] == inside[1]:
            continue
        w = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, w)
        f, tau = (
            (1 if inside[1] else -1) * c.frame.reshape(3, 3).T @ w.reshape(2, 3).T
        ).T
        force += f
        moment += np.cross(c.pos - com, f) + tau
    ids = np.flatnonzero(m.body_rootid == e.base)
    moment += np.sum(
        d.xfrc_applied[ids, 3:] + np.cross(d.xipos[ids] - com, d.xfrc_applied[ids, :3]),
        axis=0,
    )
    return moment, force


def run(e, student, balance, starts, params, controller="balance", factor=1):
    _, (_, _, u, _, _) = seed_rollout(e, student, starts["start"], params, 32)
    commands = list(starts["prefix_actions"]) + list(u)
    e.restore(copy.deepcopy(starts["standing"]))
    original_dt = e.m.opt.timestep
    e.m.opt.timestep = original_dt / factor
    rows = []
    stepdata = []
    states = []
    orig_step = mujoco.mj_step
    orig_measure = e.measure
    pending = {}

    def step(m, d):
        mujoco.mj_subtreeVel(m, d)
        pending["h"] = d.subtree_angmom[e.base].copy()
        pending["v"] = d.qvel.copy()
        pending["time"] = float(d.time)
        orig_step(m, d)
        moment, force = used_wrench(e)
        pending["moment"] = moment
        pending["force"] = force
        pending["qacc"] = d.qacc.copy()
        pending["v_after"] = d.qvel.copy()
        pending["solver_iter"] = d.solver_niter.copy()

    def measure():
        row = orig_measure()
        if "moment" in pending:
            x = metrics(e, True)
            x["link_audit"] = audit(e)
            x["illegal_pairs"] = [
                (
                    e.m.geom(c.geom1).name,
                    e.m.geom(c.geom2).name,
                    e.m.body(int(e.m.geom_bodyid[c.geom1])).name,
                    e.m.body(int(e.m.geom_bodyid[c.geom2])).name,
                )
                for c in e.d.contact
                if (
                    e.m.geom_bodyid[c.geom1] not in [0, *e.feet]
                    or e.m.geom_bodyid[c.geom2] not in [0, *e.feet]
                )
            ]
            rows.append(x)
            stepdata.append(
                dict(
                    time=pending["time"],
                    h_before=pending["h"],
                    h_after=np.array(x["H"]),
                    moment=pending["moment"],
                    force=pending["force"],
                    solver_iter=pending["solver_iter"],
                    finite_diff_acceleration_error=np.linalg.norm(
                        (pending["v_after"] - pending["v"]) / e.m.opt.timestep
                        - pending["qacc"]
                    ),
                )
            )
            pending.clear()
        return row

    mujoco.mj_step = step
    e.measure = measure
    try:
        obs, _ = e.observations()
        done = False
        for k in range(len(commands) + 300):
            u = (
                commands[k]
                if k < len(commands)
                else actions(
                    balance if controller == "balance" else student,
                    obs[None],
                    "balance" if controller == "balance" else "student",
                )[0]
            )
            (obs, _), _, done, info = e.step(u)
            states.append(e.state()["integration"])
            if done:
                break
    finally:
        mujoco.mj_step = orig_step
        del e.measure
        e.m.opt.timestep = original_dt
    dt = original_dt / factor
    dh = np.array([x["h_after"] - x["h_before"] for x in stepdata])
    imp = np.array([x["moment"] * dt for x in stepdata])
    res = np.cumsum(dh - imp, axis=0)
    # Legacy post-forward contact quadrature at same trajectory for comparison.
    moments = np.array([x["link_audit"]["moment"] for x in rows])
    legacy = (
        np.array(rows[-1]["H"])
        - np.array(stepdata[0]["h_before"])
        - np.sum(moments * dt, axis=0)
    )
    end = next(
        (i for i, x in enumerate(rows) if x["time"] >= 1.14 - 1e-8), len(rows) - 1
    )
    summary = dict(
        factor=factor,
        controller=controller,
        termination=info["termination"],
        terminal_time=rows[-1]["time"],
        first_unsafe=next((x["time"] for x in rows if not x["safe"]), None),
        terminal_used_wrench_residual=res[-1].tolist(),
        prefix_114_used_wrench_residual=res[end].tolist(),
        prefix_114_legacy_postforward_residual=(
            np.array(rows[end]["H"])
            - stepdata[0]["h_before"]
            - np.sum(moments[: end + 1] * dt, axis=0)
        ).tolist(),
        terminal_legacy_residual=legacy.tolist(),
        max_used_wrench_residual=float(np.linalg.norm(res, axis=1).max()),
        max_finite_difference_acceleration_error=float(
            max(x["finite_diff_acceleration_error"] for x in stepdata)
        ),
        five_second_success=bool(e.ever_hold_success),
        model=dict(
            timestep=original_dt,
            integrator=int(e.m.opt.integrator),
            solver=int(e.m.opt.solver),
            iterations=int(e.m.opt.iterations),
            tolerance=float(e.m.opt.tolerance),
            armature=e.m.dof_armature.tolist(),
            damping=e.m.dof_damping.tolist(),
        ),
    )
    return summary, dict(rows=rows, stepdata=stepdata, states=states)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--refine", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path("output/centroidal_bridge_capture_v2")
    starts = load(source / "starts.pt")
    cfg, data, bank, assets = setup(starts["args"]["preparation"], a.output)
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    student = Student()
    student.load_state_dict(load(starts["args"]["checkpoint"])["network"])
    balance = BalanceActor()
    balance.load_state_dict(
        load(starts["args"]["preparation"] / "balance.pt")["network"]
    )
    summaries = {}
    for label in ["61", "83"]:
        params = load(
            source / ("incumbent_2_8.pt" if label == "61" else "incumbent_4_10.pt")
        )["params"]
        for controller, factor in [("balance", 1), ("student", 1)] + (
            [("balance", 2)] if a.refine else []
        ):
            key = f"{label}_{controller}_dt{factor}"
            summary, saved = run(
                e, student, balance, starts, params, controller, factor
            )
            torch.save(saved, a.output / (key + ".pt"))
            summaries[key] = summary
            (a.output / "summary.json").write_text(json.dumps(summaries, indent=2))
            print(key, summary, flush=True)


if __name__ == "__main__":
    main()
