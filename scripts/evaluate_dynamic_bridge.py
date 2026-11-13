"""Expanded nominal suite, separate legacy/new cohorts, exact-state best/failure GIFs."""

import argparse, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup, actions, render_episode
from experiments.unified_go2.core import Environment, load, save_json, digest
from experiments.retained_go2.actor import Student
from experiments.front_catch.data import validate_state
import numpy as np
import torch, mujoco


def specs(cfg, bank, assets, reference):
    out = [("standing", "legacy", seed) for seed in range(96000, 96004)]
    out += [("standing", "new", seed) for seed in range(97600, 97608)]
    out += [("balance", "legacy", s) for s in bank["evaluation"]["balance"][:4]]
    e = Environment(assets, copy.deepcopy(cfg))
    for family in range(2):
        for delta in [-0.02, -0.01, 0.01, 0.02]:
            obs, _ = e.restore_start(
                copy.deepcopy(bank["evaluation"]["balance"][family]), "near_vertical"
            )
            safe = True
            for _ in range(3):
                act = actions(reference, obs[None])[0]
                act[[1, 4]] += delta / 2
                act[[2, 5]] -= delta / 2
                (obs, _), _, done, _ = e.step(act.clip(-1, 1))
                safe &= validate_state(e)[0] and not done and e._momentum_prefix_safe
            if safe:
                out.append(
                    ("balance", f"new_native_family_{family}", copy.deepcopy(e.state()))
                )
    out += [("bridge", "native_heldout", s) for s in bank["evaluation"]["bridge"]]
    return out


def episode(net, cfg, assets, centers, spec, max_steps=None):
    kind, cohort, start = spec
    e = Environment(assets, copy.deepcopy(cfg))
    e.capture_centers = centers
    e.cfg.seconds = 12.0 if kind == "standing" else 6.2
    obs, _ = (
        e.reset(1.0, start)
        if kind == "standing"
        else e.restore_start(
            copy.deepcopy(start), "near_vertical" if kind == "balance" else "partial"
        )
    )
    states = []
    entry = None
    done = False
    info = {}

    def record():
        s = np.empty(mujoco.mj_stateSize(e.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(e.m, e.d, s, mujoco.mjtState.mjSTATE_INTEGRATION)
        states.append(s)

    record()
    count = 0
    while not done:
        (obs, _), _, done, info = e.step(actions(net, obs[None])[0])
        record()
        count += 1
        if (
            entry is None
            and e.capture_distance < 1
            and e.row["front_support"]
            and e._momentum_prefix_safe
        ):
            s = e.momentum_sample
            entry = dict(
                pitch=s["pitch"],
                pitch_velocity=s["pitch_velocity"],
                com_support_x=s["com_support_x"],
                com_velocity_x=s["com_velocity_x"],
            )
        if max_steps and count >= max_steps:
            break
    peak = info["peak_collision_free_pitch"]
    row = dict(
        distribution=kind,
        cohort=cohort,
        seed=start if kind == "standing" else None,
        completed=done,
        certified_takeoff=bool(info["certified_standing_takeoff"]),
        peak_pitch_degrees=float(np.rad2deg(peak)),
        valid_hold=info["handstand_seconds"],
        five_second_success=bool(e.ever_hold_success),
        full_standing_success=bool(
            kind == "standing"
            and info["certified_standing_takeoff"]
            and e.ever_hold_success
        ),
        termination=info["termination"] if done else "diagnostic_truncation",
        illegal_collision=bool(info["nonfoot_collision"] or info["self_collision"]),
        front_support_fraction=info["front_support_fraction"],
        front_recontacts=info["counted_front_recontacts"],
        genuine_pushes=info["genuine_pushes"],
        max_torque_ratio=info["max_torque_ratio"],
        empirical_neighborhood_entry=entry,
        capture_claim="Empirical distance <1 is a diagnostic; only legal five-second hold demonstrates recovery",
    )
    for deg in [45, 60, 75, 85]:
        row[f"crossed_{deg}"] = bool(np.rad2deg(peak) >= deg)
    return row, e, np.array(states)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("output/retained_go2_distillation_v3/student.pt"),
    )
    p.add_argument(
        "--preparation", type=Path, default=Path("output/retained_go2_preparation_v3")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only two control steps per cohort; never report success rates",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(a.preparation, a.output)
    net = Student()
    net.load_state_dict(load(a.initial)["network"])
    reference = Student()
    reference.load_state_dict(load(a.reference)["network"])
    jobs = specs(cfg, bank, assets, reference)
    if a.smoke:
        jobs = [
            next(s for s in jobs if s[0] == k)
            for k in ["standing", "balance", "bridge"]
        ]
    rows = []
    best = -float("inf")
    failure_saved = False
    for i, spec in enumerate(jobs):
        row, e, states = episode(
            net, cfg, assets, data["centers"], spec, 2 if a.smoke else None
        )
        rows.append(row)
        print(i, spec[:2], row["peak_pitch_degrees"], row["valid_hold"], flush=True)
        if a.render and spec[0] == "standing":
            if row["peak_pitch_degrees"] > best:
                best = row["peak_pitch_degrees"]
                np.savez_compressed(
                    a.output / "best_standing_states.npz", integration=states
                )
                render_episode(
                    e, states, a.output / "best_standing.gif", "Best standing attempt"
                )
            if row["illegal_collision"] and not failure_saved:
                failure_saved = True
                np.savez_compressed(a.output / "failure_states.npz", integration=states)
                render_episode(
                    e,
                    states,
                    a.output / "failure.gif",
                    "Representative physical failure",
                )
    save_json(a.output / "episodes.json", rows)
    summaries = {}
    for key in sorted({(r["distribution"], r["cohort"]) for r in rows}):
        group = [r for r in rows if (r["distribution"], r["cohort"]) == key]
        counts = {
            k: sum(bool(r[k]) for r in group)
            for k in [
                "certified_takeoff",
                "five_second_success",
                "full_standing_success",
                "illegal_collision",
                "crossed_45",
                "crossed_60",
                "crossed_75",
                "crossed_85",
            ]
        }
        summaries["/".join(key)] = dict(
            n=len(group),
            counts=counts,
            mean_peak=float(np.mean([r["peak_pitch_degrees"] for r in group])),
            max_peak=max(r["peak_pitch_degrees"] for r in group),
            terminations={
                k: sum(r["termination"] == k for r in group)
                for k in {r["termination"] for r in group}
            },
        )
    save_json(
        a.output / "summary.json",
        dict(
            initial=str(a.initial),
            checkpoint_sha=digest(a.initial),
            reference_sha=digest(a.reference),
            preparation_sha=digest(a.preparation / "manifest.json"),
            complete_evaluation=not a.smoke,
            scope="New balance resets are native 60ms perturbation descendants of two legacy families, correlated rather than independent robustness trials.",
            cohorts=summaries,
        ),
    )


if __name__ == "__main__":
    main()
