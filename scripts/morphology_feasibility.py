"""Short paired CPU probes, not an optimal-control or PPO feasibility proof."""

import argparse
import csv
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.morphology.task import Config, Task, metadata, reference, write_json
from gym.envs.go2.leg_morphology import SCALES, SOURCE, build_model


def save_csv(path, rows):
    with Path(path).open("w") as f:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = a.output / "source"
    sources.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/task.py"),
        Path("gym/envs/go2/leg_morphology.py"),
    ):
        shutil.copyfile(path, sources / path.name)
    cfg = Config()
    task = Task(a.output / "assets", cfg)
    original, generated = build_model(SOURCE), task.models[1.0]
    fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "dof_armature",
        "dof_damping",
    )
    identity = {
        k: bool(np.array_equal(getattr(original, k), getattr(generated, k)))
        for k in fields
    }
    if not all(identity.values()):
        raise AssertionError(identity)
    write_json(
        a.output / "configuration.json",
        dict(
            **metadata(cfg),
            seeds=list(range(a.seeds)),
            identity=identity,
            probe="paired open-loop commands and near-vertical PD holds",
            limitation="Observed rotation maxima are specific to the tested commands",
        ),
    )
    rows, summaries, static = [], [], []
    for scale in SCALES:
        task.reset(scale, 0)
        static.append(task.last.copy())
        probes = [
            ("push", amp, duration, 0.0, 0.0)
            for amp in (0.6, 1.0)
            for duration in (0.12, 0.24)
        ]
        probes += [
            ("vertical", 0.0, 0.18, np.deg2rad(angle), omega)
            for angle in (80, 85, 90)
            for omega in (-0.3, 0.0, 0.3)
        ]
        for probe, amp, duration, pitch, omega in probes:
            for seed in range(a.seeds):
                task.reset(scale, seed, pitch, omega)
                hold_target = task.d.qpos[task.qids].copy()
                attempt = []
                for _ in range(round(cfg.seconds / cfg.dt)):
                    target = (
                        reference(task, amp, duration)
                        if probe == "push"
                        else hold_target
                    )
                    _, _, done, row = task.step(np.zeros(12), target)
                    row.update(
                        seed=seed,
                        probe=probe,
                        amplitude=amp,
                        duration=duration,
                        initial_pitch=pitch,
                        initial_omega=omega,
                    )
                    attempt.append(row.copy())
                    if done:
                        break
                rows.extend(attempt)
                summaries.append(
                    dict(
                        scale=scale,
                        seed=seed,
                        probe=probe,
                        amplitude=amp,
                        duration=duration,
                        initial_pitch=pitch,
                        initial_omega=omega,
                        episode_seconds=task.t,
                        success=bool(task.last["success"]),
                        max_pitch=max(r["pitch"] for r in attempt),
                        max_rear_clearance=max(r["rear_clearance"] for r in attempt),
                        rear_impulse=task.rear_impulse,
                        push_impulse=sum(
                            r["rear_step_impulse"]
                            for r in attempt
                            if 0.45 < r["time"] <= 0.45 + duration
                        ),
                        excess_push_impulse=sum(
                            r["rear_excess_step_impulse"]
                            for r in attempt
                            if 0.45 < r["time"] <= 0.45 + duration
                        ),
                        min_pitch=min(r["pitch"] for r in attempt),
                        illegal_contact_bodies=task.last["illegal_contact_bodies"],
                        peak_rear_torque=max(r["rear_peak_torque"] for r in attempt),
                        saturation=np.mean([r["saturation"] for r in attempt]),
                        front_support_fraction=np.mean(
                            [r["front_support"] for r in attempt]
                        ),
                        best_hold=task.best_hold,
                        termination=task.last["termination"],
                    )
                )
        print(f"scale={scale:.2f}: probes complete", flush=True)
    save_csv(a.output / "dynamics.csv", static)
    save_csv(a.output / "trajectories.csv", rows)
    save_csv(a.output / "attempts.csv", summaries)
    push = [r for r in summaries if r["probe"] == "push"]
    # An intentionally conservative screening gate; cannot authorize a learning claim.
    rates = {
        s: np.mean([r["success"] for r in push if r["scale"] == s]) for s in SCALES
    }
    feasible = [s for s in SCALES[:-1] if rates[s] > rates[1.0]]
    write_json(
        a.output / "screening.json",
        dict(
            demonstrated_transition=bool(feasible),
            candidate_scales=feasible,
            extended_training_recommended=False,
            standing_probe_success_rates=rates,
            conclusion="No shortened-leg curriculum justified by this screen"
            if not feasible
            else "Candidate found; independent coverage tests needed",
            caveat="Failed scripted probes do not establish policy infeasibility.",
        ),
    )


if __name__ == "__main__":
    main()
