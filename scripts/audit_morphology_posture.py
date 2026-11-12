"""Full-size posture ablation: change COM-to-support length without leg scaling."""

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.morphology.dynamics_audit import Audit, frozen_energy_barrier
from experiments.morphology.task import Config, metadata, write_json
from scripts.audit_morphology_dynamics import FEET, command, summarize, waypoints
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.output / "source"
    source.mkdir()
    for path in (
        Path(__file__),
        Path("experiments/morphology/dynamics_audit.py"),
        Path("scripts/audit_morphology_dynamics.py"),
        Path("experiments/morphology/task.py"),
        Path("gym/envs/go2/leg_morphology.py"),
    ):
        shutil.copyfile(path, source / path.name)
    task = Audit(a.output / "assets", Config())
    task.reset(1.0, 0)
    base, _, target = waypoints(task, "relative_extension")
    target_row = task.static(target, (0, 1))
    geom = []
    traces = []
    attempts = []
    poses = []
    checks = []
    for x in (0.0, 0.03, 0.06):
        standing, es = task.pose(0.34, x=x, feet=FEET, guess=base)
        crouch, ec = task.pose(0.27, x=x, feet=FEET, guess=standing)
        if max(es, ec) > 1e-5:
            raise RuntimeError("posture IK failed")
        row = task.static(standing)
        geom.append(
            dict(
                offset=x,
                **row,
                frozen_barrier=frozen_energy_barrier(
                    row["mass"], row["com_x"] - 0.1917, row["com_z"]
                ),
                target_delta_U=row["mass"]
                * 9.81
                * (target_row["com_z"] - row["com_z"]),
            )
        )
        for label, q in [("standing", standing), ("crouch", crouch)]:
            poses.append(dict(offset=x, pose=label, qpos=q.tolist(), **task.static(q)))
        for k in range(51):
            h = 0.34 + (0.27 - 0.34) * (3 * (k / 50) ** 2 - 2 * (k / 50) ** 3)
            q, error = task.pose(h, x=x, feet=FEET, guess=standing)
            checks.append(
                dict(offset=x, fraction=k / 50, ik_error=error, **task.static(q))
            )
        for duration in (0.12, 0.24):
            for knee in (-1.8, -2.2):
                for seed in range(a.seeds):
                    task.set_pose(
                        standing, np.random.default_rng(seed).uniform(-0.02, 0.02)
                    )
                    previous = standing[task.qids].copy()
                    rows = []
                    identity = dict(
                        evaluation="takeoff",
                        family="posture",
                        mode="pitch_feedback",
                        seed=seed,
                        duration=duration,
                        knee=knee,
                        offset=x,
                    )
                    for _ in range(900):
                        desired, phase = command(
                            task, standing, crouch, duration, knee, True, previous
                        )
                        previous = desired.copy()
                        row, events = task.sample(desired, phase)
                        row.update(identity)
                        rows.append(row.copy())
                        if events:
                            break
                    attempts.append(summarize(rows, task, **identity))
                    traces.extend(rows)
        print(f"full-size posture x={x:.2f}", flush=True)
    write_json(
        a.output / "configuration.json",
        dict(
            **metadata(task.cfg),
            seeds=list(range(a.seeds)),
            offsets=[0.0, 0.03, 0.06],
            scale=1.0,
            controller="same bounded pitch-feedback grid as morphology audit",
            purpose="separate changing posture from changing physical link length",
        ),
    )
    save_csv(a.output / "geometry.csv", geom)
    save_csv(a.output / "takeoff.csv", attempts)
    save_csv(a.output / "trajectories.csv", traces)
    save_csv(a.output / "path_checks.csv", checks)
    write_json(a.output / "poses.json", poses)


if __name__ == "__main__":
    main()
