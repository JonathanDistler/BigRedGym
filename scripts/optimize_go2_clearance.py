"""Optimize actual native-simulation calf clearance; no hardware interfaces."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.trajectory_optimization.shooting import NativeShooting
from experiments.trajectory_optimization.validation import replay


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--knots", type=int, default=4)
    p.add_argument("--buffer", type=float, default=0.04)
    p.add_argument("--weight", type=float, default=100.0)
    p.add_argument("--evaluations", type=int, default=20)
    p.add_argument("--max-seconds", type=float, default=180)
    p.add_argument("--horizon", type=float)
    p.add_argument("--initial-offsets", type=Path)
    p.add_argument("--offset-bound", type=float, default=0.6)
    p.add_argument("--pose-weight", type=float, default=1.0)
    p.add_argument("--target-margin", type=float, default=0.0)
    p.add_argument("--terminal-only", action="store_true")
    p.add_argument("--reference-seed", type=Path)
    p.add_argument("--preparation", type=float, default=0.5)
    a = p.parse_args()
    if a.reference_seed and a.initial_offsets:
        raise ValueError("Choose reference seed or existing offsets")
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    source_hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path("experiments/trajectory_optimization/shooting.py"),
            Path("experiments/trajectory_optimization/model.py"),
            Path("experiments/trajectory_optimization/transcription.py"),
            Path("experiments/trajectory_optimization/validation.py"),
        ]
    }
    shooting = NativeShooting(
        a.candidate,
        a.knots,
        a.buffer,
        a.weight,
        horizon=a.horizon,
        pose_weight=a.pose_weight,
        target_margin=a.target_margin,
        terminal_only=a.terminal_only,
    )
    initial = None
    if a.initial_offsets:
        with np.load(a.initial_offsets, allow_pickle=False) as data:
            initial = data["offsets"].copy()
        if (
            initial.ndim != 1
            or len(initial) % 4
            or len(initial) < 8
            or not np.isfinite(initial).all()
        ):
            raise ValueError("Invalid initial offset archive")
        if len(initial) != a.knots * 4:
            old = np.vstack([np.zeros(4), initial.reshape(-1, 4)])
            initial = np.array(
                [
                    np.interp(
                        np.linspace(0, 1, a.knots + 1)[1:],
                        np.linspace(0, 1, len(old)),
                        old[:, j],
                    )
                    for j in range(4)
                ]
            ).T.ravel()
    initialization = None
    if a.reference_seed:
        initial, initialization = shooting.reference_seed(
            a.reference_seed, a.preparation, a.offset_bound
        )
    planned = shooting.planned_clearance()
    baseline, trace, contact = shooting.simulate(np.zeros(a.knots * 4))
    best, history, message = shooting.optimize(
        a.evaluations, a.max_seconds, offset_bound=a.offset_bound, initial=initial
    )
    grid, commands = shooting.commands(best["parameters"], hold_seconds=5.0)
    validation = replay(a.candidate, command_override=(grid, commands))
    a.output.mkdir(parents=True)
    np.savez_compressed(
        a.output / "shooting_commands.npz",
        time=grid,
        targets=commands,
        offsets=best["parameters"],
        trace=best["trace"],
        planned_clearance=planned,
    )
    report = {
        "kind": "native_clearance_shooting_v1",
        "commands_asset_sha256": hashlib.sha256(
            (a.output / "shooting_commands.npz").read_bytes()
        ).hexdigest(),
        "source_candidate_sha256": hashlib.sha256(
            (a.candidate / "candidate.npz").read_bytes()
        ).hexdigest(),
        "simulation_evaluations": best["simulation_evaluations"],
        "hardware_ready": False,
        "status": "EXPERIMENT_ONLY_NOT_EXPORTABLE",
        "solver_message": message,
        "parameters": vars(a)
        | {
            "candidate": str(a.candidate),
            "output": str(a.output),
            "initial_offsets": str(a.initial_offsets) if a.initial_offsets else None,
            "reference_seed": str(a.reference_seed) if a.reference_seed else None,
        },
        "planned_minimum_calf_clearance_m": planned[:, 1:].min(axis=0).tolist(),
        "initialization": initialization,
        "baseline_cost": float(0.5 * baseline @ baseline),
        "best_cost": best["cost"],
        "baseline_minimum_calf_clearance_m": float(trace[:, 1:5].min()),
        "best_minimum_calf_clearance_m": float(best["trace"][:, 1:5].min()),
        "baseline_first_illegal_contact": contact,
        "baseline_terminal_orientation_error_rad": float(
            np.linalg.norm(trace[-1, 9:12])
        ),
        "best_terminal_orientation_error_rad": float(
            np.linalg.norm(best["trace"][-1, 9:12])
        ),
        "source_hashes_before_run": source_hashes,
        "history": history,
        "independent_replay": validation,
        "collision_free_full_replay": validation["collision_free"],
        "handstand_success": validation["handstand_success"],
        "actual_joint_legal": validation["prefix_joint_legal"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shooting_source_sha256": hashlib.sha256(
            Path("experiments/trajectory_optimization/shooting.py").read_bytes()
        ).hexdigest(),
    }
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "solver_message",
                    "baseline_cost",
                    "best_cost",
                    "baseline_minimum_calf_clearance_m",
                    "best_minimum_calf_clearance_m",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
