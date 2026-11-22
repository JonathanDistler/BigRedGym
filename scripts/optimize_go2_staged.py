"""Optimize crouched preparation; replay a full-joint handstand warm start."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.trajectory_optimization.staged import Preparation, evaluate


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", choices=["standing", "crouch"], default="crouch")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--settle", type=float, default=0.5)
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--max-seconds", type=float, default=180)
    p.add_argument("--initial", type=Path)
    p.add_argument("--trials", type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    a.output.mkdir(parents=True)
    preparation = Preparation(a.reference, a.duration, settle=a.settle)
    parameters = np.zeros(preparation.parameter_count)
    if a.initial:
        with np.load(a.initial, allow_pickle=False) as f:
            parameters = f["parameters"].copy()
    baseline, diagnostic = preparation.simulate(parameters)
    report = {
        "source_reference_sha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        "kind": "staged_native_trajectory_experiment_v1",
        "baseline_preparation_cost": float(baseline @ baseline / 2),
        "baseline_preparation": diagnostic,
        "source_initial_velocity": preparation.goal_v.tolist(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
        },
    }
    if a.optimize:
        best, history, message = preparation.optimize(
            a.iterations, a.max_seconds, parameters, a.output / "solver_checkpoint.json"
        )
        parameters = best["parameters"]
        report.update(
            best_cost=best["cost"],
            best_preparation=best["report"],
            history=history,
            solver_message=message,
        )
    np.savez_compressed(
        a.output / "preparation.npz",
        parameters=parameters,
        time=preparation.grid,
        targets=preparation.commands(parameters),
        start=preparation.start,
        goal_q=preparation.goal_q,
        goal_v=preparation.goal_v,
    )
    rows = []
    for k in range(a.trials + 1):
        row = evaluate(
            preparation.reference,
            preparation if a.start == "crouch" else None,
            parameters,
            301000 + k,
            k > 0,
        )
        rows.append(row)
        print(
            "trial",
            k,
            row["complete_success"],
            row["strict_hold_seconds"],
            row["first_illegal_contact_s"],
            flush=True,
        )
        report["rows"] = rows
        report["heldout_successes"] = sum(x["complete_success"] for x in rows[1:])
        report["heldout_trials"] = len(rows) - 1
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
