"""Solve zero-task-cost constrained trajectory feasibility; no hardware I/O."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, BFGS
from scipy.sparse import csr_matrix
from experiments.trajectory_optimization.model import Robot, Settings
from experiments.trajectory_optimization.transcription import (
    Problem,
    Y,
    V,
    A,
    TAU,
    FORCE,
)


def solve(problem, initial, max_iterations=20, max_seconds=120):
    # Eliminate exactly fixed boundary variables instead of duplicating their
    # constraints with zero-velocity contact equalities (rank-deficient SQP).
    base = problem.unpack(initial).copy()
    fixed = np.zeros_like(base, dtype=bool)
    fixed[0, Y] = True
    fixed[0, V] = True
    base[0, Y] = problem.robot.chart(problem.start)
    base[0, V] = 0
    fixed[1, Y] = True
    base[1, Y] = base[0, Y]
    fixed[-1, Y] = True
    fixed[-1, V] = True
    fixed[-1, A] = True
    base[-1, Y] = problem.robot.chart(problem.goal)
    base[-1, V] = 0
    base[-1, A] = 0
    for k in range(problem.nodes):
        for i, contact in enumerate(problem.cfg.contacts(k)):
            if not contact:
                fixed[k, 66 + i * 3 : 69 + i * 3] = True
                base[k, 66 + i * 3 : 69 + i * 3] = 0
    free = np.flatnonzero(~fixed.ravel())
    base = base.ravel()

    def unpack(z):
        full = base.copy()
        full[free] = z
        return full

    constraints = []
    for kind in ["eq", "ineq"]:
        constraints.append(
            NonlinearConstraint(
                lambda z, k=kind: problem.values(unpack(z), k),
                0,
                0 if kind == "eq" else np.inf,
                jac="2-point",
                hess=BFGS(),
                finite_diff_jac_sparsity=problem.sparsity(base, kind)[:, free],
            )
        )
    start = time.monotonic()
    history = []

    def callback(z, state):
        history.append(
            {
                "iteration": int(state.nit),
                "objective": float(state.fun),
                "constraint_violation": float(state.constr_violation),
                "elapsed_s": time.monotonic() - start,
            }
        )
        return time.monotonic() - start >= max_seconds

    lo, hi = problem.bounds()
    result = minimize(
        lambda z: 0.0,
        base[free],
        method="trust-constr",
        jac=lambda z: np.zeros_like(z),
        hess=lambda z: csr_matrix((len(z), len(z))),
        bounds=Bounds(lo[free], hi[free]),
        constraints=constraints,
        callback=callback,
        options={
            "maxiter": max_iterations,
            "gtol": 1e-7,
            "sparse_jacobian": True,
            "factorization_method": "AugmentedSystem",
            "verbose": 0,
        },
    )
    result.x = unpack(result.x)
    return result, history


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--intervals", type=int, default=12)
    p.add_argument("--duration", type=float, default=2.4)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--max-seconds", type=float, default=120)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--seed", choices=["linear", "ik"], default="ik")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    if args.iterations < 1 or args.max_seconds <= 0:
        raise ValueError("Positive solver budgets required")
    cfg = Settings(intervals=args.intervals, duration=args.duration)
    robot = Robot(cfg)
    start, goal, anchors, nominal = robot.endpoints(args.reference)
    problem = Problem(robot, start, goal, anchors, nominal)
    initial = problem.kinematic_seed() if args.seed == "ik" else problem.seed()
    before = problem.audit(initial)
    source_hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path("experiments/trajectory_optimization/model.py"),
            Path("experiments/trajectory_optimization/transcription.py"),
        ]
    }
    args.output.mkdir(parents=True)
    if args.audit_only:
        candidate = initial
        history = []
        solver = {"success": False, "message": "Seed audit only; optimizer not run"}
    else:
        result, history = solve(problem, initial, args.iterations, args.max_seconds)
        candidate = result.x
        solver = {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
        }
    after = problem.audit(candidate)
    rows = problem.unpack(candidate)
    states = np.array([robot.pose(row[Y]) for row in rows])
    targets = np.array([problem.target(row) for row in rows])
    np.savez_compressed(
        args.output / "candidate.npz",
        qpos=states,
        qvel=rows[:, V],
        acceleration=rows[:, A],
        torque=rows[:, TAU],
        forces=rows[:, FORCE].reshape(-1, 4, 3),
        targets=targets,
        contacts=np.array([cfg.contacts(k) for k in range(problem.nodes)]),
        time=np.linspace(0, cfg.duration, problem.nodes),
        start=start,
        goal=goal,
        anchors=anchors,
        nominal=nominal,
        decision=candidate,
    )
    report = {
        "status": "MESH_FEASIBLE_PENDING_REPLAY"
        if after["mesh_feasible"]
        else "NOT_FEASIBLE",
        "hardware_ready": False,
        "task_reward": None,
        "task_cost": 0.0,
        "settings": asdict(cfg),
        "seed_method": args.seed,
        "source_sha256": source_hashes,
        "solver": solver,
        "initial_audit": before,
        "candidate_audit": after,
        "history": history,
        "python": sys.version,
        "mujoco": mujoco.__version__,
        "candidate_sha256": hashlib.sha256(
            (args.output / "candidate.npz").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "limitations": [
            "Point-foot model differs from compliant spherical contacts",
            "Prescribed contact schedule; no impacts or recontacts modeled",
            "Motor speed-torque envelope is uncalibrated research assumption",
            "Coarse Euler nodes are not a continuous-time safety certificate",
            "Independent native replay and staff approval required before export",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: report[k] for k in ["status", "solver", "candidate_audit"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
