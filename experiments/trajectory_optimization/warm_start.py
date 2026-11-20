"""Initialization/mesh tools; their numerical merit is never a task reward."""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from .transcription import Y, V, A, TAU, FORCE, WIDTH


def contact_mesh(intervals, duration, release_fraction):
    """Include release exactly and concentrate nodes around support transfer."""
    release = duration * release_fraction
    left = max(2, min(intervals - 2, round(intervals * release_fraction)))
    if intervals < 4:
        raise ValueError("Contact mesh requires at least four intervals")
    # Quadratic spacing on either side of release, without duplicate nodes.
    a = release * (1 - (1 - np.linspace(0, 1, left + 1)) ** 2)
    b = release + (duration - release) * np.linspace(0, 1, intervals - left + 1) ** 2
    return tuple(np.r_[a, b[1:]])


def prolong(source, decision, destination):
    """Manifold interpolation of an audited decision; no feasibility claim."""
    old = source.unpack(decision)
    out = np.empty((destination.nodes, WIDTH))
    for j in range(WIDTH):
        out[:, j] = np.interp(destination.cfg.times, source.cfg.times, old[:, j])
    import mujoco

    for k, t in enumerate(destination.cfg.times):
        i = min(
            max(np.searchsorted(source.cfg.times, t, side="right") - 1, 0),
            source.nodes - 2,
        )
        fraction = (t - source.cfg.times[i]) / source.cfg.step(i)
        q = source.robot.pose(old[i, Y])
        tangent = np.empty(18)
        mujoco.mj_differentiatePos(
            source.robot.m, tangent, 1, q, source.robot.pose(old[i + 1, Y])
        )
        mujoco.mj_integratePos(source.robot.m, q, tangent, float(fraction))
        out[k, Y] = destination.robot.chart(q)
        out[k, FORCE].reshape(4, 3)[~destination.cfg.contacts(k)] = 0
    out[0, Y] = destination.robot.chart(destination.start)
    out[1, Y] = out[0, Y]  # explicit Euler and zero initial velocity
    out[0, V] = out[-1, V] = 0
    out[-1, Y] = destination.robot.chart(destination.goal)
    out[-1, A] = 0
    return out.ravel()


def inverse_dynamics_seed(problem, initial):
    """Nodewise constrained inverse dynamics with exact linear equalities.

    q,v remain unchanged. Solve for a,tau,stance forces, enforcing floating-base
    dynamics, no stance acceleration (when enabled), friction and motor envelope.
    Failed nodes retain their original values and are reported, never hidden.
    This repairs dynamics without claiming to repair temporal Euler defects.
    """
    rows = problem.unpack(initial).copy()
    r = problem.robot
    reports = []
    for k, row in enumerate(rows):
        M, h, _, J, *_ = r.terms(r.pose(row[Y]), row[V])
        mask = problem.cfg.contacts(k)
        active = J[mask]
        count = int(mask.sum())
        B = np.zeros((18, 12))
        B[r.vids, np.arange(12)] = 1
        E = np.column_stack([M, -B, -active.reshape(-1, 18).T])
        rhs = -h
        if problem.cfg.contact_acceleration:
            C = np.column_stack(
                [active.reshape(-1, 18), np.zeros((3 * count, 12 + 3 * count))]
            )
            E = np.vstack([E, C])
            rhs = np.r_[rhs, -r.jdot_v(r.pose(row[Y]), row[V], J)[mask].ravel()]
        desired = np.r_[row[A], row[TAU], row[FORCE].reshape(4, 3)[mask].ravel()]
        scale = np.r_[np.full(18, 20.0), r.effort, np.full(3 * count, r.mass * 9.81)]
        lo = np.r_[
            np.full(18, -200.0),
            -r.torque_envelope(row[V][r.vids]),
            np.tile([-np.inf, -np.inf, 0.0], count),
        ]
        hi = np.r_[
            np.full(18, 200.0),
            r.torque_envelope(row[V][r.vids]),
            np.tile(
                [np.inf, np.inf, problem.cfg.max_force_bodyweights * r.mass * 9.81],
                count,
            ),
        ]
        if k == problem.nodes - 1:
            # Bounds with identical endpoints are supported by SLSQP.
            lo[:18] = hi[:18] = 0
        # Eight-sided inner approximation: every feasible force is in true cone.
        cone = np.zeros((8 * count, len(desired)))
        for i in range(count):
            for j, angle in enumerate(np.arange(8) * np.pi / 4):
                cone[8 * i + j, 30 + 3 * i : 33 + 3 * i] = [
                    -np.cos(angle),
                    -np.sin(angle),
                    problem.cfg.friction * np.cos(np.pi / 8),
                ]
        result = minimize(
            lambda z: 0.5 * np.sum(((z - desired) / scale) ** 2),
            np.clip(desired, lo, hi),
            jac=lambda z: (z - desired) / scale**2,
            method="SLSQP",
            bounds=Bounds(lo, hi),
            constraints=[
                LinearConstraint(E, rhs, rhs),
                LinearConstraint(cone, 0, np.inf),
            ],
            options={"maxiter": 150, "ftol": 1e-10},
        )
        eq = float(np.max(np.abs(E @ result.x - rhs)))
        iq = float(np.maximum(0, -cone @ result.x).max(initial=0))
        accepted = eq < 1e-6 and iq < 1e-7 and np.isfinite(result.x).all()
        if accepted:
            row[A] = result.x[:18]
            row[TAU] = result.x[18:30]
            f = np.zeros((4, 3))
            f[mask] = result.x[30:].reshape(count, 3)
            row[FORCE] = f.ravel()
        reports.append(
            {
                "node": k,
                "accepted": bool(accepted),
                "equality_error": eq,
                "cone_error": iq,
                "message": str(result.message),
            }
        )
    return rows.ravel(), reports


def refine_defects(problem, decision, count):
    """Bisect intervals with largest scaled Euler/contact-transfer defects."""
    if not isinstance(count, int) or not 1 <= count <= problem.cfg.intervals:
        raise ValueError("Refinement count must be between one and interval count")
    scores = np.zeros(problem.cfg.intervals)
    for _, name, value, scale, depends in problem.blocks(decision):
        if name.startswith("explicit_euler"):
            scores[depends[0]] = max(
                scores[depends[0]], float(np.max(abs(value) / scale))
            )
    selected = np.argsort(-scores, kind="stable")[:count]
    times = problem.cfg.times
    new = np.sort(np.r_[times, [(times[i] + times[i + 1]) / 2 for i in selected]])
    return tuple(new), {
        "split_intervals": selected.tolist(),
        "scaled_euler_scores": scores.tolist(),
    }
