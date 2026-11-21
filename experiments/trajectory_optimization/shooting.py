"""Software-only native-contact target optimization; never a deployment policy.

Unlike pose-space clearance, this objective sees PD tracking error and native
compliant contacts. Costs guide search; measured contact remains a failed trial.
"""

import time
import mujoco
import numpy as np
from scipy.optimize import least_squares
from gym.envs.go2.handstand_replay import CONTROL_DT, PHYSICS_DT, KP, KD, clearance
from .validation import audit_saved, position_commands


class NativeShooting:
    def __init__(
        self,
        candidate,
        knots=4,
        buffer=0.04,
        weight=100.0,
        tail=0.4,
        horizon=None,
        pose_weight=1.0,
        target_margin=0.0,
        terminal_only=False,
    ):
        if knots < 2 or buffer <= 0 or weight <= 0 or tail < 0:
            raise ValueError("Invalid native shooting parameters")
        if not np.isfinite(pose_weight) or pose_weight <= 0:
            raise ValueError("Positive finite pose weight required")
        self.pose_scale = np.sqrt(pose_weight)
        self.terminal_only = bool(terminal_only)
        self.robot, self.problem, self.saved, self.mesh = audit_saved(candidate)
        if not np.isfinite(target_margin) or target_margin < 0:
            raise ValueError("Invalid target joint margin")
        r = self.robot
        self.lower_target = np.maximum(
            r.bounds[:, 0] + target_margin, self.saved["nominal"] - 2
        )
        self.upper_target = np.minimum(
            r.bounds[:, 1] - target_margin, self.saved["nominal"] + 2
        )
        if (
            np.any(self.lower_target >= self.upper_target)
            or np.any(self.saved["targets"][0] < self.lower_target)
            or np.any(self.saved["targets"][0] > self.upper_target)
        ):
            raise ValueError("Target margin conflicts with limits or initial target")
        self.duration = float(self.saved["time"][-1])
        self.knot_times = np.linspace(0, self.duration, knots + 1)
        self.knots, self.buffer, self.weight, self.tail = knots, buffer, weight, tail
        self.references = []
        self.horizon = self.duration + tail if horizon is None else float(horizon)
        if (
            not np.isfinite(self.horizon)
            or not 0 < self.horizon <= self.duration + tail
        ):
            raise ValueError("Horizon must be within the full native trial")
        self.grid = np.arange(int(np.ceil(self.horizon / CONTROL_DT))) * CONTROL_DT
        r = self.robot
        for t in self.grid + CONTROL_DT:
            i = min(
                max(np.searchsorted(self.saved["time"], t, side="right") - 1, 0),
                len(self.saved["time"]) - 2,
            )
            alpha = np.clip(
                (t - self.saved["time"][i])
                / (self.saved["time"][i + 1] - self.saved["time"][i]),
                0,
                1,
            )
            q = self.saved["qpos"][i].copy()
            tangent = np.empty(18)
            mujoco.mj_differentiatePos(r.m, tangent, 1, q, self.saved["qpos"][i + 1])
            mujoco.mj_integratePos(r.m, q, tangent, float(alpha))
            self.references.append(q)

    def reference_seed(self, reference, preparation=0.5, bound=1.0):
        """Fit release actions for initialization; no learned controller is replaced."""
        from pathlib import Path
        from gym.envs.go2.handstand_replay import load_trajectory
        from .model import REFERENCE_SHA

        if not np.isfinite(preparation) or preparation <= 0 or bound <= 0:
            raise ValueError("Invalid reference preparation/bound")
        data = load_trajectory(Path(reference), REFERENCE_SHA, self.robot.m)
        grid, baseline = self.commands(np.zeros(self.knots * 4), hold_seconds=0)
        initial = self.saved["targets"][0]
        wanted = []
        for t in grid:
            if t < preparation:
                u = min(t / (0.75 * preparation), 1.0)
                u = u * u * (3 - 2 * u)
                wanted.append((1 - u) * initial + u * data["initial_target"])
            else:
                k = min(round((t - preparation) / CONTROL_DT), len(data["actions"]) - 1)
                wanted.append(data["nominal"] + 2 * data["actions"][k])
        wanted = np.clip(wanted, self.lower_target, self.upper_target)
        basis = np.column_stack(
            [
                np.interp(grid, self.knot_times, np.eye(self.knots + 1)[j])
                for j in range(1, self.knots + 1)
            ]
        )
        delta = np.array(
            [
                (wanted[:, js] - baseline[:, js]).mean(axis=1)
                for js in [(1, 4), (2, 5), (7, 10), (8, 11)]
            ]
        ).T
        parameters = np.clip(
            np.linalg.lstsq(basis, delta, rcond=None)[0], -bound, bound
        )
        return parameters.ravel(), {
            "kind": "release_action_fit",
            "reference_sha256": REFERENCE_SHA,
            "preparation_seconds": preparation,
            "fit_rms_rad": float(np.sqrt(np.mean((basis @ parameters - delta) ** 2))),
        }

    def planned_clearance(self, period=PHYSICS_DT):
        if not np.isfinite(period) or period <= 0:
            raise ValueError("Positive planned sample period required")
        result = []
        r = self.robot
        for t in np.r_[np.arange(0, self.duration, period), self.duration]:
            i = min(
                max(np.searchsorted(self.saved["time"], t, side="right") - 1, 0),
                len(self.saved["time"]) - 2,
            )
            q = self.saved["qpos"][i].copy()
            tangent = np.empty(18)
            mujoco.mj_differentiatePos(r.m, tangent, 1, q, self.saved["qpos"][i + 1])
            alpha = (t - self.saved["time"][i]) / (
                self.saved["time"][i + 1] - self.saved["time"][i]
            )
            mujoco.mj_integratePos(r.m, q, tangent, float(alpha))
            r.set(q)
            result.append(np.r_[t, r.calf_heights()])
        return np.array(result)

    def commands(self, parameters, hold_seconds=None):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.knots * 4,) or not np.isfinite(parameters).all():
            raise ValueError("Expected finite sagittal target offsets")
        saved = dict(self.saved)
        hold = self.tail if hold_seconds is None else hold_seconds
        grid, _ = position_commands(saved, self.robot.speed, hold)
        offsets = np.vstack([np.zeros(4), parameters.reshape(self.knots, 4)])
        values = np.array(
            [np.interp(grid, self.knot_times, offsets[:, j]) for j in range(4)]
        ).T
        desired = np.array(
            [np.interp(grid, saved["time"], saved["targets"][:, j]) for j in range(12)]
        ).T
        for j, joints in enumerate([(1, 4), (2, 5), (7, 10), (8, 11)]):
            desired[:, joints] += values[:, j, None]
        r = self.robot
        desired = np.clip(desired, self.lower_target, self.upper_target)
        # Preserve the initial target and slew-limit optimization offsets too.
        target = saved["targets"][0].copy()
        commands = []
        for command in desired:
            target += np.clip(
                command - target,
                -0.7 * r.speed * CONTROL_DT,
                0.7 * r.speed * CONTROL_DT,
            )
            commands.append(target.copy())
        return grid, np.asarray(commands)

    def simulate(self, parameters):
        r = self.robot
        m = r.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.saved["start"]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        grid, commands = self.commands(parameters)
        samples, residuals = [], []
        first_illegal = None
        for k, target in enumerate(commands[: len(self.grid)]):
            calf = np.full(4, np.inf)
            body = np.inf
            penetration = 0.0
            speed = np.zeros(12)
            torque_excess = np.zeros(12)
            joint_excess = np.zeros(12)
            for _ in range(round(CONTROL_DT / PHYSICS_DT)):
                dq = d.qvel[r.vids].copy()
                torque = np.clip(
                    KP * (target - d.qpos[r.qids]) - KD * dq, -r.effort, r.effort
                )
                torque[(abs(dq) >= r.speed) & (torque * dq > 0)] = 0
                torque_excess = np.maximum(
                    torque_excess, np.maximum(abs(torque) - r.torque_envelope(dq), 0)
                )
                d.qfrc_applied[:] = 0
                d.qfrc_applied[r.vids] = torque
                previous_time = float(d.time)
                mujoco.mj_step(m, d)
                if abs(float(d.time) - previous_time - PHYSICS_DT) > 1e-9:
                    raise FloatingPointError(
                        "Native simulation reset or skipped a physics step"
                    )
                mujoco.mj_forward(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    raise FloatingPointError("Nonfinite native simulation")
                calf = np.minimum(
                    calf, [min(clearance(m, d, g) for g in gs) for gs in r.calf_geoms]
                )
                body = min(body, min(clearance(m, d, g) for g in r.robot_geoms))
                speed = np.maximum(speed, abs(d.qvel[r.vids]) / r.speed)
                joint_excess = np.maximum(
                    joint_excess,
                    np.maximum(
                        r.bounds[:, 0] - d.qpos[r.qids], d.qpos[r.qids] - r.bounds[:, 1]
                    ),
                )
                for c in d.contact:
                    if c.dist > 0:
                        continue
                    b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
                    if (b1 and b2) or any(b and b not in r.feet for b in (b1, b2)):
                        penetration = max(penetration, -float(c.dist))
                        if first_illegal is None:
                            first_illegal = {
                                "time_s": float(d.time),
                                "bodies": [m.body(b1).name, m.body(b2).name],
                            }
            error = np.empty(18)
            mujoco.mj_differentiatePos(m, error, 1, self.references[k], d.qpos)
            # Clearance is a strong preference, not a waiver of hard acceptance.
            residuals.extend(
                np.sqrt(self.weight) * np.maximum(self.buffer - calf, 0) / self.buffer
            )
            residuals.append(np.sqrt(self.weight) * max(0.015 - body, 0) / 0.015)
            residuals.append(np.sqrt(self.weight) * penetration / 0.005)
            # Body tracking and final posture prevent "avoid contact by doing nothing".
            pose_scale = (
                self.pose_scale
                if (
                    not self.terminal_only
                    or (k + 1) * CONTROL_DT >= max(0, self.horizon - 0.4)
                )
                else 0.0
            )
            residuals.extend(pose_scale * error[:3] / [0.10, 0.10, 0.05])
            residuals.extend(pose_scale * error[3:6] / 0.20)
            residuals.extend(pose_scale * 0.2 * error[6:] / 0.3)
            residuals.extend(5 * np.maximum(speed - 1, 0))
            residuals.extend(5 * joint_excess / 0.05)
            residuals.extend(2 * torque_excess / r.effort)
            if k >= len(self.grid) - 1:
                residuals.extend(d.qvel / np.r_[[0.25] * 3, [0.5] * 3, [1.0] * 12])
            samples.append(np.r_[d.time, calf, body, error[:6], d.qpos, d.qvel])
        # Small control regularizer only for this explicitly nonzero-cost experiment.
        residuals.extend(0.1 * np.asarray(parameters))
        return np.asarray(residuals), np.asarray(samples), first_illegal

    def optimize(
        self, max_evaluations=20, max_seconds=180, offset_bound=0.6, initial=None
    ):
        if max_evaluations < 1 or max_seconds <= 0 or offset_bound <= 0:
            raise ValueError("Positive shooting budgets/bounds required")
        started = time.monotonic()
        initial = (
            np.zeros(self.knots * 4)
            if initial is None
            else np.asarray(initial, dtype=float)
        )
        if (
            initial.shape != (self.knots * 4,)
            or not np.isfinite(initial).all()
            or np.any(abs(initial) > offset_bound)
        ):
            raise ValueError("Invalid initial shooting offsets")
        best = {"cost": np.inf}
        history = []
        evaluations = 0

        class Budget(Exception):
            pass

        def fun(x):
            nonlocal evaluations
            if time.monotonic() - started >= max_seconds and np.isfinite(best["cost"]):
                raise Budget
            evaluations += 1
            residual, trace, contact = self.simulate(x)
            cost = 0.5 * float(residual @ residual)
            if cost < best["cost"]:
                best.update(
                    cost=cost, parameters=x.copy(), trace=trace.copy(), contact=contact
                )
                history.append(
                    {
                        "evaluation": evaluations,
                        "cost": cost,
                        "elapsed_s": time.monotonic() - started,
                        "minimum_calf_clearance_m": float(trace[:, 1:5].min()),
                        "first_illegal_contact": contact,
                    }
                )
            return residual

        def jac(x):
            # Absolute 2 mrad perturbation, including at x=0; relative steps can
            # collapse to machine epsilon and see only contact-solver noise.
            baseline = fun(x)
            columns = []
            for i in range(len(x)):
                trial = x.copy()
                delta = 0.002 if x[i] + 0.002 <= offset_bound else -0.002
                trial[i] += delta
                columns.append((fun(trial) - baseline) / delta)
            return np.array(columns).T

        try:
            result = least_squares(
                fun,
                initial,
                jac=jac,
                bounds=(-offset_bound, offset_bound),
                max_nfev=max_evaluations,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            message = str(result.message)
        except Budget:
            message = (
                "Software shooting time budget reached; returning lowest measured merit"
            )
        best["simulation_evaluations"] = evaluations
        return best, history, message
