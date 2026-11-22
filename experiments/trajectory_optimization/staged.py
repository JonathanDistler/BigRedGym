"""Native-dynamics stand-up optimization and full-joint trajectory continuation.

The released trajectory is a disclosed warm start, never a checkpoint/policy.
No simulator state is used by the exported reference actor. Optimization and
independent physics scoring may inspect state; no mid-trial reset is allowed.
"""

from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from gym.envs.go2.handstand_replay import (
    Replay,
    build_model,
    load_trajectory,
    STATE_SPEC,
    CONTROL_DT,
    KP,
    KD,
    clearance,
)
from .model import Robot, Settings, REFERENCE_SHA


def smooth(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 + x * (-15 + 6 * x))


def wire_target(desired, previous, bounds, speed):
    """Float32 MotorCmd-compatible target, rounded inward at every safety bound."""
    desired = np.asarray(desired, dtype=float)
    previous = np.asarray(previous, dtype=float)
    if (
        desired.shape != (12,)
        or previous.shape != (12,)
        or not np.isfinite([desired, previous]).all()
    ):
        raise ValueError("Expected finite twelve-joint position targets")
    lo = np.maximum(bounds[:, 0], previous - 0.7 * speed * CONTROL_DT)
    hi = np.minimum(bounds[:, 1], previous + 0.7 * speed * CONTROL_DT)
    out = np.clip(desired, lo, hi).astype(np.float32)
    out = np.where(
        out.astype(float) < lo,
        np.nextafter(lo.astype(np.float32), np.float32(np.inf)),
        out,
    )
    out = np.where(
        out.astype(float) > hi,
        np.nextafter(hi.astype(np.float32), np.float32(-np.inf)),
        out,
    )
    return out.astype(float)


class Preparation:
    """Quasistatic IK/inverse-dynamics seed followed by native state matching."""

    def __init__(self, reference, duration=3.0, crouch_height=0.27, settle=0.5):
        if (
            not np.isfinite([duration, crouch_height, settle]).all()
            or duration < 0.4
            or settle < 0
        ):
            raise ValueError("Invalid preparation duration/height/settle")
        self.robot = Robot(Settings(crouch_height=crouch_height))
        r = self.robot
        self.reference = load_trajectory(Path(reference), REFERENCE_SHA, r.m)
        self.start, _, _, self.nominal = r.endpoints(Path(reference))
        mujoco.mj_setState(r.m, r.d, self.reference["initial_state"], STATE_SPEC)
        mujoco.mj_forward(r.m, r.d)
        self.goal_q = r.d.qpos.copy()
        self.goal_v = r.d.qvel.copy()
        goal_feet, _ = r.foot_kinematics()
        r.set(self.start)
        start_feet, _ = r.foot_kinematics()
        self.duration = round((duration + settle) / CONTROL_DT) * CONTROL_DT
        self.grid = np.arange(round(self.duration / CONTROL_DT)) * CONTROL_DT
        self.seed_targets, self.poses = [], []
        guess = self.start[r.qids].copy()
        for t in self.grid:
            u = smooth(t / duration)
            q = self.start.copy()
            q[:3] = (1 - u) * q[:3] + u * self.goal_q[:3]
            # Both original standing/crouch frames are level. Full quaternion
            # interpolation still avoids assuming an Euler chart in the solver.
            tangent = np.empty(18)
            mujoco.mj_differentiatePos(r.m, tangent, 1, self.start, self.goal_q)
            mujoco.mj_integratePos(
                r.m, q, np.r_[np.zeros(3), tangent[3:6], np.zeros(12)], float(u)
            )
            wanted = (1 - u) * start_feet + u * goal_feet

            def error(joints):
                q[r.qids] = joints
                r.set(q)
                return (r.foot_kinematics()[0] - wanted).ravel()

            fit = least_squares(
                error,
                guess,
                bounds=r.bounds.T,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                max_nfev=60,
            )
            if np.max(abs(error(fit.x))) > 1e-6:
                raise ValueError(
                    f"Standing-preparation IK failed at t={t}: "
                    f"residual={np.max(abs(error(fit.x)))}"
                )
            q[r.qids] = fit.x
            guess = fit.x.copy()
            r.set(q)
            _, J = r.foot_kinematics()
            bias = r.d.qfrc_bias - r.d.qfrc_passive
            # Static normal-force initialization: no force is applied in replay.
            normal_map = J[:, 2, :].T
            forces = lsq_linear(
                normal_map[:6], bias[:6], bounds=(0, 3 * r.mass * 9.81)
            ).x
            tau = bias[r.vids] - (normal_map @ forces)[r.vids]
            target = np.clip(q[r.qids] + tau / KP, *r.bounds.T)
            self.seed_targets.append(target)
            self.poses.append(q.copy())
        self.seed_targets = np.asarray(self.seed_targets)
        self.poses = np.asarray(self.poses)
        self.initial_target = self.seed_targets[0].copy()
        self.count = 0
        self.parameter_count = 36

    def commands(self, parameters):
        p = np.asarray(parameters, dtype=float)
        if p.shape == (24,):
            p = np.r_[p, np.zeros(12)]  # Preserve archived two-basis warm starts.
        if p.shape != (36,) or not np.isfinite(p).all():
            raise ValueError("Expected 36 finite endpoint/velocity/pulse parameters")
        phase = self.grid / self.duration
        a = smooth(phase)
        # Position and terminal target-slope directions. Unlike a long static
        # pause, this can match the reference's NONZERO initial pitch velocity.
        pulse_width = min(0.45, self.duration)
        pulse_phase = np.clip(
            (self.grid - (self.duration - pulse_width)) / pulse_width, 0, 1
        )
        pulse = smooth(pulse_phase) * smooth(1 - pulse_phase)
        desired = (
            self.seed_targets
            + a[:, None] * p[:12]
            + (a * (phase - 1))[:, None] * p[12:24]
            + pulse[:, None] * p[24:]
        )
        desired = np.clip(desired, self.nominal - 2, self.nominal + 2)
        target = self.initial_target.copy()
        out = []
        for row in desired:
            target = wire_target(row, target, self.robot.bounds, self.robot.speed)
            out.append(target)
        return np.asarray(out)

    def simulate(self, parameters):
        r = self.robot
        m = r.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.start
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        lowest = np.inf
        penetration = 0.0
        joint_excess = 0.0
        speed_excess = 0.0
        for target in self.commands(parameters):
            for _ in range(10):
                dq = d.qvel[r.vids].copy()
                tau = np.clip(
                    KP * (target - d.qpos[r.qids]) - KD * dq, -r.effort, r.effort
                )
                tau[(abs(dq) >= r.speed) & (tau * dq > 0)] = 0
                d.qfrc_applied[:] = 0
                d.qfrc_applied[r.vids] = tau
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                lowest = min(
                    lowest, min(clearance(m, d, g) for gs in r.calf_geoms for g in gs)
                )
                joint_excess = max(
                    joint_excess,
                    float(
                        np.maximum(
                            r.bounds[:, 0] - d.qpos[r.qids],
                            d.qpos[r.qids] - r.bounds[:, 1],
                        ).max()
                    ),
                )
                speed_excess = max(
                    speed_excess, float((abs(d.qvel[r.vids]) / r.speed - 1).max())
                )
                for c in d.contact:
                    if c.dist > 0:
                        continue
                    b1, b2 = map(int, m.geom_bodyid[[c.geom1, c.geom2]])
                    if (b1 and b2) or any(b and b not in r.feet for b in (b1, b2)):
                        penetration = max(penetration, -float(c.dist))
            # Clearance buffer is sampled at control rate for optimization;
            # actual contact checks above and independent replay remain 500 Hz.
            lowest = min(
                lowest, min(clearance(m, d, g) for gs in r.calf_geoms for g in gs)
            )
        delta = np.empty(18)
        mujoco.mj_differentiatePos(m, delta, 1, self.goal_q, d.qpos)
        velocity = d.qvel - self.goal_v
        residual = np.r_[
            np.r_[np.zeros(2), delta[2:]]
            / np.r_[[0.001] * 3, [0.002] * 3, [0.003] * 12],
            velocity / np.r_[[0.002] * 3, [0.002] * 3, [0.005] * 12],
            max(0.02 - lowest, 0) / 0.002,
            penetration / 0.0001,
            max(joint_excess, 0) / 0.001,
            max(speed_excess, 0) / 0.001,
        ]
        return residual, dict(
            pose_error=delta.tolist(),
            velocity_error=velocity.tolist(),
            minimum_calf_clearance_m=float(lowest),
            penetration_m=float(penetration),
            joint_limit_excess_rad=max(joint_excess, 0),
            speed_excess_ratio=max(speed_excess, 0),
        )

    def optimize(self, evaluations=30, seconds=180, initial=None, checkpoint=None):
        if evaluations < 1 or seconds <= 0:
            raise ValueError("Invalid solver budget")
        p = (
            np.zeros(self.parameter_count)
            if initial is None
            else np.asarray(initial, dtype=float)
        )
        if p.shape == (24,):
            p = np.r_[p, np.zeros(12)]
        started = time.monotonic()
        best = {"cost": np.inf}
        history = []

        class Budget(Exception):
            pass

        def fun(x):
            if time.monotonic() - started > seconds and np.isfinite(best["cost"]):
                raise Budget
            residual, report = self.simulate(x)
            self.count += 1
            cost = float(residual @ residual / 2)
            if cost < best["cost"]:
                best.update(cost=cost, parameters=x.copy(), report=report)
                if checkpoint is not None:
                    import json

                    Path(checkpoint).write_text(
                        json.dumps(
                            dict(
                                cost=cost,
                                parameters=x.tolist(),
                                report=report,
                                evaluations=self.count,
                                elapsed_s=time.monotonic() - started,
                            ),
                            indent=2,
                        )
                    )
                history.append(
                    dict(
                        evaluation=self.count,
                        cost=cost,
                        elapsed_s=time.monotonic() - started,
                        **report,
                    )
                )
            return residual

        def jac(x):
            base = fun(x)
            columns = []
            for i in range(self.parameter_count):
                delta = 1e-4 if x[i] + 1e-4 <= 0.3 else -1e-4
                point = x.copy()
                point[i] += delta
                columns.append((fun(point) - base) / delta)
            return np.array(columns).T

        try:
            result = least_squares(
                fun,
                p,
                jac=jac,
                bounds=(-0.3, 0.3),
                max_nfev=evaluations,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            message = str(result.message)
        except Budget:
            message = "Software time budget reached"
        return best, history, message


def applied_reference(reference, bounds, speed, previous=None):
    target = (
        reference["initial_target"].copy()
        if previous is None
        else np.asarray(previous).copy()
    )
    commands = []
    # Whole twelve-joint sequence; no sagittal averaging or low-knot projection.
    for action in reference["actions"]:
        target = wire_target(
            np.clip(
                reference["nominal"] + 2 * action,
                reference["nominal"] - 2,
                reference["nominal"] + 2,
            ),
            target,
            bounds,
            speed,
        )
        commands.append(target)
    return np.asarray(commands)


def evaluate(
    reference,
    preparation=None,
    parameters=None,
    seed=301000,
    perturbed=False,
    wire=True,
    policy=None,
    sensor_noise=0.0,
    presentation=None,
    source_start_step=0,
    skip_preparation=False,
    adapter_dry_run=False,
):
    if adapter_dry_run and (preparation is not None or policy is None):
        raise ValueError("Adapter dry run requires the standing tracking policy")
    if not 0 <= source_start_step < len(reference["actions"]) - 1:
        raise ValueError("Invalid reference phase initialization")
    r = Replay(build_model(), reference)
    preparation_steps = 0
    if preparation is not None:
        mujoco.mj_resetData(r.m, r.d)
        r.d.qpos[:] = preparation.start
        r.d.qvel[:] = 0
        r.target = preparation.initial_target.copy()
        prep_commands = preparation.commands(
            np.zeros(24) if parameters is None else parameters
        )
        if skip_preparation:
            prep_commands = np.empty((0, 12))
            # Match Cornell INTERMEDIATE's last published encoder-pose hold.
            r.target = preparation.start[r.qids].astype(np.float32).astype(float)
        preparation_steps = len(prep_commands)
    else:
        prep_commands = np.empty((0, 12))
    rng = np.random.default_rng(seed)
    if perturbed:
        r.d.qpos[r.qids] += rng.uniform(-0.003, 0.003, 12)
        r.d.qvel[r.vids] += rng.uniform(-0.03, 0.03, 12)
    mujoco.mj_forward(r.m, r.d)
    r.anchor = r.d.geom_xpos[r.foot_geoms[:2], :2].copy()
    if wire:
        previous = prep_commands[-1] if len(prep_commands) else r.target
        commands = np.vstack(
            (
                prep_commands,
                applied_reference(
                    {**reference, "actions": reference["actions"][source_start_step:]},
                    r.bounds,
                    r.speed,
                    previous,
                ),
            )
        )
    else:
        target = r.target.copy()
        rows = []
        for action in reference["actions"]:
            target += np.clip(
                np.clip(reference["nominal"] + 2 * action, *r.bounds.T) - target,
                -0.7 * r.speed * CONTROL_DT,
                0.7 * r.speed * CONTROL_DT,
            )
            rows.append(target.copy())
        commands = np.vstack((prep_commands, rows))
    if policy is not None:
        # Same eight-second control allowance as CapturePolicy evaluation.
        # After reference step 319 the actor clamps its reference, not feedback.
        tail = max(0, 400 - (len(reference["actions"]) - source_start_step))
        commands = np.vstack((commands, np.repeat(commands[-1:], tail, axis=0)))
    first_illegal = None
    min_calf = np.full(4, np.inf)
    max_joint = 0.0
    max_speed = 0.0
    torque_curve_excess = 0.0
    handoff = None
    protocol = None
    parity = dict(observation=0.0, action=0.0, target=0.0)
    if adapter_dry_run:
        from types import SimpleNamespace as NS
        import threading
        import torch
        from go2_deploy.custom_implementation import go2trajectory_lqr as protocol

        r.target = r.target.astype(np.float32).astype(float)
        runtime = dict(
            nominal=reference["nominal"],
            bounds=r.bounds,
            initial_q=r.d.qpos.copy(),
            initial_v=r.d.qvel.copy(),
            calibration=dict(
                calibrated=True,
                source="SYNTHETIC SIMULATION ONLY",
                imu=dict(
                    body_from_imu=np.eye(3),
                    world_alignment=np.eye(3),
                    gyro_bias_imu=np.array([0, -0.009, 0]),
                ),
            ),
        )
        cfg = NS(
            custom_runtime=runtime,
            default_dof_pos=torch.tensor(reference["nominal"], dtype=torch.float32),
        )
        controller = NS(
            cfg=cfg,
            _state=NS(name="INTERMEDIATE"),
            kp_mult=1.0,
            kd_mult=1.0,
            last_action=torch.zeros(2, 12),
            last_lowstate_msg_lock=threading.Lock(),
            **protocol.get_custom_buffers_dict(),
        )
        protocol.initialize_controller(controller)
        initial_command = NS(
            motor_cmd=[NS(q=float(x)) for x in r.target[protocol.ORDER]]
        )
        protocol.on_command_sent(controller, initial_command, True, float(r.d.time))
        controller._state = NS(name="CUSTOM_CTRL")
        scripted = torch.jit.script(policy)
    orientation_errors = []
    front_path = np.zeros(2)
    previous_front = r.d.geom_xpos[r.foot_geoms[:2]].copy()
    min_rear_capture = np.inf
    view = presentation(r) if presentation is not None else None
    cancelled = False
    try:
        sensor_rng = np.random.default_rng(seed + 8000000)
        for k, target in enumerate(commands):
            wall = time.monotonic()
            if view is not None and not view.sync():
                cancelled = True
                break
            if policy is not None and k >= preparation_steps:
                import torch

                rotation = r.d.xmat[r.base].reshape(3, 3)
                observed = np.r_[
                    rotation.T @ [0, 0, -1],
                    r.d.qvel[3:6] + sensor_rng.normal(0, sensor_noise, 3),
                    r.d.qpos[r.qids] - reference["nominal"],
                    r.d.qvel[r.vids] / r.speed,
                    (r.target - reference["nominal"]) / 2,
                    (k - preparation_steps + source_start_step) / 400,
                ].astype(np.float32)
                with torch.no_grad():
                    action = policy(torch.from_numpy(observed)).numpy()
                target = wire_target(
                    reference["nominal"] + 2 * action, r.target, r.bounds, r.speed
                )
            if protocol is not None:
                raw = NS(
                    tick=1000 + round(float(r.d.time) * 1000),
                    imu_state=NS(
                        quaternion=r.d.qpos[3:7].astype(np.float32),
                        gyroscope=(observed[3:6].astype(float) + [0, -0.009, 0]).astype(
                            np.float32
                        ),
                    ),
                    motor_state=[
                        NS(q=float(q), dq=float(v))
                        for q, v in zip(
                            r.d.qpos[r.qids][protocol.ORDER].astype(np.float32),
                            r.d.qvel[r.vids][protocol.ORDER].astype(np.float32),
                        )
                    ],
                )
                # Independent expected pieces from the SAME float32 wire inputs.
                # Telemetry quantization is a changed sensor stream, not code error.
                matrix = np.empty(9)
                qwire = raw.imu_state.quaternion.astype(float)
                mujoco.mju_quat2Mat(matrix, qwire / np.linalg.norm(qwire))
                observed[:3] = matrix.reshape(3, 3).T @ [0, 0, -1]
                observed[3:6] = raw.imu_state.gyroscope.astype(float) - [0, -0.009, 0]
                qwire = np.array([m.q for m in raw.motor_state])[protocol.ORDER]
                dqwire = np.array([m.dq for m in raw.motor_state])[protocol.ORDER]
                observed[6:18] = qwire - reference["nominal"]
                observed[18:30] = dqwire / r.speed
                with torch.no_grad():
                    action = policy(torch.from_numpy(observed)).numpy()
                target = wire_target(
                    reference["nominal"] + 2 * action, r.target, r.bounds, r.speed
                )
                with controller.last_lowstate_msg_lock:
                    controller.last_lowstate_msg = raw
                    protocol.on_lowstate_msg(controller, raw, float(r.d.time))
                controller.lowstate_clock = lambda: float(r.d.time)
                protocol.update_custom_buffers(controller, float(r.d.time))
                protocol_obs = np.concatenate(
                    [
                        protocol.get_custom_obs_pieces()[key](controller, raw).numpy()
                        for key in protocol.SLICES
                    ]
                )
                parity["observation"] = max(
                    parity["observation"], float(abs(protocol_obs - observed).max())
                )
                with torch.no_grad():
                    protocol_action = scripted(torch.from_numpy(protocol_obs)).numpy()
                parity["action"] = max(
                    parity["action"], float(abs(protocol_action - action).max())
                )
                applied = protocol.bound_command(
                    controller, torch.from_numpy(protocol_action) * 2
                ).numpy()
                parity["target"] = max(
                    parity["target"], float(abs(applied - target).max())
                )
                target = applied
                # Software command sink confirms the array actually applied below.
                command = NS(motor_cmd=[NS(q=float(x)) for x in target[protocol.ORDER]])
                protocol.on_command_sent(controller, command, True, float(r.d.time))
            r.target = target.copy()
            for _ in range(10):
                velocity = r.d.qvel[r.vids].copy()
                torque = np.clip(
                    KP * (target - r.d.qpos[r.qids]) - KD * velocity,
                    -r.effort,
                    r.effort,
                )
                torque[(abs(velocity) >= r.speed) & (torque * velocity > 0)] = 0
                torque_curve_excess = max(
                    torque_curve_excess,
                    float(
                        np.maximum(
                            abs(torque)
                            - r.effort * np.clip(1 - abs(velocity) / r.speed, 0, 1),
                            0,
                        ).max()
                    ),
                )
                r.d.qfrc_applied[:] = 0
                r.d.qfrc_applied[r.vids] = torque
                mujoco.mj_step(r.m, r.d)
                mujoco.mj_forward(r.m, r.d)
                r.measure(velocity)
                from gym.envs.go2.handstand_replay import TARGET_ROTATION

                rotation = r.d.xmat[r.base].reshape(3, 3)
                orientation_errors.append(
                    float(
                        np.arccos(
                            np.clip(
                                (np.trace(TARGET_ROTATION.T @ rotation) - 1) / 2, -1, 1
                            )
                        )
                    )
                )
                current_front = r.d.geom_xpos[r.foot_geoms[:2]].copy()
                front_path += np.linalg.norm(current_front - previous_front, axis=1)
                previous_front = current_front
                if r.captured:
                    min_rear_capture = min(
                        min_rear_capture,
                        float(r.d.geom_xpos[r.foot_geoms[2:], 2].min() - 0.022),
                    )
                if view is not None:
                    view.capture()
                min_calf = np.minimum(
                    min_calf,
                    [
                        min(
                            clearance(r.m, r.d, g)
                            for g in np.flatnonzero(
                                r.m.geom_bodyid == r.m.body(leg + "_calf").id
                            )
                        )
                        for leg in ("FL", "FR", "RL", "RR")
                    ],
                )
                max_joint = max(
                    max_joint,
                    float(
                        np.maximum(
                            r.bounds[:, 0] - r.d.qpos[r.qids],
                            r.d.qpos[r.qids] - r.bounds[:, 1],
                        ).max()
                    ),
                )
                max_speed = max(
                    max_speed, float((abs(r.d.qvel[r.vids]) / r.speed).max())
                )
                if r.illegal_contacts and first_illegal is None:
                    first_illegal = float(r.d.time)
            if k + 1 == preparation_steps:
                e = np.empty(18)
                mujoco.mj_differentiatePos(r.m, e, 1, preparation.goal_q, r.d.qpos)
                handoff = dict(
                    pose_error=e.tolist(),
                    velocity_error=(r.d.qvel - preparation.goal_v).tolist(),
                )
            if view is not None and view.viewer:
                time.sleep(max(0, CONTROL_DT - (time.monotonic() - wall)))
            if not r.safe or (r.best_hold >= 5 and r.takeoff):
                break
        if view is not None:
            view.capture(force=True)
    finally:
        if view is not None:
            view.close()
    return dict(
        seed=seed,
        adapter_dry_run=adapter_dry_run,
        maximum_adapter_parity_errors=parity if adapter_dry_run else None,
        orientation_rms_rad=float(np.sqrt(np.mean(np.square(orientation_errors))))
        if orientation_errors
        else None,
        front_foot_path_length_m=front_path.tolist(),
        minimum_rear_clearance_after_capture_m=float(min_rear_capture)
        if np.isfinite(min_rear_capture)
        else None,
        failure_reason=None
        if r.safe and r.takeoff and r.best_hold >= 5 and not cancelled
        else "cancelled"
        if cancelled
        else "illegal_contact"
        if first_illegal is not None
        else "joint_limit"
        if max_joint > 0.001
        else "overspeed"
        if max_speed > 1.001
        else "insufficient_strict_hold",
        start="crouch" if preparation else "standing",
        wire_float32=wire,
        source_start_step=source_start_step,
        skip_preparation=skip_preparation,
        complete_success=bool(
            r.safe and r.takeoff and r.captured and r.best_hold >= 5 and not cancelled
        ),
        cancelled=cancelled,
        strict_hold_seconds=r.best_hold,
        certified_takeoff=r.takeoff,
        front_capture=r.captured,
        illegal_contacts=r.illegal_contacts,
        first_illegal_contact_s=first_illegal,
        minimum_calf_clearance_m=[
            float(x) if np.isfinite(x) else None for x in min_calf
        ],
        max_joint_limit_excess_rad=max(max_joint, 0),
        max_speed_ratio=max_speed,
        max_provisional_torque_curve_excess_Nm=torque_curve_excess,
        seconds=float(r.d.time),
        handoff=handoff,
    )
