"""Remote-activated command-path diagnostic, independent of handstand calibration.

Default is hold-only. An explicitly configured small joint excursion returns to
the captured hold target, then holds. No IMU/COM observer or trained actor is used.
This verifies commanded/publication/measured positions, not motor protections.
"""

import threading
import numpy as np
import torch

from go2_deploy.custom_implementation.custom_config import CustomConfig
from go2_deploy.custom_implementation.go2handstand import (
    JOINTS,
    PERMUTATION,
    SPEED,
    build_fk_model,
    finite,
)


class CommandCheck(CustomConfig):
    fixed_gains = require_intermediate = True
    task_name = "go2commandcheck"
    obs_vector = ["command_joint_position"]
    custom_obs_sizes = {"command_joint_position": 12}
    default_dof_pos = [0.0] * 12

    class DeployScaling:
        command_joint_position = dof_pos_target = 1.0

    def __init__(self, joint=0, offset=0.0):
        if not isinstance(joint, int) or not 0 <= joint < 12:
            raise ValueError("Diagnostic joint must be a canonical index 0..11")
        if not np.isfinite(offset) or abs(offset) > 0.02:
            raise ValueError("Diagnostic offset must be finite and at most 0.02 rad")
        self.joint, self.offset = joint, float(offset)

    def get_custom_buffers_dict(self):
        return dict(
            check_lock=threading.Lock(),
            check_sample=None,
            check_fault=None,
            check_hold=None,
            check_sent=None,
            check_pending=None,
            check_step=0,
            check_active=False,
            check_tick=None,
            check_last_cycle=None,
        )

    def initialize_controller(self, c):
        model = build_fk_model()
        c.check_bounds = model.jnt_range[[model.joint(n).id for n in JOINTS]].copy()
        if c.cfg.ctrl_freq != 50 or c.cfg.exp_moving_avg:
            raise ValueError("Command diagnostic requires 50 Hz and EMA off")
        log_event = getattr(getattr(c, "csv_logger", None), "log_event", None)
        if log_event is not None:
            log_event(
                "command_check_config",
                {
                    "joint": JOINTS[self.joint],
                    "offset_rad": self.offset,
                    "kp": c.cfg.kp,
                    "kd": c.cfg.kd,
                    "ctrl_freq": c.cfg.ctrl_freq,
                },
            )

    def on_lowstate_msg(self, c, msg, received):
        with c.check_lock:
            try:
                q = finite([m.q for m in msg.motor_state[:12]], (12,))[PERMUTATION]
                dq = finite([m.dq for m in msg.motor_state[:12]], (12,))[PERMUTATION]
                tick = int(msg.tick)
                if not 0 <= tick < 2**32 or not np.isfinite(received):
                    raise ValueError("Invalid diagnostic timestamp")
                if c.check_tick is not None:
                    delta = (tick - c.check_tick) % 2**32
                    if delta == 0:
                        if c.check_sample is not None and (
                            not np.array_equal(q, c.check_sample[1])
                            or not np.array_equal(dq, c.check_sample[2])
                        ):
                            raise ValueError(
                                "Conflicting duplicate diagnostic timestamp"
                            )
                        return  # Frozen ticks never refresh freshness.
                    if delta >= 2**31:
                        raise ValueError("Reversed diagnostic timestamp")
                c.check_tick = tick
                c.check_sample = (float(received), q.copy(), dq.copy())
            except (ValueError, TypeError, AttributeError) as exc:
                c.check_fault = str(exc)

    def update_custom_buffers(self, c, t):
        with c.check_lock:
            sample, fault = c.check_sample, c.check_fault
        now = getattr(c, "lowstate_clock", lambda: t)()
        if fault or sample is None or not 0 <= now - sample[0] <= 0.015:
            raise ValueError(f"Diagnostic needs fresh valid encoders: {fault}")
        q, dq = sample[1:]
        if np.any(q < c.check_bounds[:, 0]) or np.any(q > c.check_bounds[:, 1]):
            raise ValueError("Diagnostic encoders outside model joint bounds")
        if np.any(abs(dq) > SPEED):
            raise ValueError("Diagnostic encoder speed exceeds model limit")
        active = getattr(c._state, "name", "") == "CUSTOM_CTRL"
        if active and not c.check_active:
            if c.check_sent is None:
                raise ValueError("Remote B / INTERMEDIATE is required before A")
            if np.max(abs(q - c.check_sent)) > 0.15 or np.max(abs(dq)) > 0.2:
                raise ValueError(
                    "Diagnostic activation requires stationary target tracking"
                )
            c.check_hold = c.check_sent.copy()
            desired = c.check_hold.copy()
            desired[self.joint] += self.offset
            if np.any(desired < c.check_bounds[:, 0]) or np.any(
                desired > c.check_bounds[:, 1]
            ):
                raise ValueError("Requested diagnostic excursion exceeds joint bounds")
            c.check_step = 0
        if active and c.check_last_cycle is not None and c.check_active:
            if abs(t - c.check_last_cycle - 0.02) > 0.002:
                raise ValueError("Diagnostic cadence differs from 50 Hz")
        c.check_active = active
        c.check_last_cycle = t if active else None

    def get_custom_obs_pieces(self):
        def joints(c, msg):
            with c.check_lock:
                return torch.tensor(c.check_sample[1], dtype=torch.float32)

        return {"command_joint_position": joints}

    def bound_command(self, c, offset, kp_mult=1.0, kd_mult=1.0):
        if kp_mult != 1 or kd_mult != 1:
            raise ValueError("Diagnostic gains must remain at their configured values")
        with c.check_lock:
            sample = c.check_sample
        now = c.lowstate_clock()
        if sample is None or not 0 <= now - sample[0] <= 0.015:
            raise ValueError("Diagnostic state stale before command")
        if c.check_pending is not None:
            raise ValueError("Previous diagnostic publication unconfirmed")
        target = finite(offset, (12,)).copy()
        if c.check_active:
            target = c.check_hold.copy()
            # Two-second out-and-back pulse; no repetition or automatic recovery.
            phase = min(c.check_step / 100, 1.0)
            target[self.joint] += self.offset * np.sin(np.pi * phase) ** 2
        if np.any(target < c.check_bounds[:, 0]) or np.any(
            target > c.check_bounds[:, 1]
        ):
            raise ValueError("Diagnostic command exceeds model bounds")
        if c.check_sent is not None and np.any(
            abs(target - c.check_sent) > 0.7 * SPEED * 0.02
        ):
            raise ValueError("Diagnostic command exceeds target slew bound")
        # The pulse is tiny; validate the actual wire representation as well.
        wire = target.astype(np.float32).astype(float)
        if np.any(wire < c.check_bounds[:, 0]) or np.any(wire > c.check_bounds[:, 1]):
            raise ValueError("Float32 diagnostic command exceeds bounds")
        if c.check_sent is not None and np.any(
            abs(wire - c.check_sent) > 0.7 * SPEED * 0.02
        ):
            raise ValueError("Float32 diagnostic command exceeds target slew bound")
        c.check_pending = wire.copy()
        return torch.tensor(wire, dtype=torch.float64)

    def on_command_sent(self, c, command, accepted, t):
        sent = finite([m.q for m in command.motor_cmd[:12]], (12,))[PERMUTATION]
        if (
            accepted is not True
            or c.check_pending is None
            or not np.array_equal(sent, c.check_pending)
        ):
            raise ValueError("Diagnostic command publication unconfirmed or changed")
        c.check_sent, c.check_pending = sent.copy(), None
        if c.check_active:
            c.check_step += 1

    def runtime_status(self, c):
        with c.check_lock:
            sample = c.check_sample
        return {
            "task": self.task_name,
            "joint": JOINTS[self.joint],
            "excursion_rad": self.offset,
            "published_steps": c.check_step,
            "pulse_complete": c.check_step >= 101,
            "max_tracking_error_rad": float(abs(sample[1] - c.check_sent).max())
            if sample is not None and c.check_sent is not None
            else None,
            "calibration_required": False,
        }


def make_runtime(joint=0, offset=0.0):
    from go2_deploy.deploy_config import DeployConfig

    cfg = DeployConfig(CommandCheck(joint, offset))
    cfg.ctrl_freq, cfg.exp_moving_avg = 50, False
    cfg.command_limits = dict(lin_vel_x=0.0, lin_vel_y=0.0, yaw_vel=0.0)
    cfg.validate()

    def zero_actor(obs):
        return torch.zeros(12, dtype=torch.float32)

    return zero_actor, cfg
