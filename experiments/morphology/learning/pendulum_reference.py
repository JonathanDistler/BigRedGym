"""Upright-angle reduced models and strict controlled-rotation validation.

theta=atan2(COM_x-pivot_x, COM_z-pivot_z), from UPWARD vertical.
Positive angular velocity/moment is world +y. A COM behind support has negative
theta; gravity decelerates its positive approach to upright. Body pitch differs.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from experiments.morphology.learning.rotation import RotationEnvironment


def acceleration(
    theta,
    omega,
    length,
    length_rate,
    mass,
    torque=0.0,
    inertia_com=0.0,
    inertia_rate=0.0,
    pivot_term=0.0,
    gravity=9.81,
):
    if length <= 0 or mass <= 0 or inertia_com < 0:
        raise ValueError("Positive length/mass and nonnegative inertia required")
    inertia = mass * length**2 + inertia_com
    rate = 2 * mass * length * length_rate + inertia_rate
    return (
        mass * gravity * length * np.sin(theta) + torque + pivot_term - rate * omega
    ) / inertia


def quintic(start, velocity, target, duration, times):
    """Continuous position/velocity/acceleration; terminal velocity/accel zero."""
    if duration <= 0:
        raise ValueError("Positive duration required")
    coefficients = np.zeros(6)
    coefficients[:3] = [start, velocity, 0.0]
    T = duration
    coefficients[3:] = np.linalg.solve(
        [
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ],
        [target - start - velocity * T, -velocity, 0.0],
    )
    t = np.clip(times, 0, T)
    position = np.polynomial.polynomial.polyval(t, coefficients)
    speed = np.polynomial.polynomial.polyval(t, np.arange(1, 6) * coefficients[1:])
    accel = np.polynomial.polynomial.polyval(
        t, np.arange(1, 5) * np.arange(2, 6) * coefficients[2:]
    )
    return position, speed, accel


class PendulumInstrument(RotationEnvironment):
    """Adds material-foot geometry/velocity; never changes actions or dynamics."""

    def measure(self):
        previous = getattr(self, "rotation_sample", None)
        row = super().measure()
        sample = getattr(self, "rotation_sample", None)
        if sample is None or sample is previous:
            return row
        for j, leg in enumerate(("FL", "FR", "RL", "RR")):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                self.m,
                self.d,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(self.foot_geoms[j]),
                velocity,
                0,
            )
            for axis, k in zip("xyz", range(3)):
                sample[f"{leg}_position_{axis}"] = float(row["feet"][j, k])
                sample[f"{leg}_velocity_{axis}"] = float(velocity[k + 3])
            sample[f"{leg}_load"] = float(row["loads"][j])
        sample["max_speed_ratio"] = float(
            np.max(abs(self.d.qvel[self.vids]) / self.speed)
        )
        sample["torque_ratio"] = float(
            np.max(abs(self.d.qfrc_applied[self.vids]) / self.effort)
        )
        sample["joint_violation"] = float(
            max(
                0.0,
                np.max(self.bounds[:, 0] - self.d.qpos[self.qids]),
                np.max(self.d.qpos[self.qids] - self.bounds[:, 1]),
            )
        )
        # Parent copied the sample to its trace before these additional fields.
        if self.record_rotation_trace:
            self.rotation_trace[-1].update(sample)
        return row


@dataclass(frozen=True)
class ReferenceGate:
    target_degrees: float = 45.0
    band_degrees: float = 5.0
    max_pitch_speed: float = 0.5
    hold_seconds: float = 0.3
    max_contact_gap_seconds: float = 0.02
    max_slip_distance: float = 0.02
    max_front_drift: float = 0.04
    required_fraction: float = 0.8

    def dictionary(self):
        return asdict(self)


def score_trajectory(trace, weight, gate=None):
    gate = gate or ReferenceGate()
    if not trace:
        return dict(success=False, peak_pitch=0.0, hold_seconds=0.0, reason="empty")
    pitch = np.array([r["pitch"] for r in trace])
    speed = np.array([r["pitch_velocity"] for r in trace])
    dt = (
        float(np.median(np.diff([r["time"] for r in trace])))
        if len(trace) > 1
        else 0.002
    )
    both = np.array(
        [r["FL_load"] > 0.05 * weight and r["FR_load"] > 0.05 * weight for r in trace]
    )
    slip = np.array(
        [
            max(
                np.hypot(r["FL_velocity_x"], r["FL_velocity_y"]),
                np.hypot(r["FR_velocity_x"], r["FR_velocity_y"]),
            )
            for r in trace
        ]
    )
    slip_distance = float(np.sum(slip * both) * dt)
    unsafe = any(
        r["nonfoot_collision"]
        or r["self_collision"]
        or r["joint_violation"] > 0.001
        or r["max_speed_ratio"] > 1.001
        or r["torque_ratio"] > 1.00001
        for r in trace
    )
    gap = longest_gap = hold = longest_hold = 0.0
    reached = False
    for p, w, supported in zip(pitch, speed, both):
        gap = 0.0 if supported else gap + dt
        longest_gap = max(longest_gap, gap)
        reached |= bool(
            p >= np.deg2rad(gate.target_degrees)
            and abs(w) <= gate.max_pitch_speed
            and supported
        )
        within = (
            abs(p - np.deg2rad(gate.target_degrees)) <= np.deg2rad(gate.band_degrees)
            and abs(w) <= gate.max_pitch_speed
            and supported
        )
        hold = hold + dt if within else 0.0
        longest_hold = max(longest_hold, hold)
    drift = max(r["front_drift"] for r in trace)
    success = bool(
        reached
        and longest_hold >= gate.hold_seconds - 1e-9
        and not unsafe
        and longest_gap <= gate.max_contact_gap_seconds + 1e-9
        and slip_distance <= gate.max_slip_distance
        and drift <= gate.max_front_drift
    )
    reasons = []
    if not reached:
        reasons.append("target_or_velocity")
    if longest_hold < gate.hold_seconds - 1e-9:
        reasons.append("hold_duration")
    if unsafe:
        reasons.append("collision_or_native_state_limit")
    if longest_gap > gate.max_contact_gap_seconds + 1e-9:
        reasons.append("front_contact_gap")
    if slip_distance > gate.max_slip_distance or drift > gate.max_front_drift:
        reasons.append("slip")
    near = np.abs(pitch - np.deg2rad(gate.target_degrees)) < np.deg2rad(
        gate.band_degrees
    )
    return dict(
        success=success,
        peak_pitch=float(pitch.max()),
        hold_seconds=longest_hold,
        reached_target_low_speed=reached,
        contact_gap_seconds=longest_gap,
        slip_distance=slip_distance,
        front_drift=float(drift),
        unsafe=bool(unsafe),
        terminal_pitch=float(pitch[-1]),
        terminal_speed=float(speed[-1]),
        near_target_speed_rms=float(np.sqrt(np.mean(speed[near] ** 2)))
        if near.any()
        else 10.0,
        reason=";".join(reasons),
    )
