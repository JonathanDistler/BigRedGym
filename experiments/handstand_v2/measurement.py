"""Physics-rate event certificates, distinct from continuous training feedback."""

import math
import numpy as np

# Fixed stage target per training block: no hidden randomized target command.
STAGES = [
    ("0", 30, 0),
    ("1A", 35, 0),
    ("1B", 40, 0),
    ("1C", 45, 0),
    ("2", 45, 0.15),
    ("3", 60, 0.15),
    ("4", 85, 0.30),
    ("5", 90, 5.0),
]


class Measurement:
    def __init__(self):
        self.gap = self.max_gap = self.slip = self.loaded_peak = self.peak = 0.0
        self.certified_peak = self.catch = self.best_catch = 0.0
        self.previous = None
        self.failed = False
        self.first_invalid = None
        self.crossings = {}
        self.dynamic_crossings = {}
        self.flight_displacement = 0.0
        self.recovered = False
        self.recovery_timer = self.best_front_recovery = 0.0
        self.handstand_timer = self.best_handstand = 0.0
        self.had_gap = False
        self.peak_time = 0.0
        self.touchdown_speed = 0.0

    def observe(self, row, time, dt, safe, takeoff, target):
        loaded = np.asarray(row["loads"][:2]) > 0.05 * row["weight"]
        feet = row["feet"][:2].copy()
        if self.previous is not None:
            oldfeet, oldloaded = self.previous
            displacement = np.linalg.norm(feet[:, :2] - oldfeet[:, :2], axis=1)
            self.flight_displacement += float(
                np.max(np.where(~loaded & ~oldloaded, displacement, 0))
            )
            self.slip += float(row.get("ground_slip_speed", 0.0)) * dt
            landing = loaded & ~oldloaded
            if takeoff and landing.any():
                self.touchdown_speed = max(
                    self.touchdown_speed,
                    float(
                        np.max(
                            np.where(landing, abs(feet[:, 2] - oldfeet[:, 2]) / dt, 0)
                        )
                    ),
                )
        self.previous = feet, loaded.copy()
        if row["pitch"] > self.peak:
            self.peak, self.peak_time = row["pitch"], time
        for degrees in (30, 35, 40, 45, 60, 75, 85, 90):
            if row["pitch"] >= math.radians(degrees) and row["pitch_velocity"] >= 0:
                self.dynamic_crossings.setdefault(
                    degrees, dict(time=time, loaded=bool(loaded.all()), safe=bool(safe))
                )
        if loaded.all():
            self.loaded_peak = max(self.loaded_peak, row["pitch"])
        if not safe:
            if self.first_invalid is None:
                self.first_invalid = dict(
                    time=time,
                    bodies=row["illegal_bodies"],
                    self_collision=bool(row["self_contacts"]),
                    self_bodies=row.get("self_bodies", ""),
                    speed_ratio=row.get("speed_ratio", 0),
                    joint_violation=row.get("joint_violation", 0),
                    torque_ratio=row.get("torque_ratio", 0),
                )
            self.failed = True
        if takeoff:
            self.gap = 0.0 if loaded.all() else self.gap + dt
            self.max_gap = max(self.max_gap, self.gap)
            self.had_gap |= self.gap > 0
        # Historical gaps remain diagnostic across new support intervals.
        certified = (
            takeoff
            and loaded.all()
            and not self.failed
            and self.slip <= 0.02
            and row["rear_clearance"] >= 0.015
            and row["front_drift"] <= 0.04
            and sum(row["loads"][2:]) < 0.04 * row["weight"]
            and self.touchdown_speed <= 1.0
        )
        if certified:
            self.certified_peak = max(self.certified_peak, row["pitch"])
            for degrees in (30, 35, 40, 45, 60, 75, 85, 90):
                if (
                    row["pitch"] >= math.radians(degrees)
                    and -0.5 <= row["pitch_velocity"] <= 2.0
                ):
                    self.crossings.setdefault(degrees, time)
        caught = (
            certified
            and target in self.crossings
            and abs(math.degrees(row["pitch"]) - target) < 7.5
            and abs(row["pitch_velocity"]) < (0.5 if target >= 85 else 0.8)
            and np.linalg.norm(row["com_velocity"]) < (0.25 if target >= 85 else 0.75)
        )
        self.catch = self.catch + dt if caught else 0.0
        self.best_catch = max(self.best_catch, self.catch)
        recovery = (
            certified
            and self.had_gap
            and abs(row["pitch_velocity"]) < 2.0
            and np.linalg.norm(row["com_velocity"]) < 0.75
        )
        self.recovery_timer = self.recovery_timer + dt if recovery else 0.0
        self.best_front_recovery = max(self.best_front_recovery, self.recovery_timer)
        self.recovered |= self.recovery_timer >= 0.02
        upright = (
            certified
            and abs(math.degrees(row["pitch"]) - 90) <= 10
            and abs(row["pitch_velocity"]) < 0.5
            and np.linalg.norm(row["com_velocity"]) < 0.25
        )
        self.handstand_timer = self.handstand_timer + dt if upright else 0.0
        self.best_handstand = max(self.best_handstand, self.handstand_timer)
        return certified


def shaping(row, safe, takeoff, target, dt, slip_speed=0.0, moment=0.0):
    """Dense supported income, separate from strict success certification."""
    zero = dict(
        progress=0.0, velocity=0.0, clearance=0.0, loading=0.0, slip=0.0, moment=0.0
    )
    if not safe or not takeoff or not row["front_support"]:
        return zero
    pitch = math.degrees(row["pitch"])
    progress = np.clip((pitch - 25) / max(target - 25, 1), 0, 1)
    desired = np.clip(math.radians(target - pitch) * 3, 0, 2)
    # Smoothly reward approach and braking; full model remains authoritative.
    return dict(
        progress=float(2 * progress * dt),
        velocity=float(
            0.3 * np.exp(-(((row["pitch_velocity"] - desired) / 1.5) ** 2)) * dt
        ),
        clearance=float(0.2 * np.clip(row["rear_clearance"] / 0.08, 0, 1) * dt),
        loading=0.1 * dt,
        slip=-min(2.0, slip_speed) * dt,
        moment=-0.01 * min(abs(moment), 20) * dt,
    )
