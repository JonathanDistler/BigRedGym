"""Passive, physics-rate evidence of real impulses; never actuates the robot."""

import numpy as np
from experiments.morphology.learning.takeoff import TakeoffTracker


class Corrections:
    def __init__(self, dt=0.002):
        self.dt = dt
        self.tracker = TakeoffTracker(dt)
        self.armed = True
        self.loaded_age = 0.0
        self.origin = None
        self.pushes = []
        self.front_events = []
        self.front_air = np.zeros(2)
        self.front_loaded = np.zeros(2, bool)
        self.above = np.zeros(3)
        self.last_time = None
        self.front_J = self.rear_J = self.work = 0.0
        self.peak = 0.0
        self.rear_returns = 0
        self.was_rear_air = False

    def observe(self, e, row, s):
        dt = self.dt
        feet = row["feet"]
        loaded = row["loads"][:2] > 0.05 * e.weight
        for i in range(2):
            if loaded[i] and not self.front_loaded[i] and self.front_air[i] >= 0.02:
                self.front_events.append(
                    dict(
                        time=float(e.d.time),
                        leg=("FL", "FR")[i],
                        pitch=float(s["pitch"]),
                        L=float(s["L_com"]),
                    )
                )
            self.front_air[i] = 0.0 if loaded[i] else self.front_air[i] + dt
        self.front_loaded = loaded.copy()
        rear_loaded = bool((row["loads"][2:] > 0.08 * e.weight).all())
        self.loaded_age = self.loaded_age + dt if rear_loaded else 0.0
        rear_air = bool(
            (row["loads"][2:] < 0.02 * e.weight).all() and row["rear_clearance"] > 0.015
        )
        if rear_loaded and self.was_rear_air:
            self.rear_returns += 1
            self.was_rear_air = False
        self.was_rear_air |= rear_air
        if not self.armed and self.loaded_age >= 0.04:
            self.tracker = TakeoffTracker(dt)
            self.armed = True
            self.origin = feet[:2, :2].copy()
        if self.origin is None:
            self.origin = feet[:2, :2].copy()
        knee = [e.names.index(n) for n in ("RL_calf_joint", "RR_calf_joint")]
        power = e.d.qfrc_applied[e.vids] * e.d.qvel[e.vids]
        legpower = [
            sum(max(power[j], 0) for j, n in enumerate(e.names) if n.startswith(leg))
            for leg in ("RL", "RR")
        ]
        enriched = dict(
            row,
            weight=e.weight,
            rear_leg_power=legpower,
            front_drift=float(np.linalg.norm(feet[:2, :2] - self.origin, axis=1).max()),
        )
        if self.armed:
            self.tracker.observe(
                enriched,
                float(e.d.time),
                e.d.qvel[e.vids[knee]],
                valid=bool(s["prefix_safe"]),
            )
            if self.tracker.event:
                self.pushes.append(
                    dict(
                        time=float(e.d.time),
                        pitch=float(s["pitch"]),
                        L=float(s["L_com"]),
                        energy=float(s["mechanical_energy"]),
                        work_min=self.tracker.certified_work_min,
                        extension_min=self.tracker.certified_extension_min,
                        excess_impulse_min=self.tracker.certified_impulse_min,
                        front_J=self.front_J,
                        rear_J=self.rear_J,
                    )
                )
                self.armed = False
        self.front_J += s["front_moment_com"] * dt
        self.rear_J += s["rear_moment_com"] * dt
        self.work += s["actuator_power"] * dt
        if s["prefix_safe"]:
            self.peak = max(self.peak, float(s["pitch"]))
            self.above += (
                dt * (s["pitch"] >= np.deg2rad([45, 60, 75])) * rear_air * loaded.any()
            )
        return dict(
            genuine_pushes=len(self.pushes),
            rear_returns=self.rear_returns,
            counted_front_recontacts=len(self.front_events),
            seconds_above45=float(self.above[0]),
            seconds_above60=float(self.above[1]),
            seconds_above75=float(self.above[2]),
        )


def joint_sample(e):
    q = e.d.qpos[e.qids]
    v = e.d.qvel[e.vids]
    tau = e.d.qfrc_applied[e.vids]
    raw = e.cfg.kp * (e.target - q) - np.asarray(e.cfg.kd) * v
    out = dict(
        saturated_fraction=float(np.mean(abs(tau) >= 0.999 * e.effort)),
        tracking_rmse=float(np.sqrt(np.mean((e.target - q) ** 2))),
    )
    for i, name in enumerate(e.names):
        out.update(
            {
                name + "_q": float(q[i]),
                name + "_v": float(v[i]),
                name + "_tau": float(tau[i]),
                name + "_raw_post": float(raw[i]),
                name + "_target": float(e.target[i]),
            }
        )
    for i, leg in enumerate(("FL", "FR", "RL", "RR")):
        for axis, value in zip("xyz", e.d.geom_xpos[e.foot_geoms[i]]):
            out[f"{leg}_world_{axis}"] = float(value)
        pos = e.d.xmat[e.base].reshape(3, 3).T @ (
            e.d.geom_xpos[e.foot_geoms[i]] - e.d.xpos[e.base]
        )
        for axis, value in zip("xyz", pos):
            out[f"{leg}_body_{axis}"] = float(value)
        out[leg + "_load"] = float(e.row["loads"][i]) if hasattr(e, "row") else 0.0
    return out
