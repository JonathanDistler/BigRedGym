"""Opt-in momentum-profile extension of the existing CPG-free energy task."""

from dataclasses import dataclass
import numpy as np
from experiments.com_energy_phase.core import (
    EnergyConfig,
    EnergyEnvironment,
    Phase,
    gaussian,
)
from experiments.momentum_phase.physics import heading_axis, sample


class ManeuverPhase(Phase):
    """One-shot event phase: the clock alone cannot initiate braking or balancing."""

    def __init__(self, times):
        super().__init__(times, True)
        self.front_loaded = False
        self.omega = 0.0

    def update(self, time, loaded, takeoff, pitch, stable):
        if takeoff and self.takeoff_time is None:
            self.takeoff_time = time
        if self.takeoff_time is None:
            p = float(
                np.interp(time, (0.0, self.times[0], self.times[2]), (0.0, 0.2, 0.399))
            )
            if not self.front_loaded:
                p = min(p, 0.199)
            if not loaded:
                p = min(p, 0.199)
        else:
            p = 0.4 + 0.199 * np.clip(
                (time - self.takeoff_time) / max(self.times[3] - self.times[2], 0.02),
                0,
                1,
            )
            # Start braking only on measured approach, not after an elapsed clock.
            if pitch >= np.deg2rad(60) or (
                pitch >= np.deg2rad(45) and self.omega > 2.5
            ):
                p = max(
                    p,
                    0.6
                    + 0.199 * np.clip((pitch - np.deg2rad(60)) / np.deg2rad(20), 0, 1),
                )
            if pitch >= np.deg2rad(80) and abs(self.omega) < 1.0:
                p = max(p, 0.8)
            if stable:
                p = 1.0
        self.progress = max(self.progress, float(p))


@dataclass
class MomentumConfig(EnergyConfig):
    condition: str = "B"
    momentum_com_cap: float = 1.3
    momentum_support_cap: float = 4.5
    sigma_momentum: float = 1.0
    momentum_weight: float = 1.0
    terminal_weight: float = 4.0
    sigma_omega: float = 0.5
    support_sigma: float = 0.04
    inertia_reference: float = 4.0
    phase_filter: bool = False
    swing_filter_tau: float = 0.005
    terminal_filter_tau: float = 0.04
    phase_enabled: bool = True
    action_tau: float = 0.0
    pumping_weight: float = 0.2  # B/C replace the old pumping proxy below.


def momentum_goal(s, progress, cfg):
    """Deficit-guided support momentum inside a measured, feasible peak envelope."""
    deficit = max(0.0, s["energy_target"] - s["potential"])
    energy_goal = np.sqrt(2 * max(s["I_support"], 1e-9) * deficit)
    pitch = s["pitch"]
    # Positive rotation until approach, then smoothly remove momentum at vertical.
    approach = np.clip((np.pi / 2 - pitch) / np.deg2rad(30), 0, 1)
    if progress < 0.2:
        activation = 0.0
    elif progress < 0.4:
        activation = float(np.clip((progress - 0.2) / 0.16, 0, 1))
    else:
        activation = 1.0
    value = min(cfg.momentum_support_cap, energy_goal) * activation * approach
    return float(value)


def terminal_reward(s, cfg):
    valid = bool(s["target_valid"] and s["calf_clearance"] > 0 and not s["collision"])
    velocity = gaussian(
        s.get("body_angular_speed", abs(s["pitch_velocity"])), cfg.sigma_omega
    )
    location = gaussian(s["support_region_distance"], cfg.support_sigma)
    rate = cfg.terminal_weight * s["r_energy"] * s["r_orientation"] * velocity * valid
    return float(rate), float(location * valid)


class MomentumEnvironment(EnergyEnvironment):
    policy_actor_dim = 61
    policy_critic_dim = 91

    def reset(self, *args, **kwargs):
        self._momentum_prefix_safe = True
        self.momentum_sample = None
        self.pitch_axis = np.array([0.0, 1.0, 0.0])
        self.record_physics = False
        self.physics_trace = []
        self.previous_physics = None
        self.rear_work = self.rear_positive_work = self.contact_work = 0.0
        self.net_impulse = self.rear_impulse = self.useful_impulse = 0.0
        self.balance_residual_com = self.balance_residual_fixed = 0.0
        self.paid_up = self.paid_rotation_ke = self.paid_impulse = 0.0
        self.swing_floor = None
        self.best_support_L = 0.0
        self.physical_hold = self.best_physical_hold = 0.0
        super().reset(*args, **kwargs)
        self.phase = ManeuverPhase(self.cfg.phase_times)
        self.pitch_axis = heading_axis(self.d.xmat[self.base].reshape(3, 3))
        self.fixed_support = np.r_[self.anchor.mean(0), 0.0]
        self.momentum_sample = sample(self, self.row)
        self.previous_physics = self.momentum_sample.copy()
        self.record_physics = True
        return self.observations()

    def features(self):
        s = self.momentum_sample
        if s is None:
            return np.zeros(14, np.float32)
        target = s["mass"] * np.linalg.norm(self.m.opt.gravity) * self.cfg.target_height
        vel = [s[f"com_velocity_{a}"] for a in "xyz"]
        x = [
            s["L_com"] / self.cfg.momentum_com_cap,
            s["L_support"] / self.cfg.momentum_support_cap
            if s["support_valid"]
            else 0.0,
            s["potential"] / max(target, 1),
            s["K_system"] / max(target, 1),
            (target - s["mechanical_energy"]) / max(target, 1),
            s["I_support"] / self.cfg.inertia_reference if s["support_valid"] else 0.0,
            s["com_support_x"] / 0.3,
            s["com_support_y"] / 0.3,
            s["com_height"] / 0.6,
            *vel,
            s["pitch_velocity"] / 3.0,
            float(s["support_valid"]),
        ]
        return np.clip(x, -5, 5).astype(np.float32)

    def observations(self):
        a, c = super().observations()
        extra = self.features()
        return np.r_[a, extra].astype(np.float32), np.r_[c, extra].astype(np.float32)

    def measure(self):
        row = super().measure()
        if not getattr(self, "record_physics", False):
            return row
        if self.d.time <= self.previous_physics["time"] + 1e-10:
            return row
        s = sample(self, row)
        old = self.previous_physics
        dt = s["time"] - old["time"]
        self.phase.front_loaded |= bool(row["loads"][:2].sum() > row["loads"][2:].sum())
        self.phase.omega = s["pitch_velocity"]
        self.rear_work += s["rear_power"] * dt
        self.rear_positive_work += s["rear_positive_power"] * dt
        self.contact_work += (
            sum(s[f"{n}_contact_power"] for n in ("front", "rear", "nonfoot")) * dt
        )
        # Trapezoidal moments retain finite-step/contact-switch residuals.
        impulse = 0.5 * (s["moment_com"] + old["moment_com"]) * dt
        impulse_fixed = 0.5 * (s["moment_fixed"] + old["moment_fixed"]) * dt
        self.net_impulse += impulse
        self.rear_impulse += 0.5 * (s["rear_moment_com"] + old["rear_moment_com"]) * dt
        self.balance_residual_com += s["L_com"] - old["L_com"] - impulse
        self.balance_residual_fixed += s["L_fixed"] - old["L_fixed"] - impulse_fixed
        # Actual H increase is required alongside measured rear-ground moment and work.
        eligible = (
            not s["collision"]
            and s["front_support"]
            and s["rear_loaded"]
            and self.phase.progress >= 0.2
            and self.phase.progress < 0.4
            and s["rear_positive_power"] > 0
            and s["pitch_velocity"] > 0
        )
        useful = min(
            max(s["L_com"] - old["L_com"], 0),
            max(0, min(s["rear_moment_com"], s["moment_com"])) * dt,
        )
        if eligible:
            self.useful_impulse += useful
        moving_valid = bool(s["support_valid"] and old["support_valid"])
        if moving_valid:
            vs = np.array([s[f"support_{a}"] - old[f"support_{a}"] for a in "xyz"]) / dt
            p = np.array([s[f"linear_momentum_{a}"] for a in "xyz"])
            transport = float(np.cross(vs, p) @ self.pitch_axis)
            moving_residual = (
                s["L_support"]
                - old["L_support"]
                - (0.5 * (s["moment_support"] + old["moment_support"]) - transport) * dt
            )
        else:
            transport = moving_residual = 0.0
        # Five-second certification runs at physics rate and rejects every bad sample.
        target = np.array(self.cfg.target_rotation).reshape(3, 3)
        from experiments.com_energy_phase.core import orientation_error

        err = orientation_error(self.d.xmat[self.base].reshape(3, 3), target)
        safe = (
            not s["collision"]
            and not self.nonfoot_collision
            and not self.self_collision
            and np.max(abs(self.d.qvel[self.vids]) / self.speed) <= 1.001
            and (self.d.qpos[self.qids] >= self.bounds[:, 0] - 0.001).all()
            and (self.d.qpos[self.qids] <= self.bounds[:, 1] + 0.001).all()
        )
        self._momentum_prefix_safe &= safe
        stable = (
            self._momentum_prefix_safe
            and self.tracker.flight_valid
            and s["front_support"]
            and err < np.deg2rad(10)
            and abs(s["pitch_velocity"]) < 0.5
            and s["body_angular_speed"] < 0.5
            and s["com_speed"] < 0.25
            and s["support_region_distance"] < 0.04
            and s["foot_drift"] < 0.1
            and s["front_slip_speed"] < 0.05
            and s["calf_clearance"] > 0
        )
        self.physical_hold = self.physical_hold + dt if stable else 0.0
        self.best_physical_hold = max(self.best_physical_hold, self.physical_hold)
        s.update(
            phase=self.phase.progress,
            rear_work=self.rear_work,
            rear_positive_work=self.rear_positive_work,
            contact_work=self.contact_work,
            net_impulse=self.net_impulse,
            rear_impulse=self.rear_impulse,
            useful_impulse=self.useful_impulse,
            momentum_residual_com=self.balance_residual_com,
            momentum_residual_fixed=self.balance_residual_fixed,
            support_transport=transport,
            moving_balance_valid=moving_valid,
            moving_balance_residual=moving_residual,
            takeoff=self.tracker.confirmed,
            flight_valid=self.tracker.flight_valid,
            prefix_safe=self._momentum_prefix_safe,
        )
        for i in range(12):
            s[f"joint_{i}"] = float(self.d.qpos[self.qids[i]])
            s[f"joint_velocity_{i}"] = float(self.d.qvel[self.vids[i]])
            s[f"target_{i}"] = float(self.target[i])
            s[f"torque_{i}"] = float(self.d.qfrc_applied[self.vids[i]])
        self.previous_physics = s.copy()
        self.momentum_sample = s
        self.physics_trace.append(s.copy())
        return row

    def step(self, action):
        self._momentum_prefix_safe = getattr(self, "_momentum_prefix_safe", True)
        if self.cfg.phase_filter:
            p = self.phase.progress
            self.cfg.action_tau = (
                0.0
                if p < 0.4
                else self.cfg.swing_filter_tau
                if p < 0.8
                else self.cfg.terminal_filter_tau
            )
        else:
            self.cfg.action_tau = 0.0
        old_terms = self.terms_sum.copy()
        (a, c), reward, done, info = super().step(action)
        s = self.momentum_sample
        s.update(
            {
                key: self.trace[-1][key]
                for key in (
                    "energy_target",
                    "energy_deficit",
                    "orientation_error",
                    "target_valid",
                    "r_energy",
                    "r_orientation",
                    "r_target",
                )
            }
        )
        s["phase"] = self.phase.progress
        target = momentum_goal(s, self.phase.progress, self.cfg)
        safe = bool(
            self._momentum_prefix_safe and not s["collision"] and s["support_valid"]
        )
        useful = safe and (
            self.tracker.flight_valid
            or (s["rear_loaded"] and s["rear_positive_power"] > 0 and s["pitch"] > 0.03)
        )
        if useful and self.swing_floor is None:
            self.swing_floor = s["potential"]
        up = (
            np.clip(
                (
                    s["potential"]
                    - (
                        self.swing_floor
                        if self.swing_floor is not None
                        else s["potential"]
                    )
                )
                / max(s["energy_target"] - s["potential"], 10.0),
                0,
                1,
            )
            if useful
            else 0.0
        )
        rotke = (
            np.clip(
                s["rotational_proxy"]
                / max(self.cfg.momentum_com_cap**2 / (2 * s["I_com"]), 0.1),
                0,
                1,
            )
            if useful and s["pitch_velocity"] > 0 and s["pitch"] < np.deg2rad(65)
            else 0.0
        )
        impulse = np.clip(self.useful_impulse / self.cfg.momentum_com_cap, 0, 1)
        rm = gaussian(s["L_support"] - target, self.cfg.sigma_momentum)
        terminal, location = terminal_reward(s, self.cfg)
        terms = dict(
            momentum_tracking=self.cfg.momentum_weight
            * rm
            * self.cfg.dt
            * float(useful and self.phase.progress >= 0.2),
            angular_impulse=0.5 * max(0, impulse - self.paid_impulse),
            upward_energy=0.5 * max(0, up - self.paid_up),
            useful_rotation_energy=0.2 * max(0, rotke - self.paid_rotation_ke),
            excessive_momentum=-0.05
            * self.cfg.dt
            * float(safe)
            * min(max(abs(s["L_support"]) - self.cfg.momentum_support_cap, 0) ** 2, 25),
            terminal=terminal * self.cfg.dt if self.cfg.condition == "C" else 0.0,
            com_support=0.5
            * location
            * self.cfg.dt
            * float(s["orientation_error"] < 0.35)
            if self.cfg.condition == "C"
            else 0.0,
        )
        self.paid_up = max(self.paid_up, up)
        self.paid_rotation_ke = max(self.paid_rotation_ke, rotke)
        self.paid_impulse = max(self.paid_impulse, impulse)
        extra = 0.0
        if self.cfg.condition in ("B", "C"):
            # Replace the old pumping proxy and double-product terminal term.
            for key in ("pumping", "target"):
                removed = self.terms_sum.get(key, 0) - old_terms.get(key, 0)
                extra -= removed
                self.terms_sum[key] = old_terms.get(key, 0)
            extra += sum(terms.values())
            for key, value in terms.items():
                self.terms_sum[key] = self.terms_sum.get(key, 0) + value
        self.reward_sum += extra
        reward += extra
        s.update(
            L_target=target,
            r_momentum=rm,
            terminal_rate=terminal,
            com_support_reward=location,
            reward=reward,
            condition=self.cfg.condition,
            action_tau=self.cfg.action_tau,
            success=self.best_physical_hold >= 5.0,
        )
        for key, value in terms.items():
            s[f"reward_{key}"] = value if self.cfg.condition != "A" else 0.0
        self.trace[-1].update(s)
        info.update(
            reward=reward,
            success=self.best_physical_hold >= 5.0,
            handstand_seconds=self.best_physical_hold,
            rear_work=self.rear_work,
            rear_positive_work=self.rear_positive_work,
            angular_impulse=self.net_impulse,
            minimum_calf_clearance=min(x["calf_clearance"] for x in self.physics_trace),
        )
        info.update({f"episode_reward_{k}": v for k, v in self.terms_sum.items()})
        return self.observations(), reward, done, info
