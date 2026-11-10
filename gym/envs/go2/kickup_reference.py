"""Opt-in, policy-free trajectory experiments; existing kick-up task is unchanged."""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class KickReference:
    settle: float = 0.20
    crouch: float = 0.40
    extend: float = 0.14
    hip_delay: float = 0.0
    knee_delay: float = 0.0
    hip_duration: float = 1.0
    knee_duration: float = 1.0
    rear_crouch_hip: float = 1.15
    rear_crouch_knee: float = -2.3
    rear_push_hip: float = 0.3
    rear_push_knee: float = -0.85
    front_crouch_hip: float = 0.75
    front_crouch_knee: float = -1.5
    front_push_hip: float = 0.75
    front_push_knee: float = -1.5
    front_gain: float = 0.65
    front_damping: float = 1.0
    feedforward: float = 1.0
    tuck_delay: float = 0.05
    tuck_duration: float = 0.20
    rear_tuck_hip: float = 1.0
    rear_tuck_knee: float = -1.8
    tuck: bool = True

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("reference parameters must be finite")
        if (
            min(
                self.crouch,
                self.extend,
                self.hip_duration,
                self.knee_duration,
                self.tuck_duration,
            )
            <= 0
        ):
            raise ValueError("reference durations must be positive")
        if (
            not 0 < self.front_gain <= 1
            or self.front_damping < 0
            or self.feedforward < 0
        ):
            raise ValueError("invalid controller gains")

    def to_dict(self):
        return asdict(self)


def blend(t, start, duration):
    u = np.clip((t - start) / duration, 0, 1)
    return u * u * (3 - 2 * u), 6 * u * (1 - u) / duration


def physical_liftoff(
    clearance, forces, loaded, weight, drift, body_contact, target=0.01
):
    """Actual sphere-ground separation and maintained front load, never force alone."""
    return bool(
        np.all(loaded)
        and np.min(clearance[2:]) >= target
        and np.max(forces[2:]) < 0.015 * weight
        and np.min(forces[:2]) >= 0.06 * weight
        and drift <= 0.04
        and not body_contact
    )


def reference(t, p, standing):
    """Continuous position and analytic velocity; hip/knee clocks independent."""
    q = standing.copy()
    v = np.zeros(12)
    a, da = blend(t, p.settle, p.crouch)
    push = p.settle + p.crouch
    for leg in range(4):
        front = leg < 2
        for offset, name in ((1, "hip"), (2, "knee")):
            j = 3 * leg + offset
            folded = getattr(p, ("front" if front else "rear") + "_crouch_" + name)
            extended = getattr(p, ("front" if front else "rear") + "_push_" + name)
            delay = 0 if front else getattr(p, name + "_delay")
            duration = 1 if front else getattr(p, name + "_duration")
            b, db = blend(t, push + delay * p.extend, p.extend * duration)
            q[j] = standing[j] + a * (folded - standing[j]) + b * (extended - folded)
            v[j] = da * (folded - standing[j]) + db * (extended - folded)
            if not front and p.tuck:
                tuck_time = (
                    push
                    + p.extend
                    * max(p.hip_duration + p.hip_delay, p.knee_duration + p.knee_delay)
                    + p.tuck_delay
                )
                c, dc = blend(t, tuck_time, p.tuck_duration)
                tucked = getattr(p, "rear_tuck_" + name)
                q[j] += c * (tucked - extended)
                v[j] += dc * (tucked - extended)
    return q, v


def torch_reference(clock, p, standing):
    """Vectorized batched equivalent of the validated NumPy generator."""
    import torch

    key = (str(standing.device), standing.dtype)
    cache = getattr(p, "_tensor_cache", {})
    if key not in cache:
        folded = []
        thrust = []
        tuck_delta = []
        delays = []
        durations = []
        for leg in range(4):
            front = leg < 2
            for offset, name in enumerate(("abduction", "hip", "knee")):
                if offset == 0:
                    folded.append(0.0)
                    thrust.append(0.0)
                    tuck_delta.append(0.0)
                    delays.append(0.0)
                    durations.append(1.0)
                    continue
                group = "front" if front else "rear"
                a = getattr(p, group + "_crouch_" + name)
                b = getattr(p, group + "_push_" + name)
                folded.append(a)
                thrust.append(b)
                tuck_delta.append(
                    0.0 if front or not p.tuck else getattr(p, "rear_tuck_" + name) - b
                )
                delays.append(0.0 if front else getattr(p, name + "_delay"))
                durations.append(1.0 if front else getattr(p, name + "_duration"))
        cache[key] = tuple(
            torch.tensor(x, device=standing.device, dtype=standing.dtype)[None, :]
            for x in (folded, thrust, tuck_delta, delays, durations)
        )
        p._tensor_cache = cache
    folded, thrust, tuck_delta, delays, durations = cache[key]

    def mix(t, start, duration):
        u = ((t - start) / duration).clamp(0, 1)
        return u * u * (3 - 2 * u), 6 * u * (1 - u) / duration

    t = clock[:, None]
    push = p.settle + p.crouch
    a, da = mix(t, p.settle, p.crouch)
    b, db = mix(t, push + delays * p.extend, durations * p.extend)
    tuck_time = (
        push
        + p.extend * max(p.hip_duration + p.hip_delay, p.knee_duration + p.knee_delay)
        + p.tuck_delay
    )
    c, dc = mix(t, tuck_time, p.tuck_duration)
    q = standing + a * (folded - standing) + b * (thrust - folded) + c * tuck_delta
    v = da * (folded - standing) + db * (thrust - folded) + dc * tuck_delta
    return q, v
