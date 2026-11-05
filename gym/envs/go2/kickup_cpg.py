"""One-shot, four-oscillator kick primitive; angles are absolute radians."""

import math
import torch


class KickupCPG:
    def __init__(self, num_envs, device, cfg):
        self.cfg = cfg
        if not math.isfinite(cfg.duration_s) or cfg.duration_s <= 0:
            raise ValueError("kickup duration_s must be positive and finite")
        poses = torch.tensor(
            [cfg.standing, cfg.crouched, cfg.extended],
            device=device,
            dtype=torch.float32,
        )
        if poses.shape != (3, 4, 3) or not torch.isfinite(poses).all():
            raise ValueError(
                "kickup poses must contain finite FL/FR/RL/RR joint triples"
            )
        self.poses = poses
        self.amplitude = torch.ones(num_envs, device=device)
        self.phase = torch.zeros(num_envs, 4, device=device)
        self.handed_off = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros_like(self.handed_off)

    def reset(self, mask):
        self.phase[mask] = 0
        self.handed_off[mask] = False
        self.failed[mask] = False

    def advance(self, dt, in_region):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("kickup dt must be positive and finite")
        # A region hit latches permanently until reset. Timeout is a failure,
        # never evidence that the robot entered the recovery set.
        self.handed_off |= in_region & ~self.failed
        active = ~(self.handed_off | self.failed)
        self.phase[active] += 2 * math.pi * dt / self.cfg.duration_s
        self.phase.clamp_(max=2 * math.pi)
        self.failed |= active & (self.phase >= 2 * math.pi - 1e-6).all(dim=1)

    def targets(self):
        # First half-cycle crouches all legs. Second half keeps front support
        # crouched while the hind legs extend rapidly against the ground.
        p = self.phase
        first = p <= math.pi
        blend = torch.where(first, (1 - p.cos()) / 2, (1 + p.cos()) / 2)
        start = torch.where(first[..., None], self.poses[0], self.poses[1])
        end = torch.where(first[..., None], self.poses[1], self.poses[2])
        pose = start + blend[..., None] * (end - start)
        pose = self.poses[0] + self.amplitude[:, None, None] * (pose - self.poses[0])
        return pose.flatten(1)
