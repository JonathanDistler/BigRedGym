"""Optional independent, force-coupled leg clocks and desired contact modes."""

import math
import torch


class ContactCPG:
    def _init_buffers(self):
        super()._init_buffers()
        c = getattr(self.cfg, "contact_cpg", None)
        self.contact_cpg_enabled = c is not None and c.enabled
        if not self.contact_cpg_enabled:
            return
        names = self.robot_layout.body_groups["feet"]
        if len(c.modes) != len(names) or any(
            mode not in ("phase", "stance", "swing", "ignore") for mode in c.modes
        ):
            raise ValueError(
                "contact_cpg.modes must specify phase/stance/swing/ignore per foot"
            )
        if (
            len(c.phase_offsets) != len(names)
            or not all(
                math.isfinite(x)
                for x in [
                    c.frequency_hz,
                    c.sigma,
                    c.zeta,
                    c.force_filter_s,
                    *c.phase_offsets,
                ]
            )
            or c.frequency_hz < 0
            or c.force_filter_s < 0
        ):
            raise ValueError("invalid contact CPG parameters")
        self.leg_phase = torch.zeros(self.num_envs, len(names), device=self.device)
        self.cpg_phase_obs = torch.zeros(
            self.num_envs, 2 * len(names), device=self.device
        )
        self.cpg_contact_obs = torch.zeros_like(self.leg_phase)
        self._cpg_load = torch.zeros_like(self.leg_phase)
        self._cpg_offsets = self.leg_phase.new_tensor(c.phase_offsets) * (2 * torch.pi)
        self._cpg_modes = torch.tensor(
            [("phase", "stance", "swing", "ignore").index(m) for m in c.modes],
            device=self.device,
        )
        self.scales["cpg_phase_obs"] = 1.0
        self.scales["cpg_contact_obs"] = 1.0
        self._update_cpg_obs()

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if self.contact_cpg_enabled:
            self.leg_phase[reset_mask] = self._cpg_offsets
            self._cpg_load[reset_mask] = 0
            self._update_cpg_obs()

    def _update_cpg_obs(self):
        self.cpg_phase_obs[:] = torch.stack(
            (self.leg_phase.sin(), self.leg_phase.cos()), dim=-1
        ).flatten(1)
        desired = (self.leg_phase.sin() > 0).to(self.leg_phase.dtype)
        desired[:, self._cpg_modes == 1] = 1
        desired[:, self._cpg_modes == 2] = 0
        desired[:, self._cpg_modes == 3] = -1
        self.cpg_contact_obs[:] = desired

    def _post_physics_step(self):
        super()._post_physics_step()
        if not self.contact_cpg_enabled:
            return
        c = self.cfg.contact_cpg
        dt = self.dt / self.cfg.control.decimation
        load = self.contact_forces[:, self.feet_indices, 2].clamp(min=0)
        load = load / (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)).clamp(
            min=1e-6
        )
        alpha = 1.0 if c.force_filter_s == 0 else -math.expm1(-dt / c.force_filter_s)
        self._cpg_load.lerp_(load, alpha)
        rate = 2 * torch.pi * c.frequency_hz - c.sigma * self._cpg_load * (
            self.leg_phase.cos() + c.zeta
        )
        self.leg_phase.add_(dt * rate).remainder_(2 * torch.pi)
        self._update_cpg_obs()

    def _reward_cpg_contact(self):
        if not self.contact_cpg_enabled:
            return torch.zeros(self.num_envs, device=self.device)
        # Continuous load matching; ignored legs contribute neither reward nor penalty.
        load = self.contact_forces[:, self.feet_indices, 2].clamp(min=0)
        weight = (9.81 * self._backend.link_mass.sum(dim=1, keepdim=True)).clamp(
            min=1e-6
        )
        active = self.cpg_contact_obs >= 0
        stance = self.cpg_contact_obs == 1
        nominal = weight / stance.sum(dim=1, keepdim=True).clamp(min=1)
        strength = (load / nominal).clamp(0, 1)
        match = torch.where(stance, strength, 1 - strength)
        return (match * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
