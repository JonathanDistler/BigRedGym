"""CPG balance residual fine-tuning with native trajectory replay resets."""

import copy
import os

import mujoco
import torch

from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.recovery_state import RecoveryBank, restore, sync


class Go2KickupRecovery(Go2Balance):
    def _init_buffers(self):
        super()._init_buffers()
        if not self.contact_cpg_enabled:
            raise ValueError("Recovery requires the verified CPG policy interface")
        if self.device != "cpu" and str(self.device) != "cpu":
            raise ValueError("Recovery replay currently requires MuJoCo CPU")
        path = os.environ.get("GO2_RECOVERY_DATASET", self.cfg.recovery.dataset)
        self.recovery_bank = RecoveryBank(path)
        self.frontier_curriculum = bool(
            getattr(self.cfg.recovery, "frontier_curriculum", False)
        )
        if self.frontier_curriculum and self.recovery_bank.version != 2:
            raise ValueError("Measured frontier curriculum requires a version-2 bank")
        self.recovery_sampling_counts = {}
        self._recovery_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Each replay may have different mass/friction; never mutate a shared model.
        b = self._backend
        b._models = [copy.copy(b._model_for_env(i)) for i in range(self.num_envs)]
        b._mjm = b._models[0]
        b._datas = [mujoco.MjData(m) for m in b._models]
        self._nominal_models = [copy.copy(m) for m in b._models]
        from gym.envs.go2.frontyield_cpu import install_batched_cpu_sync

        install_batched_cpu_sync(b)

    def _reset_idx(self, reset_mask):
        # Curriculum accounting must observe the finished episode before reset.
        super()._reset_idx(reset_mask)
        if not hasattr(self, "recovery_bank"):
            return
        self._recovery_collision[reset_mask] = False
        ids = reset_mask.nonzero().flatten().tolist()
        for i in ids:
            stage = int(self._reset_level[i])
            if stage >= 3 or self.frontier_curriculum:
                restore(self, i, self.recovery_bank.sample(stage))
                if self.frontier_curriculum:
                    index = self.recovery_bank.last_sample_index
                    stratum = int(self.recovery_bank.data["stratum"][index])
                    key = f"stage_{stage}_stratum_{stratum}"
                    self.recovery_sampling_counts[key] = (
                        self.recovery_sampling_counts.get(key, 0) + 1
                    )
            else:
                # Earlier replay levels use nominal physics, including after a
                # trajectory reset with randomized mass/friction.
                b = self._backend
                b._models[i] = copy.copy(self._nominal_models[i])
                b._mjm = b._models[0]
                m, d = b._model_for_env(i), b._datas[i]
                self._backend._link_mass_t[i] = torch.as_tensor(
                    m.body_mass[self._backend._canonical_to_native_body_np]
                )
                if stage == 2:
                    d.qvel[:2] += (torch.rand(2).numpy() * 2 - 1) * 0.25
                mujoco.mj_forward(m, d)
        if ids:
            sync(self)

    def _update_handstand_hold(self):
        if not getattr(self, "frontier_curriculum", False):
            return super()._update_handstand_hold()
        from gym.envs.go2.recovery_frontier import valid_recovery

        valid, _ = valid_recovery(self, self._recovery_collision.numpy())
        held = (
            torch.as_tensor(valid, device=self.device)
            & ~self.terminated
            & ~self.timed_out
        )
        self._hold_time[:] = torch.where(held, self._hold_time + self.dt, 0.0)
        self._best_hold[:] = torch.maximum(self._best_hold, self._hold_time)

    def training_state_dict(self):
        state = super().training_state_dict()
        if getattr(self, "frontier_curriculum", False):
            state["recovery_sampling_counts"] = dict(self.recovery_sampling_counts)
        return state

    def _native_nonfoot_contacts(self):
        b = self._backend
        feet = set(b._canonical_to_native_body_np[self.feet_indices.cpu().numpy()])
        hits = []
        for i, d in enumerate(b._datas):
            m = b._model_for_env(i)
            hits.append(
                any(
                    c.dist < -1e-5
                    and (
                        (
                            m.geom_bodyid[c.geom1] == 0
                            and m.geom_bodyid[c.geom2] not in feet
                        )
                        or (
                            m.geom_bodyid[c.geom2] == 0
                            and m.geom_bodyid[c.geom1] not in feet
                        )
                    )
                    for c in d.contact
                )
            )
        return torch.tensor(hits, device=self.device)

    def _body_contact(self):
        hit = self._native_nonfoot_contacts()
        if hasattr(self, "_recovery_collision"):
            hit = hit | self._recovery_collision
        return hit

    def _post_physics_step(self):
        super()._post_physics_step()
        if hasattr(self, "_recovery_collision"):
            self._recovery_collision |= self._native_nonfoot_contacts()

    def _com_terms(self):
        offsets, speeds = [], []
        b = self._backend
        feet = b._canonical_to_native_body_np[self.feet_indices[:2].cpu().numpy()]
        for i, d in enumerate(b._datas):
            m = b._model_for_env(i)
            mujoco.mj_subtreeVel(m, d)
            offsets.append(
                float(((d.subtree_com[1, :2] - d.xpos[feet, :2].mean(0)) ** 2).sum())
            )
            speeds.append(float((d.subtree_linvel[1] ** 2).sum()))
        return torch.tensor(offsets, device=self.device), torch.tensor(
            speeds, device=self.device
        )

    def _reward_recovery_com_velocity(self):
        return -self._com_terms()[1].clamp(max=9)

    def _reward_recovery_com_offset(self):
        return -(self._com_terms()[0] / 0.05**2).clamp(max=25)

    def _reward_recovery_hold(self):
        return self._valid_handstand() * (self._hold_time / 5).clamp(0, 1)
