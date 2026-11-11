"""Versioned native MuJoCo integration states and residual-controller memory."""

import mujoco
import numpy as np
import torch

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
CONTROLLER = (
    "leg_phase",
    "_cpg_load",
    "dof_pos_target",
    "dof_vel_target",
    "tau_ff",
    "dof_pos_history",
    "commands",
    "handstand_stage",
    "p_gains",
    "d_gains",
)
MODEL = ("body_mass", "body_inertia", "body_ipos", "geom_friction")
DERIVED = ("body_invweight0", "dof_M0", "dof_invweight0")


def capture(env, index):
    b = env._backend
    m, d = b._model_for_env(index), b._datas[index]
    state = np.empty(mujoco.mj_stateSize(m, STATE))
    mujoco.mj_getState(m, d, state, STATE)
    return dict(
        native=state,
        **{f"model_{k}": getattr(m, k).copy() for k in MODEL + DERIVED},
        **{
            f"controller_{k}": getattr(env, k)[index].detach().cpu().numpy().copy()
            for k in CONTROLLER
            if hasattr(env, k)
        },
    )


def restore(env, index, sample):
    """Restore integration inputs, regenerate solver contacts, then sync tensors."""
    b = env._backend
    m, d = b._model_for_env(index), b._datas[index]
    expected = mujoco.mj_stateSize(m, STATE)
    if sample["native"].shape != (expected,):
        raise ValueError("Recovery state does not match this MuJoCo model")
    for k in MODEL:
        getattr(m, k)[:] = sample[f"model_{k}"]
    mujoco.mj_setConst(m, d)
    for k in DERIVED:
        if f"model_{k}" in sample:
            getattr(m, k)[:] = sample[f"model_{k}"]
    mujoco.mj_setState(m, d, sample["native"], STATE)
    warm = d.qacc_warmstart.copy()
    mujoco.mj_forward(m, d)
    d.qacc_warmstart[:] = warm
    for k in CONTROLLER:
        if f"controller_{k}" in sample:
            getattr(env, k)[index] = torch.as_tensor(
                sample[f"controller_{k}"], device=env.device
            )
    b._link_mass_t[index] = torch.as_tensor(m.body_mass[b._canonical_to_native_body_np])
    b._contact_friction_t[index] = float(m.geom_friction[0, 0])
    env._update_cpg_obs()


def sync(env):
    env._backend._sync_state_from_mujoco()
    env.base_quat[:] = env.root_states[:, 3:7]
    env._refresh_base_observations(
        torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    )
    env.dof_pos_obs[:] = env.dof_pos - env.default_dof_pos


class RecoveryBank:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {k: data[k].copy() for k in data.files}
        self.version = int(self.data["version"])
        if self.version not in (1, 2):
            raise ValueError("Unsupported recovery bank version")
        if not all(
            np.isfinite(v).all() for v in self.data.values() if v.dtype.kind in "fc"
        ):
            raise ValueError("Nonfinite recovery bank")
        stages = range(6) if self.version == 2 else (3, 4, 5)
        if self.version == 2:
            required = (
                "source_id",
                "family",
                "stratum",
                "frozen_success",
                "frozen_hold_s",
            )
            if any(k not in self.data for k in required):
                raise ValueError("Incomplete measured recovery-frontier bank")
            count = len(self.data["seed"])
            if any(self.data[k].shape != (count,) for k in required):
                raise ValueError("Unaligned frontier metadata")
            if not np.isin(self.data["stratum"], (0, 1, 2)).all():
                raise ValueError("Invalid frontier stratum")
            if not np.array_equal(
                self.data["stratum"] == 0, self.data["frozen_success"]
            ):
                raise ValueError("Recoverable stratum conflicts with measured outcomes")
            if not self.data["frozen_success"][self.indices(0)].all():
                raise ValueError("Stage zero must contain demonstrated recoveries")
        for stage in stages:
            if not len(self.indices(stage)):
                raise ValueError(f"Recovery bank has no samples for stage {stage}")

    def indices(self, stage):
        return np.flatnonzero(self.data[f"stage_{stage}"])

    def sample(self, stage):
        indices = self.indices(stage)
        if self.version == 2:
            weights = (
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.7, 0.3, 0.0),
                (0.6, 0.3, 0.1),
                (0.5, 0.35, 0.15),
                (0.4, 0.4, 0.2),
            )[stage]
            present = torch.tensor(
                [bool((self.data["stratum"][indices] == k).any()) for k in range(3)]
            )
            probabilities = torch.tensor(weights) * present
            if probabilities.sum() <= 0:
                raise ValueError("No eligible recovery mixture stratum")
            stratum = int(torch.multinomial(probabilities, 1).item())
            indices = indices[self.data["stratum"][indices] == stratum]
            # First choose a source trajectory uniformly, then one of its states.
            sources = np.unique(self.data["source_id"][indices])
            source = sources[int(torch.randint(len(sources), ()).item())]
            indices = indices[self.data["source_id"][indices] == source]
        i = indices[int(torch.randint(len(indices), ()).item())]
        self.last_sample_index = int(i)
        return {
            k: v[i]
            for k, v in self.data.items()
            if k == "native" or k.startswith(("model_", "controller_"))
        }
