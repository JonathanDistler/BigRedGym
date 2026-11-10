"""Opt-in batched state copies; native stepping/contact solver are unchanged."""

from types import MethodType
import mujoco
import numpy as np
import torch
from gym.utils.torch_quat import quat_apply
from gym.envs.base.mujoco_backend_base import WXYZ_TO_XYZW


def batch_sync(b):
    for i, d in enumerate(b._datas):
        b._activate_domain(i)
        m = b._model_for_env(i)
        mujoco.mj_rnePostConstraint(m, d)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
    dofs = b._canonical_to_native_dof_np
    bodies = b._canonical_to_native_body_np
    qpos = np.stack([d.qpos for d in b._datas])
    qvel = np.stack([d.qvel for d in b._datas])
    b._dof_pos_view.copy_(torch.from_numpy(qpos[:, b._qpos_offset :][:, dofs]))
    b._dof_vel_view.copy_(torch.from_numpy(qvel[:, b._qvel_offset :][:, dofs]))
    b._contact_forces_t.copy_(
        torch.from_numpy(np.stack([d.cfrc_ext[bodies, 3:6] for d in b._datas]))
    )
    pos = np.stack([d.xpos[bodies] for d in b._datas])
    quat = np.stack([d.xquat[bodies] for d in b._datas])[:, :, WXYZ_TO_XYZW]
    vel = np.stack([d.cvel[bodies] for d in b._datas])
    com = np.stack(
        [
            d.subtree_com[b._model_for_env(i).body_rootid[bodies]]
            for i, d in enumerate(b._datas)
        ]
    )
    rbs = np.concatenate(
        (pos, quat, vel[:, :, 3:6] - np.cross(pos - com, vel[:, :, :3]), vel[:, :, :3]),
        axis=2,
    )
    b._rigid_body_states_t.copy_(torch.from_numpy(rbs))
    root = np.zeros((b._num_envs, 13))
    root[:, :3] = qpos[:, :3]
    root[:, 3:7] = qpos[:, 3:7][:, WXYZ_TO_XYZW]
    root[:, 7:10] = qvel[:, :3]
    root[:, 10:13] = quat_apply(
        torch.from_numpy(root[:, 3:7]), torch.from_numpy(qvel[:, 3:6])
    ).numpy()
    b._root_states_t.copy_(torch.from_numpy(root))


def install_batched_cpu_sync(backend):
    """Attach only to this CPU task instance, leaving all other tasks untouched."""
    backend._sync_state_from_mujoco = MethodType(batch_sync, backend)
