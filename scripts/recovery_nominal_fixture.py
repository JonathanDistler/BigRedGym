"""Original near-vertical fixture, independent of a recovery task's reset bank."""

import copy

import mujoco
import numpy as np
import torch
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.utils.torch_quat import quat_from_euler_xyz
from scripts.evaluate_kickup_handoff import Policy, recovery, states
from scripts.reproduce_transition_hold import BALANCE


@torch.inference_mode()
def evaluate_nominal(policy, dest):
    nominal = Policy(BALANCE, "go2balance", 16, True)
    try:
        nominal.reset()
        b = nominal.env._backend
        order = b._canonical_to_native_dof_np
        feet = b._canonical_to_native_body_np[nominal.env.feet_indices.numpy()]
        for i, d in enumerate(b._datas):
            m = b._model_for_env(i)
            rng = np.random.default_rng(4000 + i)
            angles = torch.tensor(
                [
                    rng.normal(0, 0.004),
                    np.pi / 2 + rng.normal(0, 0.004),
                    rng.uniform(-np.pi, np.pi),
                ]
            )
            d.qpos[3:7] = quat_from_euler_xyz(*angles).numpy()[[3, 0, 1, 2]]
            d.qpos[7 + order] += rng.normal(0, 0.004, 12)
            d.qvel[:] = 0
            d.qvel[4] = rng.normal(0, 0.02)
            mujoco.mj_forward(m, d)
            d.qpos[2] += FOOT_RADIUS + 0.001 - d.xpos[feet, 2].min()
            mujoco.mj_forward(m, d)
        samples = states(nominal.env)
        models = [copy.copy(b._model_for_env(i)) for i in range(16)]
        reset = policy.reset

        def reset_nominal_controller():
            reset()
            e = policy.env
            e.leg_phase[:] = e._cpg_offsets
            e._cpg_load.zero_()
            e.dof_vel_target.zero_()
            e.tau_ff.zero_()
            e._update_cpg_obs()

        policy.reset = reset_nominal_controller
        try:
            return recovery(policy, samples, dest, "nominal_retention", models=models)
        finally:
            policy.reset = reset
    finally:
        nominal.env._backend.close()
