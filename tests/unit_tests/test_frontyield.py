"""Opt-in reference parity and phase/rate safety in canonical joint coordinates."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from gym.envs.go2.kickup_reference import KickReference, reference, torch_reference
from gym.envs.go2.go2frontyield_config import Go2FrontYieldCfg, Go2FrontYieldRunnerCfg
from gym.utils.task_registry import task_registry


def test_tensor_reference_matches_independent_generator():
    params = KickReference(
        **json.loads(
            (
                Path(__file__).parents[2]
                / "gym/envs/go2/references/kickup_front_yield.json"
            ).read_text()
        )
    )
    times = np.linspace(0, 1.6, 801)
    standing = np.tile([0.0, 0.58, -1.16], 4)
    q, v = torch_reference(
        torch.tensor(times), params, torch.tensor(standing).repeat(len(times), 1)
    )
    for i, t in enumerate(times):
        expected, rate = reference(t, params, standing)
        np.testing.assert_allclose(q[i], expected, atol=1e-12)
        np.testing.assert_allclose(v[i], rate, atol=1e-12)


@pytest.fixture
def env():
    import gym.envs  # noqa:F401

    cfg, train = Go2FrontYieldCfg(), Go2FrontYieldRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    task_registry.convert_frequencies_to_params(cfg, train)
    e = task_registry.make_env(
        "go2frontyield", cfg, device="cpu", headless=True, backend="mujoco"
    )
    yield e
    e._backend.close()


def test_zero_actions_leave_no_motor_residual(env):
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    for clock in (0.1, 0.5, 0.8, 0.9, 1.05, 1.4):
        env._kick_clock[:] = clock
        env.set_state("dof_pos_target", torch.zeros(2, 12))
        env._compute_torques()
        assert not env.dof_pos_target.any()


def test_extreme_actor_cannot_suppress_initial_push(env):
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    env._reset_level[:] = 0
    env._kick_clock[:] = 0.86
    env.set_state("dof_pos_target", torch.full((2, 12), -100.0))
    for _ in range(100):
        env._compute_torques()
    assert not env.dof_pos_target[:, 6:].any()
    assert env.dof_pos_target[:, :6].abs().max() <= 0.0050001
    assert torch.all(env._compute_torques().abs() <= env.actuated_torque_limits)


def test_later_stage_rate_projection_preserves_extension(env):
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    env._reset_level[:] = 6
    env._kick_clock[:] = 0.86
    env.set_state("dof_pos_target", torch.tensor([[-100.0] * 12, [100.0] * 12]))
    q, v = env._kick_reference(env._kick_clock + env._physics_offset)
    previous = env.dof_pos_target.clone()
    env._compute_torques()
    h = env.dt / env.cfg.control.decimation
    residual_rate = (env.dof_pos_target - previous) / h
    for j in (7, 8, 10, 11):
        assert torch.all((v[:, j] + residual_rate[:, j]) / v[:, j] >= 0.8 - 1e-6)
    bounds = env.residual_bounds(env._kick_clock)
    assert torch.all(env.dof_pos_target.abs() <= bounds + 1e-7)


def test_batched_sync_matches_original_backend_buffers(env):
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend

    assert env._backend._sync_state_from_mujoco.__name__ == "batch_sync"
    env.step()
    b = env._backend
    names = (
        "_dof_pos_view",
        "_dof_vel_view",
        "_contact_forces_t",
        "_rigid_body_states_t",
        "_root_states_t",
    )
    fast = {k: getattr(b, k).clone() for k in names}
    MuJocoCPUBackend._sync_state_from_mujoco(b)
    for k in names:
        torch.testing.assert_close(getattr(b, k), fast[k], rtol=0, atol=1e-7)
