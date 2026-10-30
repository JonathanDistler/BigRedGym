"""Physics constraints and reset integration for empirical reachability."""

import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2reachability import Go2Reachability
from gym.envs.go2.go2reachability_config import (
    Go2ReachabilityCfg,
    Go2ReachabilityRunnerCfg,
)
from gym.envs.go2.reachability_physics import work_speed_bound
from gym.utils.task_registry import task_registry


def test_work_energy_and_motor_cap():
    assert work_speed_bound(2, np.array([4.0]), np.array([1.0]), 100) == pytest.approx(
        2
    )
    assert work_speed_bound(2, np.array([4.0]), np.array([1.0]), 1) == 1
    with pytest.raises(ValueError):
        work_speed_bound(0, [4], [1], 1)


@pytest.fixture
def env():
    cfg = Go2ReachabilityCfg()
    cfg.env.num_envs = 2
    task_registry.convert_frequencies_to_params(cfg, Go2ReachabilityRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        yield Go2Reachability(cfg, "cpu", True, backend)
    finally:
        backend.close()


def test_velocity_projection_preserves_both_support_feet(env):
    s = env._sampler
    s.load(env.root_states[0].numpy(), env.dof_pos[0].numpy())
    velocity, inertia = s.direction(np.ones(6) / np.sqrt(6))
    assert inertia > 0
    np.testing.assert_allclose(velocity[s.v[6:]], np.ones(6) / np.sqrt(6))
    # direction() itself verifies the full stacked 6D foot velocity constraint.
    assert np.isfinite(s.bound(env.dof_pos[0].numpy(), np.ones(6), inertia))


def test_reset_acceptance_and_contact_failure_latch(env):
    assert env.reset_accepted.all()
    assert not env.contact_failed.any()
    env.episode_length_buf[:] = 100
    env.contact_forces.zero_()
    env._post_physics_step()
    assert env.contact_failed.all()
    env._check_terminations_and_timeouts()
    assert env.terminated.all()
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert not env.contact_failed.any()


def test_rejected_proposals_are_replay_not_promotion(env):
    env._sampler.probe = lambda *args: False
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert not env.reset_accepted.any()
    assert not env._primary_reset.any()
    assert (env.dof_vel == 0).all()
