"""Physics constraints and reset integration for empirical reachability."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2reachability import Go2Reachability, ReachabilityDiagnostics
from gym.envs.go2.go2reachability_config import (
    Go2ReachabilityCfg,
    Go2ReachabilityRunnerCfg,
)
from gym.envs.go2.reachability_physics import work_speed_bound
from gym.utils.task_registry import task_registry
from gym.utils.robot_metrics import configure_contact_cpg
from gym.utils.helpers import class_to_dict


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
    runner = Go2ReachabilityRunnerCfg()
    configure_contact_cpg(
        SimpleNamespace(cpg=True, contact_modes=["stance", "stance", "swing", "swing"]),
        cfg,
        runner,
    )
    task_registry.convert_frequencies_to_params(cfg, runner)
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


def test_contact_loss_allows_recovery_before_terminal_fall(env):
    assert env.reset_accepted.all()
    env.episode_length_buf[:] = 100
    env.contact_forces.zero_()
    assert env.cfg.balance_failure.both_front_airborne_seconds == 0.25
    steps = math.ceil(0.25 / env.dt)
    for _ in range(steps - 1):
        env._post_decimation_step()
    env._check_terminations_and_timeouts()
    assert not env.terminated.any()
    # Recover on one foot, even with substantial slip: no terminal latch.
    env.contact_forces[:, env.feet_indices[0], 2] = 10
    env._rigid_body_lin_vel[:, env.feet_indices[:2], :2] = 1
    env._post_decimation_step()
    env._check_terminations_and_timeouts()
    assert not env.terminated.any()
    assert (env._front_airborne_time == 0).all()
    env.contact_forces.zero_()
    for _ in range(steps):
        env._post_decimation_step()
    env._check_terminations_and_timeouts()
    assert env.terminated.all()
    assert (env._reward_termination() < 0).all()
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert (env._front_airborne_time == 0).all()


def test_dense_recovery_penalties(env):
    feet = env.feet_indices[:2]
    weight = 9.81 * env._backend.link_mass.sum(dim=1)
    env.contact_forces[:, feet, 2] = (
        weight * env.cfg.reward_settings.front_support_fraction
    )[:, None]
    assert (env._reward_front_contact_loss() == 0).all()
    env.contact_forces[:, feet, 2] *= 0.5
    torch.testing.assert_close(
        env._reward_front_contact_loss(), torch.full((2,), -0.25)
    )
    env.contact_forces[:, feet, 2] = 100
    env._rigid_body_lin_vel[:, feet, :2] = 0.1
    slow = env._reward_handstand_slip().clone()
    env._rigid_body_lin_vel[:, feet, :2] *= 2
    torch.testing.assert_close(env._reward_handstand_slip(), 4 * slow)
    env.base_ang_vel.zero_()
    assert (env._reward_reachability_stability() == 0).all()
    env.base_ang_vel[:] = 1
    moderate = env._reward_reachability_stability().clone()
    env.base_ang_vel[:] = 2
    assert (env._reward_reachability_stability() < moderate).all()


def test_rejected_proposals_are_replay_not_promotion(env):
    env._sampler.probe = lambda *args: False
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert not env.reset_accepted.any()
    assert not env._primary_reset.any()
    assert (env.dof_vel == 0).all()


def test_episode_returns_span_rollouts_and_separate_falls_from_timeouts():
    d = ReachabilityDiagnostics("cpu")
    no = torch.tensor([False, False])
    d.record_rewards(torch.tensor([3.0, 4.0]), no, no)
    d.reset()  # rollout boundaries must not discard ongoing episode returns
    d.record_rewards(
        torch.tensor([-20.0, 5.0]),
        torch.tensor([True, False]),
        torch.tensor([False, True]),
    )
    d.finish()
    assert d.fall_episode_return.item() == -17
    assert d.timeout_episode_return.item() == 9
    assert d.fall_episodes.item() == d.timeout_episodes.item() == 1
    assert (d.episode_returns == 0).all()


def test_recovery_events_require_contact_restored_before_deadline(env):
    env.episode_length_buf[:] = 100
    env.contact_forces[:, env.feet_indices[:2], 2] = 10
    env._post_decimation_step()
    env.contact_forces[:, env.feet_indices[0], 2] = 0
    env._post_decimation_step()
    env.update_training_diagnostics()
    assert env._contact_loss_started.all()
    env.contact_forces[:, env.feet_indices[:2], 2] = 10
    env._post_decimation_step()
    env.update_training_diagnostics()
    assert env._contact_recovered.all()
    env.training_diagnostics.finish()
    assert env.training_diagnostics.contact_recoveries.item() == 2
    assert env.training_diagnostics.cumulative_recovery_fraction.item() == 1
    env._contact_loss_time[:] = 0.3
    env._post_decimation_step()
    assert not env._contact_recovered.any()


def test_supported_handstand_return_exceeds_immediate_fall(env):
    # A controlled reference-state check, not a claim about learned behavior.
    env.projected_gravity[:] = torch.tensor([1.0, 0.0, 0.0])
    env.base_ang_vel.zero_()
    env.base_lin_vel.zero_()
    env._rigid_body_lin_vel.zero_()
    env.contact_forces.zero_()
    weight = 9.81 * env._backend.link_mass.sum(dim=1)
    env.contact_forces[:, env.feet_indices[:2], 2] = weight[:, None] / 2
    env.torques.zero_()
    env.dof_pos_history.zero_()
    weights = class_to_dict(Go2ReachabilityRunnerCfg().critic.reward.weights)
    rate = sum(
        weight * getattr(env, "_reward_" + name)() for name, weight in weights.items()
    )
    assert (rate > 0).all()
    env.terminated[:] = True
    cost = (
        env._reward_termination()
        * env.dt
        * Go2ReachabilityRunnerCfg.critic.reward.termination_weight.termination
    )
    torch.testing.assert_close(cost, torch.full((2,), -20.0))
    assert (rate * env.cfg.env.episode_length_s > cost).all()
