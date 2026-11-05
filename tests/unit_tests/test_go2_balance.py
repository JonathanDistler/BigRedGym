"""Top balance references, shaping signals, diagnostics and perturbation gates."""

import math

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2balance import BalanceDiagnostics, Go2Balance
from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry
from tests.unit_tests.test_go2_handstand import task


@pytest.fixture
def env():
    cfg = Go2BalanceCfg()
    cfg.env.num_envs = 2
    cfg.seed = 17
    task_registry.convert_frequencies_to_params(cfg, Go2BalanceRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        yield Go2Balance(cfg, "cpu", True, backend)
    finally:
        backend.close()


def test_balance_starts_at_top_with_zero_residual_and_fixed_physics(env):
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert env.handstand_stage.tolist() == [2, 2]
    assert env.cfg.handstand_curriculum.standing_fraction == 0
    assert env.cfg.handstand_curriculum.tilt_noise_degrees[0] == 0
    assert env.cfg.domain_randomization.startup.link_mass_scale_range is None
    torch.testing.assert_close(env.dof_pos_target, torch.zeros_like(env.dof_pos_target))
    torch.testing.assert_close(env._standing_posture(), env.dof_pos)
    # Removing a policy action must retain the top pose, without standing torque.
    env.dof_vel.zero_()
    env.dof_vel_target.zero_()
    env.tau_ff.zero_()
    torch.testing.assert_close(
        env._compute_torques(), torch.zeros_like(env.torques), atol=1e-5, rtol=0
    )
    assert torch.all(env.projected_gravity[:, 0] > 0.999)
    assert torch.all(env._episode_push_limit == 0)


def test_dense_shaping_survives_one_failed_component_but_strict_score_does_not():
    original = task()
    env = Go2Balance.__new__(Go2Balance)
    env.__dict__.update(original.__dict__)
    env.cfg = Go2BalanceCfg()
    env.termination_contact_indices = torch.tensor([], dtype=torch.long)
    env.handstand_stage[:] = 2
    env.dof_pos[:, [8, 11]] = -2.0
    assert env._reward_handstand_balance().item() < 0.01
    assert env._reward_balance_orientation().item() == pytest.approx(1)
    assert env._reward_balance_front_support().item() == pytest.approx(1)
    # Being still on the floor cannot earn the stability shaping reward.
    env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    env.contact_forces.zero_()
    assert env._reward_balance_stability().item() < 1e-8


def test_diagnostics_are_raw_rollout_means_and_reset_between_rollouts():
    diagnostics = BalanceDiagnostics("cpu")
    diagnostics.update({name: torch.tensor([0.2, 0.6]) for name in diagnostics.names})
    diagnostics.update({name: torch.tensor([0.6, 1.0]) for name in diagnostics.names})
    diagnostics.finish()
    assert diagnostics.vertical.item() == pytest.approx(0.6)
    diagnostics.reset()
    diagnostics.update({name: torch.ones(2) for name in diagnostics.names})
    diagnostics.finish()
    assert diagnostics.vertical.item() == pytest.approx(1)


def test_rear_freedom_has_deadband_but_fast_motion_costs(env):
    env.dof_pos[:, [8, 11]] = env.cfg.reward_settings.kick_knee_target + 0.1
    torch.testing.assert_close(env._rear_extension(), torch.ones(2, 2))
    env.dof_pos[:, [8, 11]] += 0.5
    assert torch.all(env._rear_extension() < 0.5)
    env.dof_vel.zero_()
    env.dof_vel[:, 6:] = 0.5
    assert torch.all(env._reward_balance_rear_motion() == 0)
    env.dof_vel[:, 6:] = 3.0
    torch.testing.assert_close(env._reward_balance_rear_motion(), torch.full((2,), -4.0))


def test_front_diagnostics_report_world_velocity_and_normal_load(env):
    env._rigid_body_lin_vel[:, env.feet_indices[0]] = torch.tensor([0.3, 0.4, -0.2])
    env.contact_forces[:, env.feet_indices[0], 2] = 42.0
    env.update_training_diagnostics()
    env.training_diagnostics.finish()
    d = env.training_diagnostics
    assert d.front_left_load_n.item() == pytest.approx(42)
    assert d.front_left_slip_speed_m_s.item() == pytest.approx(0.5)
    assert d.front_left_vz_m_s.item() == pytest.approx(-0.2)


def test_all_balance_rewards_and_diagnostics_are_finite(env):
    for _ in range(5):
        env.step()
        env.update_training_diagnostics()
    env.training_diagnostics.finish()
    for name in class_to_dict(Go2BalanceRunnerCfg().critic.reward.weights):
        value = getattr(env, "_reward_" + name)()
        assert value.shape == (2,)
        assert torch.isfinite(value).all()
    assert all(
        torch.isfinite(getattr(env.training_diagnostics, name))
        for name in env.training_diagnostics.names
    )


def test_impulses_require_advanced_level_schedule_and_settled_hold(env):
    interval = math.ceil(env.cfg.balance_perturbations.interval_s / env.dt)
    env.cfg.balance_perturbations.undisturbed_fraction = 0
    mask = torch.ones(2, dtype=torch.bool)
    env.curriculum_level = 0
    env._reset_idx(mask)
    env.episode_length_buf[:] = 300
    env._hold_time[:] = 1
    before = env.root_states[:, 7:9].clone()
    env._pre_decimation_step()
    torch.testing.assert_close(env.root_states[:, 7:9], before)
    env.cfg.handstand_curriculum.replay_fraction = 0
    env.curriculum_level = 3
    env._reset_idx(mask)
    env.terminated.zero_()
    env.timed_out.zero_()
    env._hold_time[:] = torch.tensor([1.0, 0.0])
    env._best_hold[:] = 2.5
    # The runner invokes reset_idx with an empty mask on nonterminal steps.
    # It must not invalidate the eligibility of episodes awaiting a push.
    env._record_handstand_episodes(torch.zeros(2, dtype=torch.bool))
    assert env._primary_reset.all()
    env.episode_length_buf[:] = interval - 1
    before = env.root_states[:, 7:9].clone()
    env._pre_decimation_step()
    torch.testing.assert_close(env.root_states[:, 7:9], before)
    env.episode_length_buf[:] = interval
    env._pre_decimation_step()
    delta = env.root_states[:, 7:9] - before
    assert delta[0].norm().item() == pytest.approx(0.03, abs=1e-6)
    assert delta[1].norm().item() == 0
    assert env._best_hold[0].item() == 0
    assert env._hold_time[0].item() == 0
    assert env._push_count.tolist() == [1, 0]
    env.episode_length_buf[:] = 2 * interval
    env._hold_time[:] = 1
    before = env.root_states[:, 7:9].clone()
    env._pre_decimation_step()
    torch.testing.assert_close(env.root_states[0, 7:9], before[0])
    assert env._push_count.tolist() == [1, 1]


def test_push_level_cannot_promote_using_only_pre_push_hold(env):
    env.cfg.balance_perturbations.undisturbed_fraction = 0
    env.cfg.handstand_curriculum.replay_fraction = 0
    env.curriculum_level = 3
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    env._best_hold[:] = 5
    env.timed_out[:] = True
    env._record_handstand_episodes(torch.ones(2, dtype=torch.bool))
    assert env._curriculum_episodes == 2
    assert env._curriculum_successes == 0


def test_push_level_promotes_only_after_actual_push_and_recovery(env):
    c = env.cfg.handstand_curriculum
    c.replay_fraction = 0
    c.min_episodes = 2
    p = env.cfg.balance_perturbations
    p.undisturbed_fraction = 0
    env.curriculum_level = 3
    mask = torch.ones(2, dtype=torch.bool)
    for _ in range(c.required_batches):
        env._reset_idx(mask)
        env.terminated.zero_()
        env.timed_out.zero_()
        env.episode_length_buf[:] = math.ceil(p.interval_s / env.dt)
        env._hold_time[:] = 1
        env._pre_decimation_step()
        assert env._push_count.tolist() == [1, 1]
        assert torch.all(env._best_hold == 0)
        # Stand in for a subsequent continuous strict hold after the impulse.
        env._best_hold[:] = c.hold_seconds + 0.1
        env.timed_out[:] = True
        env._record_handstand_episodes(mask)
        # Avoid recording the same synthetic completed episodes again.
        env._episode_active.zero_()
    assert env.curriculum_level == 4


def test_undisturbed_episodes_at_push_levels_do_not_promote(env, monkeypatch):
    env.cfg.handstand_curriculum.replay_fraction = 0
    env.cfg.balance_perturbations.undisturbed_fraction = 0.5
    env.curriculum_level = 3
    # Draw below the undisturbed threshold for every environment.
    monkeypatch.setattr(
        torch, "rand", lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    )
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    assert torch.all(env._episode_push_limit == 0)
    assert not env._primary_reset.any()
    env._best_hold[:] = 5
    env.timed_out[:] = True
    env._record_handstand_episodes(torch.ones(2, dtype=torch.bool))
    assert env._curriculum_episodes == 0


def test_perturbation_curriculum_state_round_trips(env):
    env.curriculum_level = 3
    env.curriculum_success_rate = 0.75
    state = env.training_state_dict()
    env.curriculum_level = 0
    env.load_training_state_dict(state)
    assert env.curriculum_level == 3
    assert env.curriculum_success_rate == 0.75
    assert env.handstand_stage.tolist() == [2, 2]


def test_hold_followed_by_fall_cannot_promote(env):
    env._best_hold[:] = env.cfg.handstand_curriculum.hold_seconds + 1
    env.terminated[:] = True
    env._record_handstand_episodes(torch.ones(2, dtype=torch.bool))
    assert env._curriculum_successes == 0


def test_unloading_and_slip_are_penalties_without_termination(env):
    env.episode_length_buf[:] = 100
    env.contact_forces.zero_()
    env._rigid_body_lin_vel[:, env.feet_indices[:2], :2] = 1
    for _ in range(40):
        env._post_decimation_step()
        env._check_terminations_and_timeouts()
    assert not env.terminated.any()
    assert torch.all(env._reward_balance_front_unloading() < 0)
    env.contact_forces[:, env.feet_indices[:2], 2] = 49.05
    assert torch.all(env._reward_handstand_slip() < 0)
    assert torch.all(env._reward_handstand_balance() == 0)


@pytest.mark.parametrize("failure", ["contact", "collapse"])
def test_failure_requires_continuous_grace_and_resets(env, failure):
    env.contact_forces.zero_()
    env.episode_length_buf[:] = 0
    if failure == "contact":
        env.contact_forces[:, env.termination_contact_indices, 2] = 10
        timer = env._body_contact_time
        duration = env.cfg.balance_failure.body_contact_seconds
    else:
        env.root_states[:, 2] = 0.05
        timer = env._collapse_time
        duration = env.cfg.balance_failure.collapse_seconds
    env._post_decimation_step()
    assert torch.all(timer == 0)
    env.episode_length_buf[:] = 100
    steps = math.ceil(duration / env.dt)
    for _ in range(steps - 1):
        env._post_decimation_step()
    env._check_terminations_and_timeouts()
    assert not env.terminated.any()
    # A recovered frame clears the continuous timer.
    forces = env.contact_forces.clone()
    height = env.root_states[:, 2].clone()
    env.contact_forces.zero_()
    env.root_states[:, 2] = 0.5
    env._post_decimation_step()
    assert torch.all(timer == 0)
    env.contact_forces[:] = forces
    env.root_states[:, 2] = height
    for _ in range(steps):
        env._post_decimation_step()
    env._check_terminations_and_timeouts()
    assert env.terminated.all()
    env._reset_idx(torch.tensor([True, False]))
    assert timer[0] == 0
    assert timer[1] > 0


def test_sustained_reward_requires_current_valid_hold(env):
    env.contact_forces.zero_()
    weight = 9.81 * env._backend.link_mass.sum(dim=1)
    env.contact_forces[:, env.feet_indices[:2], 2] = weight[:, None] / 2
    env.projected_gravity[:] = torch.tensor([1., 0., 0.])
    env.base_lin_vel.zero_()
    env.base_ang_vel.zero_()
    env._rigid_body_lin_vel.zero_()
    env._rigid_body_pos[:, env.feet_indices[:2], 2] = 0
    env._rigid_body_pos[:, env.feet_indices[2:], 2] = 0.6
    env._hold_time[:] = torch.tensor([0., 2.])
    torch.testing.assert_close(env._reward_balance_sustained(), torch.tensor([0., 1.]))
    env._rigid_body_lin_vel[:, env.feet_indices[:2], 0] = 0.2
    assert torch.all(env._reward_balance_sustained() == 0)
    env._update_handstand_hold()
    assert torch.all(env._hold_time == 0)


def test_fall_cost_is_independent_of_timestep_and_excludes_timeouts(env):
    weight = Go2BalanceRunnerCfg.critic.reward.termination_weight.termination
    env.terminated[:] = torch.tensor([True, False])
    env.timed_out[:] = torch.tensor([False, True])
    for dt in (0.01, 0.02):
        env.dt = dt
        torch.testing.assert_close(
            env._reward_termination() * weight * dt, torch.tensor([-20.0, 0.0])
        )
