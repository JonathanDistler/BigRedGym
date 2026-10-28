"""Top balance references, shaping signals, diagnostics and perturbation gates."""

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
    env.episode_length_buf[:] = 299
    before = env.root_states[:, 7:9].clone()
    env._pre_decimation_step()
    torch.testing.assert_close(env.root_states[:, 7:9], before)
    env.episode_length_buf[:] = 300
    env._pre_decimation_step()
    delta = env.root_states[:, 7:9] - before
    assert delta[0].norm().item() == pytest.approx(0.03, abs=1e-6)
    assert delta[1].norm().item() == 0
    assert env._best_hold[0].item() == 0
    assert env._hold_time[0].item() == 0
    assert env._push_count.tolist() == [1, 0]
    env.episode_length_buf[:] = 600
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
        env.episode_length_buf[:] = 300
        env._hold_time[:] = 1
        env._pre_decimation_step()
        assert env._push_count.tolist() == [1, 1]
        assert torch.all(env._best_hold == 0)
        # Stand in for a subsequent continuous strict hold after the impulse.
        env._best_hold[:] = 2.1
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
