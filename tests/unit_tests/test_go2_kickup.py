"""Ground geometry, support, reset isolation and standing-start curriculum."""

import pytest
import torch

from gym.envs.go2.go2kickup_config import Go2KickupCfg, Go2KickupRunnerCfg
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.utils.helpers import class_to_dict
from gym.utils.task_registry import task_registry


@pytest.fixture
def env():
    import gym.envs  # noqa: F401

    cfg, runner = Go2KickupCfg(), Go2KickupRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    task_registry.convert_frequencies_to_params(cfg, runner)
    task = task_registry.make_env(
        "go2kickup", cfg, device="cpu", headless=True, backend="mujoco"
    )
    yield task
    task._backend.close()


def plant(env):
    env.contact_forces.zero_()
    env.contact_forces[:, env.feet_indices[:2], 2] = 60
    env._rigid_body_lin_vel.zero_()
    env._rigid_body_pos[:, env.feet_indices[:2], :2] = env._front_anchor
    env._rigid_body_pos[:, env.feet_indices[2:], 2] = FOOT_RADIUS + 0.006
    env.terminated.zero_()
    env._support_failed.zero_()
    env._kick_clock[:] = 1.0
    env._takeoff_time[:] = 0.95
    env._extension_time[:] = 0.75


def test_clearance_uses_ground_and_both_feet(env):
    plant(env)
    assert torch.allclose(env._rear_clearance(), torch.full((2, 2), 0.006))
    full = env._reward_kick_clearance().clone()
    env._rigid_body_pos[0, env.feet_indices[2], 2] = FOOT_RADIUS
    assert env._reward_kick_clearance()[0] < full[0] * 0.2
    assert env._reward_kick_clearance()[1] == full[1]
    # Folding knees cannot earn clearance with feet on the floor.
    env.dof_pos[:, [8, 11]] = -2.5
    env._rigid_body_pos[:, env.feet_indices[2:], 2] = FOOT_RADIUS
    assert torch.equal(env._reward_kick_clearance(), torch.zeros(2))
    env._rigid_body_pos[:, env.feet_indices[:2], 2] = 0.3
    assert env._rear_clearance().max() == 0


@pytest.mark.parametrize("failure", ["contact", "speed", "drift"])
def test_support_gates_success_and_dense_reward(env, failure):
    plant(env)
    assert env._valid_kick_support().all()
    if failure == "contact":
        env.contact_forces[0, env.feet_indices[0], 2] = 0
    elif failure == "speed":
        env._rigid_body_lin_vel[0, env.feet_indices[0], 0] = 1
    else:
        env._rigid_body_pos[0, env.feet_indices[0], 0] += 0.2
    assert env._valid_kick_support().tolist() == [False, True]
    for _ in range(10):
        env._update_handstand_hold()
    assert env._best_hold[0] == 0
    assert env._best_hold[1] >= env.cfg.handstand_curriculum.hold_seconds
    assert env._reward_kick_clearance()[0] < env._reward_kick_clearance()[1]


def test_reference_and_partial_reset(env):
    env._reset_idx(torch.ones(2, dtype=torch.bool))
    start = env._standing_posture().clone()
    assert torch.allclose(start, env.dof_pos, atol=1e-5)
    env._kick_clock[:] = 0.7
    crouch = env._standing_posture().clone()
    assert (crouch[:, 8] < start[:, 8]).all()
    assert torch.equal(crouch[:, :6], start[:, :6])
    env._kick_clock[:] = 1.0
    thrust = env._standing_posture().clone()
    assert (thrust[:, 8] > crouch[:, 8]).all()
    assert torch.equal(thrust[:, :6], start[:, :6])
    env._kick_clock[:] = 0.7 - 1e-5
    left = env._standing_posture().clone()
    env._kick_clock[:] = 0.7 + 1e-5
    assert (left - env._standing_posture()).abs().max() < 1e-4
    anchor = env._front_anchor[1].clone()
    env._reset_idx(torch.tensor([True, False]))
    assert env._kick_clock[0] == 0
    assert env._kick_clock[1] > 0
    assert torch.equal(anchor, env._front_anchor[1])
    assert torch.allclose(env.dof_pos_target[0], torch.zeros(12), atol=2e-7)
    for _ in range(5):
        env.step()
    runner = Go2KickupRunnerCfg()
    assert torch.isfinite(env.get_states(runner.actor.obs)).all()
    for name in class_to_dict(runner.critic.reward.weights):
        reward = getattr(env, "_reward_" + name)()
        assert reward.shape == (2,)
        assert torch.isfinite(reward).all()


def test_curriculum_batches_failure_and_persistence(env):
    c = env.cfg.handstand_curriculum
    c.min_episodes = 2
    c.required_batches = 2
    mask = torch.ones(2, dtype=torch.bool)

    def complete(success=True, failed=False):
        env._best_hold[:] = c.hold_seconds if success else 0
        env._support_failed[:] = failed
        env.terminated.zero_()
        env.timed_out.fill_(True)
        env._reset_idx(mask)

    complete(failed=True)
    assert env.curriculum_level == 0
    complete()
    assert env.curriculum_level == 0
    complete()
    assert env.curriculum_level == 1
    assert (env._goal("clearance") == 0.012).all()
    # Manual resets and stale episodes never promote.
    env.timed_out.zero_()
    env._best_hold[:] = 1
    env._reset_idx(mask)
    assert env._curriculum_episodes == 0
    saved = env.training_state_dict()
    env.load_training_state_dict(saved)
    assert env.curriculum_level == 1
    assert env._best_hold.max() == 0
    assert (env.projected_gravity[:, :2].abs() < 1e-3).all()


def test_policy_residual_is_bounded_and_slewed(env):
    env.dof_pos_target.zero_()
    env.set_state("dof_pos_target", torch.full((2, 12), 100.0))
    assert env.dof_pos_target.abs().max() <= 0.04
    for _ in range(30):
        env.set_state("dof_pos_target", torch.full((2, 12), 100.0))
    assert env.dof_pos_target[:, :6].abs().max() <= 0.12
    expected = super(type(env), env)._compute_torques()
    assert torch.equal(env._compute_torques(), expected)


def test_forty_percent_batches_promote_and_final_level_needs_no_balance(env):
    c = env.cfg.handstand_curriculum
    c.min_episodes = 10
    c.required_batches = 2
    mask = torch.ones(2, dtype=torch.bool)
    for batch in range(2):
        for episode in range(5):
            env._best_hold[:] = c.hold_seconds if episode < 2 else 0
            env.terminated.zero_()
            env.timed_out.fill_(True)
            env._reset_idx(mask)
        assert env.curriculum_success_rate == pytest.approx(0.4)
        assert env.curriculum_level == batch
    env.curriculum_level = 6
    env._reset_idx(mask)
    plant(env)
    env._rigid_body_pos[:, env.feet_indices[2:], 2] = FOOT_RADIUS + 0.25
    env.handstand_stage.zero_()
    for _ in range(10):
        env._update_handstand_hold()
    assert (env._best_hold >= c.hold_seconds).all()


def test_support_failure_latches_and_new_episode_clears_it(env):
    plant(env)
    env._rigid_body_pos[0, env.feet_indices[0], 0] += 0.2
    env._post_decimation_step()
    assert env._support_failed.tolist() == [True, False]
    plant(env)
    # plant clears the failure for fixtures; recreate a previously failed episode.
    env._support_failed[0] = True
    assert not env._valid_kick_support()[0]
    env._reset_idx(torch.tensor([True, False]))
    assert not env._support_failed.any()


def test_world_velocity_sync_and_phase_gates(env):
    plant(env)
    rear = env.feet_indices[2:]
    env._rigid_body_lin_vel[:, rear, 2] = 0.6
    assert (env._reward_kick_upward_velocity() > 0).all()
    synchronized = env._reward_kick_synchronization().clone()
    env._rigid_body_lin_vel[0, rear[0], 2] = -0.2
    assert env._reward_kick_upward_velocity()[0] == 0
    assert env._reward_kick_synchronization()[0] < synchronized[0]
    env._kick_clock[:] = 0.4  # crouch, no flight shaping
    assert env._reward_kick_clearance().max() == 0
    assert env._reward_kick_upward_velocity().max() == 0
    env._kick_clock[:] = 2.0  # finite kick window prevents lingering rewards
    assert env._reward_kick_clearance().max() == 0


def test_takeoff_timing_gates_curriculum_and_release(env):
    plant(env)
    env._takeoff_time[0] = torch.tensor([0.8, 1.0])
    assert env._reward_kick_takeoff_timing()[0] < 0
    assert env._reward_kick_release()[0] == 0
    for _ in range(10):
        env._update_handstand_hold()
    assert env._best_hold[0] == 0
    assert env._best_hold[1] > 0
    env._takeoff_time[0, 1] = -1
    assert not env._synchronized_takeoff()[0]


def prepare_push(env):
    plant(env)
    env._kick_clock[:] = 0.8
    env._takeoff_time[:] = -1
    env._extension_time[:] = -1
    env._push_impulse.zero_()
    env._useful_push_impulse.zero_()
    env._rigid_body_pos[:, env.feet_indices[2:], 2] = FOOT_RADIUS
    env.dof_vel[:, [8, 11]] = 1.0
    weight = env._body_weight()
    env.contact_forces[:, env.feet_indices[2:], 2] = weight[:, None] * 0.65
    env._pre_decimation_step()
    return weight


def test_impulse_integrates_physics_samples_and_normalizes(env):
    weight = prepare_push(env)
    for _ in range(env.cfg.control.decimation):
        env._post_physics_step()
    assert torch.allclose(
        env._push_impulse, weight[:, None].expand(-1, 2) * 0.65 * env.dt
    )
    # The runner multiplies rates by control dt: no extra dt in reward magnitude.
    assert torch.allclose(
        env._reward_kick_push_impulse(), torch.full((2,), 0.4 / 0.65), atol=1e-5
    )
    env._kick_clock[:] = 1.1
    previous = env._push_impulse.clone()
    env._pre_decimation_step()
    env._post_physics_step()
    assert torch.equal(env._push_impulse, previous)
    assert env._reward_kick_push_impulse().max() == 0


def test_static_force_unbalanced_force_spikes_and_slip(env):
    prepare_push(env)
    env.dof_vel.zero_()
    env._post_physics_step()
    assert env._reward_kick_push_impulse().max() == 0
    prepare_push(env)
    env.contact_forces[:, env.feet_indices[3], 2] = 0
    env._post_physics_step()
    assert env._reward_kick_push_impulse().max() == 0
    prepare_push(env)
    env.contact_forces[:, env.feet_indices[2:], 2] *= 10
    env._rigid_body_lin_vel[:, env.feet_indices[2:], 0] = 1
    env._post_physics_step()
    assert (env._reward_kick_force_spikes() < 0).all()
    assert (env._reward_kick_rear_slip() < 0).all()
    assert env._reward_kick_push_impulse().max() <= 1


def test_takeoff_tracking_requires_prior_load_and_upward_world_motion(env):
    prepare_push(env)
    env._post_physics_step()
    env._kick_clock[:] = 0.9
    rear = env.feet_indices[2:]
    env.contact_forces[:, rear, 2] = 0
    env._rigid_body_pos[:, rear, 2] = FOOT_RADIUS + 0.01
    env._rigid_body_lin_vel[:, rear, 2] = 0.4
    env._post_physics_step()
    assert (env._takeoff_time >= 0).all()
    first = env._takeoff_time.clone()
    env._post_physics_step()
    assert torch.equal(first, env._takeoff_time)
    env._reset_idx(torch.tensor([True, False]))
    assert (env._takeoff_time[0] == -1).all()
    assert torch.equal(first[1], env._takeoff_time[1])
    assert env._push_impulse[0].max() == 0
    env.update_training_diagnostics()
    env.training_diagnostics.finish()
    assert torch.isfinite(env.training_diagnostics.rear_left_clearance_m)


def test_training_contract():
    cfg = Go2KickupRunnerCfg()
    assert Go2KickupCfg.env.num_envs == 16
    assert cfg.algorithm.rollout_size == 8192
    assert cfg.runner.save_interval == 25
    assert cfg.runner.disable_wandb
    assert not cfg.runner.resume


def test_impulse_is_mass_invariant_and_saturates(env):
    prepare_push(env)
    for _ in range(env.cfg.control.decimation):
        env._post_physics_step()
    reference = env._reward_kick_push_impulse().clone()
    env._backend.link_mass.mul_(2)
    prepare_push(env)
    for _ in range(env.cfg.control.decimation):
        env._post_physics_step()
    assert torch.allclose(env._reward_kick_push_impulse(), reference)
    env._useful_push_impulse[:] = (
        env._body_weight()[:, None] * env.cfg.reward_settings.impulse_target
    )
    env._pre_decimation_step()
    env._post_physics_step()
    assert env._reward_kick_push_impulse().max() == 0


def test_extension_onset_and_support_gate(env):
    prepare_push(env)
    env.dof_vel[:, 11] = 0
    env._post_physics_step()
    assert (env._extension_time[:, 0] >= 0).all()
    assert (env._extension_time[:, 1] < 0).all()
    env._kick_clock[:] = 0.9
    env.dof_vel[:, 11] = 1
    env._post_physics_step()
    assert (env._event_difference(env._extension_time) > 0.09).all()
    assert (env._reward_kick_extension_timing() < 0).all()
    plant(env)
    env.contact_forces[:, env.feet_indices[0], 2] = 0
    env._rigid_body_lin_vel[:, env.feet_indices[2:], 2] = 1
    assert env._reward_kick_front_support().max() == 0
    assert env._reward_kick_clearance().max() == 0
    assert env._reward_kick_upward_velocity().max() == 0
    assert env._reward_kick_synchronization().max() == 0
