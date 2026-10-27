"""Reset geometry, success-based progression, replay, and checkpoint resume."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2_config import FOOT_RADIUS, GO2_HIP_X, GO2_FOOT_Y, LEG_LENGTH
from gym.envs.go2.go2handstand import Go2Handstand
from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg
from gym.utils.task_registry import task_registry
from gym.utils.torch_quat import quat_apply
from tests.unit_tests.test_go2_handstand import task


@pytest.fixture
def env():
    cfg = Go2HandstandCfg()
    cfg.seed = 7
    cfg.env.num_envs = 8
    task_registry.convert_frequencies_to_params(cfg, Go2HandstandRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        yield Go2Handstand(cfg, "cpu", True, backend)
    finally:
        backend.close()


def test_each_reset_level_has_valid_grounded_geometry_and_matching_pd_targets(env):
    c = env.cfg.handstand_curriculum
    c.replay_fraction = c.standing_fraction = 0
    for level in range(len(c.pitch_degrees)):
        env.curriculum_level = level
        env._reset_idx(torch.ones(8, dtype=torch.bool))
        joints = env.dof_pos.index_select(1, env.actuated_dof_indices).reshape(8, 4, 3)
        assert torch.all(env.dof_pos >= env.dof_pos_limits[:, 0])
        assert torch.all(env.dof_pos <= env.dof_pos_limits[:, 1])
        feet = env._rigid_body_pos[:, env.feet_indices] - env.root_states[:, None, :3]
        thigh, calf = joints[:, :, 1], joints[:, :, 2]
        expected = torch.zeros_like(feet)
        expected[:, :, 0] = torch.tensor(GO2_HIP_X) - LEG_LENGTH * (
            thigh.sin() + (thigh + calf).sin()
        )
        expected[:, :, 1] = torch.tensor(GO2_FOOT_Y)
        expected[:, :, 2] = -LEG_LENGTH * (thigh.cos() + (thigh + calf).cos())
        expected = quat_apply(env.base_quat[:, None].expand(-1, 4, -1), expected)
        torch.testing.assert_close(feet, expected, atol=2e-6, rtol=1e-5)
        heights = (
            env._rigid_body_pos[:, env.feet_indices, 2] - env.env_origins[:, None, 2]
        )
        torch.testing.assert_close(
            heights.amin(dim=1), torch.full((8,), FOOT_RADIUS + c.ground_clearance)
        )
        if level < 5:
            assert torch.all(heights[:, 2:] > FOOT_RADIUS + 0.05)
        else:
            torch.testing.assert_close(
                heights, torch.full_like(heights, FOOT_RADIUS + c.ground_clearance)
            )
        torch.testing.assert_close(
            env.dof_pos_target + env._standing_posture(), joints.flatten(1)
        )
        expected_stage = 2 if level < 2 else (0 if level == 5 else 1)
        assert torch.all(env.handstand_stage == expected_stage)
        # Inspect actual native collision distances; no geometry starts buried.
        for data in env._backend._datas:
            assert all(contact.dist > -0.003 for contact in data.contact)
    assert env._backend._models[0].opt.gravity[2] == pytest.approx(-9.81)


def test_curriculum_requires_two_successful_batches_and_excludes_replay(env):
    c = env.cfg.handstand_curriculum
    c.min_episodes = 4
    env._episode_active[:] = True
    env._primary_reset[:] = True
    env._reset_level[:] = 0
    env.timed_out[:] = True
    env._best_hold[:] = 0
    env._best_hold[:4] = 2.1
    mask = torch.tensor([True] * 4 + [False] * 4)
    env._record_handstand_episodes(mask)
    assert env.curriculum_level == 0
    env._record_handstand_episodes(mask)
    assert env.curriculum_level == 1
    # Old-level episodes in flight cannot advance the new level.
    env._record_handstand_episodes(mask)
    assert env._curriculum_episodes == 0
    env._reset_level[:] = 1
    env._primary_reset[:] = False
    env._record_handstand_episodes(mask)
    assert env._curriculum_episodes == 0
    # A failing batch resets the consecutive-pass count.
    env._primary_reset[:] = True
    env._best_hold.zero_()
    env._record_handstand_episodes(mask)
    assert env._curriculum_passes == 0
    assert env.curriculum_level == 1


def test_hold_is_continuous_and_rejects_termination():
    env = task()
    env.num_envs = 1
    env.device = "cpu"
    env._init_handstand_curriculum()
    env.handstand_stage[:] = 2
    env.terminated = torch.zeros(1, dtype=torch.bool)
    env.timed_out = torch.zeros_like(env.terminated)
    for _ in range(100):
        env._update_handstand_hold()
    assert env._hold_time.item() == pytest.approx(1.0)
    env.terminated[:] = True
    env._update_handstand_hold()
    assert env._hold_time.item() == 0
    assert env._best_hold.item() == pytest.approx(1.0)
    env.terminated[:] = False
    env._update_handstand_hold()
    assert env._hold_time.item() == pytest.approx(0.01)


def test_mixture_retains_primary_replay_and_standing(env):
    env.curriculum_level = 3
    levels, primary = [], []
    for _ in range(40):
        env._reset_idx(torch.ones(8, dtype=torch.bool))
        levels.extend(env._reset_level.tolist())
        primary.extend(env._primary_reset.tolist())
    assert -1 in levels and 0 in levels and 3 in levels
    assert all((level == 3) == is_primary for level, is_primary in zip(levels, primary))


def test_checkpoint_restores_curriculum_and_handles_old_checkpoints(env, tmp_path):
    from learning.runners.on_policy_runner import OnPolicyRunner

    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    actor, critic = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    runner.alg = SimpleNamespace(
        actor=actor,
        critic=critic,
        optimizer=torch.optim.Adam(actor.parameters()),
        critic_optimizer=torch.optim.Adam(critic.parameters()),
    )
    # Rollouts can replace derived buffers with inference tensors.
    with torch.inference_mode():
        env.step()
    runner.env, runner.log_dir, runner.it = env, str(tmp_path), 7
    runner.device = "cpu"
    env.curriculum_level = 3
    env._curriculum_episodes = 40
    env._curriculum_successes = 30
    env._curriculum_passes = 1
    runner.save()
    env.curriculum_level = 0
    env._curriculum_episodes = 0
    runner.load(tmp_path / "model_7.pt")
    assert runner.it == 7
    assert env.curriculum_level == 3
    assert env._curriculum_episodes == 40
    assert env._curriculum_successes == 30
    assert env._curriculum_passes == 1
    assert torch.all(env._reset_level[env._primary_reset] == 3)
    saved = torch.load(tmp_path / "model_7.pt", weights_only=True)
    del saved["environment_training_state"]
    torch.save(saved, tmp_path / "old.pt")
    runner.load(tmp_path / "old.pt")
    assert env.curriculum_level == 3


def test_disabled_curriculum_starts_standing(env):
    env.cfg.handstand_curriculum.enabled = False
    env._reset_idx(torch.ones(8, dtype=torch.bool))
    assert torch.all(env.handstand_stage == 0)
    assert torch.all(env._reset_level == -1)
    assert torch.all(env.projected_gravity[:, 2] < -0.99)
