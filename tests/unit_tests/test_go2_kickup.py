"""Kick-up support gates and residual reference integration."""

import pytest
import torch

from gym.envs.go2.go2kickup import Go2Kickup
from gym.envs.go2.go2kickup_config import Go2KickupCfg, Go2KickupRunnerCfg
from tests.unit_tests.test_go2_handstand import task


def kick_task():
    env = task()
    env.__class__ = Go2Kickup
    env.cfg = Go2KickupCfg()
    env._front_anchor = torch.zeros(1, 2, 2)
    env._front_anchor_valid = torch.ones(1, dtype=torch.bool)
    env.handstand_stage[:] = 2
    return env


@pytest.mark.parametrize("failure", ["rear_low", "front_slip", "front_drift"])
def test_success_rejects_motion_exploits(failure):
    env = kick_task()
    assert env._reward_handstand_balance().item() == pytest.approx(1)
    if failure == "rear_low":
        env._rigid_body_pos[:, 2, 2] = 0.05
    elif failure == "front_slip":
        env._rigid_body_lin_vel[:, 0, 0] = 0.2
    else:
        env._rigid_body_pos[:, 0, 0] = 0.2
    assert env._reward_handstand_balance().item() == 0


def test_kickup_mujoco_reference_rewards_and_partial_reset():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.helpers import class_to_dict
    from gym.utils.task_registry import task_registry

    cfg = Go2KickupCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    cfg.handstand_curriculum.replay_fraction = 0.0
    cfg.handstand_curriculum.standing_fraction = 0.0
    runner = Go2KickupRunnerCfg()
    task_registry.convert_frequencies_to_params(cfg, runner)
    backend = MuJocoCPUBackend()
    try:
        env = Go2Kickup(cfg, "cpu", True, backend)
        assert len(cfg.handstand_curriculum.pitch_degrees) == 7
        env.curriculum_level = 6
        env._reset_idx(torch.ones(2, dtype=torch.bool))
        env.episode_length_buf[:] = 0
        start = env._standing_posture().clone()
        env.episode_length_buf[:] = round(cfg.kick_motion.crouch_seconds / env.dt)
        crouch = env._standing_posture().clone()
        env.episode_length_buf[:] = round(
            (cfg.kick_motion.crouch_seconds + cfg.kick_motion.extend_seconds) / env.dt
        )
        thrust = env._standing_posture().clone()
        assert (crouch[:, 2] < start[:, 2]).all()
        assert (thrust[:, 8] > crouch[:, 8]).all()
        assert torch.allclose(thrust[:, :6], crouch[:, :6])
        for _ in range(5):
            env.step()
            env.update_training_diagnostics()
        assert torch.isfinite(env.get_states(runner.actor.obs)).all()
        for name in class_to_dict(runner.critic.reward.weights):
            reward = getattr(env, "_reward_" + name)()
            assert reward.shape == (2,)
            assert torch.isfinite(reward).all()
        env.curriculum_level = 0
        env._reset_idx(torch.tensor([True, False]))
        assert torch.allclose(env._standing_posture()[0], env.dof_pos[0])
        assert torch.allclose(env.dof_pos_target[0], torch.zeros(12))
        assert env._front_anchor_valid.tolist() == [False, True]
        env.training_diagnostics.finish()
        assert torch.isfinite(env.training_diagnostics.rear_foot_height_m)
    finally:
        backend.close()
