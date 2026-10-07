"""Low-height collapse must end an episode even without torso contact."""
from types import SimpleNamespace

import torch

from gym.envs.base.legged_robot import LeggedRobot
from gym.envs.go2.go2_config import Go2Cfg
from gym.envs.go2.go2trot_config import Go2TrotCfg


def test_go2_collapse_terminates_without_contact():
    for cfg_type in (Go2Cfg, Go2TrotCfg):
        env = LeggedRobot.__new__(LeggedRobot)
        env.cfg = cfg_type()
        env.root_states = torch.zeros(4, 13)
        env.root_states[:, 2] = torch.tensor([0.16, 0.20, 0.30, 0.45])
        env.contact_forces = torch.zeros(4, 1, 3)
        env.termination_contact_indices = torch.tensor([0])
        env.terminated = torch.zeros(4, dtype=torch.bool)
        env.episode_length_buf = torch.zeros(4)
        env.max_episode_length = 500
        env._check_terminations_and_timeouts()
        assert env.terminated.tolist() == [True, False, False, False]
        assert not env.timed_out.any()


def test_height_termination_is_optional_for_existing_tasks():
    env = LeggedRobot.__new__(LeggedRobot)
    env.cfg = SimpleNamespace(env=SimpleNamespace())
    env.root_states = torch.zeros(1, 13)
    env.contact_forces = torch.zeros(1, 1, 3)
    env.termination_contact_indices = torch.tensor([0])
    env.terminated = torch.zeros(1, dtype=torch.bool)
    env.episode_length_buf = torch.tensor([500])
    env.max_episode_length = 500
    env._check_terminations_and_timeouts()
    assert not env.terminated.any()
    assert env.timed_out.all()
