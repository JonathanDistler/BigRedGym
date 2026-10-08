"""Height commands stay independent of locomotion and reach policy inputs."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2_config import Go2RunnerCfg
from gym.envs.go2.go2trot_config import Go2TrotRunnerCfg
from gym.utils.interfaces.MujocoKeyboardInterface import MujocoKeyboardInterface


def test_height_arrows_change_only_height_and_survive_reset():
    env = SimpleNamespace(
        commands=torch.zeros(2, 4),
        command_ranges={"base_height": [0.30, 0.45]},
        timed_out=torch.zeros(2, dtype=torch.bool),
        max_episode_length_s=5,
        cfg=SimpleNamespace(reward_settings=SimpleNamespace(base_height_target=0.40)),
        _backend=SimpleNamespace(_viewer_key_callback=None),
    )
    env.reset = lambda: env.commands.fill_(123)
    ui = MujocoKeyboardInterface(env)
    env.commands[:, :3] = torch.tensor([0.5, -0.2, 0.3])
    locomotion = env.commands[:, :3].clone()
    ui._on_key(265)
    torch.testing.assert_close(env.commands[:, 3], torch.full((2,), 0.41))
    for _ in range(30):
        ui._on_key(264)
    torch.testing.assert_close(env.commands[:, 3], torch.full((2,), 0.30))
    for _ in range(30):
        ui._on_key(265)
    torch.testing.assert_close(env.commands[:, 3], torch.full((2,), 0.45))
    torch.testing.assert_close(env.commands[:, :3], locomotion)
    before = env.commands.clone()
    ui._on_key(ord("R"))
    torch.testing.assert_close(env.commands, before)


def test_go2_height_objective_does_not_reward_fixed_height():
    assert Go2RunnerCfg.critic.reward.weights.min_base_height == 0
    assert Go2RunnerCfg.critic.reward.weights.tracking_base_height > 0
    for cfg in (Go2RunnerCfg, Go2TrotRunnerCfg):
        assert "commands" in cfg.actor.obs
    assert "phase_obs" in Go2TrotRunnerCfg.actor.obs
    assert "phase_frequency" in Go2TrotRunnerCfg.actor.obs


@pytest.mark.parametrize("task_class", ["Go2", "Go2Trot"])
def test_height_reward_tracks_target_not_fixed_reference(task_class):
    from gym.envs.go2.go2 import Go2
    from gym.envs.go2.go2trot import Go2Trot

    task = {"Go2": Go2, "Go2Trot": Go2Trot}[task_class].__new__(
        {"Go2": Go2, "Go2Trot": Go2Trot}[task_class]
    )
    task.root_states = torch.zeros(3, 13)
    from gym.envs.go2.go2_config import Go2Cfg
    from gym.envs.go2.go2trot_config import Go2TrotCfg

    task.cfg = {"Go2": Go2Cfg, "Go2Trot": Go2TrotCfg}[task_class]()
    task.root_states[:, 2] = torch.tensor([0.30, 0.375, 0.45])
    task.commands = torch.zeros(3, 4)
    task.commands[:, 3] = task.root_states[:, 2]
    torch.testing.assert_close(task._reward_tracking_base_height(), torch.ones(3))
    task.commands[:, 3] += 0.1
    assert torch.all(task._reward_tracking_base_height() < 0.5)


def test_trot_height_reward_uses_five_centimeter_tolerance():
    from gym.envs.go2.go2trot import Go2Trot
    from gym.envs.go2.go2trot_config import Go2TrotCfg

    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.root_states = torch.zeros(3, 13)
    task.root_states[:, 2] = 0.35
    task.commands = torch.zeros(3, 4)
    task.commands[:, 3] = torch.tensor([0.35, 0.40, 0.45])
    torch.testing.assert_close(
        task._reward_tracking_base_height(), torch.exp(-torch.tensor([0., 1., 4.]))
    )
