"""Playback must support environment buffers created during checkpoint loading."""

from types import SimpleNamespace

import torch

from gym.envs.base.task_skeleton import TaskSkeleton
from scripts.play import play


def test_play_updates_inference_buffers_outside_callers_inference_mode():
    class PlaybackTask(TaskSkeleton):
        max_episode_length = 1

        def __init__(self):
            super().__init__()
            self.steps = 0
            self.exit = False
            self.contact_forces = torch.zeros(1, 1, 3)
            self.termination_contact_indices = [0]

        def step(self):
            self._reset_buffers()
            self.episode_length_buf += 1
            self._check_terminations_and_timeouts()
            self.steps += 1

    env = PlaybackTask()
    # Match the reset performed while restoring handstand checkpoint state.
    with torch.inference_mode():
        env.step()
    assert env.timed_out.is_inference()
    assert not torch.is_inference_mode_enabled()

    runner = SimpleNamespace(
        actor_cfg={"actions": [], "disable_actions": False},
        get_inference_actions=lambda: torch.zeros(1, 0),
        set_actions=lambda *args: None,
    )
    play(env, runner)

    assert env.steps == 11
    assert not torch.is_inference_mode_enabled()


def test_kickup_playback_repeats_completed_episodes():
    class KickTask:
        max_episode_length = 1
        repeat_kickup_episodes = True
        _episode_active = torch.ones(1, dtype=torch.bool)

        def step(self):
            pass

        def check_exit(self):
            pass

    resets = []
    runner = SimpleNamespace(
        actor_cfg={"actions": [], "disable_actions": False},
        get_inference_actions=lambda: torch.zeros(1, 0),
        set_actions=lambda *args: None,
        reset_envs=lambda: resets.append(True),
    )
    env = KickTask()
    play(env, runner)
    assert len(resets) == 10
    assert not env._episode_active.any()
