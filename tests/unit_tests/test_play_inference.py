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
