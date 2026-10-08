"""A height-only command must not request a moving trot or punish support."""

from types import SimpleNamespace

import torch

from gym.envs.go2.go2trot import Go2Trot
from gym.envs.go2.go2trot_config import Go2TrotCfg


def standing_task():
    task = Go2Trot.__new__(Go2Trot)
    task.cfg = Go2TrotCfg()
    task.device = "cpu"
    task.num_envs = 3
    task.commands = torch.tensor(
        [[0.0, 0.0, 0.0, 0.30], [0.0, 0.0, 0.0, 0.40], [1.0, 0.0, 0.0, 0.35]]
    )
    task.phase = torch.full((3, 1), torch.pi / 2)
    task._gait_phase_offsets = torch.tensor([0.0, torch.pi, torch.pi, 0.0])
    task._gait_dof_phase_offsets = task._gait_phase_offsets.repeat_interleave(3)
    task._gait_joint_offsets = torch.tensor(task.cfg.control.gait_joint_offsets)
    task._gait_joint_amplitudes = torch.tensor(task.cfg.control.gait_joint_amplitudes)
    task.gait_reference = torch.zeros(3, 12)
    task.feet_indices = torch.arange(4)
    task.contact_forces = torch.zeros(3, 4, 3)
    task.contact_forces[:, :, 2] = 9.81 * 10 / 4
    task._rigid_body_lin_vel = torch.zeros(3, 4, 3)
    task._backend = SimpleNamespace(link_mass=torch.full((3, 1), 10.0))
    return task


def test_height_only_reference_is_stationary_but_moving_reference_trots():
    task = standing_task()
    task._update_gait_reference()
    before = task.gait_reference.clone()
    task.phase += torch.pi
    task._update_gait_reference()
    torch.testing.assert_close(task.gait_reference[:2], before[:2])
    assert not torch.allclose(task.gait_reference[2], before[2])


def test_standing_support_rewards_four_feet_without_swing_penalty():
    task = standing_task()
    assert torch.all(task._reward_trot_support()[:2] > 0)
    torch.testing.assert_close(task._reward_swing_contact()[:2], torch.zeros(2))
    task.contact_forces[0, 0, 2] = 0
    assert task._reward_trot_support()[0] == 0


def test_height_reference_extends_legs_without_fore_aft_foot_motion():
    task = standing_task()
    task._update_gait_reference()
    thigh = task.gait_reference[:2, 1]
    calf = task.gait_reference[:2, 2]
    length = task.cfg.control.standing_leg_length
    height = length * (thigh.cos() + (thigh + calf).cos())
    height += task.cfg.control.standing_foot_radius
    fore_aft = length * (thigh.sin() + (thigh + calf).sin())
    torch.testing.assert_close(height, task.commands[:2, 3])
    torch.testing.assert_close(fore_aft, torch.zeros(2))
    assert thigh[1] < thigh[0]


def test_slip_penalty_ignores_airborne_feet_and_vertical_motion():
    task = standing_task()
    task._rigid_body_lin_vel[:, 0, 0] = 2
    task._rigid_body_lin_vel[:, 1, 2] = 100
    task.contact_forces[1, 0, 2] = 0
    torch.testing.assert_close(
        task._reward_feet_slip(), torch.tensor([-4.0, 0.0, -4.0])
    )


def test_moving_height_command_changes_reference_without_changing_gait():
    task = standing_task()
    task._update_gait_reference()
    before = task.gait_reference[2].clone()
    task.commands[2, 3] += 0.04
    task._update_gait_reference()
    assert task.gait_reference[2, 1] < before[1]
    assert task.gait_reference[2, 2] > before[2]


def test_motion_penalty_tracks_command_and_ignores_requested_vertical_motion():
    task = standing_task()
    task.base_lin_vel = torch.zeros(3, 3)
    task.base_lin_vel[:, :2] = task.commands[:, :2]
    task.base_lin_vel[:, 2] = 0.2
    task.base_ang_vel = torch.zeros(3, 3)
    torch.testing.assert_close(task._reward_unwanted_motion(), torch.zeros(3))
    task.base_lin_vel[:, 0] += 0.5
    assert torch.all(task._reward_unwanted_motion() < 0)
