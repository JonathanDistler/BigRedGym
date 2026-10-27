"""Handstand rewards distinguish support and held sequences from exploits."""

from types import SimpleNamespace

import pytest
import torch

from gym.envs.go2.go2handstand import Go2Handstand
from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg


def task():
    env = Go2Handstand.__new__(Go2Handstand)
    env.cfg = Go2HandstandCfg()
    env.dt = 0.01
    env.handstand_stage = torch.zeros(1)
    env._lean_time = torch.zeros(1)
    env._kick_time = torch.zeros(1)
    env.feet_indices = torch.arange(4)
    env.actuated_dof_indices = torch.arange(12)
    env._backend = SimpleNamespace(link_mass=torch.tensor([[10.0]]))
    env.contact_forces = torch.zeros(1, 4, 3)
    env.contact_forces[:, :2, 2] = 49.05
    env.projected_gravity = torch.tensor([[1.0, 0.0, 0.0]])
    env.dof_pos = torch.zeros(1, 12)
    env.dof_pos[:, [8, 11]] = -0.95
    env._rigid_body_pos = torch.zeros(1, 4, 3)
    env._rigid_body_pos[:, 2:, 2] = 0.6
    env._rigid_body_lin_vel = torch.zeros(1, 4, 3)
    env.base_lin_vel = torch.zeros(1, 3)
    env.base_ang_vel = torch.zeros(1, 3)
    return env


def test_lean_prefers_front_load_and_requires_real_support():
    env = task()
    assert env._reward_handstand_lean().item() == pytest.approx(1.0)
    env.contact_forces[:, :, 2] = 24.525
    assert env._reward_handstand_lean().item() < 0.1
    for forces in ([0, 0, 0, 0], [98.1, 0, 0, 0], [490.5, 490.5, 0, 0]):
        env.contact_forces[:, :, 2] = torch.tensor(forces)
        assert env._reward_handstand_lean().item() < 1e-4


def test_sequence_requires_continuous_lean_and_then_kick():
    env = task()
    for _ in range(5):
        env._advance_handstand_stage()
    env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    env._advance_handstand_stage()
    assert env._lean_time.item() == 0
    assert env._reward_handstand_kick().item() == 0
    assert env._reward_handstand_balance().item() == 0
    env.projected_gravity[:] = torch.tensor([1.0, 0.0, 0.0])
    for _ in range(11):
        env._advance_handstand_stage()
    assert env.handstand_stage.item() == 1
    for _ in range(5):
        env._advance_handstand_stage()
    assert env.handstand_stage.item() == 2
    assert env._reward_handstand_balance().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "failure",
    [
        "one_front",
        "rear_contact",
        "level",
        "inverted",
        "crouched",
        "low_rear",
        "spinning",
        "falling",
    ],
)
def test_balance_rejects_bad_poses(failure):
    env = task()
    env.handstand_stage[:] = 2
    if failure == "one_front":
        env.contact_forces[:, :2, 2] = torch.tensor([98.1, 0])
    elif failure == "rear_contact":
        env.contact_forces[:, 2:, 2] = 49.05
    elif failure == "level":
        env.projected_gravity[:] = torch.tensor([0.0, 0.0, -1.0])
    elif failure == "inverted":
        env.projected_gravity[:] = torch.tensor([-1.0, 0.0, 0.0])
    elif failure == "crouched":
        env.dof_pos[:, [8, 11]] = -2.0
    elif failure == "low_rear":
        env._rigid_body_pos.zero_()
    elif failure == "spinning":
        env.base_ang_vel[:] = 4
    elif failure == "falling":
        env.base_lin_vel[:] = 1
    assert env._reward_handstand_balance().item() < 0.01


def test_kick_accepts_either_leg_but_requires_lift_and_extension():
    env = task()
    env.handstand_stage[:] = 1
    env._rigid_body_pos[:, 2, 2] = 0
    assert env._reward_handstand_kick().item() == pytest.approx(1.0)
    env.dof_pos[:, 11] = -2
    assert env._reward_handstand_kick().item() < 0.01
    env.dof_pos[:, 11] = -0.95
    env._rigid_body_pos.zero_()
    assert env._reward_handstand_kick().item() == 0


def test_mujoco_task_steps_observes_rewards_and_resets_selected_stage():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.helpers import class_to_dict

    cfg = Go2HandstandCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    from gym.utils.task_registry import task_registry

    task_registry.convert_frequencies_to_params(cfg, Go2HandstandRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        env = Go2Handstand(cfg, "cpu", True, backend)
        runner = Go2HandstandRunnerCfg()
        for _ in range(5):
            env.step()
        assert torch.isfinite(env.get_states(runner.actor.obs)).all()
        for name in class_to_dict(runner.critic.reward.weights):
            reward = getattr(env, "_reward_" + name)()
            assert reward.shape == (2,)
            assert torch.isfinite(reward).all()
        env.handstand_stage[:] = 2
        env._lean_time[:] = 1
        env._kick_time[:] = 1
        env._reset_idx(torch.tensor([True, False]))
        assert env.handstand_stage.tolist() == [2, 2]
        assert env._lean_time.tolist() == [0, 1]
        assert env._kick_time.tolist() == [0, 1]
    finally:
        backend.close()
