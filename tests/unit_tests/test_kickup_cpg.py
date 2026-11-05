import pytest
import torch

from gym.envs.go2.kickup_cpg import KickupCPG
from gym.envs.go2.go2kickup_config import Go2KickupCfg, Go2KickupRunnerCfg


def test_one_shot_and_independent_resets():
    c = Go2KickupCfg.kickup
    oscillator = KickupCPG(2, "cpu", c)
    assert oscillator.targets()[0].tolist() == pytest.approx(sum(c.standing, []))
    oscillator.advance(c.duration_s / 2, torch.zeros(2, dtype=torch.bool))
    assert oscillator.targets()[0].tolist() == pytest.approx(sum(c.crouched, []))
    oscillator.advance(c.duration_s / 2, torch.tensor([True, False]))
    assert oscillator.handed_off.tolist() == [True, False]
    assert oscillator.failed.tolist() == [False, True]
    assert oscillator.targets()[1].tolist() == pytest.approx(sum(c.extended, []))
    oscillator.reset(torch.tensor([False, True]))
    assert oscillator.handed_off.tolist() == [True, False]
    assert oscillator.phase[1].sum() == 0
    oscillator.advance(0.01, torch.zeros(2, dtype=torch.bool))
    assert oscillator.handed_off[0]


def test_cpu_task_kicks_and_hands_off():
    from gym.envs.go2.go2kickup import Go2Kickup
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.task_registry import task_registry

    cfg, runner = Go2KickupCfg(), Go2KickupRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    cfg.kickup_curriculum.enabled = False
    task_registry.convert_frequencies_to_params(cfg, runner)
    backend = MuJocoCPUBackend()
    try:
        env = Go2Kickup(cfg, "cpu", True, backend)
        assert torch.allclose(
            env.dof_pos, env.default_dof_pos.expand_as(env.dof_pos), atol=0.1
        )
        target = env.dof_pos_target.clone()
        for _ in range(5):
            env.step()
        assert torch.isfinite(env.root_states).all()
        assert (env.kickup.phase > 0).all()
        env.kickup_in_region = lambda: torch.ones(2, dtype=torch.bool)
        env.step()
        assert env.kickup.handed_off.all()
        assert torch.equal(env.dof_pos_target, target)
        assert torch.allclose(
            env._compute_torques(), super(Go2Kickup, env)._compute_torques()
        )
    finally:
        backend.close()


def test_amplitude_scales_each_environment():
    c = Go2KickupCfg.kickup
    oscillator = KickupCPG(2, "cpu", c)
    oscillator.amplitude[:] = torch.tensor([0.25, 1.0])
    oscillator.advance(c.duration_s / 2, torch.zeros(2, dtype=torch.bool))
    standing = torch.tensor(c.standing).flatten()
    crouched = torch.tensor(c.crouched).flatten()
    assert torch.allclose(
        oscillator.targets()[0], standing + 0.25 * (crouched - standing)
    )
    assert torch.allclose(oscillator.targets()[1], crouched)


@torch.inference_mode()
def test_curriculum_promotes_only_completed_current_level_episodes():
    from gym.envs.go2.go2kickup import Go2Kickup
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.utils.task_registry import task_registry

    cfg, runner = Go2KickupCfg(), Go2KickupRunnerCfg()
    cfg.env.num_envs = 2
    cfg.seed = 7
    cfg.kickup_curriculum.min_episodes = 2
    cfg.kickup_curriculum.required_batches = 1
    task_registry.convert_frequencies_to_params(cfg, runner)
    backend = MuJocoCPUBackend()
    try:
        env = Go2Kickup(cfg, "cpu", True, backend)
        mask = torch.ones(2, dtype=torch.bool)
        env._kick_success[:] = True
        env.terminated[:] = False
        env.timed_out[:] = False
        env._reset_idx(mask)
        assert env.kickup_level == 0  # Manual resets do not promote.
        env._kick_success[:] = True
        env.timed_out[:] = True
        env._reset_idx(mask)
        assert env.kickup_level == 1
        assert (env.kickup.amplitude == cfg.kickup_curriculum.amplitudes[1]).all()
        state = env.training_state_dict()
        env.load_training_state_dict(state)
        assert env.kickup_level == 1
        assert not env._kick_success.any()
        env._kick_level[:] = len(cfg.kickup_curriculum.amplitudes) - 1
        env.kickup_level = len(cfg.kickup_curriculum.amplitudes) - 1
        env._kick_success[:] = True
        env.kickup.handed_off[:] = True
        env._best_hold[:] = 0  # Final success needs policy balance, not just pitch.
        env.timed_out[:] = True
        env._reset_idx(mask)
        assert env.kickup_success_rate == 0
    finally:
        backend.close()
