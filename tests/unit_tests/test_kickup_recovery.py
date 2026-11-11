"""Native replay, checkpoint compatibility and six-stage recovery contracts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2kickup_recovery import Go2KickupRecovery
from gym.envs.go2.recovery_state import capture, restore, sync
from gym.utils.task_registry import task_registry
from scripts.train_kickup_recovery import configuration
from scripts.reproduce_transition_hold import BALANCE

DATA = Path("datasets/go2kickup_recovery_train_v2/train.npz")


@pytest.fixture
def recovery():
    if not DATA.exists():
        pytest.skip("Generate the physical training dataset first")
    cfg, train = configuration(DATA)
    cfg.env.num_envs = 2
    cfg.seed = 7
    train.runner.device = "cpu"
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    backend = MuJocoCPUBackend()
    env = Go2KickupRecovery(cfg, "cpu", True, backend)
    try:
        yield env, train
    finally:
        backend.close()


@torch.inference_mode()
def test_checkpoint_interface_and_baseline(recovery):
    e, train = recovery
    runner = task_registry.make_alg_runner(e, train)
    saved = torch.load(BALANCE, weights_only=True, map_location="cpu")
    runner.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
    runner.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
    assert runner.get_inference_actions().shape == (2, 12)
    assert runner.get_obs(runner.actor_cfg["obs"]).shape == (2, 59)
    assert runner.get_obs(runner.critic_cfg["obs"]).shape == (2, 63)
    e.curriculum_level = 0
    e._reset_idx(torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(e.dof_pos, e._standing_posture())
    assert e.root_states[:, 2].min() > 0.5
    assert e.contact_cpg_enabled
    torch.testing.assert_close(e.leg_phase, e._cpg_offsets.expand(2, -1))
    assert e.cpg_contact_obs.tolist() == [[1, 1, 0, 0]] * 2
    torch.testing.assert_close(e.dof_pos_target, torch.zeros_like(e.dof_pos_target))
    for _ in range(10):
        runner.set_actions(
            runner.actor_cfg["actions"], runner.get_inference_actions(), False
        )
        e.step()
    assert torch.isfinite(e.root_states).all()


def test_native_replay_roundtrip_and_rewards(recovery):
    e, train = recovery
    e.cfg.handstand_curriculum.replay_fraction = 0
    for stage in range(6):
        e.curriculum_level = stage
        e._reset_idx(torch.ones(2, dtype=torch.bool))
        before = capture(e, 0)
        e.step()
        restore(e, 0, before)
        sync(e)
        after = capture(e, 0)
        np.testing.assert_allclose(after["native"], before["native"], atol=1e-12)
        for key in before:
            if key.startswith("controller_"):
                np.testing.assert_array_equal(after[key], before[key])
        d = e._backend._datas[0]
        assert np.isfinite(d.qacc).all()
        assert all(c.geom1 >= 0 and c.geom2 >= 0 for c in d.contact)
        for name in ("recovery_com_velocity", "recovery_com_offset"):
            reward = getattr(e, "_reward_" + name)()
            assert torch.isfinite(reward).all() and (reward <= 0).all()
        assert (e._reward_recovery_hold() >= 0).all()
        for _ in range(5):
            e.step()
        assert torch.isfinite(e.root_states).all()


def test_replay_native_continuation(recovery):
    e, _ = recovery
    sample = e.recovery_bank.sample(5)
    restore(e, 0, sample)
    restore(e, 1, sample)
    for i in range(2):
        mujoco.mj_step(e._backend._model_for_env(i), e._backend._datas[i])
    np.testing.assert_allclose(
        e._backend._datas[0].qpos, e._backend._datas[1].qpos, atol=1e-12
    )


def test_recovery_curriculum_is_success_gated(recovery):
    e, _ = recovery
    c = e.cfg.handstand_curriculum
    c.replay_fraction = 0
    c.min_episodes = 2
    mask = torch.ones(2, dtype=torch.bool)
    e.curriculum_level = 0
    for _ in range(c.required_batches):
        e._reset_idx(mask)
        e._best_hold[:] = 4.9
        e.timed_out[:] = True
        e.terminated.zero_()
        e._record_handstand_episodes(mask)
        e._episode_active.zero_()
    assert e.curriculum_level == 0
    for _ in range(c.required_batches):
        e._reset_idx(mask)
        e._best_hold[:] = 5.1
        e.timed_out[:] = True
        e.terminated.zero_()
        e._record_handstand_episodes(mask)
        e._episode_active.zero_()
    assert e.curriculum_level == 1


def test_com_velocity_penalty_tracks_actual_motion(recovery):
    e, _ = recovery
    e.curriculum_level = 0
    e._reset_idx(torch.ones(2, dtype=torch.bool))
    for i, d in enumerate(e._backend._datas):
        d.qvel[:] = 0
        mujoco.mj_forward(e._backend._model_for_env(i), d)
    still = e._reward_recovery_com_velocity()
    for i, d in enumerate(e._backend._datas):
        d.qvel[0] = 0.5
        mujoco.mj_forward(e._backend._model_for_env(i), d)
    moving = e._reward_recovery_com_velocity()
    torch.testing.assert_close(still, torch.zeros(2))
    torch.testing.assert_close(moving, torch.full((2,), -0.25))
