"""PPO asymmetry, terminal returns, physical limits and stage/resume invariants."""

import copy
import numpy as np
import pytest
import torch

from experiments.morphology.learning.environment import Assets, Environment
from experiments.morphology.learning.ppo import (
    ActorCritic,
    PerformanceCurriculum,
    advantages,
    update,
)
from scripts.run_physical_morphology_ppo import protocol


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("learning_assets"))


def test_asymmetric_observations_and_constant_reward_configuration(assets):
    env = Environment(assets)
    settings = copy.deepcopy(env.cfg)
    for scale in (0.8, 0.85, 0.9, 0.95, 1.0):
        actor, critic = env.reset(scale, 42)
        assert actor.shape == (44,) and critic.shape == (74,)
        assert np.isfinite(critic).all()
        assert env.row["illegal_contacts"] == 0
        assert env.cfg == settings
        assert actor[-1] == pytest.approx(scale)
    torch.set_num_threads(1)
    net = ActorCritic(44, 74)
    actor = torch.tensor(actor)
    before = net.actor(actor).detach().clone()
    _ = net.critic(torch.tensor(critic) * 2)
    torch.testing.assert_close(net.actor(actor), before)
    assert net.actor[0].in_features != net.critic[0].in_features


def test_finite_horizon_gae_and_no_episode_leakage():
    r = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.zeros_like(r)
    next_values = np.array([[0.0], [99.0], [0.0]])
    done = np.array([[False], [True], [False]])
    adv, returns = advantages(r, values, next_values, done, 1.0, 1.0)
    np.testing.assert_allclose(adv[:, 0], [3, 2, 3])
    np.testing.assert_array_equal(adv, returns)


def test_progression_without_five_second_success_and_recovery_gate():
    p = protocol("experiments/morphology/learning/protocol_90.json")
    p.update(stage_window_episodes=2, stage_passing_windows=2, minimum_stage_steps=0)
    schedule = PerformanceCurriculum("B", p)
    sample = dict(
        scale=0.9,
        takeoff=True,
        max_supported_pitch=0.4,
        front_support_fraction=0.9,
        max_speed_ratio=0.5,
        joint_limit_violation=0.0,
        recovery_seconds=0.0,
        success=False,
    )
    episode_scale = schedule.scale
    for i in range(4):
        schedule.completed(sample, i)
    assert schedule.scale == 0.95 and episode_scale == 0.9
    assert len(schedule.events) == 1
    old = len(schedule.outcomes)
    schedule.completed(sample, 10)
    assert len(schedule.outcomes) == old
    sample.update(scale=0.95, max_supported_pitch=1.0, recovery_seconds=0.0)
    for i in range(4):
        schedule.completed(sample, i + 20)
    assert schedule.scale == 0.95
    sample["recovery_seconds"] = 0.12
    for i in range(4):
        schedule.completed(sample, i + 30)
    assert schedule.scale == 1.0
    all_five = protocol("experiments/morphology/learning/protocol_80.json")
    assert all_five["scales"] == [0.8, 0.85, 0.9, 0.95, 1.0]


def test_physics_limits_latched_scale_and_state_resume(assets):
    env = Environment(assets)
    other = Environment(assets)
    env.reset(0.9, 71)
    contact_flags = env.m.geom_contype.copy()
    for _ in range(8):
        _, _, done, info = env.step(np.linspace(-1, 1, 12))
        assert env.scale == 0.9
        assert info["max_torque_ratio"] <= 1
        if done:
            break
    saved = copy.deepcopy(env.state())
    other.restore(saved)
    assert other.scale == 0.9
    np.testing.assert_array_equal(other.m.geom_contype, contact_flags)
    a = env.step(np.zeros(12))
    b = other.step(np.zeros(12))
    np.testing.assert_allclose(env.d.qpos, other.d.qpos, atol=1e-10)
    np.testing.assert_allclose(env.d.qvel, other.d.qvel, atol=1e-10)
    assert a[1] == pytest.approx(b[1], abs=1e-10)


def test_squashed_policy_update_and_parameter_preservation():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    net = ActorCritic(44, 74, hidden=(32, 16))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    actor = torch.randn(128, 44)
    critic = torch.randn(128, 74)
    with torch.no_grad():
        raw, logs, values = net.sample(actor, critic)
    before = net.actor[-1].weight.detach().clone()
    hp = dict(
        epochs=2,
        batch_size=64,
        target_kl=0.02,
        clip=0.2,
        value_weight=0.5,
        entropy_weight=0.005,
        max_grad_norm=1.0,
    )
    stats = update(
        net,
        optimizer,
        (actor, critic, raw, logs, values + torch.randn(128), torch.randn(128)),
        hp,
    )
    assert np.isfinite(list(stats.values())).all()
    assert not torch.equal(before, net.actor[-1].weight)
    # Scheduling cannot replace policy or optimizer parameters.
    parameter_ids = [id(p) for p in net.parameters()]
    schedule = PerformanceCurriculum(
        "B", protocol("experiments/morphology/learning/protocol_90.json")
    )
    schedule.stage = 1
    assert parameter_ids == [id(p) for p in net.parameters()]


def test_checkpoint_resume_preserves_optimizer_rng_and_worlds(assets, tmp_path):
    from scripts.run_physical_morphology_ppo import Run

    p = protocol("experiments/morphology/learning/protocol_90.json")
    p.update(num_envs=2, steps_per_iteration=64, evaluation_seeds=[1000])
    p["ppo"].update(epochs=1, batch_size=64)
    torch.set_num_threads(1)
    original = Run(tmp_path / "original", "B", 42, p, assets)
    original.collect_and_update()
    original.checkpoint_and_evaluate()
    resumed = Run(tmp_path / "resumed", "B", 42, p, assets)
    resumed.restore(original.directory / "checkpoint_000001.pt")
    original.collect_and_update()
    resumed.collect_and_update()
    assert original.actual_steps == resumed.actual_steps == 128
    assert original.schedule.scale == resumed.schedule.scale == 0.9
    for key, value in original.net.state_dict().items():
        torch.testing.assert_close(
            value, resumed.net.state_dict()[key], atol=1e-7, rtol=1e-6
        )
    for left, right in zip(original.envs, resumed.envs):
        np.testing.assert_allclose(left.d.qpos, right.d.qpos, atol=1e-8)
