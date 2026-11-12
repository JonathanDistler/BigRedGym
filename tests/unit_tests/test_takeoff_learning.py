"""Takeoff reward anti-shortcuts, synchronization, action mapping and gates."""

import numpy as np
import pytest
from experiments.morphology.learning.takeoff import (
    TakeoffTracker,
    TakeoffEnvironment,
    inverse_target_action,
    TakeoffCurriculum,
)
from experiments.morphology.learning.environment import Assets
from scripts.validate_takeoff_learning import negative_controls


def row(height=0.022, front=True, weight=160.0):
    feet = np.array(
        [
            [0, 0.14, 0.022],
            [0, -0.14, 0.022],
            [-0.4, 0.14, height],
            [-0.4, -0.14, height],
        ]
    )
    loads = (
        np.array([60.0, 60.0, 40.0, 40.0])
        if height == 0.022
        else np.array([80.0, 80.0, 0.0, 0.0])
    )
    return dict(
        feet=feet,
        loads=loads,
        weight=weight,
        front_support=front,
        front_drift=0.0,
        pitch=0.4,
        rear_leg_power=np.array([5.0, 5.0]),
        rear_clearance=height - 0.022,
    )


def test_grounded_lean_single_leg_and_fold_do_not_earn_takeoff():
    for result in negative_controls():
        assert result["positive_takeoff_reward"] == 0
        assert not result["takeoff"]


def test_push_then_synchronous_release_earns_once_and_credits_rotation():
    tracker = TakeoffTracker()
    reward = 0.0
    events = 0
    for k in range(50):
        terms = tracker.observe(row(), k * 0.002, np.array([2.0, 2.0]))
        reward += sum(terms.values())
        events += terms["simultaneous_liftoff"] > 0
    for k in range(50, 100):
        terms = tracker.observe(row(height=0.07), k * 0.002, np.zeros(2))
        reward += sum(terms.values())
        events += terms["simultaneous_liftoff"] > 0
    assert tracker.confirmed and tracker.max_flight_pitch == pytest.approx(0.4)
    assert events == 1 and reward > 3


def test_stale_push_or_nonfoot_invalidity_cannot_certify():
    tracker = TakeoffTracker()
    for k in range(50):
        tracker.observe(row(), k * 0.002, np.ones(2) * 2)
    for k in range(50, 300):
        tracker.observe(row(), k * 0.002, np.zeros(2))
    for k in range(300, 340):
        tracker.observe(row(0.07), k * 0.002, np.zeros(2))
    assert not tracker.confirmed
    for k in range(340, 380):
        tracker.observe(row(0.07), k * 0.002, np.ones(2) * 2, valid=False)
    assert not tracker.confirmed


def test_asynchronous_release_is_rejected():
    tracker = TakeoffTracker()
    for k in range(50):
        tracker.observe(row(), k * 0.002, np.ones(2) * 2)
    for k in range(50, 100):
        r = row()
        r["feet"][2, 2] = 0.07
        r["loads"][2] = 0
        tracker.observe(r, k * 0.002, np.zeros(2))
    for k in range(100, 150):
        tracker.observe(row(0.07), k * 0.002, np.zeros(2))
    assert not tracker.confirmed


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("takeoff"))


def test_inverse_action_and_measurement_does_not_double_count(assets):
    env = TakeoffEnvironment(assets)
    env.reset(0.9, 42)
    desired = env.target + 0.01
    action, error = inverse_target_action(env, desired)
    assert error["target_error"] < 1e-12 and error["action_projection_fraction"] == 0
    env.step(action)
    np.testing.assert_allclose(env.target, desired)
    length = len(env.tracker.window)
    env.measure()
    env.measure()
    assert len(env.tracker.window) == length
    assert env.max_torque_ratio <= 1


def test_takeoff_local_gate_does_not_demand_whole_episode_support():
    p = dict(
        scales=[0.9, 0.95, 1.0],
        progress_pitch_degrees=[20, 35],
        progress_airborne_seconds=[0.02, 0.04],
        stage_window_episodes=2,
        stage_passing_windows=2,
        stage_required_fraction=0.3,
        minimum_stage_steps=0,
    )
    schedule = TakeoffCurriculum("B", p)
    episode = dict(
        scale=0.9,
        takeoff=True,
        genuine_flight_pitch=0.5,
        genuine_airborne_seconds=0.04,
        max_speed_ratio=0.5,
        joint_limit_violation=0.0,
        front_support_fraction=0.2,
    )
    for i in range(4):
        schedule.completed(episode, i)
    assert schedule.scale == 0.95
    old = schedule.scale
    schedule.completed(episode, 5)
    assert old == schedule.scale and not schedule.outcomes


def test_stale_validation_cannot_start_training(tmp_path):
    import json
    from scripts.run_takeoff_ppo import check_validation

    path = tmp_path / "stale.json"
    path.write_text(
        json.dumps(
            dict(
                reward_negative_controls_passed=True,
                action_authority=dict(passed=True),
                reward_semantic_sha256="stale",
            )
        )
    )
    with pytest.raises(ValueError, match="differs"):
        check_validation(path)


def test_takeoff_checkpoint_resume_preserves_tracker_and_update(assets, tmp_path):
    import torch
    from scripts.run_takeoff_ppo import TakeoffRun
    from scripts.run_physical_morphology_ppo import protocol

    p = protocol("experiments/morphology/learning/protocol_takeoff.json")
    p.update(num_envs=2, steps_per_iteration=64, evaluation_seeds=[1000])
    p["ppo"].update(epochs=1, batch_size=64)
    torch.set_num_threads(1)
    run = TakeoffRun(tmp_path / "original", "B", 42, p, assets)
    run.collect_and_update()
    run.checkpoint_and_evaluate()
    restored = TakeoffRun(tmp_path / "restored", "B", 42, p, assets)
    restored.restore(run.directory / "checkpoint_000001.pt")
    assert len(run.envs[0].tracker.window) == len(restored.envs[0].tracker.window)
    run.collect_and_update()
    restored.collect_and_update()
    assert run.actual_steps == restored.actual_steps
    for key, value in run.net.state_dict().items():
        torch.testing.assert_close(
            value, restored.net.state_dict()[key], atol=1e-7, rtol=1e-6
        )
