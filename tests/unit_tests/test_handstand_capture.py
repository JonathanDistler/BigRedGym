"""Read-only diagnostics, sensor estimator, feedback coverage and strict scoring."""

from argparse import Namespace
from pathlib import Path
import os

import mujoco
import numpy as np
import pytest
import torch

from experiments.handstand_robustness.capture import (
    CapturePolicy,
    CaptureTrainingEnvironment,
    InertialObserver,
)
from gym.envs.go2.handstand_replay import STATE_SPEC, build_model, load_trajectory
from scripts.handstand_takeoff import load, save, train


def fixture_trajectory():
    path = os.getenv("GO2_HANDSTAND_TRAJECTORY")
    digest = os.getenv("GO2_HANDSTAND_SHA256")
    if not path or not digest:
        pytest.skip("Provide external hash-locked reference")
    model = build_model()
    return model, load_trajectory(Path(path), digest, model)


def policy():
    balance = dict(
        weight=np.zeros((12, 34)),
        bias=np.zeros(12),
        lower=-np.ones(12),
        upper=np.ones(12),
    )
    balance["weight"][0, 4] = 1.0
    p = CapturePolicy(np.zeros((320, 12)), np.zeros((320, 47)), balance)
    with torch.no_grad():
        p.gains[:, 0, 4] = 1.0
    return p


def test_feedback_active_before_and_after_capture_handoff():
    p = policy()
    for k in [10, 60, 80, 150]:
        x = torch.zeros(47)
        x[42] = k / 400
        perturbed = x.clone()
        perturbed[4] = 0.01
        assert (p(perturbed) - p(x)).abs().max() > 0
        assert (p(perturbed) - p(x)).abs().max() <= 0.05
        assert torch.equal(p(x, feedback=False), p(perturbed, feedback=False))
    with pytest.raises(ValueError, match="47"):
        p(torch.zeros(43))


def test_ideal_imu_integration_units_and_gravity():
    observer = InertialObserver(0.34)
    for _ in range(500):
        observer.update([1.0, 0, 9.81], np.eye(3))
    np.testing.assert_allclose(observer.velocity, [1.0, 0, 0], atol=1e-12)
    assert observer.height == pytest.approx(0.34)


def test_observer_and_logging_do_not_change_nominal_physics():
    model, t = fixture_trajectory()
    e = CaptureTrainingEnvironment(model, t, horizon=75)
    e.trace_enabled = True
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    for k, u in enumerate(t["actions"][:75]):
        e.step(u)
        mujoco.mj_getState(model, e.d, state, STATE_SPEC)
        np.testing.assert_array_equal(state, t["reference_states"][k + 1])
    assert e.result()["capture_success"]
    assert not e.result()["complete_success"]  # a short prefix never passes 5 s
    assert e.first_geometric_illegal is None
    assert e.max_observer_velocity_error < 0.001
    assert e.max_observer_height_error < 0.001
    assert e.contacts
    for event in e.contacts:
        np.testing.assert_allclose(
            event["impulse_on_geom2_world_Ns"],
            np.array(event["force_on_geom2_world_N"]) * 0.002,
        )
        assert np.linalg.norm(event["normal_geom1_to_geom2"]) == pytest.approx(1)


def test_actor_observation_excludes_true_base_translation_and_velocity():
    model, t = fixture_trajectory()
    e = CaptureTrainingEnvironment(model, t)
    sensor = e.observation()
    # An observation-contract unit test, not a purported continuous robot rollout.
    e.d.qpos[:3] += [3.0, 2.0, 1.0]
    e.d.qvel[:3] = [12.0, -4.0, 7.0]
    mujoco.mj_forward(model, e.d)
    np.testing.assert_array_equal(sensor, e.observation())


def test_capture_checkpoint_roundtrip_and_long_run_guard(tmp_path):
    p = policy()
    optimizer = torch.optim.Adam(p.parameters(), lr=1e-5)
    args = Namespace(
        trajectory_sha256="reference", seed=4110, envs=1, rollout_steps=128
    )
    path = tmp_path / "capture.pt"
    save(path, p, optimizer, 2, args, [], None)
    reloaded, data = load(path)
    x = torch.zeros(47)
    x[42] = 0.4
    x[4] = 0.01
    torch.testing.assert_close(reloaded(x), p(x))
    assert data["kind"] == "capture_residual_v1"
    assert data["curriculum"]["prefix_steps"] == 110
    args = Namespace(
        checkpoint=path,
        resume=False,
        updates=250,
        output=tmp_path,
        seed=4110,
        envs=1,
        rollout_steps=128,
        trajectory_sha256="reference",
        capture_continuation_proof=None,
    )
    with pytest.raises(ValueError, match="Molab proof"):
        train(args, {})
