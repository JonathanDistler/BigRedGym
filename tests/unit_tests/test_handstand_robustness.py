"""Independent actor, physical metrics, and optional exact trajectory regression."""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from experiments.handstand_robustness.core import Environment, Policy
from gym.envs.go2.handstand_replay import STATE_SPEC, build_model, load_trajectory


def test_sensor_policy_bounds_and_feedback():
    torch.manual_seed(4110)
    policy = Policy()
    obs = torch.zeros(4, 43)
    perturbed = obs.clone()
    perturbed[:, 3] += 0.1
    assert policy(obs).shape == (4, 12)
    assert torch.isfinite(policy(obs)).all()
    assert (policy(obs).abs() <= 1).all()
    assert not torch.equal(policy(obs), policy(perturbed))


def reference():
    path, digest = (
        os.getenv("GO2_HANDSTAND_TRAJECTORY"),
        os.getenv("GO2_HANDSTAND_SHA256"),
    )
    if not path or not digest:
        pytest.skip("External hash-locked trajectory required")
    model = build_model()
    return model, load_trajectory(Path(path), digest, model)


def test_continuous_evaluation_matches_every_reference_state():
    model, trajectory = reference()
    env = Environment(model, trajectory)
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    for k, action in enumerate(trajectory["actions"]):
        _, _, done = env.step(action)
        mujoco.mj_getState(model, env.d, state, STATE_SPEC)
        np.testing.assert_array_equal(state, trajectory["reference_states"][k + 1])
        assert not done or k == len(trajectory["actions"]) - 1
    result = env.result()
    assert result["complete_success"]
    assert result["strict_hold_seconds"] == pytest.approx(5.014)
    assert result["hold_metrics"]["duration_seconds"] == 5
    assert 0 < result["hold_metrics"]["front_leg_kinetic_J"] < 1


def test_reset_rejects_impossible_joint_positions():
    model, trajectory = reference()
    with pytest.raises(ValueError, match="Invalid reset"):
        Environment(model, trajectory, condition={"joint_position": 10})


def test_ppo_resume_restores_exact_optimizer_rng_and_normalization(
    tmp_path, monkeypatch
):
    """Exercise real PPO updates on a tiny deterministic simulator surrogate."""
    from argparse import Namespace
    from scripts import handstand_robustness as cli

    class TinyEnvironment:
        def __init__(self):
            self.index = 0

        def observation(self):
            return np.full(43, self.index / 10, np.float32)

        def step(self, action):
            self.index += 1
            return self.observation(), float(1 - np.mean(action**2)), self.index == 3

    monkeypatch.setattr(cli, "make_env", lambda *args, **kwargs: TinyEnvironment())
    monkeypatch.setattr(
        cli,
        "rollout",
        lambda *a, **kw: (
            dict(complete_success=False, strict_hold_seconds=0.01),
            None,
            None,
        ),
    )

    def arguments(output, updates, checkpoint=None):
        output.mkdir(exist_ok=True)
        return Namespace(
            output=output,
            updates=updates,
            envs=2,
            rollout_steps=4,
            eval_every=1,
            patience=10,
            seed=4110,
            checkpoint=checkpoint,
            resume=checkpoint is not None,
            trajectory=Path("unused.npz"),
            trajectory_sha256="test",
            teacher=None,
            feedback=None,
        )

    torch.set_num_threads(1)
    torch.manual_seed(4110)
    cli.train(arguments(tmp_path / "continuous", 2), {"actions": np.zeros((3, 12))})
    torch.manual_seed(4110)
    cli.train(arguments(tmp_path / "split", 1), {"actions": np.zeros((3, 12))})
    cli.train(
        arguments(tmp_path / "split", 2, tmp_path / "split/latest.pt"),
        {"actions": np.zeros((3, 12))},
    )
    full = torch.load(tmp_path / "continuous/latest.pt", weights_only=False)
    split = torch.load(tmp_path / "split/latest.pt", weights_only=False)

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        elif isinstance(a, list):
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    equal(full["policy"], split["policy"])
    equal(full["optimizer"], split["optimizer"])
    equal(full["torch_rng"], split["torch_rng"])
    equal(full["curriculum"], split["curriculum"])
    assert full["update"] == split["update"] == 2


def test_gyro_is_in_body_frame_not_inertial_principal_frame():
    model = build_model()
    data = mujoco.MjData(model)
    data.qpos[3:7] = [np.cos(0.4), 0, np.sin(0.4), 0]
    data.qvel[3:6] = [0.2, 0.4, 0.6]
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    world_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, base, world_velocity, 0
    )
    rotation = data.xmat[base].reshape(3, 3)
    np.testing.assert_allclose(
        rotation.T @ world_velocity[:3], data.qvel[3:6], atol=1e-14
    )
