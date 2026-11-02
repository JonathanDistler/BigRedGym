"""Survival reports must exclude all state after each robot's first failure."""

import json
from types import SimpleNamespace

import pytest
import torch

from gym.utils.survival_evaluation import evaluate_survival


def test_first_failure_is_latched_and_trace_is_bounded(tmp_path):
    n = 2
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        dt=0.1,
        feet_indices=[0, 1, 2, 3],
        terminated=torch.zeros(n, dtype=torch.bool),
        _best_hold=torch.zeros(n),
        root_states=torch.zeros(n, 13),
        base_lin_vel=torch.zeros(n, 3),
        base_ang_vel=torch.zeros(n, 3),
        dof_pos=torch.zeros(n, 12),
        dof_vel=torch.zeros(n, 12),
        dof_pos_target=torch.ones(n, 12),
        torques=torch.zeros(n, 12),
        _rigid_body_pos=torch.zeros(n, 4, 3),
        _rigid_body_lin_vel=torch.zeros(n, 4, 3),
        contact_forces=torch.ones(n, 4, 3) * 10,
        dof_names=[str(i) for i in range(12)],
        robot_layout=SimpleNamespace(body_groups={"feet": ["FL", "FR", "RL", "RR"]}),
        cfg=SimpleNamespace(
            handstand_curriculum=SimpleNamespace(hold_seconds=5),
            balance_perturbations=SimpleNamespace(enabled=False),
        ),
        _standing_posture=lambda: torch.ones(n, 12) * 2,
    )
    count = 0

    def step():
        nonlocal count
        count += 1
        env.terminated[:] = False
        env.terminated[0] = count == 15
        env._best_hold[:] = count / 10
        env.contact_forces[0, 1, 2] = 0 if count >= 14 else 10

    env.step = step
    runner = SimpleNamespace(
        actor_cfg={"actions": [], "disable_actions": False},
        get_inference_actions=lambda: None,
        set_actions=lambda *args: None,
    )
    report = evaluate_survival(env, runner, seconds=2, output=tmp_path)
    assert report["survival_fraction"] == 0.5
    assert report["trials"][0]["first_failure_s"] == pytest.approx(1.5)
    assert report["trials"][0]["best_hold_s"] == pytest.approx(1.5)
    assert report["trials"][0]["front_contact_loss_steps"] == 2
    assert report["trials"][1]["first_failure_s"] is None
    assert report["trials"][1]["observed_seconds"] == 2
    trace = json.loads((tmp_path / "failure_env_0.json").read_text())
    assert len(trace) == 11
    assert trace[-1]["time_s"] == pytest.approx(1.5)
    assert trace[-1]["joint_targets_rad"] == [3] * 12
    assert not (tmp_path / "failure_env_1.json").exists()
