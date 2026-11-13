"""Physics identities, event gates, bounded goals and action authority."""

import copy
import mujoco
import numpy as np
import pytest
import torch
from experiments.morphology.learning.environment import Assets
from experiments.momentum_phase.core import (
    MomentumConfig,
    MomentumEnvironment,
    ManeuverPhase,
    momentum_goal,
    terminal_reward,
)
from experiments.momentum_phase.physics import heading_axis, sample


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("momentum_assets"))


def test_mass_energy_momentum_and_axis(assets):
    e = MomentumEnvironment(assets, MomentumConfig())
    e.d.qvel[:] = np.random.default_rng(7).normal(0, 0.1, e.m.nv)
    mujoco.mj_forward(e.m, e.d)
    row = e.measure()
    s = sample(e, row)
    assert s["com_check"] < 1e-12 and s["momentum_check"] < 1e-10
    assert s["L_com"] == pytest.approx(s["momentum_y"])
    mujoco.mj_energyPos(e.m, e.d)
    mujoco.mj_energyVel(e.m, e.d)
    assert s["mechanical_energy"] == pytest.approx(e.d.energy.sum())
    assert s["I_support"] >= s["I_com"] > 0
    assert np.allclose(
        heading_axis(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])), [-1, 0, 0]
    )
    h = np.array([s[f"momentum_{a}"] for a in "xyz"])
    r = np.array([s[f"com_{a}"] - s[f"support_{a}"] for a in "xyz"])
    v = np.array([s[f"com_velocity_{a}"] for a in "xyz"])
    assert s["L_support"] == pytest.approx(
        (h + np.cross(r, s["mass"] * v)) @ e.pitch_axis
    )


def test_phase_events_and_goals():
    p = ManeuverPhase((0.02, 0.68, 0.7, 0.9, 1.3))
    p.update(5, True, False, 0, False)
    assert p.progress < 0.2
    p.front_loaded = True
    p.update(5, True, False, 0, False)
    assert p.progress < 0.4
    p.update(6, True, True, 0.5, False)
    p.update(10, True, True, 0.5, False)
    assert p.progress < 0.6
    p.omega = 0.8
    p.update(11, True, True, np.deg2rad(81), False)
    assert 0.8 <= p.progress < 1
    p.update(12, True, True, np.pi / 2, True)
    assert p.progress == 1
    cfg = MomentumConfig()
    s = dict(energy_target=90, potential=40, I_support=3, pitch=0.4)
    assert momentum_goal(s, 0.1, cfg) == 0
    assert 0 < momentum_goal(s, 0.5, cfg) <= cfg.momentum_support_cap
    s["pitch"] = np.pi / 2
    assert momentum_goal(s, 0.9, cfg) == 0
    s = dict(
        target_valid=True,
        calf_clearance=0.01,
        collision=False,
        r_energy=1,
        r_orientation=1,
        pitch_velocity=0,
        support_region_distance=0,
    )
    rate, loc = terminal_reward(s, cfg)
    assert rate == cfg.terminal_weight and loc == 1
    for key in ["collision", "target_valid"]:
        t = s.copy()
        t[key] = key == "collision"
        assert terminal_reward(t, cfg) == (0.0, 0.0)
    s["pitch_velocity"] = 10
    assert terminal_reward(s, cfg)[0] < 1e-50


def test_action_authority_reset_and_serialization(assets, tmp_path):
    e = MomentumEnvironment(assets, MomentumConfig())
    a, c = e.reset(1.0, 42)
    assert a.shape == (61,) and c.shape == (91,)
    before = e.target.copy()
    _, reward, _, info = e.step(np.full(12, 0.02))
    assert np.isfinite(reward) and e.cfg.action_tau == 0
    assert np.linalg.norm(e.target - before) > 0 and not info["success"]
    assert e.trace[-1]["reward_terminal"] == 0
    state = copy.deepcopy(e.state())
    torch.save(state, tmp_path / "environment.pt")
    clone = MomentumEnvironment(assets, copy.deepcopy(e.cfg))
    clone.restore(torch.load(tmp_path / "environment.pt", weights_only=False))
    r1 = e.step(np.full(12, 0.01))
    r2 = clone.step(np.full(12, 0.01))
    assert np.allclose(e.d.qpos, clone.d.qpos, atol=1e-12)
    assert r1[1] == pytest.approx(r2[1], abs=1e-12)
    e.reset(1.0, 43)
    assert e.rear_work == 0 and e.phase.progress == 0 and e._momentum_prefix_safe


def test_airborne_internal_motion_does_not_generate_momentum(assets):
    e = MomentumEnvironment(assets, MomentumConfig())
    e.d.qpos[2] += 2
    e.d.qvel[:] = 0
    mujoco.mj_forward(e.m, e.d)
    before = sample(e, e.measure())
    # Internal equal/opposite joint forces: no external ground moment in flight.
    for _ in range(50):
        e.d.qfrc_applied[:] = 0
        e.d.qfrc_applied[e.vids[[7, 10]]] = 2.0
        mujoco.mj_step(e.m, e.d)
        mujoco.mj_forward(e.m, e.d)
    after = sample(e, e.measure())
    assert abs(after["L_com"] - before["L_com"]) < 0.01
    assert abs(after["moment_com"]) < 1e-8
    assert not after["support_valid"]


def test_yaw_covariance_of_projected_pitch_and_momentum(assets):
    e = MomentumEnvironment(assets, MomentumConfig())
    e.record_physics = False
    p = 0.4
    c, s = np.cos(p), np.sin(p)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    e.d.qpos[3:7] = quat
    e.d.qpos[2] += 2
    e.d.qvel[:] = np.random.default_rng(9).normal(0, 0.1, e.m.nv)
    mujoco.mj_forward(e.m, e.d)
    first = sample(e, e.measure(), np.array([0.0, 1.0, 0.0]))
    yaw = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mujoco.mju_mat2Quat(quat, (yaw @ rotation).ravel())
    e.d.qpos[3:7] = quat
    e.d.qpos[:3] = yaw @ e.d.qpos[:3]
    e.d.qvel[:3] = yaw @ e.d.qvel[:3]
    mujoco.mj_forward(e.m, e.d)
    second = sample(e, e.measure(), np.array([-1.0, 0.0, 0.0]))
    for key in ("L_com", "L_support", "I_com", "I_support", "pitch", "pitch_velocity"):
        assert first[key] == pytest.approx(second[key], abs=1e-10)
