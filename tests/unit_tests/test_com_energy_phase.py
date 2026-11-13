"""Independent physics identities and safety counterexamples for opt-in task."""

import copy
import mujoco
import numpy as np
import pytest
import torch
from experiments.com_energy_phase.core import (
    EnergyConfig,
    EnergyEnvironment,
    Phase,
    coefficient,
    gaussian,
    orientation_error,
    measurements,
)
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("com_assets"))


def test_com_momentum_energy(assets):
    e = EnergyEnvironment(assets, EnergyConfig())
    e.reset(1.0, 42)
    e.d.qvel[:] = np.random.default_rng(42).normal(0, 0.2, e.m.nv)
    mujoco.mj_forward(e.m, e.d)
    e.row = e.measure()
    s = measurements(e)
    assert s["com_check"] < 1e-12
    assert s["velocity_check"] < 1e-12
    assert s["momentum_check"] < 1e-10
    assert np.isclose(s["mass"], e.m.body_subtreemass[e.base])
    mujoco.mj_energyPos(e.m, e.d)
    mujoco.mj_energyVel(e.m, e.d)
    assert np.isclose(s["potential"], e.d.energy[0], atol=1e-8)
    # Rotor armature is present in MuJoCo kinetic energy but absent in link-only sum.
    rotor = 0.5 * np.dot(e.m.dof_armature, e.d.qvel**2)
    assert np.isclose(s["kinetic"] + rotor, e.d.energy[1], atol=1e-8)
    assert not np.allclose([s[f"com_{a}"] for a in "xyz"], e.d.xipos[e.base])
    assert np.isclose(
        s["pendulum_angle"],
        np.arctan2(s["com_support_x"], s["com_height"] - s["support_height"]),
    )
    assert s["inertia_support_y"] >= s["inertia_com_y"]


def test_orientation_and_filter():
    target = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
    assert orientation_error(target, target) < 1e-8
    assert np.isclose(orientation_error(np.eye(3), target), np.pi / 2)
    wrong_yaw = np.diag([-1.0, -1.0, 1.0]) @ target
    assert np.isclose(orientation_error(wrong_yaw, target), np.pi)
    assert gaussian(0, 10) == 1 and gaussian(10, 10) == pytest.approx(np.exp(-0.5))
    assert coefficient(0.02, 0) == 1
    assert (1 - coefficient(0.01, 0.1)) ** 2 == pytest.approx(
        1 - coefficient(0.02, 0.1)
    )
    with pytest.raises(ValueError):
        coefficient(0.02, -1)


def test_phase():
    p = Phase()
    p.update(100, False, False, 0, False)
    assert p.progress < 0.2
    p.update(100, True, False, 0, False)
    assert p.progress < 0.4
    p.update(101, True, True, 0, False)
    p.update(102, True, True, 0, False)
    assert p.progress < 0.8
    p.update(103, True, True, np.pi / 2, True)
    assert p.progress == 1
    assert np.isfinite(p.encode()).all()
    assert Phase().progress == 0


def test_gating_reset_and_checkpoint(assets, tmp_path):
    cfg = EnergyConfig(energy_tau=0.1, orientation_tau=0.1, action_tau=0.03)
    e = EnergyEnvironment(assets, cfg)
    a, c = e.reset(1.0, 42)
    assert a.shape == (47,) and c.shape == (77,)
    # Make standing itself the numerical Gaussian target: physical gating must
    # reject it even when the raw rewards are essentially perfect.
    e.cfg.target_height = e.sample["com_height"]
    e.cfg.target_rotation = tuple(e.d.xmat[e.base])
    e.energy_ema = e.orientation_ema = 1.0
    # Grounded standing cannot receive target reward even with stale perfect EMAs.
    _, r, _, _ = e.step(np.zeros(12))
    assert e.trace[-1]["r_target"] == 0 and e.energy_ema == 0
    assert np.isfinite(r)
    state = copy.deepcopy(e.state())
    torch.save(state, tmp_path / "state.pt")
    clone = EnergyEnvironment(assets, copy.deepcopy(cfg))
    clone.restore(torch.load(tmp_path / "state.pt", weights_only=False))
    first = e.step(np.full(12, 0.01))
    second = clone.step(np.full(12, 0.01))
    assert np.allclose(e.d.qpos, clone.d.qpos, atol=1e-12)
    assert first[1] == pytest.approx(second[1], abs=1e-12)
    e.reset(1.0, 43)
    assert e.phase.progress == 0 and e.energy_ema == 0 and e.valid_hold == 0
    # No loaded front feet: neither flight nor a support pivot can be claimed.
    e.d.qpos[2] += 1.0
    mujoco.mj_forward(e.m, e.d)
    e.row = e.measure()
    assert not measurements(e)["pivot_valid"]


def test_action_filter_and_finite_gradient(assets):
    fast = EnergyEnvironment(assets, EnergyConfig(action_tau=0))
    slow = EnergyEnvironment(assets, EnergyConfig(action_tau=0.1))
    before = fast.target.copy()
    action = np.full(12, 0.02)
    fast.step(action)
    slow.step(action)
    assert np.linalg.norm(slow.target - before) < np.linalg.norm(fast.target - before)
    net = ActorCritic(47, 77)
    a, c = slow.observations()
    loss = (
        net.actor(torch.tensor(a)).square().mean()
        + net.critic(torch.tensor(c)).square().mean()
    )
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )


def test_reference_and_alignment(assets):
    from pathlib import Path
    from experiments.com_energy_phase.data import reference, align

    root = Path("output/recoverability_maps_20261112")
    dataset = Path("output/multimorph_bc_20261111_v2/bc_dataset.npz")
    if not root.exists() or not dataset.exists():
        pytest.skip("archived experiment assets unavailable")
    e = EnergyEnvironment(assets, EnergyConfig())
    r = reference(e, root)
    assert r["checked_candidates"] > 0 and r["measurements"]["support_valid"]
    assert r["measurements"]["com_check"] < 1e-12
    d = np.load(dataset)
    encoding, weights, events, times = align(d, r["mass"])
    assert encoding.shape == (len(d["time"]), 3)
    assert np.isfinite(encoding).all() and len(events) > 0
    assert np.all(np.diff(times) > 0)
    assert encoding[:, 2].max() < 0.8  # no invented successful handstands
    assert set(np.unique(weights)) <= {0.0, np.float32(0.1), 1.0}


def test_checked_intermediate_reset_preserves_transition_labels(assets):
    from pathlib import Path
    from experiments.com_energy_phase.data import reference

    root = Path("output/recoverability_maps_20261112")
    if not root.exists():
        pytest.skip("archived balance snapshots unavailable")
    e = EnergyEnvironment(assets, EnergyConfig())
    ref = reference(e, root)
    bank = np.load(root / "recoverable_states.npz")
    state = bank[f"{ref['sample_index']}_integration"]
    a, c = e.reset_intermediate(state, 123, {"source": str(root)})
    assert a.shape == (47,) and c.shape == (77,)
    assert e.intermediate_start and not e.tracker.confirmed
    assert e.phase.progress == 0.8
    action = np.clip((e.target - e.nominal) / e.cfg.action_scale, -1, 1)
    _, reward, _, info = e.step(action)
    assert np.isfinite(reward)
    assert not info["takeoff"] and not info["transition_success"]
    # Invalid standing snapshots cannot enter this curriculum.
    e.reset(1.0, 123)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    standing = np.empty(mujoco.mj_stateSize(e.m, spec))
    mujoco.mj_getState(e.m, e.d, standing, spec)
    with pytest.raises(ValueError):
        e.reset_intermediate(standing, 124, {"source": "negative control"})
