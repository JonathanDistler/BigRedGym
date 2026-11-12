"""Pendulum conventions, wrench signs, legacy parity, and reward safety."""

import copy

import numpy as np
import pytest
import torch

from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.rotation import (
    RotationConfig,
    RotationEnvironment,
    contact_wrenches,
    pendulum_state,
    potential_increment,
)
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from experiments.morphology.learning.ppo import ActorCritic, update


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("rotation_assets"))


def test_pendulum_angle_and_energy_conventions():
    mass = 10.0
    inertia = 0.5
    s = pendulum_state(
        np.array([-0.2, 0, 0.3]), np.array([0.1, 0, 0]), mass, inertia, 0.4
    )
    assert np.pi / 2 < s["pendulum_theta"] < np.pi
    assert s["pendulum_omega"] > 0
    assert s["potential_barrier"] == pytest.approx(
        mass * 9.81 * (np.hypot(0.2, 0.3) - 0.3)
    )
    assert s["inertia_support_y"] == pytest.approx(inertia + mass * (0.2**2 + 0.3**2))
    assert s["centroidal_rotation_energy"] == pytest.approx(0.4**2 / (2 * inertia))
    upright = pendulum_state(np.array([0.0, 0, 0.3]), np.zeros(3), mass, inertia, 0)
    assert abs(upright["pendulum_theta"]) == np.pi
    assert upright["potential_barrier"] == pytest.approx(0.0)
    # Wrong-way motion is not credited against the remaining barrier.
    reverse = pendulum_state(
        np.array([-0.2, 0, 0.3]), np.array([-0.1, 0, 0]), mass, inertia, 0.4
    )
    assert reverse["energy_deficit"] == reverse["potential_barrier"]


def test_finite_horizon_potential_telescopes_and_no_static_income():
    gamma = 0.995
    phi = [0.0, 0.4, 0.4, 0.8, 0.0]
    rewards = [potential_increment(phi[i], phi[i + 1], gamma, i == 3) for i in range(4)]
    assert sum(gamma**i * r for i, r in enumerate(rewards)) == pytest.approx(0.0)
    assert potential_increment(0.4, 0.4, gamma) < 0
    assert potential_increment(0.8, 100, gamma, True) == -0.8


def test_disabled_rotation_preserves_physics_and_rewards(assets):
    original = TakeoffEnvironment(assets, TakeoffConfig(terminate_on_goal=False))
    instrumented = RotationEnvironment(assets, RotationConfig(rotation_enabled=False))
    original.reset(1.0, 42)
    instrumented.reset(1.0, 42)
    for _ in range(25):
        a = np.zeros(12)
        _, r1, d1, _ = original.step(a)
        _, r2, d2, _ = instrumented.step(a)
        assert r1 == r2 and d1 == d2
        np.testing.assert_array_equal(original.d.qpos, instrumented.d.qpos)


def test_ground_wrench_sign_and_static_momentum(assets):
    env = RotationEnvironment(assets, RotationConfig(rotation_enabled=False))
    env.reset(1.0, 42)
    for _ in range(30):
        env.step(np.zeros(12))
    forces, moments, support = contact_wrenches(env, env.row["com"])
    assert forces[:2, 2].sum() > 0
    assert forces[:2, 2].sum() == pytest.approx(env.weight, rel=0.15)
    assert support is not None
    assert np.isfinite(moments).all()


def test_no_grounded_rotation_milestone_or_progress_income(assets):
    env = RotationEnvironment(assets, RotationConfig())
    env.reset(0.95, 42)
    for _ in range(10):
        _, _, _, info = env.step(np.zeros(12))
        assert info["reward_rotation_progress"] == 0
        assert info["reward_rotation_milestone"] == 0
        assert info["reward_rotation_arrival"] == 0
    assert not env.cfg.terminate_on_goal


def test_rotation_auxiliary_state_survives_restore(assets):
    original = RotationEnvironment(assets, RotationConfig())
    original.reset(0.95, 43)
    for _ in range(10):
        original.step(np.zeros(12))
    original._rotation_phi = 0.4
    original._milestones_paid = {45}
    saved = copy.deepcopy(original.state())
    restored = RotationEnvironment(assets, RotationConfig())
    restored.restore(saved)
    assert restored._milestones_paid == {45}
    assert restored._rotation_phi == 0.4
    _, left, ld, _ = original.step(np.zeros(12))
    _, right, rd, _ = restored.step(np.zeros(12))
    assert left == right and ld == rd
    np.testing.assert_array_equal(original.d.qpos, restored.d.qpos)


def test_csv_migration_keeps_old_rows_and_new_metrics(tmp_path):
    import csv
    from scripts.morphology_feasibility import save_csv

    path = tmp_path / "migration.csv"
    save_csv(path, [dict(iteration=30), dict(iteration=31, rotation_progress=0.2)])
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["rotation_progress"] == ""
    assert rows[1]["rotation_progress"] == "0.2"


def test_rotation_component_sums_and_logging_only_legacy_parity(assets):
    corrected = RotationEnvironment(assets, RotationConfig())
    legacy_logging = RotationEnvironment(assets, RotationConfig())
    corrected.reset(1.0, 42)
    legacy_logging.reset(1.0, 42)
    total = 0.0
    for _ in range(25):
        # Simulate the pilot's historical extra-component accumulator reset.
        for key in list(legacy_logging.terms_sum):
            if key.startswith("rotation_"):
                legacy_logging.terms_sum[key] = 0.0
        _, reward, done, _ = corrected.step(np.full(12, 0.01))
        _, prior, prior_done, _ = legacy_logging.step(np.full(12, 0.01))
        assert reward == prior and done == prior_done
        np.testing.assert_array_equal(corrected.d.qpos, legacy_logging.d.qpos)
        total += reward
    assert sum(corrected.terms_sum.values()) == pytest.approx(total, abs=1e-12)
    assert corrected.terms_sum["rotation_command"] == pytest.approx(
        25 * -0.01 * 0.02 * 0.01**2
    )


def test_phase_anchor_weights_without_changing_legacy_default():
    torch.manual_seed(9)
    net = ActorCritic(44, 74, hidden=(8,))
    obs = torch.randn(16, 44)
    priv = torch.randn(16, 74)
    with torch.no_grad():
        raw, log, value = net.sample(obs, priv)
        target = (net.actor(obs) + 0.05).tanh()
    hp = dict(
        epochs=1,
        batch_size=16,
        target_kl=100,
        clip=0.2,
        value_weight=0,
        entropy_weight=0,
        max_grad_norm=1,
        bc_weight=2,
        bc_latent_scale=0.05,
        bc_batch_size=16,
    )
    batch = (obs, priv, raw, log, value, torch.zeros(16))

    def run(demos, fraction):
        n = copy.deepcopy(net)
        torch.manual_seed(13)
        return update(
            n,
            torch.optim.Adam(n.parameters(), lr=0),
            batch,
            dict(hp, bc_phase_fraction=fraction),
            demos,
        )

    original = run((obs, target), 0)
    phase_start = run((obs, target, torch.full((16,), 0.25)), 0)
    phase_end = run((obs, target, torch.full((16,), 0.25)), 1)
    assert phase_start["bc_loss"] == original["bc_loss"]
    assert phase_end["bc_loss"] == pytest.approx(0.25 * original["bc_loss"])
    assert phase_end["bc_objective_contribution"] == pytest.approx(
        2 * phase_end["bc_loss"]
    )
    prefix = run((obs, target, torch.ones(16)), 1)
    assert prefix["bc_loss"] == original["bc_loss"]
