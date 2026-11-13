"""Native contact gates, independent vector worlds, and exact resume state."""

import copy
from pathlib import Path
import mujoco
import numpy as np
import pytest
from experiments.front_catch.core import (
    CatchConfig,
    CatchEnvironment,
    backward_displacement,
)
from experiments.front_catch.data import validate_state
from experiments.front_catch.vector import Vector
from experiments.morphology.learning.environment import Assets


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("catch_assets"))


def test_world_backward_sign():
    p = np.array([[-0.1, 0, 0], [0.2, 0, 0.0]])
    assert np.allclose(
        backward_displacement(p, np.zeros((2, 3)), [0, 1, 0]), [0.1, -0.2]
    )
    assert backward_displacement([0, -0.1, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(
        0.1
    )


def test_grounded_and_unsupported_rewards(assets):
    e = CatchEnvironment(assets, CatchConfig(catch_mode="C"))
    e.cfg.target_rotation = tuple(e.d.xmat[e.base])
    e.cfg.target_height = e.row["com"][2]
    (a, c), reward, _, info = e.step(np.zeros(12))
    assert a.shape == (73,) and c.shape == (103,) and np.isfinite(reward)
    assert (
        e.trace[-1]["terminal_reward"] == 0
        and e.trace[-1]["reward_useful_placement"] == 0
    )
    assert not info["standing_success"]
    e.tracker.confirmed = True
    e.d.qpos[2] += 0.6
    mujoco.mj_forward(e.m, e.d)
    e.step(np.zeros(12))
    assert e.trace[-1]["reward_rotation_progress"] == 0
    assert e.trace[-1]["terminal_reward"] == 0


def test_contact_events_and_impact_latch(assets):
    e = CatchEnvironment(assets, CatchConfig(catch_mode="C"))
    row = copy.deepcopy(e.row)
    s = e.momentum_sample.copy()
    s["front_force_z"] = 20.0
    row["loads"][:2] = 0
    e.contact_measure(row, s, 0.03)
    row["loads"][:2] = 20
    assert e.contact_measure(row, s, 0.04)["front_recontacts"] == 0
    e.tracker.confirmed = True
    row["loads"][0] = 0
    e.contact_measure(row, s, 0.03)
    row["loads"][0] = 20
    cs = e.contact_measure(row, s, 0.04)
    assert cs["FL_recontacts"] == 1 and cs["FR_recontacts"] == 0 and cs["caught"]
    s["front_force_z"] = 5 * e.weight
    assert not e.contact_measure(row, s, 0.002)["impact_valid"]
    s["front_force_z"] = 20
    assert not e.contact_measure(row, s, 0.04)["impact_valid"]


def test_exact_environment_resume(assets):
    e = CatchEnvironment(assets, CatchConfig(catch_mode="C"))
    for _ in range(4):
        e.step(np.full(12, 0.01))
    state = copy.deepcopy(e.state())
    clone = CatchEnvironment(assets, copy.deepcopy(e.cfg))
    clone.restore(state)
    for _ in range(4):
        a = e.step(np.full(12, 0.02))
        b = clone.step(np.full(12, 0.02))
        assert a[1] == pytest.approx(b[1], abs=1e-12)
        assert np.array_equal(e.d.qpos, clone.d.qpos)
    e.reset(1.0, 99)
    assert (
        e.recontacts.sum() == 0 and not e.impact_failed and e.start_kind == "standing"
    )
    assert validate_state(e)[0]


def test_vector_independence_and_process_equivalence(assets):
    configs = {"A": CatchConfig()}
    mix = [1, 0, 0, 0, 0, 0]
    with Vector(assets, configs, {}, 2, seed=42, workers=0) as serial:
        x, c = serial.reset("A", mix)
        assert not np.array_equal(x[0], x[1])
        assert serial.lanes["A"][0].env.d is not serial.lanes["A"][1].env.d
        assert serial.lanes["A"][0].env.cfg is not serial.lanes["A"][1].env.cfg
        with Vector(assets, configs, {}, 2, seed=42, workers=2) as parallel:
            y, d = parallel.reset("A", mix)
            assert np.array_equal(x, y) and np.array_equal(c, d)
            for _ in range(3):
                actions = np.stack([np.full(12, 0.01), np.full(12, -0.01)])
                a = serial.step("A", actions, mix)
                b = parallel.step("A", actions, mix)
                for left, right in zip(a, b, strict=True):
                    assert np.array_equal(left[0], right[0])
                    assert left[2] == pytest.approx(right[2], abs=1e-12)
            state = parallel.state("A")
            expected = parallel.step("A", actions, mix)
            parallel.restore("A", state)
            actual = parallel.step("A", actions, mix)
            assert all(
                np.array_equal(a[0], b[0])
                for a, b in zip(expected, actual, strict=True)
            )


def test_bank_split_and_native_validity(assets):
    import torch

    path = Path("output/front_catch_preparation_20261112_v3/reset_bank.pt")
    if not path.exists():
        pytest.skip("Preparation assets unavailable")
    bank = torch.load(path, weights_only=False, map_location="cpu")
    train = {s["state_id"] for group in bank["train"].values() for s in group}
    heldout = {s["state_id"] for group in bank["evaluation"].values() for s in group}
    assert not train & heldout
    e = CatchEnvironment(assets, CatchConfig(catch_mode="C"))
    for split in bank:
        for kind, states in bank[split].items():
            assert states
            e.restore_start(states[0]["state"], kind)
            assert validate_state(e)[0]
            _, _, _, info = e.step(np.zeros(12))
            assert not info["standing_success"] and info["start_kind"] == kind


def test_fast_native_quantities_match_independent_link_sums(assets):
    from experiments.momentum_phase.physics import sample
    from experiments.front_catch.physics import sample_fast
    from experiments.morphology.learning.environment import Environment

    e = CatchEnvironment(assets, CatchConfig(catch_mode="C"))
    for seed in range(4):
        e.d.qvel[:] = np.random.default_rng(seed).normal(0, 0.1, e.m.nv)
        mujoco.mj_forward(e.m, e.d)
        row = Environment.measure(e)
        old = sample(e, row)
        fast = sample_fast(e, row)
        for key in (
            "L_com",
            "L_support",
            "I_com",
            "I_support",
            "K_system",
            "potential",
            "mechanical_energy",
            "moment_com",
            "moment_support",
            "front_force_z",
            "rear_force_z",
            "calf_clearance",
            "pitch",
            "pitch_velocity",
        ):
            assert old[key] == pytest.approx(fast[key], rel=1e-10, abs=1e-10)
    for _ in range(4):
        e.step(np.full(12, 0.02))
        row = Environment.measure(e)
        old = sample(e, row)
        fast = sample_fast(e, row)
        assert old["moment_com"] == pytest.approx(fast["moment_com"], abs=1e-10)


def test_optimizer_rng_and_world_resume_is_exact(assets):
    import torch
    from experiments.morphology.learning.ppo import ActorCritic, update
    from experiments.front_catch.learning import collect, hp_defaults

    torch.set_num_threads(1)
    torch.manual_seed(22)
    net = ActorCritic(73, 103)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    hp = hp_defaults()
    hp.update(epochs=1, batch_size=4, bc_weight=0.0)
    mix = [1.0, 0, 0, 0, 0, 0]
    with Vector(assets, {"A": CatchConfig()}, {}, 2, seed=42, workers=0) as vec:
        a, c = vec.reset("A", mix)
        batch, a, c, _, _ = collect(net, vec, "A", a, c, 2, mix, hp)
        update(net, opt, batch, hp)
        saved = dict(
            network=copy.deepcopy(net.state_dict()),
            optimizer=copy.deepcopy(opt.state_dict()),
            rng=torch.get_rng_state(),
            worlds=copy.deepcopy(vec.state("A")),
            a=a.copy(),
            c=c.copy(),
        )
        batch, a, c, _, _ = collect(net, vec, "A", a, c, 2, mix, hp)
        update(net, opt, batch, hp)
        expected = copy.deepcopy(net.state_dict())
        expected_obs = a.copy()
        net.load_state_dict(saved["network"])
        opt.load_state_dict(saved["optimizer"])
        a, c = vec.restore("A", saved["worlds"])
        torch.set_rng_state(saved["rng"])
        assert np.array_equal(a, saved["a"])
        batch, a, c, _, _ = collect(net, vec, "A", a, c, 2, mix, hp)
        update(net, opt, batch, hp)
        assert np.array_equal(a, expected_obs)
        assert all(
            torch.equal(value, net.state_dict()[key]) for key, value in expected.items()
        )


def test_checkpoint_never_overwrites(tmp_path):
    from experiments.front_catch.io import atomic_checkpoint

    target = tmp_path / "checkpoint.pt"
    atomic_checkpoint(target, {"update": 1})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_checkpoint(target, {"update": 2})
    assert target.read_bytes() == original and not list(tmp_path.glob("*.tmp"))


def test_grounded_or_falling_recatch_is_not_a_useful_swing():
    from experiments.front_catch.core import useful_swing_state

    s = dict(mass=16.0, pitch=0.6, pitch_velocity=0.5, com_velocity_z=0.1)
    row = dict(rear_clearance=0.1, loads=np.array([80.0, 80.0, 0.0, 0.0]))
    assert useful_swing_state(s, row, 0.0, 0.6)
    grounded = dict(row, rear_clearance=0.0, loads=np.array([40.0, 40.0, 40.0, 40.0]))
    assert not useful_swing_state(s, grounded, 0.0, 0.6)
    assert not useful_swing_state(
        dict(s, pitch=0.2, pitch_velocity=-1.0), row, 0.0, 0.6
    )
    assert not useful_swing_state(dict(s, com_velocity_z=-1.0), row, 0.0, 0.6)
