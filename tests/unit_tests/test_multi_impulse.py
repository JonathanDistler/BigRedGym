import copy
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from experiments.multi_impulse.core import Phase, MultiConfig, MultiEnvironment
from experiments.multi_impulse.network import initialize
from experiments.multi_impulse.sensors import Corrections
from experiments.multi_impulse.vector import Vector
from experiments.front_catch.core import CatchConfig
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic, update
from experiments.front_catch.learning import collect, hp_defaults


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp("multi_assets"))


def test_phase_hysteresis_and_state_reacceleration():
    p = Phase()
    for _ in range(50):
        p.update(0.7, 0.0, True, False, True, False, 1, 0, 0.002)
    assert p.index == 4 and abs(p.weights.sum() - 1) < 1e-12
    for _ in range(10):
        p.update(0.7, 0.4, True, False, True, False, 1, 0, 0.002)
    assert p.index == 4
    for _ in range(50):
        p.update(0.4, 0.0, False, True, True, False, 1, 0, 0.002)
    assert p.index == 5


def test_free_air_is_not_push_and_real_reload_can_rearm():
    names = [
        leg + "_" + part + "_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for part in ("hip", "thigh", "calf")
    ]
    e = SimpleNamespace(
        names=names,
        vids=np.arange(12),
        weight=100.0,
        d=SimpleNamespace(
            qfrc_applied=np.ones(12) * 10, qvel=np.ones(12) * 0.5, time=0.0
        ),
    )
    row = dict(
        loads=np.array([10.0, 10.0, 0.0, 0.0]),
        feet=np.array(
            [[0.2, 0.1, 0.022], [0.2, -0.1, 0.022], [-0.2, 0.1, 0.1], [-0.2, -0.1, 0.1]]
        ),
        rear_clearance=0.078,
        front_support=True,
        pitch=0.3,
    )
    s = dict(
        prefix_safe=True,
        pitch=0.3,
        L_com=0.1,
        mechanical_energy=60,
        front_moment_com=-1,
        rear_moment_com=1,
        actuator_power=10,
    )
    c = Corrections()
    for _ in range(100):
        e.d.time += 0.002
        c.observe(e, row, s)
    assert not c.pushes
    for attempt in range(2):
        row["loads"][2:] = 40
        row["feet"][2:, 2] = 0.022
        row["rear_clearance"] = 0
        for _ in range(120):
            e.d.time += 0.002
            c.observe(e, row, s)
        row["loads"][2:] = 0
        row["feet"][2:, 2] = 0.06
        row["rear_clearance"] = 0.038
        for _ in range(20):
            e.d.time += 0.002
            c.observe(e, row, s)
    assert len(c.pushes) == 2
    assert all(p["work_min"] >= 0.05 and p["extension_min"] >= 0.03 for p in c.pushes)


def test_one_actor_preserves_B_and_adam_mapping():
    torch.set_num_threads(1)
    old = ActorCritic(73, 103)
    opt = torch.optim.Adam(old.parameters(), lr=1e-5)
    loss = sum(p.square().sum() for p in old.parameters())
    loss.backward()
    opt.step()
    parent = dict(network=old.state_dict(), optimizer=opt.state_dict())
    net, newopt = initialize(parent, False)
    a = torch.randn(16, 73)
    c = torch.randn(16, 103)
    assert torch.equal(old.actor(a), net.actor(torch.cat([a, torch.zeros(16, 20)], 1)))
    assert torch.equal(
        old.critic(c), net.critic(torch.cat([c, torch.zeros(16, 20)], 1))
    )
    assert torch.equal(
        newopt.state[net.actor.base[0].weight]["exp_avg"],
        opt.state[old.actor[0].weight]["exp_avg"],
    )


def test_physics_identical_and_no_within_episode_reset(assets):
    old = CaptureEnvironment(
        assets, CatchConfig(catch_mode="B", comparison_suite="anticipatory_v1")
    )
    new = MultiEnvironment(
        assets,
        MultiConfig(
            catch_mode="B", comparison_suite="anticipatory_v1", context_enabled=False
        ),
    )
    old.reset(1.0, 123)
    new.reset(1.0, 123)
    for _ in range(20):
        old.step(np.zeros(12))
        new.step(np.zeros(12))
        assert np.array_equal(old.d.qpos, new.d.qpos)
        assert np.array_equal(old.observations()[0], new.observations()[0][:73])
    assert new.d.time == pytest.approx(0.4) and new.observations()[0].shape == (93,)
    assert new.multi_best_hold == 0 and not new.ever_hold_success


def test_exact_next_update_boundary(assets):
    torch.set_num_threads(1)
    torch.manual_seed(100)
    parent = dict(network=ActorCritic(73, 103).state_dict())
    net, opt = initialize(parent, True)
    hp = hp_defaults()
    hp.update(epochs=1, bc_weight=0)
    cfg = MultiConfig(
        corrective=True, catch_mode="B", comparison_suite="anticipatory_v1"
    )
    mix = [1, 0, 0, 0, 0, 0]
    with Vector(assets, {"B2": cfg}, {}, 2, workers=0) as v:
        a, c = v.reset("B2", mix)
        batch, a, c, _, _ = collect(net, v, "B2", a, c, 4, mix, hp)
        update(net, opt, batch, hp)
        state = copy.deepcopy(
            dict(
                net=net.state_dict(),
                opt=opt.state_dict(),
                world=v.state("B2"),
                rng=torch.get_rng_state(),
            )
        )
        batch, expected, _, _, _ = collect(net, v, "B2", a, c, 4, mix, hp)
        update(net, opt, batch, hp)
        weights = copy.deepcopy(net.state_dict())
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        a, c = v.restore("B2", state["world"])
        torch.set_rng_state(state["rng"])
        batch, actual, _, _, _ = collect(net, v, "B2", a, c, 4, mix, hp)
        update(net, opt, batch, hp)
        assert np.array_equal(expected, actual)
        assert all(torch.equal(weights[k], net.state_dict()[k]) for k in weights)


def test_balance_bank_geometry_and_nonzero_velocity(assets):
    from pathlib import Path

    path = Path("output/multi_impulse_preparation_v1/balance_bank.pt")
    if not path.exists():
        pytest.skip("Prepared bank not present")
    from experiments.front_catch.data import validate_state

    bank = torch.load(path, weights_only=False, map_location="cpu")
    e = MultiEnvironment(assets, MultiConfig())
    for split in bank:
        for item in bank[split]["near_vertical"]:
            e.restore_start(item["state"], "near_vertical")
            assert validate_state(e)[0]
            assert np.linalg.norm(e.d.qvel) > 0
            assert item["source"]["teacher_hold"] >= 5


def test_warm_optimizer_moments_are_not_shared_between_conditions():
    from experiments.multi_impulse.network import apply_warm

    torch.set_num_threads(1)
    parent = {"network": ActorCritic(73, 103).state_dict()}
    source, source_opt = initialize(parent, True)
    sum(p.square().sum() for p in source.parameters()).backward()
    source_opt.step()
    warm = {"network": source.state_dict(), "optimizer": source_opt.state_dict()}
    left, lo = initialize(parent, True)
    right, ro = initialize(parent, True)
    apply_warm(left, lo, warm)
    apply_warm(right, ro, warm)
    key = next(iter(lo.state))
    right_key = next(iter(ro.state))
    before = ro.state[right_key]["exp_avg"].clone()
    lo.state[key]["exp_avg"].add_(1)
    assert torch.equal(before, ro.state[right_key]["exp_avg"])


def test_rejected_KL_step_restores_adam_and_weights(monkeypatch):
    import experiments.multi_impulse.learning as learning

    torch.set_num_threads(1)
    net = ActorCritic(73, 103)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    sum(p.square().sum() for p in net.parameters()).backward()
    opt.step()
    before = copy.deepcopy(opt.state_dict())
    weights = copy.deepcopy(net.state_dict())

    def bad_step(net, opt, *args):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1)
        for item in opt.state.values():
            item["exp_avg"].add_(1)
        return {"post_update_kl": 1.0}

    monkeypatch.setattr(learning, "update", bad_step)
    with pytest.raises(RuntimeError):
        learning.safe_update(net, opt, None, {"target_kl": 0.01}, None)
    assert all(torch.equal(weights[k], net.state_dict()[k]) for k in weights)
    assert all(
        torch.equal(v["exp_avg"], opt.state_dict()["state"][k]["exp_avg"])
        for k, v in before["state"].items()
    )
