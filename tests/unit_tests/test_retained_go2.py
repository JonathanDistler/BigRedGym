"""Endpoint evidence, preserved conventions, valid resets and exact PPO continuation."""

import copy, json
from pathlib import Path
import numpy as np
import torch
import pytest
from scripts.unified_go2 import setup
from experiments.unified_go2.core import adc_network, load, digest
from experiments.retained_go2.actor import Student
from experiments.retained_go2.learning import Vector
from experiments.front_catch.data import validate_state
from experiments.front_catch.learning import collect, hp_defaults
from experiments.multi_impulse.learning import safe_update

PREP = Path("output/retained_go2_preparation_v3")
MODEL = Path("output/retained_go2_distillation_v3/student.pt")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("Requires retained-skill preparation artifacts")
    torch.set_num_threads(1)
    return setup(PREP, tmp_path_factory.mktemp("retained_assets"))


def test_zero_specialization_preserves_adc_and_joint_order(prepared):
    cfg, data, bank, assets = prepared
    parent = load(PREP / "adc.pt")
    bal = load(PREP / "balance.pt")
    student = Student.from_adc(parent, bal)
    reference = adc_network(parent)
    obs = torch.tensor(data["observations"])
    assert torch.equal(student.actor(obs), reference.actor(obs))
    from experiments.morphology.learning.ppo import ActorCritic

    raw = ActorCritic(73, 103)
    raw.load_state_dict(parent["network"])
    assert torch.allclose(
        student.actor(obs), raw.actor(obs[:, :73].contiguous()), atol=2e-6
    )
    assert np.allclose(cfg.kp, parent["config"]["kp"]) and np.allclose(
        cfg.kd, parent["config"]["kd"]
    )
    assert (
        cfg.dt == parent["config"]["dt"]
        and cfg.action_scale == parent["config"]["action_scale"]
    )
    assert (
        list(assets.metadata[1.0]["names"])
        == json.loads((PREP / "compatibility.json").read_text())["joint_names"]
    )


def test_distilled_skills_and_normalization_are_preserved(prepared):
    cfg, data, bank, assets = prepared
    net = Student()
    net.load_state_dict(load(MODEL)["network"])
    before = copy.deepcopy(net.actor.state_dict())
    obs = torch.tensor(data["observations"])
    with torch.no_grad():
        prediction = net.actor(obs).tanh().numpy()
    assert np.max(abs(prediction - data["actions"])) < 2e-6
    for name in ["center", "scale", "inner_low", "inner_high", "outer_width"]:
        assert torch.equal(before[name], net.actor.state_dict()[name])
    cert = json.loads(
        Path("output/retained_go2_validation_v3/retention.json").read_text()
    )
    assert cert["initial_sha"] == digest(MODEL) and cert["passed"]
    assert (
        cert["standing_takeoffs"] == cert["standing_n"] == 4
        and cert["balance_successes"] >= 3
        and cert["balance_n"] == 4
    )


def test_all_intermediate_states_are_native_valid(prepared):
    cfg, data, bank, assets = prepared
    v = Vector(assets, cfg, bank, 2, data["centers"])
    e = v.envs[0]
    for split in bank.values():
        for states in split.values():
            for state in states:
                e.restore_start(copy.deepcopy(state), "partial")
                assert validate_state(e)[0]
    assert v.probabilities()[0] >= 0.25 and v.probabilities()[2] >= 0.25
    v.competence[2] = 0.2
    assert v.probabilities()[2] >= 0.55


def test_exact_next_ppo_update_including_optimizer_and_router(prepared):
    cfg, data, bank, assets = prepared
    torch.manual_seed(123)
    net = Student()
    net.load_state_dict(load(MODEL)["network"])
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    v = Vector(assets, cfg, bank, 2, data["centers"])
    a, c = v.reset()
    hp = hp_defaults()
    hp.update(epochs=1, bc_weight=0.0)
    batch, a, c, _, _ = collect(net, v, "", a, c, 2, None, hp)
    safe_update(net, opt, batch, hp, None)
    state = copy.deepcopy(
        dict(
            net=net.state_dict(),
            opt=opt.state_dict(),
            world=v.state(),
            rng=torch.get_rng_state(),
            a=a,
            c=c,
        )
    )
    batch, a1, c1, _, _ = collect(net, v, "", a, c, 2, None, hp)
    safe_update(net, opt, batch, hp, None)
    restored = Student()
    restored.load_state_dict(state["net"])
    opt2 = torch.optim.Adam(restored.parameters(), lr=1e-5)
    opt2.load_state_dict(state["opt"])
    v2 = Vector(assets, cfg, bank, 2, data["centers"])
    a2, c2 = v2.restore(state["world"])
    assert np.array_equal(a2, state["a"]) and np.array_equal(c2, state["c"])
    torch.set_rng_state(state["rng"])
    batch2, _, _, _, _ = collect(restored, v2, "", a2, c2, 2, None, hp)
    assert all(torch.equal(x, y) for x, y in zip(batch, batch2))
    safe_update(restored, opt2, batch2, hp, None)
    assert all(
        torch.equal(net.state_dict()[k], restored.state_dict()[k])
        for k in net.state_dict()
    )
    for key, value in opt.state_dict()["state"].items():
        for name, tensor in value.items():
            assert torch.equal(tensor, opt2.state_dict()["state"][key][name])


def test_replay_is_class_balanced_without_new_demonstrations(prepared):
    cfg, data, bank, assets = prepared
    assert (data["roles"] == 0).sum() == (data["roles"] == 1).sum()
    assert len(np.unique(data["source_frame_index"])) == 1870
    assert len(data["roles"]) == 3100


def test_changed_retention_contract_rejects_resume(prepared, monkeypatch):
    import experiments.retained_go2.learning as learning

    cfg, data, bank, assets = prepared
    v = Vector(assets, cfg, bank, 1, data["centers"])
    v.reset()
    state = copy.deepcopy(v.state())
    monkeypatch.setattr(learning, "RETENTION_HASH", "changed")
    with pytest.raises(ValueError, match="Retention evaluation contract"):
        v.restore(state)
