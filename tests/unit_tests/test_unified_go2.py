"""Focused fusion contracts and exact native-state continuation."""

import copy, json
from pathlib import Path
import numpy as np
import torch
import pytest
from experiments.unified_go2.core import Student, Vector, load, promote
from experiments.multi_impulse.core import MultiConfig
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.front_catch.data import validate_state

PREP = Path("output/unified_go2_preparation_v1")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not PREP.exists():
        pytest.skip("Run prepare_unified_go2 first")
    torch.set_num_threads(1)
    return (
        Assets(tmp_path_factory.mktemp("unified")),
        MultiConfig(**json.loads((PREP / "config.json").read_text())),
        load(PREP / "bank.pt"),
        np.load(PREP / "demonstrations.npz"),
    )


def test_adc_mapping_and_no_hidden_experts(prepared):
    torch.set_num_threads(1)
    run = load(PREP / "adc.pt")
    old = ActorCritic(73, 103)
    old.load_state_dict(run["network"])
    new = Student.from_adc(run)
    x = torch.randn(3, 93)
    c = torch.randn(3, 123)
    assert torch.allclose(
        old.actor(x[:, :73].contiguous()), new.actor.residual(x), atol=2e-6
    )
    assert torch.allclose(old.critic(c[:, :103].contiguous()), new.critic(c), atol=2e-6)
    assert list(new.actor.residual[0].weight.shape) == [256, 93]
    assert not any("expert" in n for n, _ in new.named_modules())


def test_exact_next_physical_step(prepared):
    assets, cfg, bank, data = prepared
    v = Vector(assets, cfg, bank, 2, data["centers"])
    obs, crit = v.reset()
    state = copy.deepcopy(v.state())
    action = np.zeros((2, 12))
    r1 = v.step("", action, None)
    v2 = Vector(assets, cfg, bank, 2, data["centers"])
    o2, c2 = v2.restore(state)
    assert np.array_equal(o2, obs) and np.array_equal(c2, crit)
    r2 = v2.step("", action, None)
    for a, b in zip(r1, r2):
        assert np.array_equal(a[0], b[0])
        assert a[2] == b[2] and a[3] == b[3]
    for split in bank.values():
        for states in split.values():
            for state in states:
                v2.envs[0].restore_start(state, "partial")
                assert validate_state(v2.envs[0])[0]


def test_curriculum_requires_balance_and_bridge():
    rows = [
        dict(distribution="balance", success=True),
        dict(distribution="bridge", handstand_seconds=0),
        dict(distribution="standing", certified_standing_takeoff=True),
    ]
    assert promote(0, rows, 0.75) == 1
    assert promote(1, rows, 0.75) == 1
    rows[0]["success"] = False
    assert promote(2, rows, 0.75) == 0
