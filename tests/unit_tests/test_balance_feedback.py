"""Focused observable-feedback contract and real-contact recovery checks."""

from pathlib import Path
import pytest
import torch
from experiments.balance_feedback.actor import BalanceActor, adapt_adc_observation
from experiments.multi_impulse.core import MultiConfig, MultiEnvironment
from experiments.morphology.learning.environment import Assets


@pytest.fixture(scope="module")
def saved():
    path = Path("output/balance_feedback_v2/balance_actor.pt")
    if not path.exists():
        pytest.skip("Run the bounded balance-feedback validation first")
    payload = torch.load(path, weights_only=False, map_location="cpu")
    actor = BalanceActor()
    actor.load_state_dict(payload["network"])
    return actor, payload


def test_clock_and_phase_independent_bounded_action(saved):
    actor, _ = saved
    obs = torch.zeros((2, 93))
    obs[1, 30:55] = torch.linspace(-4, 4, 25)
    obs[1, 59:] = 2
    assert torch.equal(actor(obs)[0], actor(obs)[1])
    prefix = torch.randn(4, 73)
    padded = adapt_adc_observation(prefix)
    assert torch.equal(padded[:, :73], prefix)
    assert torch.equal(actor(padded), actor(torch.cat([prefix, torch.randn(4, 20)], 1)))
    actions = actor(torch.randn(16, 93) * 100)
    assert torch.isfinite(actions).all()
    assert (actions >= actor.lower).all() and (actions <= actor.upper).all()
    with pytest.raises(ValueError):
        actor(torch.zeros(73))


def test_serialized_feedback_uses_identical_action_convention(saved):
    actor, payload = saved
    restored = BalanceActor()
    restored.load_state_dict(payload["network"])
    obs = torch.randn(4, 93)
    assert torch.equal(actor(obs), restored(obs))
    assert payload["metadata"]["dt"] == 0.02
    assert payload["metadata"]["kp"] == 60 and payload["metadata"]["kd"] == 3


def test_real_contact_balance_exceeds_one_second(saved, tmp_path):
    torch.set_num_threads(1)
    actor, payload = saved
    cfg = MultiConfig(**payload["config"])
    cfg.seconds = 1.2
    env = MultiEnvironment(Assets(tmp_path), cfg)
    states = torch.load(
        "output/balance_feedback_v2/evaluation_states.pt",
        weights_only=False,
        map_location="cpu",
    )
    obs, _ = env.restore_start(states[0][1], "near_vertical")
    for _ in range(60):
        with torch.no_grad():
            action = actor(torch.tensor(obs)).numpy()
        (obs, _), _, done, info = env.step(action)
        if done:
            break
    assert env.multi_best_hold > 1.0
    assert not info["nonfoot_collision"] and not info["self_collision"]
    assert info["max_torque_ratio"] <= 1.000001
    assert info["max_speed_ratio"] < 1
