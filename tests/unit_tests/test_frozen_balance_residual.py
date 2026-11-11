import torch
import pytest
import json
from torch import nn
from types import SimpleNamespace

from gym.envs.go2.frozen_balance_residual import FrozenBalanceResidual, ResidualAdapter
from scripts.train_frozen_residual import gae


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(59, 12)

    def act_inference(self, obs):
        return self.linear(obs)


def test_zero_head_has_exact_parity_and_frozen_teacher():
    teacher = Teacher()
    model = FrozenBalanceResidual(teacher)
    model.train()
    obs = torch.randn(64, 59)
    delta = model.correction(obs)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(model.combined(obs, delta), teacher.act_inference(obs))
    assert not model.balance.training
    assert all(not p.requires_grad for p in model.balance.parameters())
    delta.sum().backward()
    assert all(p.grad is None for p in model.balance.parameters())


def test_gate_protects_stable_state_and_bounds_difficult_state():
    model = FrozenBalanceResidual(Teacher())
    obs = torch.zeros(2, 59)
    obs[0, 3] = 1.0
    obs[1, 3] = 0.5
    latent = torch.full((2, 12), 100.0)
    delta = model.correction(obs, latent)
    assert torch.equal(delta[0], torch.zeros(12))
    assert torch.allclose(delta[1], torch.full((12,), 0.04))


def test_timeouts_bootstrap_terminal_observation_without_reset_leakage():
    rewards = torch.tensor([[1.0, 1.0], [100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[5.0, 7.0], [0.0, 0.0]])
    terminated = torch.tensor([[True, False], [True, True]])
    dones = torch.ones_like(terminated)
    advantages, returns = gae(
        rewards, values, next_values, terminated, dones, gamma=0.9
    )
    assert torch.allclose(returns[0], torch.tensor([1.0, 7.3]))
    assert torch.equal(advantages, returns)


def test_adapter_limits_added_targets_and_resets_individual_memory():
    teacher = Teacher()
    nn.init.zeros_(teacher.linear.weight)
    nn.init.zeros_(teacher.linear.bias)
    model = FrozenBalanceResidual(teacher)
    obs = torch.zeros(2, 59)
    obs[:, 3] = 0.5
    env = SimpleNamespace(
        num_envs=2,
        dof_vel_limits=torch.ones(12),
        dt=0.001,
        actuated_dof_indices=torch.arange(12),
        dof_pos_limits=torch.tensor([[-0.0005, 0.0005]] * 12),
        _standing_posture=lambda: torch.zeros(2, 12),
    )

    def set_actions(names, command, disable):
        env.dof_pos_target = command * 0.25

    runner = SimpleNamespace(
        get_obs=lambda names: obs,
        actor_cfg={"obs": [], "actions": []},
        set_actions=set_actions,
    )
    adapter = ResidualAdapter(SimpleNamespace(env=env, runner=runner), model)
    adapter.apply(torch.full((2, 12), 10.0))
    assert adapter.last.abs().max() <= 0.0005
    assert torch.equal(adapter.last, env.dof_pos_target)
    adapter.reset(torch.tensor([True, False]))
    assert not adapter.previous[0].any()
    assert adapter.previous[1].any()


def test_loading_rejects_teacher_mutation(tmp_path, monkeypatch):
    from scripts import evaluate_frozen_residual as evaluation

    policy = SimpleNamespace(
        runner=SimpleNamespace(alg=SimpleNamespace(actor=Teacher()))
    )
    model = FrozenBalanceResidual(policy.runner.alg.actor)
    checkpoint = tmp_path / "model.pt"
    monkeypatch.setattr(evaluation, "digest", lambda path: "baseline-hash")
    torch.save(
        dict(baseline_sha256="baseline-hash", controller=model.state_dict()), checkpoint
    )
    restored = evaluation.load_controller(policy, checkpoint)
    assert not restored.balance.training
    with torch.no_grad():
        model.balance.linear.bias.add_(1.0)
    torch.save(
        dict(baseline_sha256="baseline-hash", controller=model.state_dict()), checkpoint
    )
    with pytest.raises(ValueError, match="modified the frozen teacher"):
        evaluation.load_controller(policy, checkpoint)


@pytest.mark.parametrize("bad_outcomes", [True, False])
def test_training_refuses_failed_parity_before_environment_creation(
    tmp_path, bad_outcomes
):
    from scripts.train_frozen_residual import validate_baseline_parity

    parity = tmp_path / "parity.json"
    commands = tmp_path / "commands.json"
    parity.write_text(
        json.dumps(
            dict(
                exact_complete_outcome_parity=True,
                near_vertical=13 if bad_outcomes else 14,
                validation_plus_heldout=30,
                states=123,
            )
        )
    )
    commands.write_text(
        json.dumps(
            dict(
                exact_command_and_controller_parity=True,
                states=285,
                maximum_errors={"leg_phase": 0.0 if bad_outcomes else 1e-6},
            )
        )
    )
    with pytest.raises(ValueError, match="parity must pass"):
        validate_baseline_parity(parity, commands)
