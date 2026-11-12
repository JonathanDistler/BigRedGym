"""BC anchoring and explicit exploration-floor contracts."""

import copy
import numpy as np
import torch
from experiments.morphology.learning.ppo import ActorCritic, update


def test_exploration_floor_is_explicit_and_backward_compatible():
    old = ActorCritic(44, 74, hidden=(8,), log_std=np.log(0.01))
    new = ActorCritic(44, 74, hidden=(8,), log_std=np.log(0.01), log_std_min=-6)
    x = torch.zeros(4, 44)
    assert torch.allclose(old.distribution(x).scale, torch.full((4, 12), np.exp(-3)))
    assert torch.allclose(new.distribution(x).scale, torch.full((4, 12), 0.01))


def test_bc_anchor_moves_actor_toward_demonstrations_and_leaves_critic():
    torch.manual_seed(7)
    net = ActorCritic(44, 74, hidden=(8,))
    actor = torch.randn(32, 44)
    critic = torch.randn(32, 74)
    with torch.no_grad():
        raw, logs, values = net.sample(actor, critic)
        target = (net.actor(actor) + 0.1).tanh()
        before = (net.actor(actor) - torch.atanh(target)).square().mean().item()
    untouched = copy.deepcopy(net.critic.state_dict())
    hp = dict(
        epochs=4,
        batch_size=32,
        target_kl=100,
        clip=0.2,
        value_weight=0,
        entropy_weight=0,
        max_grad_norm=1,
        bc_weight=1,
        bc_latent_scale=0.05,
        bc_batch_size=32,
    )
    stats = update(
        net,
        torch.optim.Adam(net.parameters(), lr=0.001),
        (actor, critic, raw, logs, values, torch.zeros(32)),
        hp,
        (actor, target),
    )
    after = (net.actor(actor) - torch.atanh(target)).square().mean().item()
    assert after < before
    assert stats["bc_loss"] > 0 and stats["bc_weight"] == 1
    assert all(torch.equal(v, net.critic.state_dict()[k]) for k, v in untouched.items())


def test_retention_assessment_follows_training_geometry_after_promotion(
    tmp_path, monkeypatch
):
    import json
    import sys
    import pandas as pd
    from scripts.assess_takeoff_retention import main

    records = []
    for iteration in (0, 6, 8, 10):
        for scale in (1.0, 0.95 if iteration == 10 else 0.9):
            for stochastic in (False, True):
                records.append(
                    dict(
                        method="B",
                        training_seed=42,
                        iteration=iteration,
                        scale=scale,
                        stochastic=stochastic,
                        takeoff=True,
                    )
                )
    pd.DataFrame(records).to_csv(tmp_path / "probes.csv", index=False)
    (tmp_path / "outcome.json").write_text(
        json.dumps(
            [
                dict(
                    method="B",
                    seed=42,
                    updates=10,
                    stop_reason="completed bounded diagnostic",
                )
            ]
        )
    )
    monkeypatch.setattr(sys, "argv", ["assessment", "--run", str(tmp_path)])
    main()
    gate = json.loads((tmp_path / "retention_gate.json").read_text())
    assert gate["passed"]
    assert gate["checks"][0]["criteria"][1]["scales_evaluated"] == [0.9, 0.95]
    assert gate["checks"][0]["criteria"][1]["final"] == 1.0
