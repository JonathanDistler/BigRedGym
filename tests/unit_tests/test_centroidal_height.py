import copy
from pathlib import Path
import numpy as np
import pytest
from scripts.stabilize_centroidal_capture import envelope, gates


def test_support_pulse_is_zero_outside_interval():
    assert envelope(-0.1, 0.5) == 0
    assert envelope(0, 0.5) == 0
    assert envelope(0.5, 0.5) == 0
    assert envelope(0.6, 0.5) == 0
    assert 0 < envelope(0.05, 0.5) < 1
    assert envelope(0.2, 0.5) == 1


def test_logged_strict_gate_matches_original(tmp_path):
    from scripts.unified_go2 import setup, actions
    from experiments.unified_go2.core import Environment, load
    from experiments.balance_feedback.actor import BalanceActor

    prep = Path("output/retained_go2_preparation_v3")
    if not (prep / "bank.pt").exists():
        pytest.skip("Preserved simulator data required")
    cfg, data, bank, assets = setup(prep, tmp_path)
    e = Environment(assets, copy.deepcopy(cfg))
    e.capture_centers = data["centers"]
    actor = BalanceActor()
    actor.load_state_dict(load(prep / "balance.pt")["network"])
    obs, _ = e.restore_start(bank["evaluation"]["balance"][0], "near_vertical")
    for _ in range(5):
        (obs, _), _, done, _ = e.step(actions(actor, obs[None], "balance")[0])
        assert all(gates(e).values()) == bool(e.multi_hold > 0)
        assert np.isfinite(e.multi_best_hold)
