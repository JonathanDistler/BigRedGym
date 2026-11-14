"""Residual feedback, no balance-playback fallback, and portable resume selection."""

from argparse import Namespace
import copy
from pathlib import Path
import runpy

import numpy as np
import torch

from experiments.handstand_robustness.takeoff import ResidualPolicy
from scripts.handstand_takeoff import save
from scripts.verify_handstand_checkpoint import assert_equal


def policy():
    return ResidualPolicy(np.zeros((64, 12)), np.zeros((64, 43)))


def test_residual_uses_state_and_is_bounded():
    p = policy()
    with torch.no_grad():
        p.gains[:, 0, 3] = 2.0
    x = torch.zeros((2, 43))
    x[1, 3] = 0.2
    a = p(x)
    assert a[0, 0] == 0 and 0 < a[1, 0] <= 0.05
    assert (a.abs() <= 0.05).all()
    assert torch.equal(p(x, feedback=False), torch.zeros_like(a))


def test_no_recorded_balance_actions_after_takeoff_prefix():
    p = policy()
    with torch.no_grad():
        p.actions[47] = 0.1
        p.actions[48:] = 0.9
    x = torch.zeros(43)
    x[42] = 0.8
    torch.testing.assert_close(p(x), torch.full((12,), 0.1))


def test_selected_checkpoint_is_embedded_without_recursion(tmp_path):
    p = policy()
    optimizer = torch.optim.Adam(p.parameters())
    args = Namespace(trajectory_sha256="reference", seed=4110, envs=1, rollout_steps=64)
    save(tmp_path / "best.pt", p, optimizer, 1, args, [], [1, 2], select=True)
    selected = copy.deepcopy(p.state_dict())
    with torch.no_grad():
        p.gains.add_(1)
    save(tmp_path / "latest.pt", p, optimizer, 2, args, [], [1, 2])
    loaded = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert loaded["update"] == 2 and loaded["best_payload"]["update"] == 1
    assert "best_payload" not in loaded["best_payload"]
    assert_equal(loaded["best_payload"]["policy"], selected)
    assert not torch.equal(loaded["policy"]["gains"], selected["gains"])


def test_marimo_notebook_graph_is_valid_and_no_actions_run_on_import():
    from marimo._ast.load import get_notebook_status

    notebook = Path("notebooks/molab_handstand_takeoff.py")
    assert get_notebook_status(str(notebook)).status == "valid"
    app = runpy.run_path(str(notebook))["app"]
    app._maybe_initialize()


def test_long_continuation_requires_explicit_opt_in(monkeypatch, tmp_path):
    import pytest
    from scripts.handstand_takeoff import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "handstand_takeoff.py",
            "train",
            "--checkpoint",
            "unused.pt",
            "--updates",
            "200",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
