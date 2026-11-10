"""Evaluation supervisor invariants, independent of saved checkpoint contents."""

from types import SimpleNamespace

import numpy as np
import torch

from scripts import evaluate_kickup_handoff as handoff


def row():
    return dict(
        pitch_rad=np.pi / 2,
        pitch_rate=0.0,
        com_support_x=0.0,
        base_z=0.5,
        force_0=70.0,
        force_1=70.0,
        body_contact=False,
        **{f"joint_{j}": 0.0 for j in range(12)},
    )


def test_target_blend_preserves_switch_target_and_finishes():
    initial, desired = torch.zeros(16, 12), torch.ones(16, 12)
    assert torch.equal(handoff.blend_target(initial, desired, 0.0), initial)
    assert torch.equal(handoff.blend_target(initial, desired, 0.2), desired)
    assert torch.allclose(handoff.blend_target(initial, desired, 0.1), desired / 2)
    assert float(handoff.blend_target(initial, desired, 0.0001).max()) < 1e-6


def test_unknown_or_unreliable_regions_never_switch(tmp_path):
    balance = SimpleNamespace(env=SimpleNamespace(num_envs=16))
    run = {
        "control_samples": [
            (t, [None] * 16, torch.zeros(16, 12), [row() for _ in range(16)])
            for t in (0.01, 0.02, 0.03, 0.04)
        ]
    }
    atlas = [dict(successes=4, count=16, trials=[dict(success=True, initial=row())])]
    report = handoff.supervise(balance, run, atlas, tmp_path, "uncertain")
    assert report["switches"] == 0
    assert report["full_transition_successes"] == 0


def test_support_debounce_and_hold_success(monkeypatch, tmp_path):
    balance = SimpleNamespace(env=SimpleNamespace(num_envs=16))
    atlas = [dict(successes=16, count=16, trials=[dict(success=True, initial=row())])]
    run = dict(
        control_samples=[
            (t, [None] * 16, torch.zeros(16, 12), [row() for _ in range(16)])
            for t in (0.01, 0.02, 0.03, 0.04)
        ],
        models=[],
    )
    monkeypatch.setattr(
        handoff,
        "recovery",
        lambda *a, **k: {"trials": [dict(hold_s=5.1)] + [dict(hold_s=4.9)] * 15},
    )
    report = handoff.supervise(balance, run, atlas, tmp_path, "known")
    assert report["switch_times"] == [0.03] * 16
    assert report["full_transition_successes"] == 1


def test_state_samples_do_not_alias_simulator_buffers():
    d = SimpleNamespace(qpos=np.ones(19), qvel=np.ones(18))
    e = SimpleNamespace(_backend=SimpleNamespace(_datas=[d]))
    samples = handoff.states(e)
    d.qpos[:] = 0
    d.qvel[:] = 0
    assert samples[0][0].sum() == 19
    assert samples[0][1].sum() == 18


def test_zero_success_does_not_imply_zero_probability():
    lo, hi = handoff.wilson(0, 16)
    assert abs(lo) < 1e-12
    assert 0.19 < hi < 0.20
