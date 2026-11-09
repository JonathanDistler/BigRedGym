import pytest
import torch

from gym.envs.go2.kickup_cpg import KickupCPG
from types import SimpleNamespace


def config():
    standing = [[0, 0.58, -1.16]] * 4
    return SimpleNamespace(
        duration_s=1.0,
        standing=standing,
        crouched=standing[:2] + [[0, 1.15, -2.3]] * 2,
        extended=standing[:2] + [[0, 0.3, -0.85]] * 2,
    )


def test_one_shot_and_independent_resets():
    c = config()
    oscillator = KickupCPG(2, "cpu", c)
    assert oscillator.targets()[0].tolist() == pytest.approx(sum(c.standing, []))
    oscillator.advance(c.duration_s / 2, torch.zeros(2, dtype=torch.bool))
    assert oscillator.targets()[0].tolist() == pytest.approx(sum(c.crouched, []))
    oscillator.advance(c.duration_s / 2, torch.tensor([True, False]))
    assert oscillator.handed_off.tolist() == [True, False]
    assert oscillator.failed.tolist() == [False, True]
    assert oscillator.targets()[1].tolist() == pytest.approx(sum(c.extended, []))
    oscillator.reset(torch.tensor([False, True]))
    assert oscillator.handed_off.tolist() == [True, False]
    assert oscillator.phase[1].sum() == 0
    oscillator.advance(0.01, torch.zeros(2, dtype=torch.bool))
    assert oscillator.handed_off[0]


def test_amplitude_scales_each_environment():
    c = config()
    oscillator = KickupCPG(2, "cpu", c)
    oscillator.amplitude[:] = torch.tensor([0.25, 1.0])
    oscillator.advance(c.duration_s / 2, torch.zeros(2, dtype=torch.bool))
    standing = torch.tensor(c.standing).flatten()
    crouched = torch.tensor(c.crouched).flatten()
    assert torch.allclose(
        oscillator.targets()[0], standing + 0.25 * (crouched - standing)
    )
    assert torch.allclose(oscillator.targets()[1], crouched)
