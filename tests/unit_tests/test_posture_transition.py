import numpy as np
import pytest
import torch

from gym.envs.go2.kickup_reference import KickReference
from gym.envs.go2.posture_transition import BoundedCMA, PostureTransition, smooth
from gym.envs.go2.posture_transition import rate_limited_target


def test_bounded_cma_improves_objective_and_preserves_bounds():
    solver = BoundedCMA(np.full(4, 0.8), population=10, seed=83)
    initial = np.sum((solver.mean - 0.2) ** 2)
    best = initial
    for _ in range(45):
        samples = solver.ask()
        assert np.isfinite(samples).all()
        assert ((samples >= 0) & (samples <= 1)).all()
        costs = np.sum((samples - 0.2) ** 2, axis=1)
        best = min(best, costs.min())
        solver.tell(samples, costs)
        assert np.linalg.eigvalsh(solver.cov).min() > 0
    assert best < initial * 0.001


def test_transition_is_opt_in_and_does_not_mutate_reference():
    source = KickReference()
    before = source.to_dict()
    p = PostureTransition()
    q, v = p.reference(1.1, source, np.array([0.0, 0.7, -1.4] * 4))
    assert source.to_dict() == before
    assert q.shape == v.shape == (12,)
    assert np.isfinite(q).all() and np.isfinite(v).all()


def test_blending_has_exact_endpoints_and_zero_entry_slope():
    assert smooth(0.0, 1.0, 0.6) == 0
    assert smooth(1.0, 1.0, 0.6) == 0
    assert smooth(1.6, 1.0, 0.6) == 1
    assert smooth(1.00001, 1.0, 0.6) / 0.00001 < 0.001


@pytest.mark.parametrize(
    "values",
    [dict(blend_s=0), dict(front_gain=2), dict(takeover_s=1.3), dict(front_hip=np.nan)],
)
def test_invalid_parameters_are_rejected(values):
    with pytest.raises(ValueError):
        PostureTransition(**values)


def test_large_policy_changes_are_applied_within_native_velocity_limits():
    previous = torch.zeros(4, 12)
    desired = torch.full((4, 12), 2.0)
    limits = torch.tensor([20.0, 30.0, 15.0] * 4)
    for _ in range(100):
        applied = rate_limited_target(desired, previous, limits, 0.002)
        assert ((applied - previous).abs() <= limits * 0.002 + 1e-7).all()
        previous = applied
    assert torch.allclose(previous, desired)
    reverse = rate_limited_target(-desired, previous, limits, 0.002)
    assert ((reverse - previous).abs() <= limits * 0.002 + 1e-7).all()
