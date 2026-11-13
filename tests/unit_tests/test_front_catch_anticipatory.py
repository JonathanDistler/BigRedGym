"""Anticipatory reward causality, physical gates and old baseline equivalence."""

import copy
import mujoco
import numpy as np
import pytest
from experiments.front_catch.anticipatory import (
    AnticipatoryEnvironment, AnticipatoryPhase, objective_gate, placement_prediction,
)
from experiments.front_catch.core import CatchConfig, CatchEnvironment
from experiments.morphology.learning.environment import Assets


@pytest.fixture(scope='module')
def assets(tmp_path_factory):
    return Assets(tmp_path_factory.mktemp('early_catch_assets'))


def test_staggered_forecast_improvement_and_no_backward_only_reward():
    cfg = CatchConfig()
    origin = np.array([[0., .1, .022], [0., -.1, .022]])
    feet = origin.copy()
    feet[0, 0] -= .04  # One foot moves while the other can support.
    heading = np.array([1., 0, 0])
    score, _, request, _ = placement_prediction(
        np.array([-.08, 0, .3]), np.array([-.1, 0, .1]), feet, origin, heading, 9.81, cfg)
    assert score > 0 and (request > 0).all()
    # Identical displacement is useless when the predicted support lies forward.
    score, _, request, _ = placement_prediction(
        np.array([.1, 0, .3]), np.array([.1, 0, .1]), feet, origin, heading, 9.81, cfg)
    assert score == 0 and (request == 0).all()
    # Moving past the desired placement and farther away gets no credit.
    feet[:, 0] = -.3
    score, *_ = placement_prediction(np.array([-.05, 0, .3]), np.zeros(3), feet,
                                     origin, heading, 9.81, cfg)
    assert score == 0


def test_online_rising_gate_allows_unilateral_but_rejects_fall_and_collision():
    s = dict(mass=16., pitch=.5, pitch_velocity=1., com_velocity_z=.2, calf_clearance=.02)
    row = dict(rear_clearance=.1, loads=np.array([80., 0., 0., 0.]))
    cs = dict(any_front_loaded=True, impact_valid=True, head_clearance=.03)
    def gate(state=s, contact=cs, safe=True, started=True, phase=.4, r=row):
        return objective_gate(state, r, contact, started, safe, 0., .5, phase)
    assert gate()  # Does not wait for bilateral recontact or peak pitch.
    assert not gate(contact=dict(cs, any_front_loaded=False))
    assert not gate(state=dict(s, pitch_velocity=-.01))
    assert not gate(state=dict(s, com_velocity_z=-.01))
    assert not gate(state=dict(s, calf_clearance=0))
    assert not gate(contact=dict(cs, head_clearance=0))
    assert not gate(contact=dict(cs, impact_valid=False))
    assert not gate(safe=False) and not gate(started=False) and not gate(phase=.2)
    assert not gate(r=dict(row, rear_clearance=0))


def test_event_phase_starts_reposition_at_takeoff_without_pitch_threshold():
    p = AnticipatoryPhase((.2, .6, .74, .95, 1.4))
    p.front_loaded = True
    p.update(10., True, False, .3, False)
    assert p.progress < .4
    p.omega = 2.
    p.current_support = False
    p.update(10.02, True, True, .3, False)
    assert .4 <= p.progress < .6
    assert np.isfinite(p.encode()).all()
    p.update(10.04, True, True, .4, False)
    assert p.progress >= .4
    fresh = AnticipatoryPhase(p.times)
    assert fresh.progress == 0 and fresh.takeoff_time is None


def test_original_A_is_unchanged(assets):
    old = CatchEnvironment(assets, CatchConfig(catch_mode='A'))
    new = AnticipatoryEnvironment(assets, CatchConfig(catch_mode='A', comparison_suite='anticipatory_v1'))
    for _ in range(6):
        a = old.step(np.full(12, .01))
        b = new.step(np.full(12, .01))
        assert np.array_equal(a[0][0], b[0][0])
        assert a[1] == b[1]
        assert new.trace[-1]['reward_rear_sweep'] == 0
        assert new.trace[-1]['reward_early_placement'] == 0


def test_rigid_rear_motion_does_not_create_relative_momentum(assets):
    e = AnticipatoryEnvironment(assets, CatchConfig(catch_mode='C', comparison_suite='anticipatory_v1'))
    e.d.qvel[:] = 0
    e.d.qvel[:6] = [.1, -.05, .2, .3, .4, -.2]
    mujoco.mj_forward(e.m, e.d)
    mujoco.mj_subtreeVel(e.m, e.d)
    relative, coordination, _, _ = e.rear_motion()
    assert abs(relative) < 1e-10 and coordination == 0


def test_new_state_resume_and_episode_reset(assets):
    e = AnticipatoryEnvironment(assets, CatchConfig(catch_mode='C', comparison_suite='anticipatory_v1'))
    for _ in range(3):
        e.step(np.zeros(12))
    e.early_paid, e.sweep_paid = .25, .15
    clone = AnticipatoryEnvironment(assets, copy.deepcopy(e.cfg))
    clone.restore(copy.deepcopy(e.state()))
    for _ in range(3):
        a, b = e.step(np.zeros(12)), clone.step(np.zeros(12))
        assert a[1] == b[1] and np.array_equal(a[0][0], b[0][0])
    e.reset(1., 99)
    assert e.early_paid == 0 and e.sweep_paid == 0 and e.first_front_motion is None


def test_B_C_same_base_objective_and_actions(assets):
    b = AnticipatoryEnvironment(assets, CatchConfig(catch_mode='B', comparison_suite='anticipatory_v1'))
    c = AnticipatoryEnvironment(assets, CatchConfig(catch_mode='C', comparison_suite='anticipatory_v1'))
    for _ in range(5):
        left, right = b.step(np.full(12, .01)), c.step(np.full(12, .01))
        assert np.array_equal(left[0][0], right[0][0])
        assert left[1] == pytest.approx(right[1] - c.trace[-1]['reward_rear_sweep'])
        assert b.cfg.catch_mode == 'B'
