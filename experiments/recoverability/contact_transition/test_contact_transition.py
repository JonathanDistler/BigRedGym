import numpy as np
import pytest
from experiments.recoverability.test_core import fixture
from experiments.recoverability.contact_transition.core import (Limits,TIMES,LOW,HIGH,NEUTRAL,
    Monitor,neutral,commands_at,initialize,modify,rollout)
from experiments.recoverability.contact_transition.run import TRAIN,TEST
import curriculum
from experiments.recoverability.run import KICK


def test_action_spline_is_bounded_continuous_and_validated():
    p=neutral();p[2:,0]=100
    np.testing.assert_array_equal(commands_at(p,0.),NEUTRAL)
    assert np.all(commands_at(p,.6)<=HIGH) and np.all(commands_at(p,.6)>=LOW)
    assert abs(commands_at(p,.55+1e-7)[0]-commands_at(p,.55-1e-7)[0])<1e-5
    with pytest.raises(ValueError):commands_at(np.zeros((2,2)),.5)
    p[0,0]=np.nan
    with pytest.raises(ValueError):commands_at(p,.5)


def test_contact_commands_never_edit_physical_state(fixture):
    assets,cfg,c=fixture;w,m,actor=initialize(assets,cfg,1.,992001,'maintained',Limits())
    q=w.d.qpos.copy();v=w.d.qvel.copy();t=w.d.time
    p=neutral();p[:,5]=.5;p[:,4]=.6;p[:,2]=-.2
    for strategy in ('load_yield','constant_yield','no_yield','no_anchor'):
        action=modify(w,np.zeros(12),p,.65,strategy)
        assert action.shape==(12,) and np.isfinite(action).all() and (abs(action)<=1).all()
        np.testing.assert_array_equal(w.d.qpos,q);np.testing.assert_array_equal(w.d.qvel,v)
        assert w.d.time==t


def synthetic_gate(fixture,mode):
    assets,cfg,c=fixture;w,m,actor=initialize(assets,cfg,1.,992002,mode,Limits())
    w.tracker.confirmed=True;m.failed=False;m.first_failure=None
    m.goal_best=.12;m.cross30=dict(com_support_x=-.19);m.slip=.01;m.drift=.01
    m.min_transition_calf=.01;m.max_requested=1.;m.final=dict(pitch=50.,front_support=1,omega=.2)
    return m


def test_reference_gate_rejects_peak_only_and_unsafe_sequences(fixture):
    m=synthetic_gate(fixture,'maintained');assert m.result_transition()['accepted']
    m.goal_best=0.;m.progress_peak=55;assert not m.result_transition()['accepted']
    m.goal_best=.12;m.failed=True;assert not m.result_transition()['accepted']
    m.failed=False;m.slip=.021;assert not m.result_transition()['accepted']
    m.slip=.01;m.max_requested=1.6;assert not m.result_transition()['accepted']
    m.max_requested=1.;m.final['pitch']=34;assert not m.result_transition()['accepted']


def test_transient_unloading_requires_small_gap_and_reacquisition(fixture):
    m=synthetic_gate(fixture,'maintained');m.worst_gap=.02
    assert not m.result_transition()['accepted']
    m.mode='transient';m.total_gap=.04;m.max_touchdown=.4
    assert m.result_transition()['accepted']
    m.final['front_support']=0;assert not m.result_transition()['accepted']
    m.final['front_support']=1;m.worst_gap=.09;assert not m.result_transition()['accepted']


def test_frozen_policy_reset_replay_is_exact_and_seed_sets_disjoint(fixture):
    assets,cfg,c=fixture
    net,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8
    r1,t1,a1=rollout(net,assets,cfg,neutral(),992003,limits=Limits(horizon=.12),trace=True)
    r2,t2,a2=rollout(net,assets,cfg,neutral(),992003,limits=Limits(horizon=.12),trace=True)
    assert r1==r2 and t1==t2 and a1==a2
    assert set(TRAIN).isdisjoint(TEST)
    assert all('observation' in action and len(action['observation'])==44 for action in a1)


def test_unloading_is_counted_from_takeoff_before_thirty_degrees(fixture):
    assets,cfg,c=fixture;w,m,actor=initialize(assets,cfg,1.,992004,'transient',Limits(),True)
    w.tracker.confirmed=True
    row=w.physical_row();row['pitch']=np.deg2rad(20);row['front_support']=False
    for _ in range(10):m.observe(row,.002)
    assert m.cross30 is None
    assert m.worst_gap==pytest.approx(.02) and m.total_gap==pytest.approx(.02)
