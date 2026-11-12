import copy
import numpy as np
import pytest
from experiments.recoverability.transition import envelope, decomposition, snapshot, restore, SPEC
from experiments.recoverability.run import FEATURES, SCALES
from experiments.recoverability.reverse_curriculum import ReverseCurriculum
from experiments.recoverability.test_core import fixture


def test_feature_distance_decomposition():
    row=dict(zip(FEATURES,np.zeros(len(FEATURES)))); row['time']=.2
    region=[dict(cell=0,features=row.copy())]
    query=row.copy();query['com_z']=.3;query['pitch']=20
    d=decomposition(query,region)
    assert d['features'][0]['name']=='com_z'
    assert d['distance']==pytest.approx(3)
    assert not d['filter_accepts']
    assert envelope(row,region)[2]


def test_exact_state_and_controller_memory_restore(fixture):
    from experiments.recoverability.physics import World,Metrics,feasible_state
    from experiments.recoverability.controllers import BalanceAdapter
    import mujoco
    assets,cfg,c=fixture;w=World(assets,cfg)
    feasible_state(w,c.env._standing_posture()[0].numpy(),91001)
    a=BalanceAdapter(c,w);m=Metrics(w)
    for _ in range(10):
        row=w.tick(a.target());a.advance_memory(row['loads'],.002);m.observe(row,.002)
    s=snapshot(w,m,a);bw,bm,ba=restore(assets,cfg,s,c)
    np.testing.assert_array_equal(w.d.qpos,bw.d.qpos)
    np.testing.assert_array_equal(w.d.qvel,bw.d.qvel)
    np.testing.assert_array_equal(a.observation().numpy(),ba.observation().numpy())
    assert w.d.ncon==bw.d.ncon
    np.testing.assert_allclose(w.d.qacc_warmstart,bw.d.qacc_warmstart,atol=0,rtol=0)
    assert bm.drift==m.drift
    ba.target();np.testing.assert_array_equal(w.d.qpos,bw.d.qpos)


def test_reverse_gate_and_morphology_isolation():
    good=dict(success=True,hold=5.,collision=False,first_failure=None,loaded_slip=.01,
              loaded_drift=.01,max_speed=.9,max_torque=.9,max_joint=0.,scale=1.)
    cur=ReverseCurriculum();bad=dict(good,collision=True)
    assert not cur.admit('bad',{},bad)
    assert cur.admit('native',{'state':1},good)
    assert not cur.advance([dict(good,scale=.9)]*20)
    assert not cur.advance([bad]*20)
    assert cur.advance([good]*20)
    assert cur.horizon==.08
    with pytest.raises(ValueError):cur.sample(np.random.default_rng(0),.9)
    with pytest.raises(ValueError):cur.admit('orphan',{},good,predecessor='missing')


def test_region_proposal_families_are_reconstructed_without_pose_changes():
    from experiments.recoverability.transition import cell_parameters
    assert cell_parameters(dict(family='front_config',x=-.05,y=0))==dict(front_delta=-.05)
    assert cell_parameters(dict(family='com_shift',x=.02,y=0))==dict(com_shift=.02)
    assert cell_parameters(dict(family='pitch_omega',x=90,y=.1))==dict(pitch=90,omega=.1)
    with pytest.raises(ValueError):cell_parameters(dict(family='unknown',x=0,y=0))


def test_reverse_transition_rejects_slipping_edge():
    good=dict(success=True,hold=5.,collision=False,first_failure=None,loaded_slip=.01,
              loaded_drift=.01,max_speed=.9,max_torque=.9,max_joint=0.)
    cur=ReverseCurriculum();assert cur.admit('balance',{},good)
    assert not cur.admit_transition('earlier',{},good,dict(good,loaded_slip=.03),'balance')
    assert cur.admit_transition('earlier',{},good,good,'balance')
    with pytest.raises(ValueError):cur.admit_transition('scaled',{},good,good,'balance',.9)
