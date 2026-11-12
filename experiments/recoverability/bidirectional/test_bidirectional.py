import json
import numpy as np
import pytest
import torch
import mujoco
from experiments.recoverability.test_core import fixture
from experiments.recoverability.bidirectional.core import prepare,telemetry,physical_hash,restore_contact
from experiments.recoverability.bidirectional.fine_tune import BalanceLearner,AdaptiveBands,verified_gate,reward


def test_action_integrated_reset_is_reproducible_and_exact(fixture):
    assets,cfg,c=fixture
    w,m,a,meta,saved=prepare(c,assets,cfg,85,'com_plus',991001,'heldout')
    assert meta['valid'] and not meta['standing_reachable']
    w2,m2,a2,meta2,saved2=prepare(c,assets,cfg,85,'com_plus',991001,'heldout')
    assert physical_hash(w)==physical_hash(w2)
    bw,bm,ba=restore_contact(c,assets,cfg,saved)
    np.testing.assert_array_equal(w.d.qpos,bw.d.qpos)
    np.testing.assert_array_equal(w.d.qvel,bw.d.qvel)
    np.testing.assert_array_equal(a.observation(),ba.observation())
    assert w.d.ncon==bw.d.ncon
    assert bm.slip==m.slip and bm.failed==m.failed


def test_angular_momentum_matches_mujoco_subtree(fixture):
    assets,cfg,c=fixture
    w,m,a,meta,saved=prepare(c,assets,cfg,90,'pitch_plus',991002,'train')
    values=telemetry(w);mujoco.mj_subtreeVel(w.m,w.d)
    assert values['pitch_angular_momentum']==pytest.approx(w.d.subtree_angmom[w.base,1],abs=2e-7)
    assert values['FL_normal']>=0 and values['FR_normal']>=0
    assert values['FL_friction_utilization']>=0


def test_clone_actor_parity_baseline_frozen_and_normalizer_frozen(fixture):
    assets,cfg,c=fixture
    w,m,a,meta,saved=prepare(c,assets,cfg,90,'nominal',991003,'train')
    learner=BalanceLearner(c);obs=a.observation()
    torch.testing.assert_close(learner.actor.act_inference(obs),c.actor.act_inference(obs))
    before={k:v.clone() for k,v in learner.actor.state_dict().items() if 'obs_rms' in k}
    loss=learner.distribution(obs).loc.square().mean();loss.backward()
    assert any(p.grad is not None for p in learner.actor.NN.parameters())
    assert c.network_hash()==c.digest
    for k,v in before.items():torch.testing.assert_close(learner.actor.state_dict()[k],v)
    assert not learner.actor.training


def test_no_pilot_without_verified_validation(tmp_path):
    path=tmp_path/'gate.json';path.write_text(json.dumps(dict(fine_tuning_eligible=False)))
    with pytest.raises(ValueError):verified_gate(path)


def test_curriculum_requires_independent_safe_validation():
    row=dict(success=True,collision=False,max_speed=.5,max_torque=.8,max_joint=0.,loaded_slip=.01,seed=1)
    cur=AdaptiveBands();assert not cur.observe([row]*20)
    cur=AdaptiveBands();assert cur.observe([dict(row,seed=i) for i in range(20)]);assert cur.stage==1
    assert not cur.observe([dict(row,seed=100+i,collision=True) for i in range(20)])
    fixed=AdaptiveBands(False);assert not fixed.observe([dict(row,seed=i) for i in range(20)])


def test_unloading_is_not_automatic_failure_penalty():
    sample=dict(omega=0.,com_support_x=0.,tilt=0.,slip_speed=0.,torque_ratio=0.)
    assert reward(sample,dict(front_support=False),.01,False)>-1
    assert reward(sample,dict(front_support=False),.01,True)<-9


def test_action_interventions_do_not_edit_physical_state(fixture):
    from experiments.recoverability.bidirectional.run import modify_action
    assets,cfg,c=fixture
    w,m,a,meta,saved=prepare(c,assets,cfg,90,'nominal',991004,'train')
    q=w.d.qpos.copy();v=w.d.qvel.copy();action=np.zeros(12)
    np.testing.assert_array_equal(modify_action(w,action,'baseline',.5),action)
    for method in ('anchor','yield','rear_early','rear_late','anchor_rear','yield_rear'):
        result=modify_action(w,action,method,.8)
        assert result.shape==(12,) and np.isfinite(result).all() and (abs(result)<=1).all()
        np.testing.assert_array_equal(w.d.qpos,q);np.testing.assert_array_equal(w.d.qvel,v)


def test_physical_morphology_audit(fixture):
    from experiments.recoverability.bidirectional.verify_models import audit_models
    assets,cfg,c=fixture
    records=audit_models(assets,cfg)
    assert len(records)==5 and all(r['verified'] for r in records)
    assert records[0]['mass']<records[-1]['mass']
