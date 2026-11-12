import copy,json
from pathlib import Path
import numpy as np
import mujoco
from experiments.recoverability.test_core import fixture
from experiments.recoverability.physics import World
from experiments.recoverability.configuration_planning.dynamics import integration,segment
from experiments.recoverability.slip_permissive.metrics import DualMetrics,clone
from experiments.recoverability.slip_permissive.archive import classify
from experiments.recoverability.slip_permissive.search import SlipPermissiveShooting
from experiments.recoverability.slip_permissive.run import DEVELOPMENT,HELDOUT


def prepared(fixture):
    assets,cfg,c=fixture;w=World(assets,cfg);w.reset(1.,202642000);m=DualMetrics(w);w.monitor=m
    m.head_min=m.calf_min=.02;m.begin_balance();return w,m,c


def test_slip_drift_only_relaxed(fixture):
    w,m,c=prepared(fixture);m.slip=3.;m.drift=4.;m.entry.update(slip=3.,drift=4.)
    m.strict_balance_best=m.exploratory_balance_best=5.
    g=m.gates();assert g['exploratory_success'] and not g['strict_success']
    assert not g['strict_entry'] and g['exploratory_entry']


def test_collision_and_limits_never_relaxed(fixture):
    for name,value in [('failed',True),('max_speed',1.01),('max_torque',1.01),('max_joint',.002)]:
        w,m,c=prepared(fixture);setattr(m,name,value);m.exploratory_balance_best=5.
        assert not m.gates()['exploratory_success']


def test_slip_cap_applies_entry_not_balance(fixture):
    w,m,c=prepared(fixture);m.entry.update(slip=.01,drift=.01);m.slip=.06;m.drift=.03
    m.strict_balance_best=m.exploratory_balance_best=5.
    assert m.gates()['strict_success']


def test_contact_impact_clearances_retained(fixture):
    for key,value in [('longest_gap',.081),('total_gap',.121),('impact',.76),('head',.004),('calf',.004)]:
        w,m,c=prepared(fixture);m.entry[key]=value;m.exploratory_balance_best=5.
        assert not m.gates()['exploratory_success']


def test_dual_observation_never_changes_dynamics(fixture):
    w,m,c=prepared(fixture);state=integration(w);target=w.nominal+.01
    plain=segment(w,state,target,.04)
    measured=segment(w,state,target,.04,m)
    np.testing.assert_array_equal(plain['integration'],measured['integration'])


def test_exact_clone_preserves_integration_and_memory(fixture):
    w,m,c=prepared(fixture);before=integration(w);s,n,a=clone(w,m,c)
    np.testing.assert_array_equal(integration(s)['integration'],before['integration'])
    np.testing.assert_array_equal(integration(w)['integration'],before['integration'])
    np.testing.assert_array_equal(s.target,w.target);assert n.anchor is not m.anchor
    assert n.slip==m.slip and n.drift==m.drift


def test_native_action_envelope(fixture):
    w,m,c=prepared(fixture);assets,cfg,_=fixture
    path=json.loads(Path('output/configuration_connectivity_20261112_v1/dense_configuration_path.json').read_text())['nodes']
    p=SlipPermissiveShooting(assets,cfg,integration(w),[path[0]],'sliding',[.1]);x,lo,hi=p.guess()
    assert np.all(p.command_low>=w.nominal-cfg.action_scale)
    assert np.all(p.command_high<=w.nominal+cfg.action_scale)
    assert np.all(x>=lo) and np.all(x<=hi)


def test_no_peak_only_or_collision_relabel():
    for row in [dict(peak=54,loaded_slip=.2),dict(peak=90,loaded_slip=.2,first_failure={'bodies':'Head','self_contacts':0})]:
        result=classify(row);assert not result['full_handstand_proven'];assert not result['solely_slip_drift_for_complete_handstand']


def test_fresh_matched_seed_sets():
    from experiments.recoverability.configuration_planning.plan import DEVELOPMENT as olddev,HELDOUT as oldtest
    assert len(DEVELOPMENT)*2==16 and len(HELDOUT)*2==64
    assert set(DEVELOPMENT).isdisjoint(HELDOUT)
    assert set(DEVELOPMENT+HELDOUT).isdisjoint(olddev+oldtest)


def test_success_is_achievement_prefix_not_future_credit(fixture):
    w,m,c=prepared(fixture);m.exploratory_balance_best=5.
    assert m.gates()['exploratory_success']
    m.exploratory_success_time=5.;m.failed=True
    # A physically valid five-second sequence existed before a later,
    # optional strict-observer continuation failed. Failure stays recorded.
    assert m.gates()['exploratory_success'] and not m.gates()['hard_safe']


def test_pilot_never_admits_unsafe_prefixes():
    from experiments.recoverability.slip_permissive.pilot import eligibility
    r=dict(hard_safe=False,safe_supported_peak=54.,exploratory_entry=True)
    assert not eligibility([r]*64)


def test_native_optimization_cost_does_not_depend_on_slip_or_drift():
    from experiments.recoverability.slip_permissive.refine import score,parameters
    r=dict(final=dict(pitch=45,com_support_x=-.12,omega=1.),safe_supported_peak=45.,hard_safe=True,
           head_clearance=.02,all_calf_clearance=.02,longest_contact_gap=.02,landing_impact_velocity=.2,
           loaded_slip=.01,loaded_drift=.01)
    changed=copy.deepcopy(r);changed.update(loaded_slip=100.,loaded_drift=100.)
    assert score(r)==score(changed)
    p=parameters(np.c_[np.zeros((3,4)),[.6,.8,1.1]].flatten())
    assert np.all(p[:,5]==0.) and p[:,4].min()>=.6
