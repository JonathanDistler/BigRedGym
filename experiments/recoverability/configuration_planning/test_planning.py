import numpy as np
import mujoco
import pytest
from experiments.recoverability.test_core import fixture
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem
from experiments.recoverability.configuration_planning.atlas import seeds
from experiments.recoverability.configuration_planning.dynamics import integration,segment,put,ValidationMetrics
from experiments.recoverability.physics import World
from experiments.recoverability.configuration_planning.plan import DEVELOPMENT,HELDOUT


@pytest.fixture(scope='module')
def solved(fixture):
    assets,cfg,c=fixture;p=ConfigurationProblem(assets,cfg)
    row=p.solve(90,.525,0.,(.32,0,0,0,0),seeds()[0],200)
    assert row['geometric_cell']
    return p,row


def test_static_equilibrium_and_physical_margins(solved):
    p,row=solved;test=p.static(np.array(row['qpos']))
    assert test['feasible'] and test['equilibrium']
    assert test['force_residual_N']<.05 and test['moment_residual_Nm']<.01
    assert test['torque_ratio']<1 and min(test['friction_margin_N'])>0
    assert abs(sum(test['normal_forces'])-p.w.weight)<.1
    assert row['clearances']['head']>=.0049 and row['clearances']['front_calf']>=.0049


def test_geometry_jacobian_against_finite_difference(solved):
    p,row=solved
    # Reinitialize cache because force tests mutate the scratch MjData.
    p.last_x=None;x=np.array(row['qpos'])[p.qvars];f,J=p.evaluate(x)
    for column in (0,2,3,10):
        step=x.copy();step[column]+=1e-6
        numerical=(p.evaluate(step)[0]-f)/1e-6
        np.testing.assert_allclose(numerical[:8],J[:8,column],atol=5e-4,rtol=2e-5)


def test_static_failure_is_not_a_dynamic_impossibility_claim(solved):
    p,row=solved;q=np.array(row['qpos']);q[p.w.qids[8]]-=.10
    test=p.static(q)
    assert not test['equilibrium']
    assert test['moment_residual_Nm']>.01 or test['force_residual_N']>.05
    assert 'dynamically_impossible' not in test


def test_fast_native_dynamics_matches_instrumented_motor_path(fixture):
    assets,cfg,c=fixture;w=World(assets,cfg);w.reset(1.,993001)
    start=integration(w);target=w.nominal+.01
    result=segment(w,start,target,.04)
    put(w,start)
    for tick in range(20):
        if tick%10==0:
            desired=start['target']+((tick+10)/20)*(target-start['target'])
            command=w.target+np.clip(desired-w.target,-.7*w.speed*cfg.dt,.7*w.speed*cfg.dt)
        w.tick(command)
    np.testing.assert_allclose(w.d.qpos,result['qpos'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(w.d.qvel,result['qvel'],atol=1e-12,rtol=0)


def test_landing_metric_keeps_preimpact_velocity(fixture):
    assets,cfg,c=fixture;w=World(assets,cfg);m=ValidationMetrics(w);m.active=True
    w.d.qvel[2]=-1.;mujoco.mj_forward(w.m,w.d);row=w.physical_row();row['loads'][:2]=0
    m.observe(row,.002)
    w.d.qvel[2]=0;mujoco.mj_forward(w.m,w.d);row=w.physical_row();row['loads'][:2]=w.weight/2
    m.observe(row,.002)
    assert m.impact>=1.-1e-8


def test_development_and_final_seeds_are_independent():
    assert len(DEVELOPMENT)==16 and len(HELDOUT)==64
    assert set(DEVELOPMENT).isdisjoint(HELDOUT)


def test_contact_quadrature_is_measurement_only(fixture):
    assets,cfg,c=fixture;w=World(assets,cfg);w.reset(1.,993002)
    start=integration(w);target=w.nominal+.01
    plain=segment(w,start,target,.04)
    measured=segment(w,start,target,.04,quadrature=True)
    np.testing.assert_array_equal(plain['qpos'],measured['qpos'])
    np.testing.assert_array_equal(plain['qvel'],measured['qvel'])
    assert measured['quadrature']['slip']>=0


def test_impact_detects_geometric_touch_before_loaded_threshold(fixture):
    assets,cfg,c=fixture;w=World(assets,cfg);m=ValidationMetrics(w);m.active=True
    m.previous_vertical[:]=-.9;w.d.qpos[2]-=.002;mujoco.mj_forward(w.m,w.d)
    row=w.physical_row();row['loads'][:2]=0.
    m.observe(row,.002)
    assert m.last_touching.any() and not m.last_loaded.any()
    assert m.impact>=.9
