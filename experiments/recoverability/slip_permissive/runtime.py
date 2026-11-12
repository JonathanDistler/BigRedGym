"""Continuous standing-to-entry-to-balance replay; both gates share dynamics."""
import numpy as np
import torch
from experiments.recoverability.physics import World
from experiments.recoverability.configuration_planning.dynamics import integration,segment
from experiments.recoverability.controllers import BalanceAdapter
from experiments.recoverability.slip_permissive.metrics import DualMetrics,clone,balance


def standing(net,assets,cfg,seed,prefix=.34,perturb=.018,stochastic=False):
    w=World(assets,cfg);actor,_=w.reset(1.,seed);m=DualMetrics(w);w.monitor=m
    m.standing_start=True
    rng=np.random.default_rng(seed);target=w.nominal+rng.uniform(-perturb,perturb,12)
    for _ in range(20):row=w.tick(target);m.observe(row,.002)
    w.row=w.physical_row();actor,_=w.observations();generator=torch.Generator().manual_seed(seed+7000000)
    while float(w.d.time)<prefix-1e-9 and not m.failed:
        with torch.no_grad():
            distribution=net.distribution(torch.tensor(actor));raw=distribution.loc
            if stochastic:raw=raw+distribution.scale*torch.randn(12,generator=generator)
        (actor,_),_,_,_=w.step(raw.tanh().numpy())
    state=integration(w);state['prefix_slip']=m.slip
    return w,m,state


def handoff(w,m,c,seed,stochastic):
    q,v,time=w.d.qpos.copy(),w.d.qvel.copy(),float(w.d.time)
    adapter=BalanceAdapter(c,w)
    # Controller memory is explicit; the kick has no CPG memory. Initial CPG
    # offsets are identical in scratch and live handoffs, previous action is
    # the actual last motor target, and neither operation reanchors the feet.
    scratch,probe,probe_adapter=clone(w,m,c,adapter)
    probe_result=balance(scratch,probe,probe_adapter,seed,stochastic)
    assert np.array_equal(q,w.d.qpos) and np.array_equal(v,w.d.qvel) and time==w.d.time
    result=balance(w,m,adapter,seed,stochastic)
    result['exact_state_probe']=dict(strict_success=probe_result['strict_success'],exploratory_success=probe_result['exploratory_success'],
        strict_hold=probe_result['strict_hold_seconds'],exploratory_hold=probe_result['exploratory_hold_seconds'])
    result['probe_live_match']=bool(result['exploratory_success']==probe_result['exploratory_success'] and
                                  abs(result['exploratory_hold_seconds']-probe_result['exploratory_hold_seconds'])<1e-8)
    return result


def evaluate(net,c,assets,cfg,plan,seed,stochastic=False):
    w,m,state=standing(net,assets,cfg,seed,plan.get('prefix',.34),.018,stochastic)
    m.phase='entry';m.active=m.takeoff_time is not None
    if not m.failed:
        # Execute each control interval without resetting a floating-base node.
        for command,duration in zip(plan['commands'],plan['durations']):
            count=round(duration/.02)
            initial=state['target'].copy()
            for j in range(count):
                desired=initial+((j+1)/count)*(np.array(command)-initial)
                state=segment(w,state,desired,.02,m)
                row=w.physical_row();pitch=np.degrees(row['pitch'])
                if m.failed:break
                if 85<=pitch<=95 and row['front_support']:
                    result=handoff(w,m,c,seed+9000000,stochastic)
                    result.update(seed=seed,stochastic=stochastic,scope='standing-start',plan=plan['name'])
                    return result,m.trace,m.states
            if m.failed:break
    result=m.results();result.update(seed=seed,stochastic=stochastic,scope='standing-start',plan=plan['name'])
    result['exact_state_probe']=None
    return result,m.trace,m.states
