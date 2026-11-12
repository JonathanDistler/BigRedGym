"""Frozen balance-basin mapping and contact-aware forward reachability validation."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import mujoco
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK, dump, csv_write, sha, model_audit, wilson
from experiments.recoverability.controllers import SavedController
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.bidirectional.core import (BANDS,CONDITIONS,World,ContactMetrics,
    prepare,compact_snapshot,restore_contact,continuation,physical_hash)

REGION=Path('output/recoverability_maps_20261112')
METHODS=('baseline','anchor','yield','rear_early','rear_late','anchor_rear','yield_rear')


def controller():
    meta=json.loads((REGION/'frozen_controller.json').read_text())
    c=SavedController(meta['checkpoint'],'go2balance')
    assert c.meta['sha256']==meta['sha256']
    return c


def basin(c,assets,cfg,out,seeds_per_split):
    rows=[]; bank={}; unique={}; successful=set()
    for band in BANDS:
        for split,base in (('train',202611120),('heldout',202611520)):
            for index in range(seeds_per_split):
                seed=base+index
                for condition in CONDITIONS:
                    w,m,a,reset,saved=prepare(c,assets,cfg,band,condition,seed,split)
                    state_id=reset.get('state_id')
                    if saved is not None:
                        unique.setdefault(state_id,len(unique));bank.setdefault(state_id,saved)
                    for stochastic in (False,True):
                        if saved is not None:
                            bw,bm,ba=restore_contact(c,assets,cfg,saved)
                            result=continuation(c,bw,bm,ba,stochastic,seed+30000)
                            if result['success'] and not stochastic:successful.add(state_id)
                            # Keep every condition's first seed for transparent diagnostics.
                            if index==0:csv_write(out/f'basin_{band}_{split}_{condition}_{int(stochastic)}.csv',bm.trace)
                        else:result=dict(m.result(),success=False,hold=0.,reset_rejected=True)
                        rows.append(dict(band=band,split=split,seed=seed,condition=condition,
                                         stochastic=stochastic,reset=reset,**result))
                    print('basin',band,split,seed,condition,[r['success'] for r in rows[-2:]],flush=True)
                    dump(out/'basin_trials.json',rows)
    torch.save(bank,out/'start_states.pt')
    torch.save({k:v for k,v in bank.items() if k in successful},out/'recoverable_states_unique.pt')
    groups=[]
    for band in BANDS:
        for split in ('train','heldout'):
            for stochastic in (False,True):
                for condition in ('all',)+CONDITIONS:
                    selected=[r for r in rows if r['band']==band and r['split']==split and r['stochastic']==stochastic and (condition=='all' or r['condition']==condition)]
                    n=len(selected);valid=sum(r['reset']['valid'] for r in selected);k=sum(r['success'] for r in selected)
                    groups.append(dict(band=band,split=split,stochastic=stochastic,condition=condition,n=n,valid=valid,successes=k,
                                       unconditional_probability=k/max(n,1),conditional_probability=k/max(valid,1),wilson95=wilson(k,n)))
    dump(out/'recovery_probability_maps.json',groups)
    nominal=[r for r in rows if r['band']==90 and not r['stochastic'] and r['condition']=='nominal']
    lower=[r for r in rows if r['band']<90 and not r['stochastic']]
    gate=dict(controller_compatible=True,valid_reset_proposals=all(r['reset']['valid'] for r in nominal),
              nominal_successes=sum(r['success'] for r in nominal),nominal_tests=len(nominal),
              lower_band_successes=sum(r['success'] for r in lower),lower_band_tests=len(lower),
              unique_valid_states=len(bank),unique_recoverable_states=len(successful),
              fine_tuning_eligible=len(nominal)>=4 and all(r['reset']['valid'] for r in nominal) and sum(r['success'] for r in nominal)/len(nominal)>=.8,
              transition_ppo_eligible=False,ppo_updates=0)
    dump(out/'validation_gate.json',gate)
    return rows,bank


def modify_action(w,action,method,time):
    if method=='baseline':return action
    desired=np.clip(w.nominal+w.cfg.action_scale*action,*w.bounds.T)
    if method.startswith('anchor'):
        # Velocity/position feedback at planted front feet; only motor targets change.
        front=np.arange(6);jac=[]
        error=w.anchor-w.d.geom_xpos[w.foot_geoms[:2],:2]
        for i,g in enumerate(w.foot_geoms[:2]):
            jp=np.zeros((3,w.m.nv));mujoco.mj_jacGeom(w.m,w.d,jp,None,int(g))
            jac.append(jp[:2,w.vids[front]])
        change=np.linalg.lstsq(np.vstack(jac),error.flatten(),rcond=1e-5)[0]
        desired[:6]=w.d.qpos[w.qids[:6]]+np.clip(change,-.12,.12)
    elif method.startswith('yield'):
        q=w.d.qpos[w.qids[:6]]
        desired[:6]=q+.4*(desired[:6]-q)
    rear=('rear' in method)
    threshold=.58 if method=='rear_early' else .78
    if rear and time>=threshold:
        desired[[7,10]]-=.12;desired[[8,11]]+=.20
    return np.clip((desired-w.nominal)/w.cfg.action_scale,-1,1)


def forward_trial(net,c,assets,cfg,seed,stochastic,method,scale,out,label,save_trace=True):
    w=World(assets,cfg);actor,_=w.reset(scale,seed)
    m=ContactMetrics(w);m.require_takeoff=True;w.monitor=m
    generator=torch.Generator().manual_seed(seed+70000)
    states=[];closest=None;control_steps=0
    for control in range(125): # identical 2.5 s maximum interaction cap across methods/scales
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor));raw=dist.loc
            if stochastic:raw=raw+dist.scale*torch.randn(12,generator=generator)
        action=modify_action(w,raw.tanh().numpy(),method,float(w.d.time))
        (actor,_),_,done,_=w.step(action);control_steps+=1
        sample=w.last_monitor_sample
        if not m.failed and m.slip<=.02 and m.drift<=.04 and w.physical_row()['front_support']:
            if closest is None or sample['pitch']>closest[0]['pitch']:
                closest=(sample.copy(),compact_snapshot(w,m))
        if control%5==0 and not m.failed:states.append(sample.copy())
        if m.failed or done:break
    if save_trace:csv_write(out/f'forward_{label}.csv',m.trace)
    result=dict(seed=seed,stochastic=stochastic,method=method,scale=scale,
                takeoff=bool(w.tracker.confirmed),control_steps=control_steps,**m.result())
    if closest is not None:
        result['candidate']=closest[0];result['candidate_state_id']=None
    return result,closest,states


def forward(net,c,assets,cfg,out,seeds_per_split,include_morphology=True):
    rows=[]; candidates={};regions=[]
    for split,base in (('train',202612120),('heldout',202612520)):
        for i in range(seeds_per_split):
            seed=base+i
            for stochastic in (False,True):
                for method in METHODS:
                    label=f'{split}_{seed}_{int(stochastic)}_{method}_1.0'
                    result,closest,states=forward_trial(net,c,assets,cfg,seed,stochastic,method,1.,out,label)
                    result['split']=split;rows.append(result)
                    if closest is not None:candidates[label]=dict(snapshot=closest[1],features=closest[0],split=split,scale=1.,method=method,seed=seed,stochastic=stochastic)
                    regions.extend(dict(split=split,method=method,scale=1.,seed=seed,stochastic=stochastic,**s) for s in states)
                    print('forward',label,result['peak'],result['loaded_slip'],result['collision'],flush=True)
                    dump(out/'forward_trials.json',rows)
    if include_morphology:
        # Match the baseline/yield intervention budget for every model, including 1.00 above.
        for scale in (.8,.85,.9,.95):
            for split,base in (('train',202612120),('heldout',202612520)):
                for i in range(seeds_per_split):
                    for stochastic in (False,True):
                        for method in ('baseline','yield'):
                            seed=base+i;label=f'{split}_{seed}_{int(stochastic)}_{method}_{scale}'
                            result,closest,states=forward_trial(net,c,assets,cfg,seed,stochastic,method,scale,out,label,save_trace=i==0)
                            result['split']=split;rows.append(result)
                            print('morph forward',label,result['peak'],result['loaded_slip'],flush=True)
                            dump(out/'forward_trials.json',rows)
    torch.save(candidates,out/'forward_candidates.pt');dump(out/'forward_region.json',regions)
    return rows,candidates


def morphology(c,assets,cfg,out,seeds_per_split):
    rows=[];audit=[]
    for scale in (.8,.85,.9,.95,1.):
        w=World(assets,cfg);w.reset(scale,0)
        m=w.m
        audit.append(dict(scale=scale,mass=float(m.body_mass.sum()),inertia_min=float(m.body_inertia[1:].min()),
                          inertia_positive=bool((m.body_inertia[1:]>0).all()),
                          geom_count=m.ngeom,body_count=m.nbody,joint_limits=w.bounds.tolist(),
                          actuator_effort=w.effort.tolist(),actuator_speed=w.speed.tolist(),
                          model_body_pos=m.body_pos.tolist(),model_body_mass=m.body_mass.tolist(),
                          model_body_inertia=m.body_inertia.tolist(),geom_size=m.geom_size.tolist(),
                          standing_com=m.body_mass.sum() and w.d.subtree_com[w.base].tolist()))
        for split,base in (('train',202611120),('heldout',202611520)):
            for i in range(seeds_per_split):
                for band in (90,85):
                    for stochastic in (False,True):
                        bw,bm,ba,reset,saved=prepare(c,assets,cfg,band,'nominal',base+i,split,scale)
                        result=continuation(c,bw,bm,ba,stochastic,base+i+30000) if saved is not None else bm.result()
                        rows.append(dict(scale=scale,split=split,seed=base+i,band=band,stochastic=stochastic,reset=reset,**result))
                        print('morph recovery',scale,band,split,stochastic,result['success'],flush=True)
    dump(out/'morphology_model_audit.json',audit);dump(out/'morphology_recovery.json',rows)
    return rows


def bridge(c,assets,cfg,out,candidates,basin_rows,bank):
    tested=[]
    for label,entry in candidates.items():
        w,m,a=restore_contact(c,assets,cfg,entry['snapshot'])
        result=continuation(c,w,m,a,False,entry['seed']+90000)
        tested.append(dict(label=label,**{k:v for k,v in entry.items() if k!='snapshot'},recovery=result))
        print('counterfactual',label,result['success'],flush=True)
    dump(out/'handoff_candidates.json',tested)
    overlap=[r for r in tested if r['recovery']['success'] and r['recovery']['first_failure'] is None and r['recovery']['loaded_slip']<=.02 and r['recovery']['loaded_drift']<=.04]
    dump(out/'bridge_gate.json',dict(overlap=bool(overlap),validated_handoff_labels=[r['label'] for r in overlap],
                                   tested=len(tested),transition_ppo_eligible=bool(overlap),ppo_updates=0))
    return tested


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('basin','forward','all'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds-per-split',type=int,default=2)
    a=p.parse_args()
    if a.seeds_per_split<2:raise ValueError('At least two independent seeds per split')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8.
    assets=Assets(a.output/'assets');c=controller()
    try:
        audit=model_audit(c,World(assets,cfg));dump(a.output/'compatibility.json',audit)
        assert audit['compatible']
        source={str(p):sha(p) for p in Path('experiments/recoverability').rglob('*.py')}
        for path in source:
            target=a.output/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        dump(a.output/'manifest.json',dict(arguments=vars(a),source_sha256=source,kick_sha256=sha(KICK),
                balance_sha256=c.meta['sha256'],torch_version=torch.__version__,mujoco_version=mujoco.__version__,
                numpy_version=np.__version__,ppo_updates=0,reset_claim='local IK proposals, action-perturbed; not standing reachable'))
        rows,bank=basin(c,assets,cfg,a.output,a.seeds_per_split) if a.mode in ('basin','all') else ([],{})
        if a.mode in ('forward','all'):
            forwards,candidates=forward(net,c,assets,cfg,a.output,a.seeds_per_split)
            bridge(c,assets,cfg,a.output,candidates,rows,bank)
            morphology(c,assets,cfg,a.output,a.seeds_per_split)
    finally:c.close()

if __name__=='__main__':main()
