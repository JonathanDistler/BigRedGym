"""Exact-state recovery diagnostics and bounded action-space transition search.

No physical interpolation, contact edits, or training is performed by this runner.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import (FEATURES, SCALES, KICK, feature, dump, csv_write, sha, model_audit)
from experiments.recoverability.physics import World, Metrics, feasible_state
from experiments.recoverability.controllers import SavedController, BalanceAdapter
from experiments.morphology.learning.environment import Assets

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
UNITS = ['deg','rad/s','rad/s','m','m','m/s','m/s','N','N','m'] + ['rad']*12 + ['rad/s']*12


def cell_parameters(cell):
    if cell['family'] == 'pitch_omega':
        return dict(pitch=cell['x'], omega=cell['y'])
    if cell['family'] == 'front_config':
        return dict(front_delta=cell['x'])
    if cell['family'] == 'com_shift':
        return dict(com_shift=cell['x'])
    raise ValueError('Unknown proposal family: ' + cell['family'])


def envelope(row, region):
    points = np.array([feature(r['features']) for r in region])
    f = feature(row)
    delta = abs(points-f)
    nearest = int(np.argmin(delta.max(1)))
    local = points[[r['cell'] == region[nearest]['cell'] for r in region]]
    return nearest, delta[nearest], bool(np.all(f >= local.min(0)-1e-6) and np.all(f <= local.max(0)+1e-6))


def snapshot(w, metrics, adapter=None):
    state = np.empty(mujoco.mj_stateSize(w.m, SPEC))
    mujoco.mj_getState(w.m, w.d, state, SPEC)
    memory = {k: copy.deepcopy(v) for k,v in vars(metrics).items() if k != 'w'}
    saved = dict(state=state, target=w.target.copy(), metrics=memory, scale=w.scale)
    if adapter is not None:
        saved['adapter'] = {k: copy.deepcopy(getattr(adapter,k)) for k in ('previous','phase','load','clock') if hasattr(adapter,k)}
    return saved


def restore(assets, cfg, saved, c):
    w = World(assets,cfg)
    w.reset(saved['scale'],0)
    mujoco.mj_setState(w.m,w.d,saved['state'],SPEC)
    mujoco.mj_forward(w.m,w.d)
    w.target = saved['target'].copy()
    m = Metrics(w)
    for k,v in saved['metrics'].items():
        setattr(m,k,copy.deepcopy(v))
    # A new five-second validation interval; retain all previous safety history.
    m.hold = m.best_hold = 0.0
    a = BalanceAdapter(c,w)
    for k,v in saved.get('adapter',{}).items():
        setattr(a,k,copy.deepcopy(v))
    return w,m,a


def probe(assets,cfg,c,saved,seconds=6.5):
    w,m,a = restore(assets,cfg,saved,c)
    initial_obs = a.observation().numpy().copy()
    trace=[]
    for tick in range(round(seconds/w.m.opt.timestep)):
        if tick%5 == 0:
            target=a.target()
        row=w.tick(target)
        a.advance_memory(row['loads'],w.m.opt.timestep)
        sample=m.observe(row,w.m.opt.timestep)
        if tick%10 == 0: trace.append(sample)
        if m.failed or m.best_hold >= 5: break
    return dict(**m.result(),initial_observation=initial_obs.tolist()),trace


def decomposition(row,region):
    idx,delta,member=envelope(row,region)
    ranked=np.argsort(-delta)
    return dict(time=row['time'],distance=float(delta.max()),nearest_entry=idx,filter_accepts=member,
                features=[dict(name=FEATURES[j],unit=UNITS[j],actual=row[FEATURES[j]],
                reference=region[idx]['features'][FEATURES[j]],scale=float(SCALES[j]),normalized_mismatch=float(delta[j])) for j in ranked])


def replay(net,assets,cfg,seed,stochastic,region):
    w=World(assets,cfg); actor,_=w.reset(1.0,seed)
    m=Metrics(w); m.require_takeoff=True; w.monitor=m
    rng=torch.Generator().manual_seed(seed+1000000)
    best=None; trace=[]
    for control in range(400):
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor)); raw=dist.loc
            if stochastic: raw=raw+dist.scale*torch.randn(12,generator=rng)
        (actor,_),_,done,_=w.step(raw.tanh().numpy())
        row=w.last_monitor_sample; trace.append(row)
        report=decomposition(row,region)
        if best is None or report['distance'] < best[0]['distance']:
            best=(report,snapshot(w,m))
        if done: break
    return best,trace


def optimize(assets,cfg,c,saved,region,seed,budget=32,horizon=.24):
    """CEM over symmetric hip/thigh/calf target increments, integrated at 500 Hz.

    Feasibility and frozen-policy recovery dominate pitch. Search starts from an
    exact integrated state. Resetting optimizer trials is offline branching only.
    """
    rng=np.random.default_rng(seed); mean=np.zeros((3,6)); std=np.full((3,6),.18)
    records=[]; best=None
    for iteration in range(3):
        population=[]
        for candidate in range(budget):
            actions=mean.copy() if candidate==0 else np.clip(rng.normal(mean,std),-.5,.5)
            w,m,_=restore(assets,cfg,saved,c); trace=[]
            for tick in range(round(horizon/w.m.opt.timestep)):
                knot=min(2,int(tick*w.m.opt.timestep/horizon*3))
                # FL/FR and RL/RR use paired sagittal commands; abduction signs mirrored.
                u=actions[knot]; delta=np.r_[u[:3],u[:3],u[3:],u[3:]]
                delta[[3,9]] *= -1
                target=np.clip(saved['target']+delta,w.bounds[:,0],w.bounds[:,1])
                row=w.tick(target); sample=m.observe(row,w.m.opt.timestep)
                if tick%5==0: trace.append(sample)
                if m.failed: break
            d=decomposition(sample,region)
            safe=not m.failed and m.drift<=.04 and m.slip<=.04
            # Exact frozen policy evaluation bypasses the empirical filter.
            result,_=probe(assets,cfg,c,snapshot(w,m)) if safe else ({'success':False,'hold':0},[])
            score=100*result['success']+10*result['hold']-d['distance']-30*m.slip-30*m.drift-100*(not safe)
            rec=dict(iteration=iteration,candidate=candidate,score=score,distance=d['distance'],safe=safe,
                     recovery=result,actions=actions.tolist(),**m.result())
            # Keep transition safety success distinct from balance-policy success.
            rec['balance_success']=result['success']
            records.append(rec); population.append((score,actions))
            if best is None or score>best[0]: best=(score,rec,trace,snapshot(w,m))
        elite=np.array([x[1] for x in sorted(population,key=lambda x:x[0],reverse=True)[:max(2,budget//4)]])
        mean=elite.mean(0); std=np.maximum(elite.std(0),.025)
    return records,best


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--switching',type=Path,default=Path('output/recoverability_switching_verified_20261112'))
    p.add_argument('--region-dir',type=Path,default=Path('output/recoverability_maps_20261112'))
    p.add_argument('--budget',type=int,default=8); p.add_argument('--search-starts',type=int,default=3)
    p.add_argument('--bank-only',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
    region=json.loads((a.region_dir/'validated_region.json').read_text())
    meta=json.loads((a.region_dir/'frozen_controller.json').read_text())
    net,_,cfg,_=curriculum.load_source(KICK); cfg.seconds=8.
    assets=Assets(a.output/'assets'); c=SavedController(meta['checkpoint'],'go2balance')
    try:
        audit=model_audit(c,World(assets,cfg)); dump(a.output/'compatibility.json',audit)
        assert audit['compatible'] and c.meta['sha256']==meta['sha256']
        reports=[]; starts=[]
        for trial in ([] if a.bank_only else json.loads((a.switching/'switching.json').read_text())['trials']):
            (report,saved),trace=replay(net,assets,cfg,trial['seed'],trial['stochastic'],region)
            original=list(csv.DictReader((a.switching/f"switch_{trial['seed']}_{int(trial['stochastic'])}.csv").open()))
            error=max(abs(float(x['pitch'])-y['pitch']) for x,y in zip(original,trace))
            assert len(original)==len(trace) and error<1e-7, (trial['seed'],error)
            result,cf=probe(assets,cfg,c,saved)
            label=f"{trial['seed']}_{int(trial['stochastic'])}"
            csv_write(a.output/f'kick_{label}.csv',trace); csv_write(a.output/f'probe_{label}.csv',cf)
            reports.append(dict(seed=trial['seed'],stochastic=trial['stochastic'],replay_pitch_error=error,**report,counterfactual=result))
            starts.append((report['distance'],label,saved))
            print('replay',label,report['distance'],result['success'],flush=True)
        dump(a.output/'handoff_analysis.json',reports)
        # Test every bank state with its original controller memory, then nearby
        # physical states generated by integration (never qpos interpolation).
        bank=np.load(a.region_dir/'recoverable_states.npz'); boundary=[]
        trials=json.loads((a.region_dir/'recovery_trials.json').read_text())
        cells=json.loads((a.region_dir/'recovery_cells.json').read_text())
        for i,entry in enumerate(region):
            cell=cells[entry['cell']]
            params=cell_parameters(cell)
            w=World(assets,cfg); feasible_state(w,c.env._standing_posture()[0].numpy(),entry['seed'],**params)
            anchor=w.d.geom_xpos[w.foot_geoms[:2],:2].copy()
            mujoco.mj_setState(w.m,w.d,bank[f'{i}_integration'],SPEC); mujoco.mj_forward(w.m,w.d)
            w.target=bank[f'{i}_target'].copy(); m=Metrics(w); m.anchor=anchor; m.drift=entry['features']['drift']
            adapter=BalanceAdapter(c,w); adapter.previous=torch.tensor(w.target,dtype=torch.float32)-adapter.reference
            if c.dim==59:
                adapter.phase=torch.tensor(bank[f'{i}_phase']); adapter.load=torch.tensor(bank[f'{i}_filtered_load'])
            saved=snapshot(w,m,adapter); result,_=probe(assets,cfg,c,saved)
            boundary.append(dict(entry=i,kind='bank_continuation',filter_accepts=True,result=result))
            if i%9==0:
                for direction in (-1,1):
                    bw,bm,ba=restore(assets,cfg,saved,c)
                    # 40 ms bounded motor command perturbation; preserve resulting state and memory.
                    for tick in range(20):
                        target=ba.target() if tick%5==0 else target
                        target=target.copy(); target[[1,4]]+=direction*.015
                        row=bw.tick(target); ba.advance_memory(row['loads'],bw.m.opt.timestep); sample=bm.observe(row,bw.m.opt.timestep)
                    pert=snapshot(bw,bm,ba); rr,tr=probe(assets,cfg,c,pert)
                    boundary.append(dict(entry=i,kind='integrated_perturbation',direction=direction,filter_accepts=envelope(sample,region)[2],result=rr))
                    csv_write(a.output/f'boundary_{i}_{direction}.csv',tr)
            print('bank',i,result['success'],flush=True)
        dump(a.output/'filter_validation.json',boundary)
        all_search=[]
        for _,label,saved in sorted(starts,key=lambda x:x[0])[:a.search_starts]:
            records,best=optimize(assets,cfg,c,saved,region,91000+a.search_starts,a.budget)
            dump(a.output/f'search_{label}.json',records); csv_write(a.output/f'transition_{label}.csv',best[2])
            np.savez_compressed(a.output/f'reference_{label}.npz',state=best[3]['state'],actions=np.array(best[1]['actions']))
            all_search.extend(records)
        dump(a.output/'decision.json',dict(feasible_transition=any(r['balance_success'] and r['safe'] for r in all_search),
             filter_false_negatives=sum(not r['filter_accepts'] and r['result']['success'] for r in boundary),
             rejected_tests=sum(not r['filter_accepts'] for r in boundary),bank_successes=sum(r['result']['success'] for r in boundary if r['kind']=='bank_continuation'),
             ppo_updates=0,scope='Bounded search only; no standing-to-handstand claim'))
        dump(a.output/'manifest.json',dict(kick_sha256=sha(KICK),balance_sha256=c.meta['sha256'],arguments=vars(a),region_entries=len(region),source_sha256=sha(__file__)))
    finally: c.close()

if __name__=='__main__': main()
