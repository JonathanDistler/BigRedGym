"""CEM full-dynamics launch and continuation search. This runner never trains PPO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK,dump,csv_write,sha
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.bidirectional.verify_models import audit_models
from experiments.recoverability.contact_transition.core import (Limits,TIMES,LOW,HIGH,NEUTRAL,neutral,rollout)

TRAIN=(202620120,202620121)
TEST=tuple(range(202620520,202620528))


def candidate_priors():
    base=neutral();priors=[base]
    for yield_gain in (.45,.7,1.):
        for rear_time in (.45,.65,.85):
            p=base.copy();p[1:,4]=yield_gain
            p[TIMES>=rear_time,2]=-.15;p[TIMES>=rear_time,3]=.25
            priors.append(p)
    for front_thigh,front_calf in ((-.2,.2),(.2,-.2),(-.35,.35),(.35,-.35)):
        p=base.copy();p[2:,0]=front_thigh;p[2:,1]=front_calf;p[2:,2]=-.15;p[2:,3]=.25;priors.append(p)
    return priors


def search(net,assets,cfg,out,mode,strategy,population,iterations,limits):
    rng=np.random.default_rng(202620000+(mode=='transient')*200+sum(map(ord,strategy)))
    mean=neutral();std=np.tile(np.array([.16,.18,.18,.2,.16,.15,.025]),(len(TIMES),1))
    records=[];best=None;priors=candidate_priors()
    for iteration in range(iterations):
        samples=[]
        for candidate in range(population):
            if iteration==0 and candidate<len(priors):p=priors[candidate].copy()
            elif candidate==0:p=mean.copy()
            elif candidate==1 and best is not None:p=best[1].copy()
            else:
                # Correlated neighboring knots reduce violent target changes.
                noise=rng.normal(size=mean.shape)
                noise[1:]=.65*noise[1:]+.35*noise[:-1]
                p=np.clip(mean+std*noise,LOW,HIGH)
            rs=[]
            for seed in TRAIN:
                r,_,_=rollout(net,assets,cfg,p,seed,mode,strategy,limits=limits)
                rs.append(r)
            # Worst-case training score discourages single-reset exploitation.
            score=min(r['score'] for r in rs)
            rec=dict(iteration=iteration,candidate=candidate,mode=mode,strategy=strategy,score=score,
                     accepted=sum(r['accepted'] for r in rs),results=rs,parameters=p.tolist())
            records.append(rec);samples.append((score,p))
            if best is None or score>best[0]:best=(score,p.copy(),rec)
        elites=np.array([v[1] for v in sorted(samples,key=lambda t:t[0],reverse=True)[:max(3,population//5)]])
        mean=.2*mean+.8*elites.mean(0)
        std=np.maximum(.25*std+.75*elites.std(0),[.025,.025,.03,.03,.03,.025,.008])
        label=f'{mode}_{strategy}'
        dump(out/f'search_{label}.json',records)
        np.savez_compressed(out/f'best_{label}.npz',parameters=best[1],times=TIMES)
        print('CEM',label,iteration,'score',best[0],'pitch',[r['progress_peak'] for r in best[2]['results']],
              'slip',[r['loaded_slip'] for r in best[2]['results']],
              'safe',[r['physical_safe'] for r in best[2]['results']],flush=True)
    return best


def export_rollout(out,label,net,assets,cfg,p,seed,mode,strategy,scale=1.,stochastic=False,limits=None,perturbation=.018):
    result,trace,actions=rollout(net,assets,cfg,p,seed,mode,strategy,scale,stochastic,limits,True,perturbation)
    csv_write(out/f'{label}_trajectory.csv',trace)
    dump(out/f'{label}_actions.json',actions)
    csv_write(out/f'{label}_actions.csv',[dict(control=r['control'],time=r['time'],
        **{f'action{j}':r['action'][j] for j in range(12)},**{f'target{j}':r['target'][j] for j in range(12)}) for r in actions])
    return result


def validation(net,assets,cfg,out,bests,limits):
    rows=[]
    for (mode,strategy),best in bests.items():
        p=best[1]
        for seed in TEST:
            for stochastic in (False,True):
                label=f'validation_{mode}_{strategy}_{seed}_{int(stochastic)}'
                r=export_rollout(out,label,net,assets,cfg,p,seed,mode,strategy,stochastic=stochastic,limits=limits)
                rows.append(dict(r,mode=mode,strategy=strategy,seed=seed,stochastic=stochastic,scale=1.))
        print('validation',mode,strategy,sum(r['accepted'] for r in rows if r['mode']==mode and r['strategy']==strategy),flush=True)
        dump(out/'validation.json',rows)
    return rows


def ablations(net,assets,cfg,out,bests,limits):
    rows=[]
    # The same selected parameters and fresh reset cohort distinguish mechanisms.
    for mode in ('maintained','transient'):
        eligible=[(key,best) for key,best in bests.items() if key[0]==mode]
        key,best=max(eligible,key=lambda item:item[1][0]);p=best[1]
        for ablation in ('baseline','full','launch_only','continuation_only','front_only','rear_only','no_yield','no_anchor'):
            modified=p.copy();strategy=key[1]
            if ablation=='baseline':modified=neutral()
            if ablation=='launch_only':modified[TIMES>.65]=NEUTRAL
            if ablation=='continuation_only':modified[TIMES<.65]=NEUTRAL
            if ablation=='front_only':modified[:,2:4]=0.
            if ablation=='rear_only':modified[:,:2]=0.;modified[:,4]=1.;modified[:,5]=0.
            if ablation in ('no_yield','no_anchor'):strategy=ablation
            for seed in TEST[:4]:
                label=f'ablation_{mode}_{ablation}_{seed}'
                r=export_rollout(out,label,net,assets,cfg,modified,seed,mode,strategy,limits=limits)
                rows.append(dict(r,ablation=ablation,mode=mode,seed=seed))
            dump(out/'ablations.json',rows)
    return rows


def morphology(net,assets,cfg,out,bests,limits):
    rows=[]
    for mode in ('maintained','transient'):
        key,best=max([(k,v) for k,v in bests.items() if k[0]==mode],key=lambda item:item[1][0])
        for scale in (.8,.85,.9,.95,1.):
            for method,p in (('baseline',neutral()),('optimized',best[1])):
                for seed in TEST[:4]:
                    r=export_rollout(out,f'morphology_{mode}_{scale}_{method}_{seed}',net,assets,cfg,p,seed,mode,key[1],scale,limits=limits)
                    rows.append(dict(r,mode=mode,scale=scale,method=method,seed=seed))
            print('morphology',mode,scale,flush=True)
            dump(out/'morphology.json',rows)
    return rows


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--population',type=int,default=20);p.add_argument('--iterations',type=int,default=5)
    p.add_argument('--strategies',nargs='+',choices=('load_yield','constant_yield','no_yield'),default=['load_yield','no_yield'])
    p.add_argument('--diagnose-only',action='store_true')
    p.add_argument('--search-dir',type=Path,help='Reuse completed CEM splines; write fresh validation outputs')
    p.add_argument('--validation-only',action='store_true',help='Validate frozen splines without repeating mechanism/morphology comparisons')
    a=p.parse_args()
    if a.population<4 or a.iterations<1:raise ValueError('Require population >=4 and iterations >=1')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8.
    assets=Assets(a.output/'assets');limits=Limits()
    model_audit=audit_models(assets,cfg);dump(a.output/'model_verification.json',model_audit)
    assert all(r['verified'] for r in model_audit)
    sources={str(f):sha(f) for f in Path('experiments/recoverability/contact_transition').glob('*.py')}
    for source in sources:
        target=a.output/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    dump(a.output/'manifest.json',dict(arguments=vars(a),limits=asdict(limits),training_seeds=TRAIN,test_seeds=TEST,
        kick_sha256=sha(KICK),source_sha256=sources,ppo_updates=0,bc_updates=0,
        morphology_schedule_enabled=False,physics='Full MuJoCo; original motors, friction and all collisions'))
    baselines=[]
    for seed in TRAIN+TEST[:2]:
        for stochastic in (False,True):
            r=export_rollout(a.output,f'baseline_{seed}_{int(stochastic)}',net,assets,cfg,neutral(),seed,'maintained','load_yield',stochastic=stochastic,limits=limits,perturbation=.012 if seed in TRAIN else .018)
            baselines.append(dict(seed=seed,stochastic=stochastic,**r))
    dump(a.output/'baselines.json',baselines)
    if a.diagnose_only:return
    bests={}
    for mode in ('maintained','transient'):
        for strategy in a.strategies:
            if a.search_dir is None:
                bests[(mode,strategy)]=search(net,assets,cfg,a.output,mode,strategy,a.population,a.iterations,limits)
            else:
                parameters=np.load(a.search_dir/f'best_{mode}_{strategy}.npz')['parameters']
                records=json.loads((a.search_dir/f'search_{mode}_{strategy}.json').read_text())
                best=max(records,key=lambda r:r['score'])
                bests[(mode,strategy)]=(best['score'],parameters,best)
    rows=validation(net,assets,cfg,a.output,bests,limits)
    if not a.validation_only:
        ablations(net,assets,cfg,a.output,bests,limits);morphology(net,assets,cfg,a.output,bests,limits)
    accepted=[]
    for key in bests:
        cohort=[r for r in rows if (r['mode'],r['strategy'])==key and not r['stochastic']]
        # No physical-failure exceptions on the eight independent perturbed resets.
        repeatable=sum(r['accepted'] for r in cohort)>=7 and all(r['physical_safe'] for r in cohort)
        if repeatable:accepted.append(dict(mode=key[0],strategy=key[1],accepted=sum(r['accepted'] for r in cohort),n=len(cohort)))
    gate=dict(repeatable_reference_found=bool(accepted),references=accepted,ppo_enabled=False,bc_enabled=False,
              training_authorized=False,ppo_updates=0,bc_updates=0,full_handstand_validated=False,
              acceptance_limits=asdict(limits),morphology_schedule_enabled=False)
    dump(a.output/'reference_gate.json',gate)
    if accepted:
        dump(a.output/'gated_learning_proposal.json',dict(status='prepared only; no training authorized',
            reference=accepted,bc_smoke_updates=2,ppo_smoke_updates=2,pilot_updates=10,
            prerequisites=['independent replay of reference gate','BC held-out tracking without physical regression',
                           'separate authorization before PPO','original morphology final evaluation'],
            extended_ppo_enabled=False))
    assert sha(KICK)==json.loads((a.output/'manifest.json').read_text())['kick_sha256']

if __name__=='__main__':main()
