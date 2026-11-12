"""Deterministic low-dimensional native-action refinement, without foot anchors."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.contact_transition.core import TIMES,neutral,modify
from experiments.recoverability.slip_permissive.runtime import standing,handoff
from experiments.recoverability.slip_permissive.run import DEVELOPMENT,HELDOUT

KNOTS=np.array([.3,.7,1.3])

def parameters(x):
    x=np.asarray(x).reshape(3,-1);p=neutral()
    for j in range(4):p[:,j]=np.interp(TIMES,KNOTS,x[:,j])
    if x.shape[1]==5:p[:,4]=np.interp(TIMES,KNOTS,x[:,4])
    p[:,5]=0. # no anchoring, including in reused priors
    return p


def rollout(net,c,assets,cfg,x,seed,stochastic=False,query=True):
    w,m,_=standing(net,assets,cfg,seed,.04,.018,False);actor,_=w.observations()
    generator=torch.Generator().manual_seed(seed+7000000);p=parameters(x);actions=[]
    for control in range(80):
        if m.failed:break
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor));raw=dist.loc
            if stochastic:raw=raw+dist.scale*torch.randn(12,generator=generator)
        command=modify(w,raw.tanh().numpy(),p,float(w.d.time)-.04,'no_anchor')
        (actor,_),_,_,_=w.step(command)
        actions.append(dict(time=float(w.d.time),action=command.tolist(),target=w.target.tolist()))
        pitch=np.degrees(w.physical_row()['pitch'])
        if query and 85<=pitch<=95 and w.physical_row()['front_support'] and not m.failed:
            r=handoff(w,m,c,seed+9000000,stochastic);break
    else:r=m.results()
    if m.balance_started is None:r=m.results()
    r.update(seed=seed,stochastic=stochastic,scope='standing-start',plan='native_refinement')
    return r,m.trace,m.states,actions


def score(r):
    # Slip and drift intentionally have ZERO weight. Reward controlled body
    # progression and COM placement; collisions cannot become demonstrations.
    f=r['final'];peak=r['safe_supported_peak']
    cost=(90-min(90,peak))**2/100 + (90-np.clip(f['pitch'],0,90))**2/200
    cost+=40*(f['com_support_x']**2)+3*max(0,abs(f['omega'])-1.5)**2
    cost+=100*int(not r['hard_safe'])+200*max(0,.005-r['head_clearance'])
    cost+=200*max(0,.005-r['all_calf_clearance'])+30*max(0,r['longest_contact_gap']-.08)
    cost+=50*max(0,r['landing_impact_velocity']-.75)
    return float(cost)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--max-evaluations',type=int,default=80);p.add_argument('--warm-plan',type=Path);p.add_argument('--front-yield',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);assets=Assets(a.output/'assets');c=controller();history=[];plans=[]
    try:
        old=json.loads(Path('output/contact_transition_search_20261112_v1/search_maintained_no_yield.json').read_text())
        prior=next(r for r in old if r['iteration']==2 and r['candidate']==19)
        warm=np.array([[np.interp(t,TIMES,np.array(prior['parameters'])[:,j]) for j in range(4)] for t in KNOTS]).flatten()
        initializations=[('neutral',np.zeros(12)),('prior',warm)]
        if a.warm_plan:
            seed=np.array(json.loads(a.warm_plan.read_text())['parameters']).reshape(3,4)
            if a.front_yield:seed=np.c_[seed,np.ones(3)]
            initializations=[('warm_yield',seed.flatten())]
        for label,start in initializations:
            def objective(x):
                r,_,_,_=rollout(net,c,assets,cfg,x,202643020,query=False)
                value=score(r);history.append(dict(initialization=label,parameters=np.asarray(x).tolist(),score=value,result=r))
                if len(history)%10==0:dump(a.output/'optimization.json',history);print('refine',len(history),value,r['safe_supported_peak'],flush=True)
                return value
            result=minimize(objective,start,method='Powell',bounds=([(-.55,.55)]*4+[(.3,1.3)])*3 if a.front_yield else [(-.55,.55)]*12,options=dict(maxfev=a.max_evaluations,xtol=.02,ftol=.001))
            # Powell may end on an inferior line trial; retain best *measured* point.
            best=min((h for h in history if h['initialization']==label),key=lambda h:h['score'])
            plans.append(dict(name=label,parameters=best['parameters'],training_score=best['score'],solver_message=str(result.message)))
        dump(a.output/'optimization.json',history);dump(a.output/'plans.json',plans)
        development=[]
        for plan in plans:
            for seed in (tuple(s+2000 for s in DEVELOPMENT) if a.front_yield else DEVELOPMENT):
                for noise in (False,True):
                    r,_,_,_=rollout(net,c,assets,cfg,plan['parameters'],seed,noise);r['plan']=plan['name'];development.append(r)
            dump(a.output/'development.json',development)
        def rank(plan):
            rows=[r for r in development if r['plan']==plan['name']]
            return (sum(r['exploratory_success'] for r in rows),sum(r['hard_safe'] for r in rows),-np.mean([score(r) for r in rows]))
        selected=max(plans,key=rank);dump(a.output/'selected_plan.json',selected);heldout=[]
        for seed in (tuple(s+2000 for s in HELDOUT) if a.front_yield else HELDOUT):
            for noise in (False,True):
                r,trace,states,actions=rollout(net,c,assets,cfg,selected['parameters'],seed,noise);heldout.append(r)
                if len(heldout)<=2:csv_write(a.output/f'heldout_{int(noise)}.csv',trace);torch.save(states,a.output/f'heldout_motion_{int(noise)}.pt');dump(a.output/f'actions_{int(noise)}.json',actions)
            dump(a.output/'heldout.json',heldout)
        # A separate pilot can only consume collision-free, native-valid full
        # references with repeatable progression, not prefixes of unsafe trials.
        admitted=sum(r['hard_safe'] and r['safe_supported_peak']>=45 and r['exploratory_entry'] for r in heldout)
        dump(a.output/'training_gate.json',dict(admissible_references=admitted,total=len(heldout),smoke_ready=admitted>=48,
             smoke_updates=2,pilot_updates=10,pilot_requires_smoke_pass=True,extended_ppo=False,updates_run=0))
    finally:c.close()

if __name__=='__main__':main()
