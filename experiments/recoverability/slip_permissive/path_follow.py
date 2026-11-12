"""Full-motion native path continuation, with free translating front feet."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.configuration_planning.reverse import bank_start
from experiments.recoverability.contact_transition.core import modify
from experiments.recoverability.slip_permissive.refine import parameters,score
from experiments.recoverability.slip_permissive.runtime import standing,handoff


def rollout(net,c,assets,cfg,warm,path,x,seed,noise=False):
    w,m,_=standing(net,assets,cfg,seed,.04);actor,_=w.observations();generator=torch.Generator().manual_seed(seed+7000000)
    actions=[];onset,ramp,front,rear,knee,thigh,feedback=x
    for step in range(90):
        if m.failed:break
        t=float(w.d.time)-.04
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor));raw=dist.loc
            if noise:raw=raw+dist.scale*torch.randn(12,generator=generator)
        action=modify(w,raw.tanh().numpy(),parameters(warm),t,'no_anchor')
        desired=w.nominal+cfg.action_scale*action
        alpha=np.clip((t-onset)/ramp,0,1)
        # Feed-forward whole-body path plus pitch feedback; no foot XY target.
        pitch=float(np.degrees(w.physical_row()['pitch']));reference=30+60*alpha
        pose=np.array([np.interp(reference,np.arange(30,91),path[:,j]) for j in range(12)])
        pose[[2,5]]+=knee;pose[[1,4]]+=thigh+feedback*np.deg2rad(pitch-reference)
        desired[:6]=(1-alpha*front)*desired[:6]+alpha*front*pose[:6]
        desired[6:]=(1-alpha*rear)*desired[6:]+alpha*rear*pose[6:]
        command=np.clip((desired-w.nominal)/cfg.action_scale,-1,1)
        (actor,_),_,_,_=w.step(command);actions.append(dict(time=float(w.d.time),action=command.tolist(),target=w.target.tolist()))
        row=w.physical_row()
        if 85<=np.degrees(row['pitch'])<=95 and row['front_support'] and not m.failed:
            r=handoff(w,m,c,seed+9000000,noise);break
    else:r=m.results()
    if m.balance_started is None:r=m.results()
    r.update(seed=seed,stochastic=noise,plan='geometric_continuation',scope='standing-start')
    return r,m.trace,m.states,actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--max-evaluations',default=60,type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);assets=Assets(a.output/'assets');c=controller()
    try:
        warm=json.loads(Path('output/slip_permissive_refine_20261112_v1/selected_plan.json').read_text())['parameters']
        nodes=json.loads(Path('output/configuration_connectivity_20261112_v1/dense_configuration_path.json').read_text())['nodes'];path=np.array([r['qpos'][7:] for r in nodes])
        region=Path('output/recoverability_maps_20261112');bank=np.load(region/'recoverable_states.npz');entries=json.loads((region/'validated_region.json').read_text())
        bw,_,_=bank_start(c,assets,cfg,entries[0],0,bank)
        # Blend the last 15 degrees into the real policy-compatible terminal shape.
        for i in range(46,61):
            alpha=(i-45)/15;path[i]=(1-alpha)*path[i]+alpha*bw.d.qpos[bw.qids]
        dump(a.output/'joint_path.json',path.tolist());history=[]
        def objective(x):
            r,_,_,_=rollout(net,c,assets,cfg,warm,path,x,202646020);value=score(r);history.append(dict(parameters=np.asarray(x).tolist(),score=value,result=r))
            if len(history)%10==0:dump(a.output/'optimization.json',history);print('path',len(history),value,r['safe_supported_peak'],flush=True)
            return value
        minimize(objective,[.52,.5,.8,.7,0.,0.,.3],method='Powell',bounds=[(.4,.75),(.25,.9),(.2,1.),(.2,1.),(-.2,.4),(-.4,.4),(0.,1.)],options=dict(maxfev=a.max_evaluations,xtol=.02,ftol=.001))
        best=min(history,key=lambda h:h['score']);dump(a.output/'optimization.json',history);dump(a.output/'selected_plan.json',best);results=[]
        for cohort,seeds in [('development',range(202646120,202646128)),('heldout',range(202646520,202646552))]:
            rows=[]
            for seed in seeds:
                for noise in (False,True):
                    r,trace,states,actions=rollout(net,c,assets,cfg,warm,path,best['parameters'],seed,noise);rows.append(r)
                    if cohort=='heldout' and len(rows)<=2:csv_write(a.output/f'heldout_{int(noise)}.csv',trace);torch.save(states,a.output/f'heldout_motion_{int(noise)}.pt');dump(a.output/f'actions_{int(noise)}.json',actions)
                dump(a.output/f'{cohort}.json',rows)
    finally:c.close()

if __name__=='__main__':main()
