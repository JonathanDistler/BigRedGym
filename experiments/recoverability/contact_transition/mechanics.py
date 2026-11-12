"""Matched physical probes of the kick policy's terminal leg configuration."""
import argparse
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.rotation import contact_wrenches
from experiments.recoverability.contact_transition.core import initialize,Limits

CASES=('baseline','hold_targets','hold_front','rear_counterrotate','rear_retract','front_fold','front_extend')


def trial(net,assets,cfg,seed,case):
    w,m,actor=initialize(assets,cfg,1.,seed,'transient',Limits(),True,.018)
    native_observe=m.observe
    def observe(row,dt):
        sample=native_observe(row,dt)
        if m.record and (m.count%5==0 or m.failed):
            force,moment,support=contact_wrenches(w,row['com'])
            s=m.trace[-1]
            for i,group in enumerate(('front','rear','nonfoot')):
                for j,axis in enumerate('xyz'):
                    s[f'{group}_world_force_{axis}']=float(force[i,j])
                    s[f'{group}_moment_com_{axis}']=float(moment[i,j])
            s['gravity_moment_front_y']=float(w.mass*9.81*(row['com'][0]-row['feet'][:2,0].mean()))
        return sample
    m.observe=observe;held=None
    for control in range(80):
        with torch.no_grad():action=net.distribution(torch.tensor(actor)).loc.tanh().numpy()
        desired=np.clip(w.nominal+w.cfg.action_scale*action,*w.bounds.T)
        if held is None and m.cross30 is not None:held=w.target.copy()
        if held is not None:
            if case=='hold_targets':desired=held.copy()
            if case=='hold_front':desired[:6]=held[:6]
            if case=='rear_counterrotate':desired[[7,10]]+=.20;desired[[8,11]]-=.25
            if case=='rear_retract':desired[[7,10]]-=.20;desired[[8,11]]+=.25
            if case=='front_fold':desired[[1,4]]-=.15;desired[[2,5]]-=.20
            if case=='front_extend':desired[[1,4]]-=.10;desired[[2,5]]+=.20
        command=np.clip((desired-w.nominal)/w.cfg.action_scale,-1,1)
        (actor,_),_,_,_=w.step(command)
        if m.failed:break
    result=m.result_transition();result['case']=case;result['seed']=seed
    return result,m.trace


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8
    assets=Assets(a.output/'assets');rows=[]
    for case in CASES:
        for seed in (202621120,202621121,202621520,202621521):
            r,t=trial(net,assets,cfg,seed,case);rows.append(r)
            csv_write(a.output/f'mechanics_{case}_{seed}.csv',t)
        print('mechanics',case,[(round(r['peak'],2),round(r['loaded_slip'],4),r['physical_safe']) for r in rows[-4:]],flush=True)
        dump(a.output/'mechanics.json',rows)

if __name__=='__main__':main()
