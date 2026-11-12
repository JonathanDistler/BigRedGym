"""Targeted front-leg extension / launch-height hypothesis, fully dynamic."""
import argparse
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.contact_transition.core import TIMES,neutral,rollout,Limits


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8
    assets=Assets(a.output/'assets');rows=[];best=None
    for onset in (.35,.55,.75):
        for thigh in (-.30,-.15,0.):
            for calf in (.2,.4,.6):
                parameters=neutral()
                active=TIMES>=onset
                parameters[active,0]=thigh;parameters[active,1]=calf
                results=[]
                for seed in (202622120,202622121):
                    r,t,actions=rollout(net,assets,cfg,parameters,seed,'transient','load_yield',trace=True)
                    results.append(r)
                score=min(r['score'] for r in results)
                entry=dict(onset=onset,thigh=thigh,calf=calf,score=score,results=results,parameters=parameters.tolist())
                rows.append(entry)
                if best is None or score>best['score']:
                    best=entry;csv_write(a.output/'best_extension_trajectory.csv',t)
                    dump(a.output/'best_extension_actions.json',actions)
                    np.savez_compressed(a.output/'best_extension.npz',parameters=parameters,times=TIMES)
                print('extension',onset,thigh,calf,[(round(r['peak'],1),round(r['loaded_slip'],3),r['physical_safe'],(r['first_failure'] or {}).get('bodies')) for r in results],flush=True)
                dump(a.output/'extension_sweep.json',rows)

if __name__=='__main__':main()
