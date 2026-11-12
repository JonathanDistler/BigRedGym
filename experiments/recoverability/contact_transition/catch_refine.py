"""Mechanics-directed late front-leg extension/catch refinement of launch splines."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.contact_transition.core import TIMES,LOW,HIGH,rollout


def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--search-records',type=Path);group.add_argument('--source-candidate',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source=(json.loads(a.source_candidate.read_text()) if a.source_candidate is not None else
            max(json.loads(a.search_records.read_text()),key=lambda r:r['score']))
    dump(a.output/'source_candidate.json',source)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8;assets=Assets(a.output/'assets')
    base=np.array(source['parameters']);rows=[];best=None
    for onset in (.75,.95):
        for thigh in (0.,.2,.4):
            for calf in (.2,.4,.6):
                for anchor in (0.,.3):
                    parameters=base.copy();active=TIMES>=onset
                    parameters[active,0]+=thigh;parameters[active,1]+=calf
                    parameters[active,5]=anchor;parameters=np.clip(parameters,LOW,HIGH)
                    results=[]
                    for seed in (202620120,202620121):
                        r,_,_=rollout(net,assets,cfg,parameters,seed,'transient','no_yield')
                        results.append(r)
                    score=min(r['score'] for r in results)
                    entry=dict(onset=onset,thigh=thigh,calf=calf,anchor=anchor,score=score,results=results,parameters=parameters.tolist())
                    rows.append(entry)
                    if best is None or score>best['score']:
                        best=entry;np.savez_compressed(a.output/'best_catch.npz',parameters=parameters,times=TIMES)
                    print('catch',onset,thigh,calf,anchor,[(round(r['peak'],1),round(r['loaded_slip'],3),r['physical_safe'],r['accepted']) for r in results],flush=True)
                    dump(a.output/'catch_search.json',rows)
    tests=[]
    for seed in range(202623520,202623528):
        r,trace,actions=rollout(net,assets,cfg,np.array(best['parameters']),seed,'transient','no_yield',trace=True,perturbation=.018)
        tests.append(dict(seed=seed,**r));csv_write(a.output/f'catch_validation_{seed}.csv',trace)
        dump(a.output/f'catch_actions_{seed}.json',actions)
    dump(a.output/'catch_validation.json',tests)
    dump(a.output/'reference_gate.json',dict(repeatable_reference_found=sum(r['accepted'] for r in tests)>=7 and all(r['physical_safe'] for r in tests),
        tested=len(tests),accepted=sum(r['accepted'] for r in tests),ppo_updates=0,bc_updates=0,training_authorized=False))

if __name__=='__main__':main()
