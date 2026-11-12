"""Deterministic duration/defect homotopy for the atlas-justified launch branch."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.configuration_planning.shooting import MultipleShooting
from experiments.recoverability.configuration_planning.plan import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=16);p.add_argument('--path-constraints',action='store_true')
    p.add_argument('--exact-start',type=Path);p.add_argument('--seed-base',type=int,default=202631120)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8
    assets=Assets(a.output/'assets');source=json.loads((a.source/'plan_D.json').read_text())
    start=torch.load(a.exact_start or a.source/'exact_start_D.pt',weights_only=False);current=source
    start['prefix_slip']=.006
    for index,weight in enumerate((100.,300.) if a.path_constraints else (30.,100.)):
        planner=MultipleShooting(assets,cfg,start,source['waypoints'],'D',[.34,.30,.30]);planner.defect_weight=weight
        planner.path_constraints=a.path_constraints
        current=planner.solve(a.max_nfev,current);current['waypoints']=source['waypoints'];current['defect_weight']=weight
        dump(a.output/f'refinement_{index}.json',current)
        print('refinement',weight,current['final_cost'],current['defects'],flush=True)
    dump(a.output/'plan_D.json',current)
    c=controller()
    try:
        development=[]
        for seed in range(a.seed_base,a.seed_base+16):
            row,trace,states=evaluate(net,c,assets,cfg,current,seed)
            development.append(row);csv_write(a.output/f'development_{seed}.csv',trace)
            if seed==a.seed_base:torch.save(states,a.output/'motion_D.pt')
        dump(a.output/'development.json',development)
        heldout=[]
        for seed in range(a.seed_base+400,a.seed_base+464):
            row,trace,states=evaluate(net,c,assets,cfg,current,seed)
            heldout.append(row)
            if seed<a.seed_base+404:csv_write(a.output/f'heldout_{seed}.csv',trace)
            dump(a.output/'heldout.json',heldout)
        dump(a.output/'learning_gate.json',dict(development=16,heldout=64,
            physical_development=sum(r['physical_valid'] for r in development),physical_heldout=sum(r['physical_valid'] for r in heldout),
            recovered_heldout=sum(r['arrival_recoverable'] for r in heldout),valid_heldout=sum(r.get('complete_valid',False) for r in heldout),
            repeatable_reference=sum(r.get('complete_valid',False) for r in heldout)>=58 and all(not r['collision'] for r in heldout),
            bc_enabled=False,ppo_enabled=False,updates=0))
    finally:c.close()

if __name__=='__main__':main()
