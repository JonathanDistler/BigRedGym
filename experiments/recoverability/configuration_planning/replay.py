"""Independent frozen-command 16+64 replay with geometric impact/contact auditing."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write,sha,model_audit
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.physics import World
from experiments.recoverability.configuration_planning.plan import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-base',type=int,default=202634120);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    plan=json.loads((a.source/'plan_D.json').read_text());dump(a.output/'plan_D.json',plan)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8;assets=Assets(a.output/'assets');c=controller()
    try:
        compatibility=model_audit(c,World(assets,cfg));dump(a.output/'compatibility.json',compatibility);assert compatibility['compatible']
        all_rows=[]
        for split,offset,n in (('development',0,16),('heldout',400,64)):
            rows=[]
            for seed in range(a.seed_base+offset,a.seed_base+offset+n):
                row,trace,states=evaluate(net,c,assets,cfg,plan,seed)
                rows.append(row)
                if seed<a.seed_base+offset+4:csv_write(a.output/f'{split}_{seed}.csv',trace)
                if split=='development' and seed==a.seed_base:torch.save(states,a.output/'motion_D.pt')
            dump(a.output/f'{split}.json',rows);all_rows.extend(rows)
            print('frozen replay',split,n,'physical',sum(r['physical_valid'] for r in rows),'complete',sum(r.get('complete_valid',False) for r in rows),flush=True)
        held=all_rows[16:];dump(a.output/'learning_gate.json',dict(development=16,heldout=64,
            physical_heldout=sum(r['physical_valid'] for r in held),recovered_heldout=sum(r['arrival_recoverable'] for r in held),
            valid_heldout=sum(r.get('complete_valid',False) for r in held),repeatable_reference=sum(r.get('complete_valid',False) for r in held)>=58 and all(not r['collision'] for r in held),
            bc_enabled=False,ppo_enabled=False,updates=0))
        dump(a.output/'manifest.json',dict(source_plan_sha256=sha(a.source/'plan_D.json'),kick_sha256=sha(KICK),balance_sha256=c.meta['sha256'],
            source_sha256={str(f):sha(f) for f in Path('experiments/recoverability/configuration_planning').glob('*.py')},
            development_seeds=list(range(a.seed_base,a.seed_base+16)),heldout_seeds=list(range(a.seed_base+400,a.seed_base+464)),
            contact_definition='geometric gap and loaded-support gap separate; pre/post actual contact velocity at 500 Hz',
            optimizer_iterations=0,ppo_updates=0,bc_updates=0))
    finally:c.close()

if __name__=='__main__':main()
