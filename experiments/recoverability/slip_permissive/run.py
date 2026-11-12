"""Opt-in offline full-motion search; never starts neural training."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write,model_audit
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.configuration_planning.reverse import bank_start
from experiments.recoverability.slip_permissive.search import SlipPermissiveShooting
from experiments.recoverability.slip_permissive.runtime import standing,evaluate

DEVELOPMENT=tuple(range(202641120,202641128))
HELDOUT=tuple(range(202641520,202641552))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--max-nfev',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False)
    assets=Assets(a.output/'assets');c=controller()
    try:
        w,_,_=standing(net,assets,cfg,202641020,.04)
        audit=model_audit(c,w);dump(a.output/'compatibility.json',audit);assert audit['compatible']
        path=json.loads(Path('output/configuration_connectivity_20261112_v1/dense_configuration_path.json').read_text())['nodes']
        region=Path('output/recoverability_maps_20261112');entries=json.loads((region/'validated_region.json').read_text());bank=np.load(region/'recoverable_states.npz')
        bw,_,_=bank_start(c,assets,cfg,entries[0],0,bank)
        final=copy.deepcopy(path[-1]);final['qpos']=bw.d.qpos.tolist();final['desired_pitch']=90.;final['static']={}
        plans=[];development=[]
        for name,prefix,durations in [('standing_fast',.04,[.30,.30,.40]),('launch_fast',.34,[.20,.30,.40]),('standing_slow',.04,[.50,.50,.60])]:
            w,m,start=standing(net,assets,cfg,202641020,prefix)
            rows=[copy.deepcopy(path[0]),copy.deepcopy(path[30]),copy.deepcopy(final)]
            planner=SlipPermissiveShooting(assets,cfg,start,rows,'sliding',durations)
            plan=planner.solve(a.max_nfev);plan.update(name=name,prefix=prefix,waypoints=rows)
            dump(a.output/f'plan_{name}.json',plan);plans.append(plan)
            r,trace,states=evaluate(net,c,assets,cfg,plan,202641021)
            dump(a.output/f'nominal_{name}.json',r);csv_write(a.output/f'nominal_{name}.csv',trace);torch.save(states,a.output/f'motion_{name}.pt')
            print(name,'cost',plan['final_cost'],'safe pitch',r['safe_peak'],'failure',r['first_failure'],flush=True)
        # All candidates get matched development budgets; selection never uses held-out seeds.
        for plan in plans:
            for seed in DEVELOPMENT:
                for noise in (False,True):
                    r,_,_=evaluate(net,c,assets,cfg,plan,seed,noise);development.append(r)
            dump(a.output/'development.json',development)
        def score(plan):
            rows=[r for r in development if r['plan']==plan['name']]
            return (sum(r['exploratory_success'] for r in rows),sum(r['hard_safe'] for r in rows),np.mean([r['safe_supported_peak'] for r in rows]))
        best=max(plans,key=score);dump(a.output/'selected_plan.json',best)
        heldout=[]
        for seed in HELDOUT:
            for noise in (False,True):
                r,trace,states=evaluate(net,c,assets,cfg,best,seed,noise);heldout.append(r)
                if len(heldout)<=2:csv_write(a.output/f'heldout_{int(noise)}.csv',trace);torch.save(states,a.output/f'heldout_motion_{int(noise)}.pt')
            dump(a.output/'heldout.json',heldout)
        dump(a.output/'training_gate.json',dict(extended_ppo=False,bc_updates=0,ppo_updates=0,
            diagnostic_pilot_ready=bool(sum(r['hard_safe'] and r['safe_supported_peak']>=45 and r['exploratory_entry'] for r in heldout)>=48),
            reason='Physical trajectory validation precedes any separate two-update smoke and gated ten-update pilot.'))
    finally:c.close()

if __name__=='__main__':main()
