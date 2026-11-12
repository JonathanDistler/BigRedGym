"""Physics-rate bottlenecks, safe-start action search, morphology and reverse seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.transition import *
from experiments.recoverability.reverse_curriculum import ReverseCurriculum


def detailed_replay(net,assets,cfg,seed,stochastic,region):
    w=World(assets,cfg);actor,_=w.reset(1.,seed)
    m=Metrics(w);m.require_takeoff=True;w.monitor=m
    rng=torch.Generator().manual_seed(seed+1000000)
    observed=m.observe; best=[];safe=[]
    def observe(row,dt):
        sample=observed(row,dt);d=decomposition(sample,region)
        if not best or d['distance']<best[0]['distance']:
            best[:]=[d,snapshot(w,m)]
        if not m.failed and m.drift<=.04 and m.slip<=.02 and row['front_support']:
            if not safe or d['distance']<safe[0]['distance']:safe[:]=[d,snapshot(w,m)]
        return sample
    m.observe=observe
    # snapshot must not serialize the local observer function.
    for control in range(400):
        with torch.no_grad():
            dist=net.distribution(torch.tensor(actor));raw=dist.loc
            if stochastic:raw=raw+dist.scale*torch.randn(12,generator=rng)
        (actor,_),_,done,_=w.step(raw.tanh().numpy())
        if done:break
    for container in (best,safe):
        if container:container[1]['metrics'].pop('observe',None)
    return best,safe


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget',type=int,default=8);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    region_dir=Path('output/recoverability_maps_20261112')
    region=json.loads((region_dir/'validated_region.json').read_text())
    meta=json.loads((region_dir/'frozen_controller.json').read_text())
    net,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8
    assets=Assets(a.output/'assets');c=SavedController(meta['checkpoint'],'go2balance')
    try:
        reports=[];starts=[]
        for stochastic in (False,True):
            for seed in range(84000,84010):
                best,safe=detailed_replay(net,assets,cfg,seed,stochastic,region)
                result,_=probe(assets,cfg,c,best[1])
                label=f'{seed}_{int(stochastic)}'
                reports.append(dict(seed=seed,stochastic=stochastic,**best[0],counterfactual=result,safe_start=safe[0] if safe else None))
                torch.save(dict(closest=best,safe=safe),a.output/f'snapshots_{label}.pt')
                if safe:starts.append((safe[0]['distance'],label,safe[1]))
                print('physics minimum',label,best[0]['distance'],flush=True)
        dump(a.output/'physics_handoff_analysis.json',reports)
        for _,label,saved in sorted(starts,key=lambda x:x[0])[:3]:
            records,best=optimize(assets,cfg,c,saved,region,92000,a.budget,horizon=.4)
            dump(a.output/f'safe_search_{label}.json',records);csv_write(a.output/f'transition_{label}.csv',best[2])
            torch.save(best,a.output/f'best_transition_{label}.pt')
            print('safe search',label,best[1]['balance_success'],flush=True)
        # Paired original/scaled-model reset proposals. No cross-model state copy.
        morphology=[];cur=ReverseCurriculum()
        for scale in (1.,.95,.9,.85,.8):
            for pitch in (90.,88.,85.,80.,70.,60.,35.):
                w=World(assets,cfg)
                accepted,screen=feasible_state(w,c.env._standing_posture()[0].numpy(),93000,pitch=pitch,scale=scale)
                m=Metrics(w);initial=m.observe(w.physical_row(),0)
                result,trace=probe(assets,cfg,c,snapshot(w,m)) if accepted else ({'success':False},[])
                row=dict(scale=scale,pitch=pitch,accepted=accepted,screen=screen,initial=initial,recovery=result)
                morphology.append(row)
                if accepted and result['success']:
                    cur.admit(f'{scale}_{pitch}',snapshot(w,m),result,scale=scale)
                    csv_write(a.output/f'reverse_{scale}_{pitch}.csv',trace)
                print('morphology',scale,pitch,accepted,result['success'],flush=True)
        dump(a.output/'morphology_validation.json',morphology)
        torch.save(cur,a.output/'reverse_curriculum.pt')
        dump(a.output/'reverse_curriculum.json',dict(horizon=cur.horizon,admitted=[dict(key=k,scale=v['scale']) for k,v in cur.admitted.items()],training_updates=0))
    finally:c.close()

if __name__=='__main__':main()
