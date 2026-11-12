"""Action-generated signed-rate coverage supplement; no velocity resets."""
import argparse
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.bidirectional.core import (BANDS,World,ContactMetrics,compact_snapshot,
    continuation,physical_hash)
from experiments.recoverability.physics import feasible_state
from experiments.recoverability.controllers import BalanceAdapter


def pulse(c,assets,cfg,band,seed,delta,duration):
    w=World(assets,cfg);valid,screen=feasible_state(w,c.env._standing_posture()[0].numpy(),seed,pitch=band)
    m=ContactMetrics(w);a=BalanceAdapter(c,w);base=w.target.copy()
    if not valid:return None
    target=np.clip(base+delta,w.bounds[:,0],w.bounds[:,1])
    a.previous=torch.tensor(target,dtype=torch.float32)-a.reference
    for _ in range(round(duration/.002)):
        row=w.tick(target);a.advance_memory(row['loads'],.002);sample=m.observe(row,.002)
        if m.failed:break
    safe=not m.failed and m.slip<=.02 and m.drift<=.04 and abs(sample['pitch']-band)<=3
    return w,m,a,sample,safe


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=48);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    _,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8;assets=Assets(a.output/'assets');c=controller()
    try:
        candidates=[];tests=[];bank={};rng=np.random.default_rng(202617120)
        for band in BANDS:
            endpoints=[]
            for index in range(a.budget):
                # Independent bounded sagittal front/rear commands, mirrored left/right.
                u=np.zeros(4) if index==0 else rng.uniform(-.35,.35,4)
                delta=np.zeros(12);delta[[1,4]]=u[0];delta[[2,5]]=u[1];delta[[7,10]]=u[2];delta[[8,11]]=u[3]
                duration=(.04,.08,.12)[index%3]
                trial=pulse(c,assets,cfg,band,202617120,delta,duration)
                if trial is None:continue
                w,m,adapter,sample,safe=trial
                candidates.append(dict(band=band,index=index,actions=delta.tolist(),duration=duration,safe=safe,
                                       actual_pitch=sample['pitch'],omega=sample['omega'],**m.result()))
                if safe:endpoints.append((sample['omega'],delta,duration))
            for sign in (-1,1):
                if not endpoints:continue
                selected=max(endpoints,key=lambda t:sign*t[0])
                print('selected rate',band,sign,selected[0],flush=True)
                for seed in (202617520,202617521):
                    jitter=np.random.default_rng(seed).uniform(-.003,.003,12)
                    for stochastic in (False,True):
                        trial=pulse(c,assets,cfg,band,seed,selected[1]+jitter,selected[2])
                        if trial is None:continue
                        w,m,adapter,sample,safe=trial
                        state_id=physical_hash(w);saved=compact_snapshot(w,m,adapter)
                        if safe:bank.setdefault(state_id,saved)
                        result=continuation(c,w,m,adapter,stochastic,seed+80000) if safe else m.result()
                        tests.append(dict(band=band,requested_sign=sign,actual_sign=int(np.sign(sample['omega'])),
                            actual_omega=sample['omega'],actual_pitch=sample['pitch'],safe=safe,state_id=state_id,
                            seed=seed,split='heldout',stochastic=stochastic,**result))
                        csv_write(a.output/f'rate_{band}_{sign}_{seed}_{int(stochastic)}.csv',m.trace)
            dump(a.output/'rate_search.json',candidates);dump(a.output/'rate_evaluation.json',tests)
        torch.save(bank,a.output/'rate_states_unique.pt')
    finally:c.close()

if __name__=='__main__':main()
