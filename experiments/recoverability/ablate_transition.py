"""Paired motor-space front/rear ablations of searched transition references."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.transition import (KICK,Assets,SavedController,restore,probe,snapshot,decomposition,dump,csv_write)


def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    _,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8;assets=Assets(a.output/'assets')
    region=json.loads(Path('output/recoverability_maps_20261112/validated_region.json').read_text())
    meta=json.loads(Path('output/recoverability_maps_20261112/frozen_controller.json').read_text())
    c=SavedController(meta['checkpoint'],'go2balance');results=[]
    try:
        for path in sorted(a.search.glob('safe_search_*.json')):
            label=path.stem.removeprefix('safe_search_');records=json.loads(path.read_text())
            best=max(records,key=lambda r:r['score']);actions=np.array(best['actions'])
            saved=torch.load(a.search/f'snapshots_{label}.pt',weights_only=False)['safe'][1]
            for mode in ('hold_target','front_only','rear_only','coupled'):
                w,m,_=restore(assets,cfg,saved,c);trace=[];supports=[]
                for tick in range(200):
                    knot=min(2,int(tick*.002/.4*3));u=actions[knot]
                    delta=np.r_[u[:3],u[:3],u[3:],u[3:]];delta[[3,9]]*=-1
                    if mode=='hold_target':delta[:]=0
                    if mode=='front_only':delta[6:]=0
                    if mode=='rear_only':delta[:6]=0
                    row=w.tick(np.clip(saved['target']+delta,w.bounds[:,0],w.bounds[:,1]))
                    sample=m.observe(row,.002);trace.append(sample);supports.append(row['front_support'])
                    if m.failed:break
                cf,_=probe(assets,cfg,c,snapshot(w,m))
                results.append(dict(seed=label,mode=mode,**m.result(),balance_success=cf['success'],
                    distance=decomposition(sample,region)['distance'],support_fraction=float(np.mean(supports)),
                    final_com_support_x=sample['com_support_x'],
                    min_calf_clearance=min(min(s['FL_calf_clearance'],s['FR_calf_clearance']) for s in trace)))
                csv_write(a.output/f'{label}_{mode}.csv',trace)
        dump(a.output/'ablations.json',results)
    finally:c.close()

if __name__=='__main__':main()
