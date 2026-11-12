"""Fresh matched native-policy baseline, without optimization or training."""
import argparse
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.slip_permissive.refine import rollout
from experiments.recoverability.slip_permissive.run import HELDOUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);assets=Assets(a.output/'assets');c=controller();rows=[]
    try:
        for seed in HELDOUT:
            for noise in (False,True):
                r,trace,states,_=rollout(net,c,assets,cfg,np.zeros(12),seed,noise);r['plan']='frozen kick baseline';rows.append(r)
                if len(rows)<=2:csv_write(a.output/f'heldout_{int(noise)}.csv',trace);torch.save(states,a.output/f'heldout_motion_{int(noise)}.pt')
            dump(a.output/'heldout.json',rows)
    finally:c.close()

if __name__=='__main__':main()
