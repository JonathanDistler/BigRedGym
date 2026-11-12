"""Physics-rate loaded contact and calf-clearance trajectories for fixed seeds."""
import argparse
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.recoverability.transition import Assets,World,Metrics,KICK,csv_write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);cfg.seconds=8;assets=Assets(a.output/'assets')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True)
    keys=['pitch','force_FL','force_FR','slip_speed','slip','drift','FL_calf_clearance','FR_calf_clearance']
    units=['deg','N','N','m/s','m','m','m','m']
    for seed in (84000,84001,84009):
        w=World(assets,cfg);actor,_=w.reset(1.,seed);m=Metrics(w);m.require_takeoff=True;w.monitor=m
        trace=[];original=m.observe
        def observe(row,dt):
            s=original(row,dt);trace.append(s);return s
        m.observe=observe;rng=torch.Generator().manual_seed(seed+1000000)
        for _ in range(400):
            with torch.no_grad():
                dist=net.distribution(torch.tensor(actor));raw=dist.loc+dist.scale*torch.randn(12,generator=rng)
            (actor,_),_,done,_=w.step(raw.tanh().numpy())
            if done:break
        csv_write(a.output/f'contacts_{seed}.csv',trace)
        for ax,key,unit in zip(axes.flat,keys,units):
            ax.plot([s['time'] for s in trace],[s[key] for s in trace],label=str(seed))
            if m.first_failure:ax.axvline(m.first_failure['time'],color='red',alpha=.25)
            ax.set_ylabel(f'{key} ({unit})');ax.grid(alpha=.2)
    axes[0,0].legend();axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
    fig.suptitle('Frozen stochastic kick: red lines mark first physical safety failure')
    fig.tight_layout();fig.savefig(a.output/'contact_collision_trajectories.png');plt.close(fig)

if __name__=='__main__':main()
