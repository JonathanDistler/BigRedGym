"""Generate physical-unit distance tables and contact trajectory figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.recoverability.transition import decomposition


def report(out, switching, region_dir, physics_analysis=None):
    region=json.loads((region_dir/'validated_region.json').read_text())
    reports=[]; table=[]
    for path in sorted(switching.glob('switch_*.csv')):
        rows=list(csv.DictReader(path.open()))
        samples=[{k:float(v) for k,v in r.items() if k not in ('mode','bodies','safe')} for r in rows]
        best=min((decomposition(r,region) for r in samples),key=lambda r:r['distance'])
        reports.append(dict(rollout=path.stem,**best))
        for rank,f in enumerate(best['features']):
            table.append(dict(rollout=path.stem,time=best['time'],rank=rank+1,**f))
    if physics_analysis is not None:
        reports=[dict(rollout=f"switch_{r['seed']}_{int(r['stochastic'])}",**r)
                 for r in json.loads(physics_analysis.read_text())]
        table=[dict(rollout=r['rollout'],time=r['time'],rank=i+1,**f)
               for r in reports for i,f in enumerate(r['features'])]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'distance_features.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=table[0].keys()); writer.writeheader();writer.writerows(table)
    lines=['# Handoff bottleneck at sampled closest approach','',
           'Distances use all 34 features and the original normalized L-infinity metric. '+
           ('Closest times use exact frozen-policy replay sampled at every 2 ms physics step.' if physics_analysis else
            'Closest times are at the saved 50 Hz kick control samples.'),'',
           '| Rollout | Time (s) | Distance | Five largest mismatches: actual → nearest bank value (units; normalized error) |',
           '|---|---:|---:|---|']
    for r in reports:
        terms=['%s: %.4g → %.4g %s (%.3f)'%(f['name'],f['actual'],f['reference'],f['unit'],f['normalized_mismatch']) for f in r['features'][:5]]
        lines.append('| %s | %.3f | %.3f | %s |'%(r['rollout'],r['time'],r['distance'],'; '.join(terms)))
    (out/'distance_report.md').write_text('\n'.join(lines)+'\n')
    for group,paths in [('kick',list(switching.glob('switch_*_1.csv'))),('transition',list(out.glob('transition_*.csv'))),('boundary',list(out.glob('boundary_*.csv')))]:
        if not paths:continue
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
        for path in paths:
            rows=list(csv.DictReader(path.open()))
            if not rows:continue
            t=[float(r['time']) for r in rows]
            for ax,key in zip(axes.flat,['pitch','force_FL','force_FR','slip','drift','safe']):
                values=[float(r[key]) if key!='safe' else float(r[key]=='True') for r in rows]
                ax.plot(t,values,alpha=.5,label=path.stem)
                ax.set_ylabel(dict(pitch='Pitch (deg)',force_FL='FL normal load (N)',force_FR='FR normal load (N)',slip='Loaded slip (m)',drift='Front drift (m)',safe='Collision/limit safety (1=safe)')[key]);ax.grid(alpha=.2)
        axes[-1,0].set_xlabel('Simulation time (s)'); axes[-1,1].set_xlabel('Simulation time (s)')
        fig.tight_layout();fig.savefig(out/f'{group}_contacts.png');plt.close(fig)
    (out/'sampled_distance_analysis.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--switching',type=Path,default=Path('output/recoverability_switching_verified_20261112'))
    p.add_argument('--region-dir',type=Path,default=Path('output/recoverability_maps_20261112'))
    p.add_argument('--physics-analysis',type=Path)
    a=p.parse_args();report(a.output,a.switching,a.region_dir,a.physics_analysis)
