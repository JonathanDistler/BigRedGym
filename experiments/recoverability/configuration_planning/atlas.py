"""Deterministic pitch-height-COM atlas; no dynamic reachability claim from IK."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,sha
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem

PITCHES=tuple(range(30,91,5))
HEIGHTS=(.30,.375,.45,.525,.60)
COMS=(-.20,-.10,-.05,0.,.05)
PROFILES=((.32,0.,0.,0.,0.),(.36,0.,2.,2.,.04),(.28,.04,-2.,-2.,-.04))


def seeds():
    rows=json.loads(Path('output/recoverability_maps_20261112/validated_region.json').read_text())
    balance=np.array([rows[0]['features'][f'q{j}'] for j in range(12)])
    alternatives=[balance.copy() for _ in PROFILES]
    alternatives[1][[7,10]]=[1.0,1.0];alternatives[1][[8,11]]=[-1.8,-1.8]
    alternatives[2][[7,10]]=[1.5,1.5];alternatives[2][[8,11]]=[-2.2,-2.2]
    return alternatives


def run(assets,cfg,out,max_nfev=60):
    problem=ConfigurationProblem(assets,cfg);rows=[];initial=seeds();previous={}
    for pitch in PITCHES:
        for height in HEIGHTS:
            for com in COMS:
                for index,profile in enumerate(PROFILES):
                    seed=initial[index].copy();seed[[1,4]]+=np.deg2rad(90-pitch)
                    if (height,com,index) in previous:seed=previous[(height,com,index)]
                    row=problem.solve(pitch,height,com,profile,seed,max_nfev)
                    row.update(id=len(rows),profile_index=index,dynamically_reachable='not established')
                    if row['geometric_cell']:
                        previous[(height,com,index)]=np.array(row['joints'])
                        row['static']=problem.static(np.array(row['qpos']))
                        row['single_FL_static']=problem.static(np.array(row['qpos']),'single_FL')
                        if row['static']['equilibrium']:
                            fraction=row['static']['load_fraction_FL']
                            row['redistribution_static']={str(target):problem.static(np.array(row['qpos']),'B',target) for target in (.2,.5,.8)}
                    else:row['static']=dict(feasible=False,evaluated=False)
                    rows.append(row)
        subset=[r for r in rows if r['pitch']==pitch]
        print('atlas',pitch,'geometric',sum(r['geometric_cell'] for r in subset),'static',sum(r['static']['feasible'] for r in subset),'/',len(subset),flush=True)
        dump(out/'atlas.json',rows)
    # A graph in configuration space is an initialization, not a dynamic path.
    path=[];parent=None
    for pitch in PITCHES:
        choices=[r for r in rows if r['pitch']==pitch and r['geometric_cell']]
        if not choices:continue
        def cost(r):
            q=np.array(r['joints']);distance=0 if parent is None else float(np.linalg.norm(q-np.array(parent['joints'])))
            return distance+10*abs(r['com_relative'][0])+.5*(not r['static']['feasible'])
        chosen=min(choices,key=cost);path.append(chosen);parent=chosen
    dump(out/'configuration_path.json',dict(nodes=path,geometric_only=True,dynamic_validation_required=True,
        complete_pitch_coverage=len(path)==len(PITCHES),maximum_joint_step=max((float(np.max(abs(np.array(b['joints'])-a['joints']))) for a,b in zip(path,path[1:])),default=None)))
    return rows


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=60)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);_,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets')
    problem=ConfigurationProblem(assets,cfg)
    if a.smoke:
        q=seeds()[0];r=problem.solve(90,.525,0.,PROFILES[0],q,a.max_nfev)
        r['static']=problem.static(np.array(r['qpos']));dump(a.output/'smoke.json',r);print(r['geometric_cell'],r['foot_error'],r['com_target_error'],r['clearances'],r['static'],flush=True)
        return
    sources={str(p):sha(p) for p in Path('experiments/recoverability/configuration_planning').glob('*.py')}
    for path in sources:
        target=a.output/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    dump(a.output/'manifest.json',dict(pitches=PITCHES,heights=HEIGHTS,COMs=COMS,profiles=PROFILES,max_nfev=a.max_nfev,
        source_sha256=sources,checkpoint_sha256=sha(KICK),classification='geometry vs static; IK is not reachability',ppo_updates=0,bc_updates=0))
    run(assets,cfg,a.output,a.max_nfev)

if __name__=='__main__':main()
