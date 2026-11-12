"""Dense geometric continuation and explicit lateral load-redistribution IK."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem


def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    _,_,cfg,_=curriculum.load_source(KICK);problem=ConfigurationProblem(Assets(a.output/'assets'),cfg)
    atlas=json.loads((a.atlas/'atlas.json').read_text());initial=next(r for r in atlas if r['pitch']==30 and r['height']==.45 and r['profile_index']==0 and r['static']['feasible'])
    previous=np.array(initial['joints']);path=[];interiors=[]
    for pitch in range(30,91):
        row=problem.solve(pitch,.45,0.,(.32,0,0,0,0),previous,120)
        row['static']=problem.static(np.array(row['qpos']))
        if row['geometric_cell']:previous=np.array(row['joints'])
        path.append(row)
        if len(path)>1:
            q0=np.array(path[-2]['qpos']);q1=np.array(row['qpos']);delta=np.zeros(problem.m.nv)
            mujoco.mj_differentiatePos(problem.m,delta,1.,q0,q1)
            for alpha in (.25,.5,.75):
                q=q0.copy();mujoco.mj_integratePos(problem.m,q,delta,alpha);problem.set(q)
                _,_,clear=problem.clearances();feet=problem.d.geom_xpos[problem.w.foot_geoms[:2]]
                contacts=np.array([[0.,.16,problem.radius],[0.,-.16,problem.radius]])
                interiors.append(dict(pitch=pitch-1+alpha,clearances=clear,
                    front_contact_error=float(np.max(np.linalg.norm(feet-contacts,axis=1))),
                    com_x=float(problem.d.subtree_com[problem.w.base,0])))
        if pitch%10==0:print('continuation',pitch,row['geometric_cell'],row['static']['feasible'],flush=True)
    dump(a.output/'dense_configuration_path.json',dict(nodes=path,interiors=interiors,
        all_nodes_geometric=all(r['geometric_cell'] for r in path),all_nodes_static=all(r['static']['feasible'] for r in path),
        dense_collision_free=all(r['clearances']['all_nonfoot']>=.0049 and r['clearances']['self_penetration']<1e-6 for r in interiors),
        contact_interpolation_error_max=max(r['front_contact_error'] for r in interiors),
        dynamically_reachable=False,classification='kinematic initialization, not an executed trajectory'))
    redistribution=[]
    for pitch in (30,45,60,90):
        bases=[r for r in atlas if r['pitch']==pitch and r['profile_index']==0 and r['static']['feasible']]
        base=min(bases,key=lambda r:abs(r['height']-.45))
        for fraction in (.2,.5,.8):
            lateral=(fraction-.5)*.32
            row=problem.solve(pitch,base['height'],0.,(.32,0,(fraction-.5)*10,0,lateral),np.array(base['joints']),150)
            row['force_test']=problem.static(np.array(row['qpos']),'B',fraction)
            row['target_fraction']=fraction;redistribution.append(row)
    dump(a.output/'load_redistribution.json',redistribution)

if __name__=='__main__':main()
