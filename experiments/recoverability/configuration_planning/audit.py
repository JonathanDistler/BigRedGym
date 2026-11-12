"""State-unit defect audits and reverse-state features read from exact snapshots."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump
from experiments.recoverability.physics import World
from experiments.recoverability.bidirectional.core import physical_hash,telemetry
from experiments.recoverability.configuration_planning.dynamics import SPEC,segment,put,integration


def defects(assets,cfg,plan,start):
    w=World(assets,cfg);sequential=start;rows=[]
    for k,(node,command,duration) in enumerate(zip(plan['nodes'],plan['commands'],plan['durations'])):
        if k==0:independent=start
        else:
            previous=plan['nodes'][k-1]
            mujoco.mj_resetData(w.m,w.d);w.d.qpos[:]=previous['qpos'];w.d.qvel[:]=previous['qvel']
            w.d.time=start['time']+sum(plan['durations'][:k]);mujoco.mj_forward(w.m,w.d)
            w.target=np.array(plan['commands'][k-1]);independent=integration(w)
        end=segment(w,independent,np.array(command),duration)
        delta=np.zeros(w.m.nv);mujoco.mj_differentiatePos(w.m,delta,1.,np.array(node['qpos']),end['qpos'])
        velocity=end['qvel']-node['qvel']
        sequential=segment(w,sequential,np.array(command),duration)
        matrix=w.d.xmat[w.base].reshape(3,3)
        rows.append(dict(node=k,root_position_error_m=float(np.linalg.norm(delta[:3])),root_rotation_error_rad=float(np.linalg.norm(delta[3:6])),
            joint_error_max_rad=float(abs(delta[6:]).max()),root_linear_velocity_error_mps=float(np.linalg.norm(velocity[:3])),
            root_angular_velocity_error_radps=float(np.linalg.norm(velocity[3:6])),joint_velocity_error_max_radps=float(abs(velocity[6:]).max()),
            single_unrolled_pitch=float(np.degrees(np.arctan2(-matrix[2,0],matrix[0,0]))),
            single_unrolled_height=float(w.d.qpos[2]),single_unrolled_com_relative_x=float(w.d.subtree_com[w.base,0]-w.d.geom_xpos[w.foot_geoms[:2],0].mean())))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--plans',type=Path,required=True);p.add_argument('--refined',type=Path,required=True)
    p.add_argument('--reverse',type=Path,required=True);p.add_argument('--contact-refined',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    _,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets');checks={}
    for mode in 'ABCD':
        plan=json.loads((a.plans/f'plan_{mode}.json').read_text());start=torch.load(a.plans/f'exact_start_{mode}.pt',weights_only=False)
        checks[mode]=defects(assets,cfg,plan,start)
    start=torch.load(a.plans/'exact_start_D.pt',weights_only=False)
    checks['D_refined']=defects(assets,cfg,json.loads((a.refined/'plan_D.json').read_text()),start)
    if a.contact_refined:checks['D_contact_refined']=defects(assets,cfg,json.loads((a.contact_refined/'plan_D.json').read_text()),start)
    dump(a.output/'dynamics_defects_physical_units.json',checks)
    raw=json.loads((a.reverse/'predecessors.json').read_text());features=[]
    for path in sorted(a.reverse.glob('predecessor_*.pt')):
        _,entry,duration=path.stem.split('_');entry=int(entry);duration=float(duration)
        saved=torch.load(path,weights_only=False);w=World(assets,cfg)
        mujoco.mj_setState(w.m,w.d,saved['state'],SPEC);mujoco.mj_forward(w.m,w.d);w.target=saved['target'].copy()
        row=w.physical_row();original=next(r for r in raw if r['entry']==entry and r['duration']==duration)
        features.append(dict(entry=entry,duration=duration,physical_state_id=physical_hash(w),recovery=original['recovery'],
            pitch=float(np.degrees(row['pitch'])),body_height=float(w.d.qpos[2]),com_relative_x=float(row['com'][0]-row['feet'][:2,0].mean()),
            com_velocity=row['com_velocity'].tolist(),angular_velocity=w.d.qvel[3:6].tolist(),joints=w.d.qpos[w.qids].tolist(),
            contact_forces=row['loads'].tolist(),front_support=bool(row['front_support']),
            prefix_slip=float(saved['metrics']['slip']),prefix_drift=float(saved['metrics']['drift']),
            **telemetry(w)))
    dump(a.output/'reverse_state_features.json',features)

if __name__=='__main__':main()
