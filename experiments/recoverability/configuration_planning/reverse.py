"""Forward-time physical predecessor probes from verified balance-bank states."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.physics import World,feasible_state
from experiments.recoverability.controllers import BalanceAdapter
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.bidirectional.core import compact_snapshot,restore_contact,continuation
from experiments.recoverability.configuration_planning.dynamics import ValidationMetrics,SPEC
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem


def bank_start(c,assets,cfg,entry,index,bank):
    w=World(assets,cfg);feasible_state(w,c.env._standing_posture()[0].numpy(),entry['seed'])
    anchor=w.anchor.copy();mujoco.mj_setState(w.m,w.d,bank[f'{index}_integration'],SPEC);mujoco.mj_forward(w.m,w.d)
    w.target=bank[f'{index}_target'].copy();m=ValidationMetrics(w);m.anchor=anchor
    m.drift=entry['features']['drift'];m.slip=entry['features']['slip'];m.active=True;m.phase='reverse'
    a=BalanceAdapter(c,w);a.previous=torch.tensor(w.target,dtype=torch.float32)-a.reference
    a.phase=torch.tensor(bank[f'{index}_phase']);a.load=torch.tensor(bank[f'{index}_filtered_load']);a.clock=float(w.d.time)
    w.monitor=m;m.observe(w.physical_row(),0.)
    return w,m,a


def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    _,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets');c=controller();problem=ConfigurationProblem(assets,cfg)
    region=Path('output/recoverability_maps_20261112');entries=json.loads((region/'validated_region.json').read_text());bank=np.load(region/'recoverable_states.npz')
    atlas=json.loads((a.atlas/'atlas.json').read_text());results=[];baselines=[]
    try:
        for index in (0,3,6):
            w,m,adapter=bank_start(c,assets,cfg,entries[index],index,bank)
            saved=compact_snapshot(w,m,adapter);bw,bm,ba=restore_contact(c,assets,cfg,saved)
            baseline=continuation(c,bw,bm,ba,False,202632120+index)
            baselines.append(dict(entry=index,**baseline));dump(a.output/'bank_validation.json',baselines)
            if not baseline['success']:raise RuntimeError('Original recovery bank continuation failed; predecessor probes blocked')
            choices=[r for r in atlas if r['pitch']==85 and r['static']['feasible']]
            goal=min(choices,key=lambda r:abs(r['height']-w.d.qpos[2])+np.linalg.norm(np.array(r['joints'])-w.d.qpos[w.qids])*.05)
            for duration in (.15,.30,.60):
                w,m,adapter=bank_start(c,assets,cfg,entries[index],index,bank)
                initial=w.target.copy();goal_joint=np.array(goal['joints'])
                tau=np.array(goal['static']['joint_torques']);start_time=float(w.d.time)
                angle0=np.degrees(w.physical_row()['pitch']);best_pitch=angle0;best=compact_snapshot(w,m,adapter)
                for tick in range(round(duration/.002)):
                    if tick%5==0:
                        alpha=min(1.,(tick*.002+.01)/duration)
                        desired=initial+alpha*(goal_joint+tau/cfg.kp-initial)
                        actual=np.degrees(w.physical_row()['pitch']);reference=angle0+alpha*(85-angle0)
                        desired[[1,4]]+=.7*np.deg2rad(actual-reference)+.1*w.d.qvel[4]
                        target=w.target+np.clip(desired-w.target,-.7*w.speed*.01,.7*w.speed*.01)
                        adapter.previous=torch.tensor(target,dtype=torch.float32)-adapter.reference
                    row=w.tick(target);adapter.advance_memory(row['loads'],.002);w.measure()
                    angle=np.degrees(row['pitch'])
                    if not m.failed and m.slip<=.02 and m.drift<=.04 and angle<best_pitch:
                        best_pitch=angle;best=compact_snapshot(w,m,adapter)
                    if m.failed:break
                bw,bm,ba=restore_contact(c,assets,cfg,best);features=bw.physical_row()
                height=float(bw.d.qpos[2]);omega=bw.d.qvel[3:6].copy();joints=bw.d.qpos[bw.qids].copy()
                velocity=features['com_velocity'].copy();forces=features['loads'].copy()
                probe=continuation(c,bw,bm,ba,False,202632520+index)
                results.append(dict(entry=index,duration=duration,initial_pitch=float(angle0),arrival_pitch=float(best_pitch),
                    arrival_com_relative_x=float(features['com'][0]-features['feet'][:2,0].mean()),arrival_body_height=height,
                    arrival_angular_velocity=omega.tolist(),arrival_joints=joints.tolist(),arrival_com_velocity=velocity.tolist(),arrival_forces=forces.tolist(),
                    physical_prefix=m.outcome(),recovery=probe,predecessor_found=bool(best_pitch<angle0-.1 and probe['success'])))
                # Preserve actual q/v and controller state, not interpolated proposals.
                torch.save(best,a.output/f'predecessor_{index}_{duration}.pt')
                csv_write(a.output/f'reverse_{index}_{duration}.csv',m.trace)
                dump(a.output/'predecessors.json',results)
                print('reverse',index,duration,'pitch',best_pitch,'recoverable',probe['success'],flush=True)
    finally:c.close()

if __name__=='__main__':main()
