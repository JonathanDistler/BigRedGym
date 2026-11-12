"""Atlas-initialized native-dynamics multiple shooting and gated robustness."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write,sha
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.bidirectional.core import continuation,restore_contact,compact_snapshot
from experiments.recoverability.controllers import BalanceAdapter
from experiments.recoverability.configuration_planning.geometry import ConfigurationProblem
from experiments.recoverability.configuration_planning.dynamics import capture,segment,integration
from experiments.recoverability.configuration_planning.shooting import MultipleShooting

DEVELOPMENT=tuple(range(202630120,202630136))
HELDOUT=tuple(range(202630520,202630584))


def waypoint_guesses(problem,atlas,start,mode):
    problem.set(start['qpos'],start['qvel'])
    feet=problem.d.geom_xpos[problem.w.foot_geoms[:2]].copy();mid=feet[:,:2].mean(0)
    width=float(feet[0,1]-feet[1,1]);stagger=float(feet[0,0]-feet[1,0])
    pitches=(30,35,45) if mode=='D' else (35,45)
    guesses=[]
    for index,pitch in enumerate(pitches):
        choices=[r for r in atlas if r['geometric_cell'] and r['pitch']==pitch]
        choices=sorted(choices,key=lambda r:np.linalg.norm(np.array(r['joints'])-start['qpos'][problem.w.qids])+
            12*abs(r['com_target']+.05)+3*abs(r['height']-start['qpos'][2]))
        refined=[]
        for candidate in choices[:3]:
            lateral=.04 if mode=='B' and index==0 else 0.
            profile=(width,stagger,2. if mode=='B' and index==0 else 0.,0.,lateral)
            row=problem.solve(pitch,candidate['height'],candidate['com_target'],profile,np.array(candidate['joints']),100)
            if row['geometric_cell']:refined.append(row)
        row=(min(refined,key=lambda r:np.linalg.norm(np.array(r['joints'])-start['qpos'][problem.w.qids])) if refined else copy.deepcopy(choices[0]))
        row['matches_source_contacts']=bool(refined)
        q=np.array(row['qpos']);q[:2]+=mid
        if mode=='C' and index==0:q[2]+=.025
        row['qpos']=q.tolist();guesses.append(row)
    return guesses


def evaluate(net,c,assets,cfg,plan,seed,noise=False,query=True):
    kind='launch' if plan['mode']=='D' else '30'
    w,m,start=capture(net,assets,cfg,seed,kind,.018,noise)
    if start is None:return dict(seed=seed,mode=plan['mode'],prefix_valid=False,physical_valid=False,arrival_recoverable=False,**m.outcome()),m.trace,m.states
    m.phase='planned';m.active=True
    for command,duration in zip(plan['commands'],plan['durations']):
        start=segment(w,start,np.array(command),duration,m)
        if m.failed:break
    row=w.physical_row();pitch=np.degrees(row['pitch']);comx=float(row['com'][0]-row['feet'][:2,0].mean())
    if plan['mode']=='A':mode_ok=m.longest_gap<=.004
    elif plan['mode']=='B':mode_ok=m.longest_gap<=.12 and all(e['mode']!='unloaded' for e in m.contact_modes if e['phase']=='planned')
    else:mode_ok=m.longest_gap<=.08 and m.total_gap<=.12 and m.impact<=.75
    physical=(not m.failed and 40<=pitch<=50 and row['rear_clearance']>.015 and row['front_support'] and
        m.slip<=.02 and m.drift<=.04 and m.head_min>=.005 and m.calf_min>=.005 and mode_ok)
    # This is always an exact-state query, independent of IK/static classification.
    a=BalanceAdapter(c,w);saved=compact_snapshot(w,m,a)
    recovery=dict(success=False,not_queried=True)
    if query and not m.failed:
        bw,bm,ba=restore_contact(c,assets,cfg,saved);bm.phase='balance_counterfactual'
        recovery=continuation(c,bw,bm,ba,False,seed+90000)
    result=dict(seed=seed,mode=plan['mode'],prefix_valid=True,physical_valid=bool(physical),arrival_recoverable=bool(recovery['success']),
                complete_valid=bool(physical and recovery['success']),terminal_pitch=float(pitch),terminal_com_x=comx,
                terminal_com_velocity=row['com_velocity'].tolist(),terminal_body_height=float(w.d.qpos[2]),
                terminal_joints=w.d.qpos[w.qids].tolist(),terminal_angular_velocity=w.d.qvel[3:6].tolist(),
                terminal_contact_forces=row['loads'].tolist(),recovery=recovery,**m.outcome())
    return result,m.trace,m.states


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=8);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    net,_,cfg,_=curriculum.load_source(KICK);net.eval().requires_grad_(False);cfg.seconds=8
    assets=Assets(a.output/'assets');problem=ConfigurationProblem(assets,cfg);atlas=json.loads((a.atlas/'atlas.json').read_text())
    modes=('A',) if a.smoke else ('A','B','C','D');plans=[]
    for mode in modes:
        w,m,start=capture(net,assets,cfg,202629120,'launch' if mode=='D' else '30')
        if start is None:raise RuntimeError('Nominal physical kick prefix unavailable')
        waypoints=waypoint_guesses(problem,atlas,start,mode)
        durations=[.20,.18,.20] if mode=='D' else [.06,.18] if mode=='C' else [.12,.18]
        planner=MultipleShooting(assets,cfg,start,waypoints,mode,durations)
        result=planner.solve(2 if a.smoke else a.max_nfev);result['waypoints']=waypoints
        plans.append(result);dump(a.output/f'plan_{mode}.json',result)
        torch.save(start,a.output/f'exact_start_{mode}.pt')
        print('shooting',mode,'cost',result['initial_cost'],result['final_cost'],'defects',result['defects'],flush=True)
    c=controller()
    try:
        development=[]
        for plan in plans:
            for seed in (DEVELOPMENT[:1] if a.smoke else DEVELOPMENT):
                result,trace,states=evaluate(net,c,assets,cfg,plan,seed,query=True)
                development.append(result);csv_write(a.output/f'development_{plan["mode"]}_{seed}.csv',trace)
                if seed==DEVELOPMENT[0]:torch.save(states,a.output/f'motion_{plan["mode"]}.pt')
            dump(a.output/'development.json',development)
            print('development',plan['mode'],sum(r.get('complete_valid',False) for r in development if r['mode']==plan['mode']),flush=True)
        if a.smoke:return
        # Selection uses development only; held-out cohort remains untouched.
        def score(plan):
            rows=[r for r in development if r['mode']==plan['mode']]
            return (sum(r.get('complete_valid',False) for r in rows),sum(r['physical_valid'] for r in rows),
                    sum(r['supported_peak']-200*r['loaded_slip']-50*r['collision'] for r in rows))
        selected=max(plans,key=score);heldout=[]
        for seed in HELDOUT:
            result,trace,states=evaluate(net,c,assets,cfg,selected,seed,query=True)
            heldout.append(result)
            if seed<HELDOUT[0]+4:csv_write(a.output/f'heldout_{seed}.csv',trace)
            dump(a.output/'heldout.json',heldout)
        good=sum(r.get('complete_valid',False) for r in heldout)
        dump(a.output/'learning_gate.json',dict(selected_mode=selected['mode'],development_perturbations_per_mode=16,heldout=64,
             valid_heldout=good,repeatable_reference=good>=58 and all(not r['collision'] for r in heldout),
             bc_enabled=False,ppo_enabled=False,updates=0,extended_ppo=False,
             reason='Physical feasibility and original balance-policy recovery are required; no training is launched'))
        dump(a.output/'manifest.json',dict(atlas_sha256=sha(a.atlas/'atlas.json'),checkpoint_sha256=sha(KICK),
             development_seeds=DEVELOPMENT,heldout_seeds=HELDOUT,solver='bounded deterministic multiple shooting',max_nfev=a.max_nfev,
             state_resets_during_validation=0,optimizer_nodes='hypotheses only; not executed resets',ppo_updates=0,bc_updates=0))
    finally:c.close()

if __name__=='__main__':main()
