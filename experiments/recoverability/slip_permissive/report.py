"""Measured same-rollout comparisons, plots and actual MuJoCo playback."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MPLCONFIGDIR','/tmp/go2_sliding_matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/go2_sliding_cache')
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import dump,sha
from experiments.recoverability.configuration_planning.visualize import animation


def summarize(rows):
    n=len(rows)
    return dict(n=n,strict_success=sum(r['strict_success'] for r in rows),exploratory_success=sum(r['exploratory_success'] for r in rows),
      physical_takeoffs=sum(r['physical_takeoff_time'] is not None for r in rows),
      crossings={a:sum(r['crossings'][a] is not None for r in rows) for a in ('45','60','75')},
      upright_arrivals=sum(r['near_upright_arrival'] is not None for r in rows),
      handoffs=sum(r['balance_handoff_time'] is not None for r in rows),
      hard_safe=sum(r['hard_safe'] for r in rows),collisions=sum(bool(r['first_failure'] and (r['first_failure']['bodies'] or r['first_failure']['self_contacts'])) for r in rows),
      native_limit_violations=sum(r['max_joint']>=.001 or r['max_speed']>1.001 or r['max_torque']>1.00001 for r in rows),
      mean_peak=float(np.mean([r['peak'] for r in rows])),max_peak=max(r['peak'] for r in rows),
      mean_safe_supported_peak=float(np.mean([r['safe_supported_peak'] for r in rows])),max_safe_supported_peak=max(r['safe_supported_peak'] for r in rows),
      mean_slip_m=float(np.mean([r['loaded_slip'] for r in rows])),mean_drift_m=float(np.mean([r['loaded_drift'] for r in rows])),
      mean_terminal_com_relative_m=float(np.mean([r['final']['com_support_x'] for r in rows])),
      mean_terminal_com_height_m=float(np.mean([r['final']['com_z'] for r in rows])),
      min_head_clearance_m=min(r['head_clearance'] for r in rows),min_calf_clearance_m=min(r['all_calf_clearance'] for r in rows),
      max_actual_torque_ratio=max(r['max_torque'] for r in rows),max_speed_ratio=max(r['max_speed'] for r in rows))


def trace(path):
    rows=list(csv.DictReader(path.open()));keys=rows[0].keys()
    return {k:np.array([float(r[k]) if r[k] not in ('','True','False') else np.nan for r in rows]) for k in keys if k not in ('phase','contact_mode','bodies')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    base=Path('output');summary={}
    dirs={'multiple_shooting':base/'slip_permissive_search_20261112_v1','native_refinement':base/'slip_permissive_refine_20261112_v1','saved_native_replay':base/'slip_permissive_exact_replay_20261112_v2','balance_start':base/'slip_permissive_archive_20261112_v3'}
    for label,directory in [('yield_refinement','slip_permissive_yield_20261112_v1'),('geometric_continuation','slip_permissive_path_20261112_v1'),('fresh_baseline','slip_permissive_baseline_20261112_v1')]:
        if (base/directory/'heldout.json').exists():dirs[label]=base/directory
    for label,d in dirs.items():
        name='balance_start_comparison.json' if label=='balance_start' else 'comparison.json' if label=='saved_native_replay' else 'heldout.json'
        rows=json.loads((d/name).read_text());summary[label]=summarize(rows)
        if label not in ('balance_start','saved_native_replay'):
            summary[label]['deterministic']=summarize([r for r in rows if not r['stochastic']]);summary[label]['stochastic']=summarize([r for r in rows if r['stochastic']])
    dump(a.output/'summary.json',summary)
    # Paired observers use precisely the same trajectory, not two new samples.
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    sources={'frozen kick':(dirs['fresh_baseline']/'heldout_0.csv' if 'fresh_baseline' in dirs else dirs['saved_native_replay']/'baseline.csv'),'sliding native refinement':dirs['native_refinement']/'heldout_0.csv','sliding multiple shooting':dirs['multiple_shooting']/'heldout_0.csv'}
    for label in ('yield_refinement','geometric_continuation'):
        if label in dirs:sources[label]=dirs[label]/'heldout_0.csv'
    fields=[('pitch','Body pitch (deg)'),('com_support_x','COM minus moving front support (m)'),('slip','Loaded front slip (m)'),('drift','Front-foot displacement (m)'),('head_clearance','Head clearance (m)'),('all_front_calf_clearance','Front calf clearance (m)')]
    for label,path in sources.items():
        t=trace(path)
        for ax,(key,title) in zip(axes.flat,fields):ax.plot(t['time'],t[key],label=label);ax.set(xlabel='Simulator time (s)',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].axhline(90,c='k',ls=':');axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/'pitch_COM_slip_clearance.png',dpi=170);plt.close(fig)
    t=trace(dirs['native_refinement']/'heldout_0.csv');fig,axes=plt.subplots(3,2,figsize=(12,10))
    for leg in ('FL','FR'):
        for ax,field in zip(axes[0],('normal','friction_utilization')):ax.plot(t['time'],t[f'{leg}_{field}'],label='refined '+leg)
    baseline=trace(sources['frozen kick'])
    for leg in ('FL','FR'):
        for ax,field in zip(axes[0],('normal','friction_utilization')):ax.plot(baseline['time'],baseline[f'{leg}_{field}'],ls=':',label='kick '+leg)
    axes[0,0].set_ylabel('Normal force (N)');axes[0,1].set_ylabel('Friction utilization Ft / mu Fn')
    for j in range(12):axes[1,0].plot(t['time'],t[f'torque{j}']);axes[1,1].plot(t['time'],t[f'target{j}'])
    axes[1,0].set_ylabel('Applied joint torque (N m)');axes[1,1].set_ylabel('Native joint target (rad)')
    axes[2,0].plot(t['time'],t['pitch_angular_momentum'],label='whole robot');axes[2,0].plot(t['time'],t['rear_pitch_momentum'],label='rear legs');axes[2,0].set_ylabel('Pitch momentum (kg m²/s)')
    axes[2,1].plot(t['com_x'],t['com_z']);axes[2,1].set(xlabel='World COM x (m)',ylabel='World COM z (m)')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend();axes[2,0].legend();fig.tight_layout();fig.savefig(a.output/'forces_torques_momentum.png',dpi=170);plt.close(fig)
    peak=int(np.argmax(t['pitch']))
    dump(a.output/'representative_peak_mechanics.json',{key:float(t[key][peak]) for key in ('time','pitch','com_support_x','com_vx','com_vz','omega','slip','drift','head_clearance','all_front_calf_clearance','FL_normal','FR_normal','FL_tangent','FR_tangent','FL_friction_utilization','FR_friction_utilization','pitch_angular_momentum','rear_pitch_momentum','actual_torque_ratio','requested_torque_ratio')})
    assets=Assets(a.output/'assets')
    for label in ('native_refinement','multiple_shooting','yield_refinement','geometric_continuation'):
        if label not in dirs:continue
        states=torch.load(dirs[label]/'heldout_motion_0.pt',weights_only=False)
        for state in states:state['caption']=f"t={state['time']:.3f} s; base x={state['qpos'][0]:.3f} m"
        animation(assets.models[1.],states,a.output/f'actual_{label}.gif',f'ACTUAL MuJoCo: {label}; no upright success')
    paths=[Path('experiments/recoverability/physics.py'),Path('experiments/recoverability/contact_transition/core.py'),Path('experiments/recoverability/configuration_planning/shooting.py'),Path('logs/go2balance/Oct31_23-29-27_/model_1500.pt'),Path('output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt')]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    import mujoco
    model=assets.models[1.];data=mujoco.MjData(model)
    for label in ('native_refinement','yield_refinement','geometric_continuation','fresh_baseline'):
        if label not in dirs:continue
        states=torch.load(dirs[label]/'heldout_motion_0.pt',weights_only=False);times=[];heights=[];feet=[]
        for state in states:
            data.qpos[:]=state['qpos'];mujoco.mj_forward(model,data)
            times.append(state['time']);heights.append(state['qpos'][2]);feet.append(float(data.geom_xpos[assets.metadata[1.]['foot_geoms'][0],0]))
        axes[0].plot(times,heights,label=label);axes[1].plot(times,feet,label=label)
    axes[0].set(xlabel='Time (s)',ylabel='Floating-base height (m)');axes[1].set(xlabel='Time (s)',ylabel='FL foot world x (m)')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'body_height_foot_translation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,label in zip(axes,('native_refinement','geometric_continuation')):
        if label not in dirs:continue
        tr=trace(dirs[label]/'heldout_0.csv')
        for index,leg in enumerate(('FL','FR')):
            ax.step(tr['time'],tr['geometric_contact_'+leg]+index*2,where='post',label=leg+' geometric')
            ax.step(tr['time'],(tr[leg+'_normal']>.05*assets.models[1.].body_mass.sum()*9.81)+index*2,where='post',ls=':',label=leg+' loaded')
        ax.set(ylabel=label);ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(a.output/'contact_modes.png',dpi=170);plt.close(fig)
    effort=np.array(assets.metadata[1.]['effort']);speed=np.array(assets.metadata[1.]['speed']);bounds=np.array(assets.metadata[1.]['bounds'])
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    tr=trace(dirs['native_refinement']/'heldout_0.csv')
    for j in range(12):
        axes[0].plot(tr['time'],tr[f'torque{j}']/effort[j]);axes[1].plot(tr['time'],tr[f'v{j}']/speed[j]);axes[2].plot(tr['time'],(tr[f'q{j}']-bounds[j,0])/(bounds[j,1]-bounds[j,0]))
    for ax in axes[:2]:ax.axhline(1,c='k',ls=':');ax.axhline(-1,c='k',ls=':')
    axes[2].axhline(0,c='k',ls=':');axes[2].axhline(1,c='k',ls=':')
    for ax,title in zip(axes,('Actual torque / native effort','Joint velocity / native speed','Joint position within original range')):ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(a.output/'motor_envelopes.png',dpi=170);plt.close(fig)
    dump(a.output/'preserved_inputs_sha256.json',{str(path):sha(path) for path in paths})

if __name__=='__main__':main()
