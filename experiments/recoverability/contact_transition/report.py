"""Torque/contact/COM/clearance figures and measured transition-feasibility report."""
import argparse
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.recoverability.run import dump,csv_write
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.contact_transition.core import TIMES,NEUTRAL


def read(p):return json.loads(p.read_text())
def load_csv(p):return list(csv.DictReader(p.open()))
def avg(rows,k):return float(np.mean([r[k] for r in rows]))


def reconstructed_clearance(actions,model):
    """Geometry only: never use reconstructed contact forces as dynamics evidence."""
    data=mujoco.MjData(model);ground=np.flatnonzero(model.geom_bodyid==0)
    groups={'head':[], 'front_calf':[],'nonfoot':[]}
    for geom in range(model.ngeom):
        body=model.geom_bodyid[geom]
        if not body or not model.geom_contype[geom]:continue
        name=model.body(body).name
        if not name.endswith('_foot'):groups['nonfoot'].append(geom)
        if name.startswith('Head'):groups['head'].append(geom)
        if name.startswith(('FL_calf','FR_calf')):groups['front_calf'].append(geom)
    rows=[]
    for step in actions:
        if 'qpos' not in step:continue
        data.qpos[:]=step['qpos'];data.qvel[:]=step['qvel'];mujoco.mj_forward(model,data)
        row=dict(time=step['time'],base_height=float(data.qpos[2]))
        for label,geoms in groups.items():
            row[label+'_clearance']=min((mujoco.mj_geomDistance(model,data,int(g),int(f),1.,None) for g in geoms for f in ground),default=1.)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True);p.add_argument('--mechanics',type=Path,required=True)
    p.add_argument('--comparison-dir',type=Path,help='Read completed ablations and morphology comparisons from this directory')
    p.add_argument('--extension',type=Path);p.add_argument('--catch',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    comparison=a.comparison_dir or a.validation
    validation=read(a.validation/'validation.json');ablations=read(comparison/'ablations.json');morph=read(comparison/'morphology.json')
    seeds=sorted({r['seed'] for r in validation});seed=seeds[0];curves={}
    basepath=a.validation/f'baseline_{seed}_0_trajectory.csv'
    curves['baseline']=load_csv(basepath)
    for mode in ('maintained','transient'):
        for strategy in ('load_yield','no_yield'):
            path=a.validation/f'validation_{mode}_{strategy}_{seed}_0_trajectory.csv'
            if path.exists():curves[f'{mode}/{strategy}']=load_csv(path)
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    keys=('pitch','com_support_x','FL_normal','FL_tangent','FL_friction_utilization','slip','FL_calf_clearance','front_gap')
    units=('degrees','m','N','N','ratio','m','m','s')
    for label,rows in curves.items():
        for ax,key,unit in zip(axes.flat,keys,units):
            ax.plot([float(r['time_from_launch']) for r in rows],[float(r[key]) for r in rows],label=label)
            ax.set_ylabel(f'{key} ({unit})');ax.grid(alpha=.2)
    axes[0,0].axhspan(45,55,color='green',alpha=.1);axes[2,1].axhline(.02,color='red',linestyle='--')
    axes[0,0].legend(fontsize=7);axes[-1,0].set_xlabel('Launch time (s)');axes[-1,1].set_xlabel('Launch time (s)')
    fig.suptitle('Diagnostic candidates: peaks do not imply accepted references');fig.tight_layout();fig.savefig(a.output/'contact_progression.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for label,rows in curves.items():
        axes[0].plot([float(r['com_support_x']) for r in rows],[float(r['com_z']) for r in rows],label=label)
        axes[1].plot([float(r['time_from_launch']) for r in rows],[float(r['com_vx']) for r in rows],label=label)
    axes[0].set_xlabel('COM x from front-foot midpoint (m)');axes[0].set_ylabel('COM height (m)')
    axes[1].set_xlabel('Launch time (s)');axes[1].set_ylabel('COM world x velocity (m/s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(a.output/'com_trajectories.png');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True)
    efforts=read(a.validation/'model_verification.json')[-1]['actuator_effort']
    joints=(1,2,4,5,8,11)
    for label,rows in curves.items():
        for ax,j in zip(axes.flat,joints):
            ax.plot([float(r['time_from_launch']) for r in rows],[float(r[f'torque{j}']) for r in rows],label=label)
            ax.set_ylabel(f'Joint {j} torque (N m)');ax.grid(alpha=.2)
    for ax,j in zip(axes.flat,joints):
        ax.axhline(efforts[j],color='red',ls='--',alpha=.3);ax.axhline(-efforts[j],color='red',ls='--',alpha=.3)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'joint_torque_limits.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for label,rows in curves.items():
        for ax,key in zip(axes,('requested_torque_ratio','actual_torque_ratio')):
            ax.plot([float(r['time_from_launch']) for r in rows],[float(r[key]) for r in rows],label=label)
            ax.set_ylabel(key+' (ratio)');ax.grid(alpha=.2);ax.axhline(1.,color='red',ls='--')
    axes[0].axhline(1.5,color='orange',ls=':');axes[0].legend(fontsize=7);axes[-1].set_xlabel('Launch time (s)')
    fig.tight_layout();fig.savefig(a.output/'requested_applied_torque.png');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True)
    for label,rows in curves.items():
        for ax,j in zip(axes.flat,joints):
            ax.plot([float(r['time_from_launch']) for r in rows],[float(r[f'target{j}']) for r in rows],label=label)
            ax.set_ylabel(f'Joint {j} target (rad)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'optimized_joint_sequences.png');plt.close(fig)
    assets=Assets(a.output/'geometry_assets');fig,axes=plt.subplots(1,3,figsize=(15,4))
    for label in curves:
        path=a.validation/(f'baseline_{seed}_0_actions.json' if label=='baseline' else f"validation_{label.replace('/','_')}_{seed}_0_actions.json")
        rows=reconstructed_clearance(read(path),assets.models[1.])
        csv_write(a.output/f"geometry_{label.replace('/','_')}.csv",rows)
        for ax,key in zip(axes,('base_height','head_clearance','front_calf_clearance')):
            ax.plot([r['time'] for r in rows],[r[key] for r in rows],label=label);ax.set_ylabel(key+' (m)');ax.set_xlabel('Launch time (s)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.output/'body_height_head_calf_clearance.png');plt.close(fig)
    mech=read(a.mechanics/'mechanics.json');fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for case in ('baseline','hold_targets','hold_front','rear_retract','front_fold'):
        rows=load_csv(a.mechanics/f'mechanics_{case}_202621520.csv')
        rows=[r for r in rows if r.get('front_moment_com_y') not in ('',None)]
        for ax,key in zip(axes.flat,('pitch_angular_momentum','rear_pitch_momentum','front_moment_com_y','gravity_moment_front_y')):
            unit='kg m²/s' if 'momentum' in key else 'N m'
            ax.plot([float(r['time_from_launch']) for r in rows],[float(r[key]) for r in rows],label=case);ax.set_ylabel(key+' ('+unit+')');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/'momentum_contact_moments.png');plt.close(fig)
    methods=[]
    for mode in ('maintained','transient'):
        for strategy in sorted({r['strategy'] for r in validation}):
            for stochastic in (False,True):
                cohort=[r for r in validation if r['mode']==mode and r['strategy']==strategy and r['stochastic']==stochastic]
                if not cohort:continue
                methods.append(dict(mode=mode,strategy=strategy,stochastic=stochastic,n=len(cohort),
                    accepted=sum(r['accepted'] for r in cohort),physical_safe=sum(r['physical_safe'] for r in cohort),
                    takeoff=sum(r['takeoff'] for r in cohort),progress_peak=avg(cohort,'progress_peak'),slip=avg(cohort,'loaded_slip'),
                    drift=avg(cohort,'loaded_drift'),front_gap=avg(cohort,'worst_front_gap'),
                    final_com_support_x=float(np.mean([r['final']['com_support_x'] for r in cohort])),
                    failures=[(r['first_failure'] or {}).get('bodies','') for r in cohort]))
    searches=[]
    for path in a.search.glob('search_*.json'):searches.extend(read(path))
    summary=dict(reference_gate=read(a.validation/'reference_gate.json'),search_candidates=len(searches),
        search_branches=sum(len(r['results']) for r in searches),validation_branches=len(validation),
        ablation_branches=len(ablations),morphology_branches=len(morph),methods=methods,
        transient_branches_with_unloading=sum(r['worst_front_gap']>0 for record in searches for r in record['results'] if r['mode']=='transient'),
        mechanics=[dict(case=case,n=len([r for r in mech if r['case']==case]),
            safe=sum(r['physical_safe'] for r in mech if r['case']==case),
            peak=avg([r for r in mech if r['case']==case],'peak'),slip=avg([r for r in mech if r['case']==case],'loaded_slip')) for case in sorted({r['case'] for r in mech})],
        ppo_updates=0,bc_updates=0,morphology_schedule_enabled=False)
    if a.extension:
        ext=read(a.extension/'extension_sweep.json');summary['extension_branches']=sum(len(r['results']) for r in ext)
    if a.catch:
        catch=read(a.catch/'catch_search.json');tests=read(a.catch/'catch_validation.json')
        summary['catch_branches']=sum(len(r['results']) for r in catch);summary['catch_validation']=tests
        summary['catch_reference_gate']=read(a.catch/'reference_gate.json')
    dump(a.output/'summary.json',summary)
    lines=['# Contact-aware launch/continuation feasibility','',f"CEM: {len(searches)} candidates / {summary['search_branches']} paired-reset branches. Held-out original-model validation: {len(validation)} branches.",'',
        '| Mode | Front strategy | Noise | Accepted | Physically safe full horizon | Mean supported progression (degrees) | Loaded slip (m) |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in methods:lines.append(f"| {r['mode']} | {r['strategy']} | {r['stochastic']} | {r['accepted']}/{r['n']} | {r['physical_safe']}/{r['n']} | {r['progress_peak']:.2f} | {r['slip']:.4f} |")
    lines+=['','No peak-only result is accepted. Check reference_gate.json for the finite-reference gate; it does not certify balance-policy recovery or standing-to-handstand fusion.']
    (a.output/'measured_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
