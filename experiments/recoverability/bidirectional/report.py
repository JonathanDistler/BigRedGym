"""Measured recovery maps, contact comparisons, gap and morphology report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.recoverability.run import dump
from experiments.recoverability.bidirectional.core import BANDS,CONDITIONS
from experiments.recoverability.bidirectional.run import METHODS


def read(path):return json.loads(path.read_text())
def mean(rows,key):return float(np.mean([r[key] for r in rows])) if rows else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--rates',type=Path);p.add_argument('--smoke',type=Path)
    a=p.parse_args();out=a.results/'report';out.mkdir(exist_ok=False)
    basin=read(a.results/'basin_trials.json');maps=read(a.results/'recovery_probability_maps.json')
    forward=read(a.results/'forward_trials.json');handoffs=read(a.results/'handoff_candidates.json')
    morph=read(a.results/'morphology_recovery.json')
    fig,axes=plt.subplots(2,2,figsize=(17,9))
    for ax,(split,stochastic) in zip(axes.flat,[(s,t) for s in ('train','heldout') for t in (False,True)]):
        matrix=np.array([[next(r['unconditional_probability'] for r in maps if r['band']==b and r['split']==split and r['stochastic']==stochastic and r['condition']==case) for case in CONDITIONS] for b in BANDS])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(CONDITIONS)),CONDITIONS,rotation=70);ax.set_yticks(range(len(BANDS)),BANDS)
        ax.set_title(f'{split}, '+('stochastic' if stochastic else 'deterministic'));ax.set_ylabel('Reset pitch band (degrees)')
        for i in range(len(BANDS)):
            for j in range(len(CONDITIONS)):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Five-second recovery fraction (2 seeds/cell)')
    fig.savefig(out/'recovery_probability_maps.png',bbox_inches='tight');plt.close(fig)
    # Same fresh held-out stochastic seed across interventions, original morphology.
    seed=202612520
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    contact_summary=[]
    keys=('FL_normal','FL_tangent','FL_friction_utilization','slip','com_support_x','pitch_angular_momentum','rear_pitch_momentum','FL_calf_clearance')
    units=('N','N','ratio','m','m','kg m²/s','kg m²/s','m')
    for method in ('baseline','anchor','yield','rear_early','yield_rear'):
        path=a.results/f'forward_heldout_{seed}_1_{method}_1.0.csv'
        rows=list(csv.DictReader(path.open()));t=np.array([float(r['time']) for r in rows])
        peak=max(rows,key=lambda r:float(r['pitch']));pre=[r for r in rows if float(r['time'])<=float(peak['time'])]
        first=next((r for r in rows if float(r['slip_speed'])>.08 and float(r['FL_normal'])>10),None)
        contact_summary.append(dict(method=method,seed=seed,peak_time=float(peak['time']),peak_pitch=float(peak['pitch']),
            slip_at_peak=float(peak['slip']),first_slip_time=float(first['time']) if first else None,
            max_prepeak_friction=max(float(r['FL_friction_utilization']) for r in pre),
            mean_prepeak_FL_normal=float(np.mean([float(r['FL_normal']) for r in pre])),
            mean_prepeak_FL_tangent=float(np.mean([float(r['FL_tangent']) for r in pre])),
            max_prepeak_requested_torque=max(float(r['requested_torque_ratio']) for r in pre)))
        for ax,key,unit in zip(axes.flat,keys,units):
            ax.plot(t,[float(r[key]) for r in rows],label=method);ax.set_ylabel(f'{key} ({unit})');ax.grid(alpha=.2)
    axes[0,0].legend();axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'front_contact_before_after.png');plt.close(fig)
    # Front leg configurations, actual/requested torques and controlled rotation.
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    for method in ('baseline','anchor','yield','rear_early','yield_rear'):
        rows=list(csv.DictReader((a.results/f'forward_heldout_{seed}_1_{method}_1.0.csv').open()))
        for ax,key in zip(axes.flat,('pitch','omega','q1','q4','requested_torque_ratio','actual_torque_ratio')):
            ax.plot([float(r['time']) for r in rows],[float(r[key]) for r in rows],label=method);ax.set_ylabel(key);ax.grid(alpha=.2)
    axes[0,0].legend();fig.tight_layout();fig.savefig(out/'configuration_rotation_torque.png');plt.close(fig)
    forward_region=read(a.results/'forward_region.json')
    good=[r for r in basin if r['success'] and not r['stochastic']]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,ykey,label in ((axes[0],'com_support_x','COM x relative to front support (m)'),(axes[1],'omega','Pitch rate (rad/s)')):
        ax.scatter([r['pitch'] for r in forward_region],[r[ykey] for r in forward_region],s=3,alpha=.2,label='Forward samples')
        ax.scatter([r['reset']['actual_pitch'] for r in good],[r['reset']['actual_com_support_x'] if ykey=='com_support_x' else r['reset']['actual_omega'] for r in good],s=20,label='Recovered start states')
        ax.set_xlabel('Body pitch (degrees)');ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'forward_backward_regions.png');plt.close(fig)
    method_rows=[]
    for method in METHODS:
        for split in ('train','heldout'):
            s=[r for r in forward if r['method']==method and r['scale']==1. and r['split']==split]
            method_rows.append(dict(method=method,split=split,n=len(s),takeoffs=sum(r['takeoff'] for r in s),
                collisions=sum(r['collision'] for r in s),loaded_slip=mean(s,'loaded_slip'),loaded_drift=mean(s,'loaded_drift'),
                supported_peak=mean(s,'supported_peak'),min_calf_clearance=mean(s,'min_calf_clearance'),
                physics_steps=sum(r['physics_steps'] for r in s)))
    # No morphology schedule is activated without a measured multi-objective advantage.
    morphology=[]
    for scale in (.8,.85,.9,.95,1.):
        s=[r for r in forward if r['scale']==scale and r['method']=='baseline' and r['split']=='heldout']
        rec=[r for r in morph if r['scale']==scale and r['split']=='heldout']
        morphology.append(dict(scale=scale,n=len(s),takeoffs=sum(r['takeoff'] for r in s),loaded_slip=mean(s,'loaded_slip'),
              supported_peak=mean(s,'supported_peak'),loaded_drift=mean(s,'loaded_drift'),
              min_calf_clearance=mean(s,'min_calf_clearance'),recoveries=sum(r['success'] for r in rec),recovery_tests=len(rec)))
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,label in zip(axes,('loaded_slip','supported_peak','recoveries'),('Loaded slip (m)','Supported peak (degrees)','Recovery successes / 8 held-out cases')):
        ax.plot([r['scale'] for r in morphology],[r[key] for r in morphology],'o-');ax.set_xlabel('Leg scale');ax.set_ylabel(label);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'morphology_tradeoffs.png');plt.close(fig)
    candidates=[r['features'] for r in handoffs]
    ranges={}
    for key,reset_key,unit in (('pitch','actual_pitch','degrees'),('com_support_x','actual_com_support_x','m'),('omega','actual_omega','rad/s')):
        f=[r[key] for r in candidates];b=[r['reset'][reset_key] for r in good]
        ranges[key]=dict(unit=unit,forward_safe_range=[min(f),max(f)] if f else None,recovered_start_range=[min(b),max(b)] if b else None)
    summary=dict(basin_trials=len(basin),valid_prepared_states=len({r['reset']['state_id'] for r in basin if r['reset']['valid']}),
                 successful_unique_starts=len({r['reset']['state_id'] for r in good}),
                 nominal_gate=read(a.results/'validation_gate.json'),forward_trials=len(forward),
                 methods=method_rows,morphology=morphology,contact_diagnosis=contact_summary,
                 physical_region_ranges=ranges,handoff_tests=len(handoffs),handoff_successes=sum(r['recovery']['success'] for r in handoffs),
                 morphology_schedule_enabled=False,transition_ppo_enabled=False,
                 interaction_budget='All balance continuations cap at 6.5 s; forward branches cap at 2.5 s, terminating physical failure. Realized step counts reported.',
                 telemetry_rate_hz=100,safety_slip_integration_hz=500)
    if a.rates:
        rates=read(a.rates/'rate_evaluation.json');summary['signed_rate_tests']=len(rates)
        summary['signed_rate_successes']=sum(r['success'] for r in rates)
        summary['signed_rate_sign_coverage']={str(b):sorted({r['actual_sign'] for r in rates if r['band']==b and r['safe']}) for b in BANDS}
    if a.smoke:
        summary['smoke']=read(a.smoke/'smoke_gate.json')
    dump(out/'summary.json',summary)
    lines=['# Bidirectional recoverability validation','',f"Basin trials: {len(basin)}. Unique recoverable starts: {summary['successful_unique_starts']}. Exact-state handoff successes: {summary['handoff_successes']}/{len(handoffs)}.",'',
           '## Recovery by pitch band','', '| Band | Train deterministic | Held-out deterministic | Train stochastic | Held-out stochastic |','|---|---:|---:|---:|---:|']
    for band in BANDS:
        values=[]
        for stochastic,split in ((False,'train'),(False,'heldout'),(True,'train'),(True,'heldout')):
            rows=[r for r in basin if r['band']==band and r['split']==split and r['stochastic']==stochastic]
            values.append(f"{sum(r['success'] for r in rows)}/{len(rows)}")
        lines.append('| '+str(band)+' | '+' | '.join(values)+' |')
    lines+=['','## Held-out forward interventions','', '| Method | Takeoffs | Collisions | Slip (m) | Supported peak (degrees) |','|---|---:|---:|---:|---:|']
    for r in method_rows:
        if r['split']=='heldout':lines.append(f"| {r['method']} | {r['takeoffs']}/{r['n']} | {r['collisions']}/{r['n']} | {r['loaded_slip']:.4f} | {r['supported_peak']:.2f} |")
    lines+=['','Initial states in the recovery map are locally valid IK proposals followed by motor integration. They are not asserted to be reachable from standing. Two seeds per map cell provide descriptive probabilities, not a robust basin certificate.',
            '', 'No empirical overlap was validated. Morphology schedule and transition PPO remain disabled. See summary.json for physical ranges, actual interaction counts, force diagnostics and separate smoke-test gate.']
    (out/'measured_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
