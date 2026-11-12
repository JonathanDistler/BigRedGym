"""Measured configuration feasibility, trajectory, contact and reachability reports."""
import argparse
import csv
import json
from pathlib import Path
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiments.recoverability.run import dump,csv_write
from experiments.recoverability.configuration_planning.atlas import PITCHES,HEIGHTS,COMS


def read(p):return json.loads(p.read_text())
def average(rows,key):return float(np.mean([r[key] for r in rows])) if rows else None

def stats(rows):
    return dict(n=len(rows),physical=sum(r['physical_valid'] for r in rows),recoveries=sum(r['arrival_recoverable'] for r in rows),
        complete=sum(r.get('complete_valid',False) for r in rows),collisions=sum(r['collision'] for r in rows),
        actuator_violations=sum(r['max_speed']>1.001 or r['max_torque']>1.00001 or r['max_joint']>=.001 for r in rows),
        peak=average(rows,'peak'),supported_peak=average(rows,'supported_peak'),slip=average(rows,'loaded_slip'),
        head_clearance_min=min(r['head_clearance'] for r in rows),calf_clearance_min=min(r['all_calf_clearance'] for r in rows),
        impact_max=max(r['landing_impact_velocity'] for r in rows),contact_gap_max=max(r['longest_contact_gap'] for r in rows),
        endpoint_pitch=[min(r.get('terminal_pitch',0) for r in rows),max(r.get('terminal_pitch',0) for r in rows)],
        endpoint_COMx=[min(r.get('terminal_com_x',0) for r in rows),max(r.get('terminal_com_x',0) for r in rows)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--connectivity',type=Path,required=True)
    p.add_argument('--plans',type=Path,required=True);p.add_argument('--refined',type=Path,required=True)
    p.add_argument('--contact-refined',type=Path,required=True);p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--reverse',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    atlas=read(a.atlas/'atlas.json');path=read(a.connectivity/'dense_configuration_path.json')
    redistribution=read(a.connectivity/'load_redistribution.json');development=read(a.plans/'development.json');heldout=read(a.replay/'heldout.json')
    backward=read(a.audit/'reverse_state_features.json');good_back=[r for r in backward if r['recovery']['success']]
    # Every scalar and all twelve coordinates/torques are exportable in physical units.
    flat=[]
    for r in atlas:
        row={k:r[k] for k in ('id','pitch','height','com_target','com_y_target','profile_index','geometric','in_cell','geometric_cell','foot_error','com_target_error')}
        row.update(r['clearances']);row['static']=r['static']['feasible'];row['reasons']=';'.join(r['reasons'])
        row.update({f'q{j}_rad':r['joints'][j] for j in range(12)})
        if 'joint_torques' in r['static']:
            row.update({f'tau{j}_Nm':r['static']['joint_torques'][j] for j in range(12)})
            row.update({k:r['static'][k] for k in ('torque_ratio','torque_margin_Nm','force_residual_N','moment_residual_Nm','load_fraction_FL')})
            row['friction_margin_N']=min(r['static']['friction_margin_N']);row['friction_utilization']=max(r['static']['friction_utilization'])
        flat.append(row)
    csv_write(a.output/'atlas_physical_values.csv',flat)
    fig,axes=plt.subplots(1,len(COMS),figsize=(19,4),sharey=True)
    cmap=ListedColormap(['#d9d9d9','#4c9bd1','#34a36b'])
    for ax,com in zip(axes,COMS):
        values=np.zeros((len(HEIGHTS),len(PITCHES)))
        for i,h in enumerate(HEIGHTS):
            for j,pitch in enumerate(PITCHES):
                rows=[r for r in atlas if r['height']==h and r['pitch']==pitch and r['com_target']==com]
                values[i,j]=2 if any(r['static']['feasible'] for r in rows) else 1 if any(r['geometric_cell'] for r in rows) else 0
        im=ax.imshow(values,cmap=cmap,vmin=0,vmax=2,origin='lower',aspect='auto')
        ax.set_xticks(range(0,len(PITCHES),3),PITCHES[::3]);ax.set_yticks(range(len(HEIGHTS)),HEIGHTS)
        ax.set_xlabel('Pitch (degrees)');ax.set_title(f'COM x target {com:+.2f} m')
    axes[0].set_ylabel('Body height (m)');fig.colorbar(im,ax=axes.ravel().tolist(),ticks=[0,1,2],label='0: no local solution; 1: geometry; 2: geometry + statics')
    fig.savefig(a.output/'pitch_height_COM_atlas.png',bbox_inches='tight');plt.close(fig)
    static=[r for r in atlas if r['static']['feasible']];fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,label in zip(axes,('torque_ratio','torque_margin_Nm','friction_utilization'),('Max joint torque / motor limit','Minimum joint torque margin (N m)','Max Coulomb friction utilization')):
        values=[max(r['static'][key]) if isinstance(r['static'].get(key),list) else r['static'].get(key) for r in static] if key!='friction_utilization' else [max(r['static']['friction_utilization']) for r in static]
        ax.scatter([r['pitch'] for r in static],values,c=[r['height'] for r in static],cmap='viridis');ax.set_xlabel('Pitch (degrees)');ax.set_ylabel(label);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'static_torque_friction.png');plt.close(fig)
    failures=collections.Counter(reason for r in atlas for reason in r['reasons']);fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(failures.keys(),failures.values());ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Local solver attempts failing constraint');fig.tight_layout();fig.savefig(a.output/'configuration_constraint_failures.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for ax,indices,label in zip(axes.flat,((1,2),(4,5),(7,8),(10,11)),('FL','FR','RL','RR')):
        for j in indices:ax.plot([r['pitch'] for r in path['nodes']],[r['joints'][j] for r in path['nodes']],label=f'q{j}')
        ax.set_ylabel(label+' joints (rad)');ax.legend();ax.grid(alpha=.2)
    axes[-1,0].set_xlabel('Geometric path pitch (degrees)');axes[-1,1].set_xlabel('Geometric path pitch (degrees)')
    fig.suptitle('Static configuration continuation - not a dynamic replay');fig.tight_layout();fig.savefig(a.output/'geometric_joint_path.png');plt.close(fig)
    curves={}
    for mode in 'ABCD':
        pth=next(a.plans.glob(f'development_{mode}_*.csv'));curves[mode]=list(csv.DictReader(pth.open()))
    curves['D final']=list(csv.DictReader(next(a.replay.glob('development_*.csv')).open()))
    fig,axes=plt.subplots(4,2,figsize=(14,13),sharex=True)
    keys=('pitch','com_support_x','slip','head_clearance','all_front_calf_clearance','load_FL','load_FR','pitch_angular_momentum')
    units=('degrees','m','m','m','m','N','N','kg m²/s')
    for label,rows in curves.items():
        for ax,key,unit in zip(axes.flat,keys,units):
            ax.plot([float(r['time']) for r in rows],[float(r[key]) for r in rows],label=label);ax.set_ylabel(f'{key} ({unit})');ax.grid(alpha=.2)
    axes[0,0].axhline(45,color='green',ls='--');axes[1,0].axhline(.02,color='red',ls='--');axes[0,0].legend(fontsize=8)
    axes[-1,0].set_xlabel('Time from standing reset (s)');axes[-1,1].set_xlabel('Time from standing reset (s)')
    fig.tight_layout();fig.savefig(a.output/'dynamic_contact_COM_clearance.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    final=curves['D final'];t=[float(r['time']) for r in final]
    axes[0].plot(t,[max(float(r['FL_friction_utilization']),float(r['FR_friction_utilization'])) for r in final]);axes[0].set_ylabel('Friction utilization')
    for j in (1,2,4,5):axes[1].plot(t,[float(r[f'torque{j}']) for r in final],label=f'joint {j}')
    axes[1].legend();axes[1].set_ylabel('Applied torque (N m)')
    axes[2].plot(t,[float(r['geometric_contact_FL']) for r in final],label='FL geometric contact')
    axes[2].plot(t,[float(r['geometric_contact_FR'])+.05 for r in final],label='FR geometric contact + .05')
    axes[2].plot(t,[float(r['geometric_flight_gap']) for r in final],label='full flight gap (s)');axes[2].legend();axes[2].set_xlabel('Time (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'friction_torques_geometric_contacts.png');plt.close(fig)
    defect=read(a.audit/'dynamics_defects_physical_units.json');fig,axes=plt.subplots(1,2,figsize=(13,4))
    labels=list(defect)
    for ax,key,unit in zip(axes,('root_position_error_m','root_rotation_error_rad'),('m','rad')):
        ax.bar(labels,[max(r[key] for r in defect[label]) for label in labels]);ax.set_ylabel(key+' ('+unit+')');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(a.output/'shooting_defects.png');plt.close(fig)
    forward=[r for r in heldout if r.get('terminal_pitch') is not None and not r['collision']]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,label in zip(axes,('terminal_com_x','terminal_body_height'),('COM x from front support (m)','Body height (m)')):
        ax.scatter([r['terminal_pitch'] for r in forward],[r[key] for r in forward],label='Actual forward terminals (rejected)')
        ax.scatter([r['pitch'] for r in good_back],[r['com_relative_x'] if key=='terminal_com_x' else r['body_height'] for r in good_back],label='5 s policy recovery only; excess slip')
        ax.set_xlabel('Pitch (degrees)');ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'forward_backward_gap.png');plt.close(fig)
    summary=dict(atlas_attempts=len(atlas),geometric=sum(r['geometric_cell'] for r in atlas),static=sum(r['static']['feasible'] for r in atlas),
        failures=dict(failures),dense_path={k:v for k,v in path.items() if k not in ('nodes','interiors')},
        redistribution_static=sum(r['geometric_cell'] and r['force_test']['feasible'] for r in redistribution),redistribution_tests=len(redistribution),
        development={mode:stats([r for r in development if r['mode']==mode]) for mode in 'ABCD'},
        earlier_heldout=stats(read(a.plans/'heldout.json')),slower_heldout=stats(read(a.refined/'heldout.json')),
        contact_refined_heldout=stats(read(a.contact_refined/'heldout.json')),final_heldout=stats(heldout),
        reverse_tests=len(backward),reverse_recoveries=len(good_back),reverse_pitch_range=[min(r['pitch'] for r in good_back),max(r['pitch'] for r in good_back)],
        learning_gate=read(a.replay/'learning_gate.json'),ppo_updates=0,bc_updates=0,
        classification='IK geometry and static feasibility found; executable robust dynamic bridge not established')
    baseline=read(a.reverse/'bank_validation.json')
    summary['reverse_slip_audit']=[dict(entry=r['entry'],duration=r['duration'],pitch=r['pitch'],
        cumulative_recovery_slip_m=r['recovery']['loaded_slip'],
        matched_baseline_slip_m=next(b['loaded_slip'] for b in baseline if b['entry']==r['entry']),
        excess_slip_m=r['recovery']['loaded_slip']-next(b['loaded_slip'] for b in baseline if b['entry']==r['entry'])) for r in good_back]
    summary['low_slip_reverse_catches']=sum(r['excess_slip_m']<=.02 for r in summary['reverse_slip_audit'])
    if forward and good_back:
        summary['separation']=dict(pitch_gap_degrees=min(r['pitch'] for r in good_back)-max(r['terminal_pitch'] for r in forward),
            COMx_gap_m=min(r['com_relative_x'] for r in good_back)-max(r['terminal_com_x'] for r in forward),
            height_gap_m=min(r['body_height'] for r in good_back)-max(r['terminal_body_height'] for r in forward),
            forward_pitch=[min(r['terminal_pitch'] for r in forward),max(r['terminal_pitch'] for r in forward)],
            forward_COMx=[min(r['terminal_com_x'] for r in forward),max(r['terminal_com_x'] for r in forward)],
            forward_height=[min(r['terminal_body_height'] for r in forward),max(r['terminal_body_height'] for r in forward)])
        summary['state_ranges']={}
        pairs=[('omega_y_body_radps',lambda r:r['terminal_angular_velocity'][1],lambda r:r['angular_velocity'][1]),
               ('COM_vx_mps',lambda r:r['terminal_com_velocity'][0],lambda r:r['com_velocity'][0]),
               ('COM_vz_mps',lambda r:r['terminal_com_velocity'][2],lambda r:r['com_velocity'][2]),
               ('FL_force_N',lambda r:r['terminal_contact_forces'][0],lambda r:r['contact_forces'][0]),
               ('FR_force_N',lambda r:r['terminal_contact_forces'][1],lambda r:r['contact_forces'][1])]
        for name,f,b in pairs:
            summary['state_ranges'][name]=dict(forward=[min(map(f,forward)),max(map(f,forward))],backward=[min(map(b,good_back)),max(map(b,good_back))])
        summary['closest_joint_configuration_gap_rad']=min(float(np.linalg.norm(np.array(f['terminal_joints'])-b['joints'])) for f in forward for b in good_back)
    dump(a.output/'summary.json',summary)
    lines=['# Configuration and structured transition planning','',f"Atlas: {len(atlas)} attempts, {summary['geometric']} geometric cells, {summary['static']} static cells.",
        '', '| Mode | Development n | Physical stage success | Balance recovery | Supported peak (degrees) | Slip (m) |','|---|---:|---:|---:|---:|---:|']
    for mode,r in summary['development'].items():lines.append(f"| {mode} | {r['n']} | {r['physical']} | {r['recoveries']} | {r['supported_peak']:.2f} | {r['slip']:.4f} |")
    lines+=['',f"Final frozen-command held-out evaluation: {summary['final_heldout']['complete']}/64 valid; {summary['reverse_recoveries']}/{summary['reverse_tests']} reverse arrivals recover.",
            '', 'No BC/PPO or forward extension beyond 45 degrees is enabled. Geometric continuation is not evidence of dynamic reachability.']
    (a.output/'measured_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
