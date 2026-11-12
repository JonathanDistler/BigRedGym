"""Same-trajectory archival reclassification and exact saved-state balance tests."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write,model_audit
from experiments.recoverability.bidirectional.run import controller
from experiments.recoverability.configuration_planning.reverse import bank_start
from experiments.recoverability.bidirectional.core import compact_snapshot
from experiments.recoverability.slip_permissive.metrics import DualMetrics,balance
from experiments.recoverability.physics import World
from experiments.recoverability.controllers import BalanceAdapter
from experiments.recoverability.transition import SPEC


def classify(record):
    peak=record.get('safe_peak',record.get('progress_peak',record.get('supported_peak',record.get('peak',0.))))
    failure=record.get('first_failure')
    hard=not failure and record.get('max_speed',0)<=1.001 and record.get('max_joint',0)<.001 and record.get('max_torque',0)<=1.00001
    slip=record.get('loaded_slip',0);drift=record.get('loaded_drift',0)
    reasons=[]
    if failure and (failure.get('bodies') or failure.get('self_contacts')):reasons.append('collision')
    if not hard and 'collision' not in reasons:reasons.append('joint_or_actuator_or_prior_safety_failure')
    if peak<85:reasons.append('no_near_upright_arrival')
    if record.get('worst_front_gap',record.get('longest_contact_gap',0))>.08:reasons.append('support_gap')
    if record.get('total_front_gap',record.get('total_contact_gap',0))>.12:reasons.append('cumulative_support_gap')
    final=record.get('final') or {}
    com=record.get('terminal_com_x',final.get('com_support_x'))
    if com is not None and com<-.16:reasons.append('COM_behind_previous_progression_target')
    if 'recovery' in record and not record['recovery'].get('success',False):reasons.append('failed_balance_recovery')
    # All archived standing candidates in this study are below 85 degrees;
    # missing high-rate history is not fabricated to assert a five-second hold.
    return dict(safe_or_supported_peak=peak,loaded_slip=slip,loaded_drift=drift,
        slip_or_drift_rejection=bool(slip>.02 or drift>.04),other_rejections=reasons,
        solely_slip_drift_for_complete_handstand=bool(slip>.02 or drift>.04) and not reasons,
        full_handstand_proven=False,diagnostic_45_crossing=bool(hard and peak>=45))


def saved_world(c,assets,cfg,saved):
    w=World(assets,cfg);mujoco.mj_setState(w.m,w.d,saved['state'],SPEC);mujoco.mj_forward(w.m,w.d)
    w.target=saved['target'].copy();m=DualMetrics(w)
    # Retain physical safety history/anchors, but initialize paired hold clocks.
    for key in ('anchor','slip','drift','failed','first_failure','max_torque','max_speed','max_joint'):
        if key in saved['metrics']:setattr(m,key,saved['metrics'][key])
    adapter=BalanceAdapter(c,w)
    for key,value in saved.get('adapter',{}).items():setattr(adapter,key,value.clone() if torch.is_tensor(value) else value)
    w.monitor=m;m.observe(w.physical_row(),0.);return w,m,adapter


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);records=[]
    for path in Path('output/contact_transition_search_20261112_v1').glob('search_*.json'):
        for candidate in json.loads(path.read_text()):
            for index,r in enumerate(candidate['results']):
                records.append(dict(source=str(path),iteration=candidate['iteration'],candidate=candidate['candidate'],branch=index,**classify(r)))
    for directory,name in [('output/contact_transition_validation_20261112_v2','validation.json'),
                           ('output/configuration_final_replay_20261112_v1','heldout.json')]:
        path=Path(directory)/name
        for index,r in enumerate(json.loads(path.read_text())):records.append(dict(source=str(path),branch=index,**classify(r)))
    dump(a.output/'archived_standing_comparison.json',records)
    dump(a.output/'archive_summary.json',dict(records=len(records),
        standing_success_strict=sum(r['full_handstand_proven'] for r in records),standing_success_exploratory=0,
        solely_slip_drift_complete=sum(r['solely_slip_drift_for_complete_handstand'] for r in records),
        safe_45_crossings=sum(r['diagnostic_45_crossing'] for r in records),
        caveat='Stored summaries are the same recorded rollouts; below-85 trajectories cannot become complete handstands. No missing contacts are invented.'))
    _,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets');c=controller()
    try:
        audit=model_audit(c,World(assets,cfg));dump(a.output/'compatibility.json',audit);assert audit['compatible']
        results=[];region=Path('output/recoverability_maps_20261112')
        entries=json.loads((region/'validated_region.json').read_text());bank=np.load(region/'recoverable_states.npz')
        starts=[]
        for index in (0,3,6):
            w,m,adapter=bank_start(c,assets,cfg,entries[index],index,bank)
            starts.append((f'bank_{index}',compact_snapshot(w,m,adapter)))
        for path in sorted(Path('output/configuration_reverse_20261112_v1').glob('predecessor_*.pt')):
            starts.append((path.stem,torch.load(path,weights_only=False)))
        for name,saved in starts:
            for stochastic in (False,True):
                w,m,adapter=saved_world(c,assets,cfg,saved)
                r=balance(w,m,adapter,202640120+len(results),stochastic)
                results.append(dict(source=name,scope='balance-start; not standing-to-handstand',stochastic=stochastic,**r))
                csv_write(a.output/f'{name}_{int(stochastic)}.csv',m.trace)
                dump(a.output/'balance_start_comparison.json',results)
                print('archive balance',name,stochastic,r['strict_hold_seconds'],r['exploratory_hold_seconds'],r['hard_safe'],flush=True)
    finally:c.close()

if __name__=='__main__':main()
