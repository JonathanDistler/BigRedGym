"""Prepare bounded diagnostic learning configuration; does not launch training."""
import argparse,json
from pathlib import Path
from experiments.recoverability.run import dump


def eligibility(rows):
    valid=[r for r in rows if r['hard_safe'] and r['safe_supported_peak']>=45 and r['exploratory_entry']]
    return len(valid)>=.75*len(rows) and bool(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=json.loads(a.evaluation.read_text());ready=eligibility(rows)
    dump(a.output/'bounded_pilot.json',dict(enabled=ready,run_automatically=False,updates_run=0,
         smoke=dict(updates=2,requires=['checkpoint/model parity','full integrated demonstration states','no illegal collisions','native motor compliance','independent reset validation']),
         pilot=dict(updates=10,requires=['smoke finite gradients/loss','no safety regression versus matched frozen baseline','held-out progression nonregression']),
         dense_reward=dict(pitch_progress=1.,COM_error=-2.,angular_rate_error=-.1,clearance_violation=-10.,recoverable_arrival=10.,loaded_slip=0.,foot_drift=0.),
         forbidden=['floating-base projection','invalid-collision demonstrations','extended PPO'],
         reset_source='Exact full integration/controller snapshots from admitted collision-free references only',
         reason='Needs at least 75% admissible complete trajectories with supported 45-degree progression and retained non-slip contact conditions. This is an opt-in diagnostic gate, not proof of a handstand.'))

if __name__=='__main__':main()
