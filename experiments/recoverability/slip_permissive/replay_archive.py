"""Replay saved native action arrays; assert every stored state is reproduced."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import curriculum
from experiments.morphology.learning.environment import Assets
from experiments.recoverability.run import KICK,dump,csv_write
from experiments.recoverability.slip_permissive.runtime import standing


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);net,_,cfg,_=curriculum.load_source(KICK);assets=Assets(a.output/'assets');rows=[]
    for path in sorted(Path('output/contact_transition_validation_20261112_v2').glob('*_actions.json')):
        seed=int(path.stem.split('_')[-3]);actions=json.loads(path.read_text());w,m,_=standing(net,assets,cfg,seed,.04,.012 if seed in (202620120,202620121) else .018)
        error=0.
        for frame in actions:
            w.step(np.array(frame['action']))
            error=max(error,float(np.max(abs(w.d.qpos-np.array(frame['qpos'])))),float(np.max(abs(w.d.qvel-np.array(frame['qvel'])))))
        assert error<1e-9,(str(path),error)
        r=m.results();r.update(source=str(path),seed=seed,max_state_error=error,scope='saved standing-start native commands')
        rows.append(r);dump(a.output/'comparison.json',rows)
        if len(rows)==1:csv_write(a.output/'baseline.csv',m.trace);torch.save(m.states,a.output/'baseline_motion.pt')
    dump(a.output/'summary.json',dict(rollouts=len(rows),max_state_error=max(r['max_state_error'] for r in rows),
         strict_success=sum(r['strict_success'] for r in rows),exploratory_success=sum(r['exploratory_success'] for r in rows)))

if __name__=='__main__':main()
