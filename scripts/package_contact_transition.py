"""Portable source/checkpoint/result bundle; refuses to overwrite an archive."""
import argparse
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--results',type=Path,action='append',default=[]);a=p.parse_args()
    paths=set()
    for dirname in ('gym','learning','resources','experiments/morphology','experiments/recoverability'):
        paths.update(f for f in Path(dirname).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc')
    paths.update(f for f in Path('scripts').glob('*.py'))
    for name in ('curriculum.py','setup.py','requirements.txt','docs/go2_contact_transition_20261112.md','notebooks/molab_contact_transition.ipynb'):
        if Path(name).is_file():paths.add(Path(name))
    for directory in ('logs/go2balance/Oct31_23-29-27_','logs/go2backwards_reachability/Oct30_22-11-01_'):
        paths.update(f for f in Path(directory).rglob('*') if f.is_file() and ('files' in f.parts or f.name=='model_1500.pt' or f.suffix in ('.json','.yaml','.pkl')))
    paths.add(Path('output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt'))
    paths.update(f for f in Path('output/recoverability_maps_20261112').glob('*') if f.is_file())
    for directory in a.results:
        paths.update(f for f in directory.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(paths)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(paths):z.write(f,str(f))
        z.writestr('contact_transition_bundle_manifest.json',json.dumps(manifest,indent=2))
        z.writestr('contact_transition_requirements.txt',''.join(n+'=='+version(n)+'\n' for n in ('numpy','torch','mujoco','matplotlib','pytest','PyYAML','wandb','pillow')))
    print(json.dumps(dict(files=len(paths),bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))

if __name__=='__main__':main()
