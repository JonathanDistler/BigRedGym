from pathlib import Path
import zipfile,json,hashlib
root=Path('.')
paths=set()
for dirname in ('gym','learning','resources','experiments/morphology','experiments/recoverability'):
 for p in Path(dirname).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',):paths.add(p)
for name in ('curriculum.py','setup.py','requirements.txt'):
 if Path(name).exists():paths.add(Path(name))
# Saved-source import utilities depend on scripts, and saved policy sources are required.
paths.update(p for p in Path('scripts').glob('*.py'))
for dirname in ('logs/go2balance/Oct31_23-29-27_', 'logs/go2backwards_reachability/Oct30_22-11-01_'):
 for p in Path(dirname).rglob('*'):
  if p.is_file() and ('files' in p.parts or p.name=='model_1500.pt' or p.suffix in ('.json','.yaml','.pkl')):paths.add(p)
paths.add(Path('output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt'))
for dirname in ('output/recoverability_maps_20261112','output/recoverability_switching_verified_20261112'):
 paths.update(p for p in Path(dirname).glob('*') if p.is_file())
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
with zipfile.ZipFile('output/recoverability_transition_molab_20261112_v3.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(paths):z.write(p,str(p))
 z.writestr('transition_source_manifest.json',json.dumps(manifest,indent=2))
 from importlib.metadata import version
 z.writestr('transition_requirements.txt',''.join(name+'=='+version(name)+'\n' for name in ('numpy','torch','mujoco','matplotlib','pytest','PyYAML','wandb','pillow')))
print(len(paths),Path('output/recoverability_transition_molab_20261112_v3.zip').stat().st_size)
