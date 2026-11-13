# Molab launch cells (Python 3.13 notebook, Python 3.11 child processes)

Upload `adc_front_capture_molab_v2.zip`. Run each block as a separate marimo Python
cell. Training happens in a subprocess; the notebook interpreter stays unchanged.
All paths below are relative to the extracted package. Do not run two training
cells against the same output directory concurrently.

## Setup cell

```python
import os, sys, subprocess, shutil, zipfile, pathlib, json, hashlib
archive = pathlib.Path('adc_front_capture_molab_v2.zip').resolve()
root = pathlib.Path('adc_front_capture').resolve()
if not root.exists():
    root.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            destination = (root / item.filename).resolve()
            if not destination.is_relative_to(root):
                raise ValueError('Unsafe archive path')
        bundle.extractall(root)
manifest = json.loads((root / 'adc_bundle_manifest.json').read_text())
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
child_env = dict(os.environ, WANDB_MODE='disabled', WANDB_DISABLED='true',
                 OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                 MUJOCO_GL='egl', UV_CACHE_DIR=str(root / '.uv-cache'),
                 UV_PYTHON_INSTALL_DIR=str(root / '.uv-python'))
child_env.pop('PYTHONPATH', None)
uv = shutil.which('uv')
if uv is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--target',
                    str(root / '.uv-bootstrap'), 'uv'], check=True)
    uv = str(root / '.uv-bootstrap' / 'bin' / 'uv')
subprocess.run([uv, 'python', 'install', '3.11'], env=child_env, check=True)
python = root / '.venv' / 'bin' / 'python'
if not python.exists():
    subprocess.run([uv, 'venv', '--python', '3.11', str(root / '.venv')], env=child_env, check=True)
subprocess.run([uv, 'pip', 'install', '--python', str(python), 'torch==2.13.0',
                '--index-url', 'https://download.pytorch.org/whl/cpu'], env=child_env, check=True)
subprocess.run([uv, 'pip', 'install', '--python', str(python), '-r',
                str(root / 'deployment/adc_front_capture/requirements.txt')], env=child_env, check=True)
def run(*args):
    subprocess.run([str(python), *map(str, args)], cwd=root, env=child_env, check=True)
prep = 'output/adc_front_capture_preparation_v1'
run('-m', 'pytest', 'tests/unit_tests/test_adc_front_capture.py', '-q')
```

## Benchmark cell

```python
run('scripts/benchmark_adc_front_capture.py', '--preparation', prep,
    '--output', 'output/molab_adc_benchmark', '--workers', '4')
selection = json.loads((root / 'output/molab_adc_benchmark/selection.json').read_text())
print(selection)  # Actual notebook CPU/affinity, throughput and memory; no GPU claim.
```

## Stage 1: bounded matched training cell

```python
# Re-run probes on the actual installed runtime before accepting elevated gains.
run('scripts/probe_adc_front_capture.py', '--preparation', prep,
    '--output', 'output/molab_adc_probe')
run('scripts/run_adc_front_capture.py', '--preparation', prep,
    '--output', 'output/molab_adc_stage1', '--updates', '30',
    '--num-envs', selection['num_envs'], '--workers', selection['workers'],
    '--horizon', '32', '--seeds', '42', '43',
    '--initial-stiffness', '67.5', '--probe', 'output/molab_adc_probe/probe.json',
    '--interaction-budget', '245760', '--max-seconds', '3300')
```

For 16 environments this is 122,880 training interactions across four conditions
and two seeds; for 32 environments, 245,760. Evaluation interactions are additional
and logged separately. `--max-seconds` stops at a complete matched update boundary.
The first several updates need not reach 45 degrees. Two evaluations below 50%
certified standing takeoff stop a run and save a checkpoint. Nonfinite PPO objectives
and failed exact restoration raise errors; restart from the last committed boundary.

## Resume / Stage 2 cell

```python
run_dir = root / 'output/molab_adc_stage1'
latest = sorted(run_dir.glob('checkpoint_*.pt'))[-1]
run('scripts/run_adc_front_capture.py', '--output', 'output/molab_adc_stage1',
    '--resume', latest.relative_to(root), '--updates', '100',
    '--interaction-budget', '819200', '--workers', selection['workers'],
    '--max-seconds', '3300')
```

`--updates 100` means **100 cumulative**, not 100 additional updates. Use 30 first;
continue toward 100–300 only after reviewing matched standing and intermediate
metrics and numerical stability. The runner preserves horizon, epochs, architecture,
RNGs, actor/critic, optimizer, lane dynamics and physical state. The target extension
is recorded in a new session budget file. Source/runtime mismatches refuse resume.

For nominal fine-tuning add `--nominal-finetune` to a continuation and extend the
cumulative target by an explicitly bounded number of updates. This switch persists
in later checkpoints. Evaluate and render under nominal dynamics regardless of
training gains. A larger multiseed Stage 3 needs a new explicit interaction budget
and evidence of improved collision-free progression/recovery; no automatic launch.

## Visualize / report cell

```python
latest = sorted(run_dir.glob('checkpoint_*.pt'))[-1]
run('scripts/render_adc_front_capture.py', latest.relative_to(root),
    '--output', 'output/molab_adc_video', '--seed', '96000')
run('scripts/report_adc_front_capture.py', latest.relative_to(root),
    '--output', 'output/molab_adc_report')
import marimo as mo
mo.image((root / 'output/molab_adc_video/comparison.gif').read_bytes())
```

Each GIF has exact MuJoCo geometry, COM/contact markers, full state NPZ, CSV and
500 Hz physics trace. All panels share the same initial standing seed and control
time; an ended episode holds its final frame and is explicitly labelled.

## Complete backup cell

```python
latest = sorted(run_dir.glob('checkpoint_*.pt'))[-1]
backup = pathlib.Path('adc_complete_backup.tar.gz')
run('scripts/backup_front_catch.py', 'output/molab_adc_stage1', '--all',
    '--checkpoint', latest.relative_to(root), '--bundle', archive,
    '--extra', 'output/molab_adc_video', '--extra', 'output/molab_adc_report',
    '--extra', 'output/molab_adc_probe', '--extra', 'output/molab_adc_benchmark',
    '--output', backup)
mo.download((root / backup).read_bytes(), filename=backup.name,
            label='Download complete source, data, checkpoints and results')
```

For a downloaded **old** combined Molab backup, import explicitly:

```python
run('scripts/import_adc_checkpoint.py', '/path/to/downloaded_backup.tar.gz',
    '--member', 'output/old_run/checkpoint_000010.pt',
    '--output', 'output/imported/checkpoint_000010.pt')
# Start a NEW matched experiment with --initial output/imported/checkpoint_000010.pt
# --initial-mode B_seed43. Re-run probes against that exact actor first.
```

Importing old policy weights is a disclosed new experiment; exact continuation of
old code requires its archived source. No local checkpoint_000010.pt is assumed.

To continue a selected **earlier** candidate, explicitly fork its full checkpoint:

```python
run('scripts/fork_adc_run.py', 'output/molab_adc_stage1/checkpoint_000010.pt',
    '--output', 'output/molab_adc_selected10')
run('scripts/run_adc_front_capture.py', '--output', 'output/molab_adc_selected10',
    '--resume', 'output/molab_adc_selected10/checkpoint_000010.pt',
    '--updates', '100', '--interaction-budget', '819200',
    '--workers', selection['workers'], '--max-seconds', '3300')
```

This preserves all conditions and their actor/critic/Adam/RNG/world states at the
same interaction boundary. The source/runtime contract still applies. It does not
allow a candidate to receive extra uncounted training experience.
