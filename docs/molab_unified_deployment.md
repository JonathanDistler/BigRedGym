# Unified Molab deployment

This package changes deployment infrastructure only. The CPU MuJoCo model,
`anticipatory_v1` rewards, PPO update/collection implementation, observation/action
architecture, actor initialization and preparation data are copied unchanged.
The notebook kernel can remain Python 3.13 at `/tmp/uv-venv/bin/python`.

## Upload and setup

Upload `output/bigredgym_molab_complete.zip`. Paste the contents of
`MOLAB_START.py` into a Molab cell and run it. The source ZIP includes the same
cell plus a simplified three-cell notebook at
`notebooks/molab_unified_launcher.ipynb`.

The standard-library-only cell searches the current upload directory, `/marimo`
and `/tmp` to depth four for the exact ZIP filename. Different ambiguous ZIPs
produce an actionable error; `os.environ['BIGREDGYM_BUNDLE']` selects an explicit
path. Extraction goes to `/marimo/bigredgym_molab`, or the absolute directory
specified by `BIGREDGYM_PROJECT`. It rejects unsafe paths and checks every file
against the packaged SHA-256 manifest. Existing identical files remain untouched;
different source files produce an error and require a new project directory.
User outputs, previous checkpoints and caches are never deleted by setup.
Repeated setup checks installed package versions first and skips index resolution
when the pinned environment is already present; validation still runs.

The bootstrap finds uv on PATH or `/usr/local/bin/uv`, downloads managed
CPython **3.11.14** into the project, and creates `.venv311` without activation or
system-site packages. It installs the fully pinned 52-package CPU dependency
lock, checks dependencies and imports, loads the preserved checkpoint, instantiates
the actual MuJoCo Go2 model, validates preparation data and runs focused tests.
All model/training/report processes use the absolute `.venv311/bin/python` path;
the notebook interpreter is not replaced or asserted to be Python 3.11.

Package downloads require network access. If uv is unavailable, the setup error
provides an installation command. The package does not contain Python binaries
or wheels. It supports Linux CPU training; GPU/MJX is outside this deployment.

## Training and equal budgets

Paste `MOLAB_TRAIN.py` in a second cell. Editable values are NUM_ENVS, WORKERS,
UPDATES, HORIZON, SEEDS, MAX_RUNTIME_SECONDS and NEW_RUN. Defaults reproduce:

```text
/marimo/bigredgym_molab/.venv311/bin/python scripts/run_front_catch.py
  --suite anticipatory_v1
  --preparation output/front_catch_preparation_20261112_v3
  --output output/front_catch_anticipatory_molab_v1
  --num-envs 16 --workers 4 --updates 10 --horizon 32
  --seeds 42 43 --eval-every 2 --checkpoint-every 2 --max-seconds 1200
```

The supervisor executes that command with the project as cwd and W&B disabled,
adds automatic resumption, periodic backups, measured reports and final export.
One actor/critic per condition and training seed is shared across its parallel
worlds. A=fixed support, B=earlier placement, C=earlier placement plus rear sweep.

| Worlds | Horizon at default matched budget | Training steps/condition/seed |
|---|---|---|
| 8 | 64 | 5,120 |
| 16 | 32 | 5,120 |
| 32 | 16 | 5,120 |
| 64 | 8 | 5,120 |

The cell treats HORIZON as the reference horizon at 16 worlds. Changing world
count therefore preserves 512 interactions per update and 30,720 total for
three conditions, two seeds and ten updates. Evaluation transitions are counted
separately. The unchanged trainer's 20-update/200k-transition pilot cap remains.
The runtime limit stops only between complete matched rounds; resume to finish
the planned budget. Higher world counts are optional and depend on actual CPU
and memory capacity, not a promise of better throughput.

The historical trainer CLI accepts only 16/32 worlds. For 8/64, the separate
`deployment/molab/entrypoint.py` expands only that argument choice at invocation.
For eight training worlds it also creates a separate 16-slot evaluation vector,
because the original evaluator runs nine distinct held-out episodes. The training
world count and budget remain eight. No algorithm, rewards, actuator constraints,
physics or source files are changed by this adapter. Standard 16/32-world runs
invoke `scripts/run_front_catch.py` directly.

Set BENCHMARK_FIRST=True for an optional isolated CPU benchmark of 8/16/32/64.
It uses equal 128-transition warmups and equal 512-transition collection/PPO
batches for every count, records throughput/RSS/PPO update time, and suggests the
smallest count within 5% of fastest. It does not change the requested training
count automatically. Benchmark optimizer updates are private timing probes and
are never used as actor initialization. Choose a count/worker value supported by
the measured host; aggregate RSS may double-count shared pages.

## Resume and checkpoints

The supervisor records `.molab/active_run.json` and an immutable configuration
snapshot per run. Repeating the training cell resumes the active run from the
latest complete checkpoint. A completed run prints `RUN ALREADY COMPLETE` and
never starts at iteration zero. Changed environment count, horizon, seed or total
update budget is rejected on resume; use NEW_RUN=True for a distinct run.

Full native checkpoints contain actor/critic tensors, optimizer state, per-run
Torch RNG, global NumPy/Torch RNG, per-environment RNG, MuJoCo integration and
warmstart state, phase/reward credit state, curriculum and committed history.
The existing trainer has no learning-rate/PPO scheduler, so there is no separate
scheduler state to save. Source/input/runtime version checks remain enforced.

The existing trainer atomically publishes exclusive checkpoint names. `.tmp`
files are ignored. A corrupt committed `.pt` is preserved under a unique
`quarantine/` filename before selecting an earlier valid checkpoint. An attempt
with no complete checkpoint is preserved and a uniquely named restart directory
is used; it is never silently wiped. Process-level file locks prevent concurrent
setup or duplicate training supervisors.

To resume without editing any configuration:

```python
import subprocess
subprocess.run([
    '/marimo/bigredgym_molab/.venv311/bin/python',
    '/marimo/bigredgym_molab/deployment/molab/workflow.py',
    'train', '--resume-only', '--periodic-backups', '--export-directory', '/marimo'
], cwd='/marimo/bigredgym_molab', check=True)
```

## Reports, export and restoration

Paste `MOLAB_BACKUP.py` to generate fresh measured reports and an immutable ZIP
under `/marimo/bigredgym_backup_<UTC>_<unique>.zip`. A download link/path is printed.
It contains all new full checkpoints, standing/intermediate evaluation metrics,
CSV and PNG/PDF figures, effective configurations/seeds, exact commands, source
revision/checksums and installed dependency/interpreter metadata. Sources,
preparation data and preserved initialization are included for a reproducible
restore; `.venv311`, Python caches, unrelated historical runs and unfinished
`.tmp` files are excluded. The export manifest hashes the actual archived bytes.

Training also emits periodic recovery ZIPs containing the latest full checkpoint
and committed output/source metadata, followed by a full all-checkpoint export
when the process stops. Older checkpoints/exports are not deleted or overwritten.
Download the latest full backup before ending the session. A hard session/storage
termination can lose anything that was not downloaded; an upload-only ZIP cannot
provide remote persistent storage by itself.

To restore, upload the backup, extract its `project/` contents into the same
project layout, run `deployment/molab/bootstrap.py` using the notebook interpreter,
and invoke the resume-only command above. Relative output/checkpoint/trace paths
are preserved. The virtual environment is recreated rather than copied. Use
`workflow.py restore-paths --run-dir output/<run>` to rebuild the active pointer
if needed. Preserve the original run subdirectory name and source version; do not
edit checkpoint bytes to migrate paths.

## Validation record

Local validation is performed by building the ZIP, extracting it into a fresh
`/tmp` project, invoking its bootstrap without the repository on PYTHONPATH,
checking all manifest hashes/imports/assets and loading the actor. The deployment
focused tests cover setup/idempotency helpers, matched budgets, immutable exports,
corrupt checkpoint handling and the existing front-catch correctness checks.
A one-update matched A/B/C smoke uses seeds 42/43, saves full checkpoints, verifies
resume without restarting at zero, generates reports and validates the exported
ZIP. Machine-readable run/environment/validation metadata accompanies the output.
No extended local PPO run or Molab/cloud run is part of setup validation.
