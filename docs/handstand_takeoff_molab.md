# Molab standing-takeoff workflow

The original 5.014-second reference is unchanged. The new student executes a
continuous, certified standing-to-takeoff prefix with actual state feedback.
It has **not** learned the complete handstand, and PPO has not demonstrated a
success-rate improvement over its retained nominal demonstration prior.

The current experiment stops at 10 cumulative PPO updates. No 50-, 200-, 500-
or 5000-update training run was launched. The prepared notebook permits an
explicitly started pilot up to 50. A 200–500 continuation command is provided
below as an opt-in option, but is not recommended on this unchanged objective.

## Evidence and diagnosis

The old student's observation ordering, fixed mean/std normalization, 12-joint
FL/FR/RL/RR ordering, action scale 2, 50 Hz control, 500 Hz integration and PD
limits match the reference. Old imitation action RMS error is 0.00662 overall
and 0.00783 in the first 24 steps. In its own rollout the maximum normalized
command discrepancy reaches 0.87; torque reaches its real configured bound and
rear-calf collisions end the episode at 0.46 s. This is measured covariate shift
and insufficient corrective coverage, not a demonstrated joint-map bug.

The old dataset has only 320 correlated nominal samples; 48 cover the first
0.96 s. Long near-vertical coverage dominates the loss. Static action MSE did
not constrain stabilizing feedback derivatives. No old simulator PPO training
had occurred, so its failure cannot be attributed to observed PPO optimization.
The old reward gave nearly +1 per standing control step and no explicit powered
takeoff event. Its Gaussian latent std ~0.0498 would perturb position targets
by approximately 0.10 rad before tanh/slew limiting. That exploration scale is
large relative to this precise maneuver; it was not experimentally swept here.

The successful transfer retains the exact nominal actions as an explicit prior,
and learns a phase-conditioned affine neural feedback residual on 42 sensor
channels. The inputs remain the existing 43-channel sensor vector: gravity,
body-frame gyro, joint offsets/normalized speeds, previous filtered target and
phase. Residuals are limited to ±0.05 normalized action (±0.10 rad before target
slew limiting). Gravity/target/joint deltas use fixed scales 0.02; gyro uses 0.1.
No COM, simulator contacts or future measurements enter this actor. Critic and
reward use simulation information. The target/speed/torque/collision constraints
and the strict handstand gate are unchanged.

For the teacher, reuse the preserved projected trajectory feedback controller.
It failed as a full handstand controller, but its first 48 control steps passed
16/16 standing-start perturbation trials. Each label is the actual bounded
feedback command on that visited state. Only continuous safe prefixes with the
original powered takeoff certificate are retained; these are takeoff
correction labels, not handstand recovery labels. The training reset bounds are
joint position ±0.002 rad and joint velocity ±0.02 rad/s; invalid resets are
rejected by the existing environment. No mid-episode resets are hidden.

BC trains 200 epochs on these 768 verified state/action pairs. At exact nominal
observations the feedback delta is zero, preserving the demonstration prior.
A gyro-offset mechanism test (+0.1 rad/s) changes actions by up to 0.00678;
disabling residual feedback gives zero response. This establishes state
dependency, not physical disturbance recovery. The prior-only ablation also
passes the small held-out takeoff family, so no robustness superiority is claimed.

The first phase is explicitly limited to 48 steps. For a full-maneuver test,
phase clamps at the last trained takeoff state and live feedback continues.
It never silently advances through recorded balance outputs. The saved original
balance actor remains in `artifacts/balance.pt` for later capture/balance work;
it is not deployed after the prefix.

## Measured local results

Generated tables and individual rollouts are in ignored outputs:

- `output/takeoff_diagnosis/diagnose.json`
- `output/takeoff_bc/bc.json`, plus `demonstrations.npz`
- `output/takeoff_old_student_eval/evaluate.json`
- `output/takeoff_verified_smoke/resume_verification.json`
- `output/takeoff_pilot/train.json` and `curve.json`
- `output/takeoff_pilot_eval/evaluate.json`
- `output/takeoff_full_eval/evaluate.json`
- `output/takeoff_benchmark/benchmark.json`
- `output/takeoff_report.md`

BC passes nominal plus 4/4 held-out takeoff starts. PPO retains nominal and both
fixed validation seeds at all 10 updates; expanded evaluation passes nominal
plus 8/8 held-out starts. Held-out reset bounds are ±0.003 rad and ±0.03 rad/s,
with seeds 90111–90118, disjoint from training demonstrations. This is a narrow
family and a small sample. Nominal simultaneous rear-foot clearance reaches
0.646 m, peak pitch is 59.63 degrees by 0.96 s, both rear contacts are lost,
front capture is detected, and illegal collision count is zero. The original
powered rear-release certificate is required, not just airborne feet.

All three full-continuation trials certify takeoff first but later collide with
rear calves at about 1.88 s. Full handstand success is 0/3, strict hold 0 s.
The per-episode `takeoff_success` field includes safety through the chosen
horizon, so those later collisions make it false in full-continuation reports;
`certified_takeoff` retains the earlier physical event.

The real MuJoCo smoke performs update 1–2, restarts the Python process, restores
and continues to update 3, then compares against uninterrupted update 1–3.
Policy, optimizer, Python/NumPy/Torch RNGs, observation buffers, curriculum and
selected best state match exactly. A regression test separately verifies that
selected weights are embedded without recursively embedding checkpoints.
Checkpoints are saved atomically every update. Resetting all training worlds at
update boundaries makes restart exact without an unsaved partial simulation.
Partial rollout episodes are bootstrapped for GAE and never counted as successful
full episodes. Updates 1–5 randomize joints ±0.001 rad/±0.01 rad/s; later updates
use ±0.002/±0.02. There are no intermediate starts in this pilot.

New optimized reward components per step are tracking `0.02*exp(-20*q_MSE -
20*quaternion_error)`, clearance up to +0.02 with certified takeoff/capture,
+2 once for newly certified takeoff, -2 on safety failure, and
`-0.001*mean(residual_action**2)`. The legacy reward is diagnostic only. In a
64-transition update, tracking sums to about 1.28, clearance 0.30 and event
reward 2.0. This makes the powered event material, with a dense tracking signal
before it. No reward is inferred from floating feet after an illegal collision.

PPO uses four epochs, clip 0.2, gamma 0.99, GAE lambda 0.95, Adam 1e-4,
minibatches 128 and gradient-norm cap 1.0. The exploration std acts on the bounded
residual, reducing initial joint-target noise to roughly 0.005 rad. Rewards,
observations, applied torques, losses and gradient norms are checked for finite
values. Final critic loss is about 0.865, approximate KL 0.00396, and clipping
fraction 0.00391. Gradient norms are recorded before clipping; a value above 1
is not evidence that clipping failed. No torque ratio exceeds 1.0. Some motors
saturate during the reference itself; thermal/current/landing limits remain
unqualified for hardware.

Local CPU measurements: serial environment counts 1/4/16, rollout lengths
64/128, 107–114 simulation transitions/s, ~99% of one CPU core, peak RSS
745–757 MiB. Full PPO including fixed validation costs about 31 training
transitions/s. The pilot's RSS reaches about 756 MiB; the final smoke with
embedded best-state persistence reaches about 759 MiB. These are process peaks,
not whole-WSL consumption. Counts are serial native MuJoCo worlds with batched
actor inference, not independent parallel workers. Hold-only kinetic-energy
scoring is skipped during 0.96-second takeoff training; the original 500 Hz
strict checks remain active. These numbers are not comparable to a GPU backend
or the earlier full-metric 54.9 transitions/s without accounting for that work.

**Molab throughput is NOT MEASURED.** No connected Molab runtime or relevant
plugin was available. The notebook writes fresh `output/molab_benchmark` and
training measurements on the user's runtime. Its graph and idle execution are
validated locally; cloud package installation and service quotas are not tested.

## Upload, extract and install

Upload `notebooks/molab_handstand_takeoff.py` as a notebook in Molab. In its first
section upload `artifacts/handstand_takeoff_bundle.tar.gz` and paste the digest
from the adjacent `.tar.gz.sha256` file. Click the steps in order. Extraction
verifies the outer digest, all file hashes and the immutable reference, rejects
unsafe paths and links, and is repeatable without overwriting a prior extraction.

The notebook installs a separate pinned Python 3.11 environment using
`uv python install 3.11` and `uv sync --frozen --no-dev --python 3.11`. It removes
notebook-owned VIRTUAL_ENV, Conda, PYTHON* and UV_* selectors from subprocesses.
Every command runs from the extracted source root. The existing `uv.lock`
remains authoritative: MuJoCo 3.11.0, Torch 2.13.0, NumPy 2.4.6.

The notebook uses the documented marimo
[run buttons](https://docs.marimo.io/api/inputs/run_button/),
[file upload](https://docs.marimo.io/api/inputs/file/) and
[downloads](https://docs.marimo.io/api/media/download/). Expensive work is
button-gated and synchronous with a 15-minute subprocess timeout; no training
runs automatically on notebook import. Current Molab documentation describes
4 CPU/32 GB default compute and shutdown after a maximum 12-hour session or
90-minute idle interval; check the runtime UI for current limits. Download
checkpoints before leaving the session. Native CPU MuJoCo does not require a GPU.
See the [official Molab guide](https://docs.marimo.io/guides/molab/).

Terminal alternative after uploading the archive and sidecar:

```bash
sha256sum -c handstand_takeoff_bundle.tar.gz.sha256
mkdir -p go2_takeoff
# Only extract the verified, trusted bundle.
tar -xzf handstand_takeoff_bundle.tar.gz -C go2_takeoff
cd go2_takeoff
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path.cwd()
m = json.loads((root / 'bundle_manifest.json').read_text())
for name, digest in m['files'].items():
    path = (root / name).resolve()
    assert path.is_relative_to(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
env = {k: v for k, v in os.environ.items()
       if k not in {'VIRTUAL_ENV','CONDA_PREFIX','PYTHONPATH','PYTHONHOME',
                    'PYTHONEXECUTABLE'} and not k.startswith('UV_')}
env.update(UV_PYTHON='3.11', UV_PROJECT_ENVIRONMENT=str(root / '.venv'))
subprocess.run(['uv','python','install','3.11'], cwd=root, env=env, check=True)
subprocess.run(['uv','sync','--frozen','--no-dev','--python','3.11'],
               cwd=root, env=env, check=True)
PY
export HANDSTAND_PYTHON="$PWD/.venv/bin/python"
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use a server-backed Molab runtime, not a browser-only WebAssembly notebook.
No repository cloning, origin/main checkout, PR mutation, SDK installation or
hardware connection is involved.

## Exact commands

All commands below run from the extracted source root. Locally, substitute the
existing Python with:
`export HANDSTAND_PYTHON=/home/jonathandistler/Distler_Gym/BigRedGym/.venv/bin/python`.
The takeoff CLI is newly implemented and tested; these are real supported
arguments. The nominal replay command retains its original interface.

```bash
# Immutable regression:
"$HANDSTAND_PYTHON" scripts/replay_go2_handstand.py \
  --trajectory artifacts/handstand_reference/trajectory.npz \
  --trajectory-sha256 21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92

# Diagnose the original student; rebuild corrective BC if desired:
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py diagnose \
  --checkpoint artifacts/baseline_student.pt --output output/diagnosis
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py bc \
  --teacher artifacts/takeoff_teacher.npz --demonstrations 16 --epochs 200 \
  --trials 4 --output output/bc
# The bundle's artifacts/student.pt is the already validated BC warm start.

# Two simulator updates, restart/resume to cumulative 3, exact comparison:
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint artifacts/student.pt --updates 2 --envs 1 --rollout-steps 64 \
  --trials 2 --output output/smoke
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/smoke/latest.pt --resume --updates 3 --envs 1 \
  --rollout-steps 64 --trials 2 --output output/smoke
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint artifacts/student.pt --updates 3 --envs 1 --rollout-steps 64 \
  --trials 2 --output output/continuous
"$HANDSTAND_PYTHON" scripts/verify_handstand_checkpoint.py \
  --left output/smoke/latest.pt --right output/continuous/latest.pt \
  --output output/smoke/resume_verification.json

# Measure this host, then attempt the bounded pilot:
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py benchmark \
  --checkpoint output/smoke/latest.pt --env-counts 1 4 16 --lengths 64 128 \
  --output output/benchmark
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/smoke/latest.pt --resume --updates 10 --envs 1 \
  --rollout-steps 64 --trials 2 --output output/pilot

# Held-out standing takeoff and separate full-continuation failure test:
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py evaluate \
  --checkpoint output/pilot/best.pt --trials 8 --output output/takeoff_eval
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py evaluate \
  --checkpoint output/pilot/latest.pt --trials 2 --full-maneuver \
  --output output/full_eval

# Optional continuation up to cumulative 50 (not run in this session):
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/pilot/latest.pt --resume --updates 50 --envs 1 \
  --rollout-steps 64 --trials 2 --output output/pilot

# Prepared 200-update option; NOT recommended on the unchanged objective.
# Explicit opt-in plus healthy simulator checkpoint is required. Do not run by default.
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/pilot/latest.pt --resume --updates 200 --envs 1 \
  --rollout-steps 64 --trials 2 --allow-long-takeoff-continuation \
  --output output/continuation
```

The source config `configs/handstand_takeoff_continuation.json` records the
200-update proposal and remaining capture-prefix gate. It is not evidence that
this longer run happened. The CLI caps any continuation at 500; 5000 is not
available in this takeoff-only workflow.

For the existing local downloadable checkpoint, use
`output/takeoff_portable/latest.pt` (cumulative update 10) or
`output/takeoff_portable/best.pt` (selected update 1). The latter is retained
because later checkpoints tie on the current success-first selection metric;
no reward-only best checkpoint is substituted. Export embeds the previously
selected full checkpoint without changing numerical actor/optimizer/RNG state.
After a runtime restart, re-extract/install and validate smoke/benchmark, upload
your trusted `.pt` through the notebook, verify its hash, then paste the restored
path into the resume field. Keep seed 4110, envs 1 and rollout length 64 for
exact continuation. Changing counts/lengths requires a fresh validation run.

## Reports and portable assets

Regenerate the current local report:

```bash
"$HANDSTAND_PYTHON" scripts/report_handstand_takeoff.py \
  --bc output/takeoff_bc/bc.json --pilot output/takeoff_pilot/train.json \
  --baseline output/takeoff_old_student_eval/evaluate.json \
  --evaluation output/takeoff_pilot_eval/evaluate.json \
  --full-evaluation output/takeoff_full_eval/evaluate.json \
  --benchmark output/takeoff_benchmark/benchmark.json \
  --output output/takeoff_report.md
```

Rebuild the upload bundle from this experimental checkout:

```bash
"$HANDSTAND_PYTHON" scripts/package_handstand_robustness.py \
  --reference artifacts/handstand_reference/trajectory.npz \
  --checkpoint output/takeoff_bc/bc.pt \
  --baseline-checkpoint output/bc_sensor_pilot/bc.pt \
  --feedback output/feedback_corrected/controller.npz \
  --teacher /home/jonathandistler/Distler_Gym/BigRedGym/output/retained_go2_preparation_v3/balance.pt \
  --output artifacts/handstand_takeoff_bundle.tar.gz
```

The old `artifacts/handstand_robustness_bundle.tar.gz` is preserved. The new
bundle reuses its pinned source/assets and adds the takeoff workflow, validated
corrective teacher, original baseline, BC warm start and notebook. Large data,
results and checkpoint archives stay ignored, not in source history.

Downloadable complete PPO checkpoints are also provided separately as
`artifacts/handstand_takeoff_checkpoints.tar.gz` with its SHA256 sidecar. This
archive contains latest update 10, selected best update 1, full optimizer/RNG
state, curves, individual evaluation results and the resume-verification receipt.
The inner `output/takeoff_portable/export.json` verifies checkpoint hashes.
Extract it into the source root and use the preserved relative checkpoint paths.
Reproduce the download archive locally with:

```bash
tar -czf artifacts/handstand_takeoff_checkpoints.tar.gz \
  output/takeoff_portable output/takeoff_pilot/train.json \
  output/takeoff_verified_smoke/resume_verification.json \
  output/takeoff_pilot_eval/evaluate.json output/takeoff_full_eval/evaluate.json \
  output/takeoff_benchmark/benchmark.json output/takeoff_report.md
(cd artifacts && sha256sum handstand_takeoff_checkpoints.tar.gz > handstand_takeoff_checkpoints.tar.gz.sha256)
```

## Stop/go decision

Takeoff transfer and real simulator PPO persistence now work. Current evidence
supports a small next-stage experiment, not a long budget on the same solved
prefix. The smallest useful corrective change is to validate corrective teacher
rollouts through roughly 1.28 s, add that physically continuous capture segment
to the student prior/labels, and extend the training horizon only after it
passes. The present controller is explicitly clamped to its 0.96 s prefix.
A stronger velocity-reset or push benchmark should demonstrate a feedback
benefit before asserting robustness. No hardware readiness or full autonomous
handstand success is claimed, and no robot commands are sent.
