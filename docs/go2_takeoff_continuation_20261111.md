# Shared BC PPO continuation, updates 10–30

The source experiment is `output/multimorph_bc_ppo_retention_20261111_v1`.
Continuation artifacts are in `output/multimorph_bc_ppo_continuation_20261111_v2`.
The original checkpoints, data, and results remain untouched. The first attempt
(`...continuation_20261111_v1`) stopped before any training: the existing saver
cannot write an empty trace when a restored checkpoint was already evaluated.
The continuation wrapper copies the update-10 checkpoint and probes it without
calling that saver again. New updates use the existing saver normally.

## Protocol and interpretation

Both methods resume updates 11–30, seeds 42/43, 1,024 training transitions per
update, eight worlds, identical rewards, action interfaces, actor/critic sizes,
PPO parameters and BC data. The checkpoint protocol is passed unchanged and its
equality is checked by the existing restore method. `base_training_steps` in that
protocol describes the original ten-update experiment; the bounded wrapper
explicitly executes twenty further updates without changing that metadata.
Every five updates, both methods receive the same full-scale reset/noise seeds
58000–58007, deterministic and stochastic, with goal termination disabled and
the original eight-second maximum horizon. Current curriculum geometry is also
probed. Evaluations do not consume training RNG. These are development resets,
not a new independent final test family.

The unchanged safety stops are KL > .03, clipping > .30, or deterministic
full-scale takeoff more than 25 percentage points below baseline for three
successive evaluation points. Partners stop together if a stop is triggered.

BC coefficient for update k uses the saved schedule at iteration k−1:
`10*(1-min((k-1)/20,1)) + 2*min((k-1)/20,1)`. Thus update 11 uses 6,
update 20 uses 2.4, and updates 21–30 use 2. It never drops abruptly to zero.
Action drift is measured on the original BC training frames relative to the
original BC actor, not relative to the update-10 actor. Demo latent differences
use the same fixed .05 scale throughout. There are no learned running
observation/reward normalization statistics in this implementation; fixed
observation transformations and the PPO minibatch advantage standardization
remain unchanged.

Actor/critic tensors, Adam tensors/parameter groups and Torch RNG are checked
against each source checkpoint in `restore_audit.json`. Existing restore also
loads NumPy RNG, episode counts, observations, curriculum windows and complete
MuJoCo integration/tracker state. Morphology changes happen at episode reset;
they do not reconstruct the network or optimizer. Worlds finish their latched
morphology episodes before changing, so interaction counts come from the actual
per-transition scale counters, not simply from promotion timestamps.

Promotion remains two passing 32-episode windows, at least 30% qualifying and
8,192 interactions at the stage. Certified takeoff plus 20°/.02 s is required
at 90%; 35°/.04 s at 95%, with the existing speed/joint-limit checks. No forced
promotion or extra budget is applied. If 100% is reached late, meaningful
full-scale training must be reserved in a separately authorized larger budget;
this twenty-update continuation does not silently extend it.

`probes.csv` contains all existing episode metrics. `rotation_diagnostics.csv`
independently replays every saved checkpoint at true full scale using the same
seeds/noise. Added diagnostics are evaluation-only and sample each physics step:
pitch velocity and world-y subtree angular momentum at certified liftoff and
peak body pitch; COM x displacement from the current front-foot midpoint;
front support at peak; RMS target-to-actual joint error; first nonfoot contact
time. Momentum units are kg m²/s, impulse Ns, angles rad, velocity rad/s,
tracking rad, COM/clearance m and times s. Liftoff means the existing certificate
time, not an independently redefined contact event. No-takeoff liftoff values
are missing, not zero. First-contact time −1 means no nonfoot collision.
The useful impulse is the unchanged certificate's minimum rear-leg impulse;
it is not a measured total external angular impulse. Episode peaks can occur
after useful flight; compare them with `genuine_flight_pitch` and duration.

Figures use actual cumulative training transitions, with individual training
seeds and mean ± one sample standard deviation across the two seeds. Eight
resets per seed cannot establish a statistically reliable method advantage.
In the rotation diagnostic figure, dashed fine curves show seed 42 and dotted
fine curves show seed 43; thick curves show their mean.

## Exact local reproduction

Run from the repository root, using a new output directory:

```bash
env -u PYTHONPATH WANDB_MODE=disabled .venv/bin/python scripts/continue_takeoff_retention.py \
  --source output/multimorph_bc_ppo_retention_20261111_v1 \
  --output output/multimorph_bc_ppo_continuation_REPRO \
  --start-update 10 --additional-updates 20

env -u PYTHONPATH .venv/bin/python scripts/diagnose_takeoff_continuation.py \
  --run output/multimorph_bc_ppo_continuation_REPRO
env -u PYTHONPATH MPLCONFIGDIR=/tmp/takeoff-mpl .venv/bin/python scripts/report_takeoff_retention.py \
  --run output/multimorph_bc_ppo_continuation_REPRO --step-axes
env -u PYTHONPATH MPLCONFIGDIR=/tmp/takeoff-mpl .venv/bin/python scripts/report_takeoff_continuation.py \
  --run output/multimorph_bc_ppo_continuation_REPRO
```

## Molab Online commands

Upload `takeoff_continuation_molab_bundle.zip`. These commands prepare a fresh
checkout and verify every overlaid source/data/checkpoint file before use.
Run shell blocks in a terminal, or with `subprocess.run(["bash", "-lc", SCRIPT],
check=True)` in a manually executed notebook cell. Nothing launches on import.

```bash
git clone --branch ppo_kickup_e2e --single-branch https://github.com/JonathanDistler/BigRedGym.git BigRedGym_continuation
cd BigRedGym_continuation
python - <<'PY'
import hashlib, json, pathlib, subprocess, zipfile
root = pathlib.Path.cwd()
with zipfile.ZipFile('../takeoff_continuation_molab_bundle.zip') as z:
    manifest = json.loads(z.read('molab_manifest.json'))
    subprocess.run(['git','checkout','--detach',manifest['base_commit']], check=True)
    for name, digest in manifest['files'].items():
        target = (root / name).resolve()
        assert target.is_relative_to(root)
        content = z.read(name)
        assert hashlib.sha256(content).hexdigest() == digest, name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
print('Verified source, datasets, BC actor, and both methods/seeds/checkpoints')
PY
python -m venv .venv
.venv/bin/python -m pip install torch==2.13.0 mujoco==3.11.0 numpy==2.4.6 pandas==3.0.5 matplotlib==3.11.1
export WANDB_MODE=disabled
export MPLCONFIGDIR=/tmp/takeoff-mpl
unset PYTHONPATH
```

Explicit bounded continuation from update 30 (run only after reviewing the
decision below; this does not launch a million-transition experiment):

```bash
WANDB_MODE=disabled .venv/bin/python -u scripts/continue_takeoff_retention.py \
  --source output/multimorph_bc_ppo_continuation_20261111_v2 \
  --output output/molab_bounded_next --start-update 30 --additional-updates 20 \
  > output/molab_bounded_next.log 2>&1 &
echo $!
tail -f output/molab_bounded_next.log
```

The wrapper accepts at most twenty additional updates per invocation. It does
not reinitialize either policy or change BC decay. Use the local reproduction
command above instead to reproduce updates 10–30. Monitor structured results:

```bash
cat output/molab_bounded_next/restore_audit.json
cat output/molab_bounded_next/outcome.json
tail -n 5 output/molab_bounded_next/B_seed42/training.csv
.venv/bin/python scripts/diagnose_takeoff_continuation.py --run output/molab_bounded_next
.venv/bin/python scripts/report_takeoff_retention.py --run output/molab_bounded_next --step-axes
.venv/bin/python scripts/report_takeoff_continuation.py --run output/molab_bounded_next
.venv/bin/python scripts/evaluate_takeoff_checkpoint.py \
  --checkpoint output/multimorph_bc_ppo_continuation_20261111_v2/B_seed42/checkpoint_000030.pt \
  --output output/molab_heldout_B42
```

The last command uses the evaluator's fresh 40000–40031 reset family,
full-horizon deterministic evaluation at current and full geometry. Repeat with
A and seed 43 for a matched held-out comparison. The diagnostic command covers
deterministic and stochastic periodic probes.

Complete backup and browser download, from a manually executed Molab Python cell:

```python
import pathlib, zipfile
import marimo as mo
root = pathlib.Path('BigRedGym_continuation')  # use Path('.') if already inside
backup = root.parent / 'takeoff_full_backup.zip'
with zipfile.ZipFile(backup, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('output', 'experiments', 'scripts', 'docs', 'notebooks'):
        for f in (root / folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f, f.relative_to(root))
mo.download(backup.read_bytes(), filename=backup.name)
```

The supplied bundle includes physical assets, original update-10 experiment,
the continuation, shared BC actor and exact dataset, validated reward gate,
source, results, and per-file SHA256 manifest. No W&B import is needed. CPU
MuJoCo is used; a Molab GPU does not accelerate these simulation loops.
Local extraction/manifest/runtime checks are reported below. Actual Molab
dependency installation and execution are not claimed tested here.

## Separate rotation curriculum design if the present run plateaus

Preserve this experiment as the morphology comparison. A new experiment should
use the supplied variable-length inverted-pendulum model already documented in
`docs/go2_morphology_dynamics_audit.md`, with
`theta_ddot = -(g/L)*sin(theta) - 2*(L_dot/L)*theta_dot` and theta measured from
downward vertical. Theta is COM-to-front-support angle, not body pitch.
Shorter static legs do not produce the dynamic shortening term or automatically
lower the measured energy barrier. The rear push adds an external moment, so
the torque-free equation applies only after rear unloading, with retained front
support, small pivot drift, and no other contacts.

Start a separate registered rotation experiment from a common preserved actor
and retain the BC floor and takeoff certificates. First measure COM-support
angle, effective L/L_dot, angular momentum about both COM and the front pivot,
and the model residual on eligible intervals. Use these to distinguish weak
liftoff angular momentum, adverse COM position, front slip, and poorly timed
articulated shortening. Match total interactions and initialization for any
new paired ablation. Do not revise the reward in this completed comparison.

Candidate new performance gates are certified supported rotation at 35°, 45°,
60°, 75°, and 85°, followed by .25/.5/1/5-second stable holds, each with held-out
takeoff retention, low pivot drift, contact validity and native actuator limits.
Calibrate reachable intermediate targets first; these thresholds are a proposal,
not validated criteria. Add progress only during certified useful rotation;
avoid rewarding standing lean or clearance without COM progress. Test timed
post-takeoff retraction/posture targets as an ablation before changing physical
morphology again. This proposal does not train a controller or modify the
current rewards, action interface, or curriculum.

## Completed result and decision

**Choose B: preserve this comparison and prepare the separate rotation-focused
curriculum.** No further training was launched. Reliable takeoff survived the
BC decay, but the morphology curriculum did not gain useful rotation or reach
100%. Fixed geometry has a modest, seed-dependent improvement; it is inaccurate
to claim both methods are exactly flat at 33°. These data do not justify a
large Molab budget for the unchanged morphology protocol.

All four runs completed updates 11–30 without a safety stop. Each received
20,480 additional training transitions, 30,720 cumulative. Total additional
training was 81,920 transitions; cumulative matched training was 122,880.
The evaluation checkpoints are 10, 15, 20, 25 and 30. At every point, every
method/seed achieved 8/8 deterministic and stochastic certified takeoffs at
full scale and current training scale. Five-second handstand success is zero.

Full-scale deterministic means (± sample standard deviation across two training
seeds; eight matched resets within each seed):

| Metric | Fixed update 10 | Fixed update 30 | Curriculum update 10 | Curriculum update 30 |
|---|---:|---:|---:|---:|
| Peak body/certified-flight pitch | 32.95 ± 1.78° | 34.12 ± .87° | 33.12 ± 1.29° | 31.10 ± 1.28° |
| Rear clearance | 16.95 ± 2.32 cm | 18.43 ± 1.56 cm | 18.57 ± 1.35 cm | 14.37 ± 2.62 cm |
| Certified flight duration | .2286 ± .0041 s | .2356 ± .0009 s | .2288 ± .0035 s | .2251 ± .0009 s |
| Certified takeoff | 100% | 100% | 100% | 100% |
| Five-second handstand | 0% | 0% | 0% | 0% |

The fixed seed-42 pitch is 31.69° → 34.74°; fixed seed 43 is
34.20° → 33.50°. Curriculum seed 42 is 32.22° → 30.19°; seed 43 is
34.03° → 32.00°. Final stochastic pitch is 33.67 ± .51° fixed versus
30.59 ± 1.10° curriculum, with 100% takeoff in both. Final 95% curriculum
deterministic pitch is only 28.34° on average.

Actual training interactions by physical morphology, including original updates:

| Method/seed | 90% | 95% | 100% | Additional updates 11–30 |
|---|---:|---:|---:|---|
| Fixed 42 | 0 | 0 | 30,720 | 20,480 at 100% |
| Fixed 43 | 0 | 0 | 30,720 | 20,480 at 100% |
| Curriculum 42 | 8,917 | 21,803 | 0 | 20,480 at 95% |
| Curriculum 43 | 8,652 | 22,068 | 0 | 20,480 at 95% |

Both original 90% → 95% events are retained (at interactions 8,638/8,445).
The last old-scale worlds account for the difference from actual scale counts.
Each curriculum has twelve completed 95% windows. Maximum qualifying fractions
are 0% and 6.25%, below 30%; no promotion was forced. Thus no full-scale training
reservation was applicable in this bounded run. Evaluation still uses the
physical 100% model, not a normalized or shortened surrogate.

BC coefficients are 6 → 2 and remain 2 through update 30. Across continuation
updates, BC losses are .00388–.00567. Final action RMS shifts from the original
BC actor are .00193/.00205 for fixed seeds 42/43 and .00235/.00201 for curriculum
seeds 42/43. These are normalized tanh action units on demonstration frames,
not a guarantee about unseen observations. Final cumulative demo Gaussian KL
is .234/.271 fixed and .341/.249 curriculum. PPO post-update KL is
.01288–.02082 fixed and .01166–.02250 curriculum; clipping is 2.34–13.84% fixed
and 4.62–12.63% curriculum. BC anchoring prevents observed takeoff forgetting
here but has not solved the rotation objective.

Final deterministic full-scale physical diagnostics, ordered seeds 42/43:

| Metric | Fixed | Curriculum |
|---|---|---|
| Pitch velocity at certified liftoff (rad/s) | 1.557 / 1.463 | 1.186 / 1.351 |
| Pitch velocity at peak pitch (rad/s) | .0181 / .0092 | .0106 / .0187 |
| World-y angular momentum about COM at liftoff (kg m²/s) | .882 / .828 | .661 / .759 |
| World-y angular momentum about COM at peak (kg m²/s) | .0670 / .0448 | .0192 / .0385 |
| Minimum rear-leg certified useful impulse (Ns) | .0584 / .0548 | .0509 / .0541 |
| COM x relative to front midpoint at peak (m) | −.2016 / −.2024 | −.1942 / −.2022 |
| Front support retained at peak | 100% / 100% | 100% / 100% |
| Episode front-support fraction | .863 / .862 | .868 / .878 |
| Joint target tracking RMS (rad) | .1258 / .1210 | .1153 / .1213 |
| Episode mean torque saturation fraction | .00112 / .00036 | .00033 / .00039 |
| First nonfoot collision (s) | 1.071 / 1.090 | 1.058 / 1.047 |
| Episode failure/end time (s) | 1.080 / 1.098 | 1.068 / 1.058 |

Every final episode eventually has a nonfoot collision, although the earlier
takeoff remains physically certified. At peak pitch the COM is still about
20 cm behind front support and angular velocity has nearly vanished. This
supports investigating impulse/COM position and articulated rotation timing,
not interpreting clearance as a completed inversion. Low motor saturation
alone does not prove adequate tracking or useful rotation authority.

`figures/rotation_per_seed.csv`, `rotation_summary.csv`, `final_rotation_per_seed.csv`,
`morphology_interactions.csv`, `per_seed.csv`, `summary.csv`, and `training.csv`
provide the measured comparison. PNG/PDF pairs are `retention`, `stability`,
`bc_anchor_drift`, `curriculum`, and `rotation_diagnostics`. Normal checkpoint
evaluations/traces, all five saved checkpoints per run, source snapshots,
episode logs, curriculum windows, and optimizer/RNG states are preserved.

## Verification

The 17 targeted takeoff/BC/PPO tests passed, including uninterrupted-versus-resumed
updates and simulator/optimizer/RNG restore tests. All changed scripts passed
Ruff and Python compilation. Independent diagnostics cover all 320 full-scale
episodes and exactly reproduce the periodic probe pitch, clearance, certified
pitch, takeoff time, takeoff, success, and nonfoot-collision outcomes.
Probe evaluation used 26,098 transitions; the independent diagnostic replay
used 17,619 more. Ordinary checkpoint evaluation counts are also retained in
each checkpoint. None of these are counted as PPO training interactions.

`source_restore_verification.json` checks all four original update-10 states;
`final_restore_verification.json` checks all four update-30 states. Both include
actor/critic, all Adam tensors/groups, curriculum/windows, Torch/NumPy RNG,
observations, episode counts and MuJoCo integration state. They consume zero
training interactions. No learned normalization state is missing.

A 538-file preflight bundle was extracted into a separate temporary checkout
and every manifest SHA256 verified. Using only its extracted sources, BC data,
physical assets and checkpoints, all four update-30 states passed the complete
restore audit (`clean_bundle_restore_verification.json`), with zero training
interactions. The extracted checkpoint evaluator also rebuilt the assets and
certified takeoff for B/42 on reset 40000 at both 95% and 100%
(`clean_bundle_smoke_evaluation.csv`). That single-reset smoke test checks
portability, not statistical performance. The final delivery bundle adds these
audits and the completed report and figures. Package versions match the local
environment; Molab installation itself remains untested.

Recreate the final source/checkpoint/results archive with a new filename:

```bash
env -u PYTHONPATH .venv/bin/python scripts/package_takeoff_molab.py \
  --output output/takeoff_continuation_molab_bundle_NEW.zip \
  --extra-root output/multimorph_bc_20261111_v1 \
  --extra-root output/multimorph_bc_20261111_v2 \
  --extra-root output/multimorph_bc_ppo_retention_20261111_v1 \
  --extra-root output/multimorph_bc_ppo_continuation_20261111_v2

env -u PYTHONPATH .venv/bin/python scripts/verify_takeoff_resume.py \
  --run output/multimorph_bc_ppo_continuation_20261111_v2 \
  --update 30 --output /tmp/takeoff_resume_check_NEW
```
