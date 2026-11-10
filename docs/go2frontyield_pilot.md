# Bounded residual PPO pilot

`go2frontyield` is a separate opt-in task. It preserves `go2kickup`, the original
reference JSON, prior curriculum classes, and all earlier checkpoints. The
optimized reference is copied into `front_yield.parameters` in each saved
effective configuration, alongside a source/JSON snapshot. It is never scaled
down by curriculum amplitude.

## Integration and residual coordinates

`Go2FrontYield` uses a batched Torch version of the independently validated
NumPy reference. It preserves forward loading, front yielding, separate rear
hip/knee clocks, analytic velocity feedforward, and post-launch retraction.
Motor gains come from the validated JSON, with the original joint/velocity and
torque limits. Default episodes last 1.6 s and start standing.

Actor actions retain the older interface: 12 normalized `dof_pos_target` actions
in FL/FR/RL/RR canonical order. They are scaled once into joint radians, then
bounded by phase and the episode's curriculum stage. Bounds are applied before
adding the reference and before the backend's native joint-order mapping. The
applied joint residual, rather than the unbounded actor output, is fed back in
the observation. Velocity feedforward follows the reference; the actor cannot
change the reference clock or front controller gains.

At initial stage zero the front/rear limits in radians are:

| Phase | Front joints | Rear joints |
| --- | ---: | ---: |
| Loading | 0.005 | 0 |
| Rear extension | 0.005 | 0 |
| Retraction | 0.010 | 0.005 |
| Stabilization | 0.020 | 0.040 |

Phase transitions blend over 50 ms and residual position slew is at most
0.5 rad/s. Every later stage still has a strict extension bound: rear bounds
are 0.005, 0.010, 0.015, 0.020, 0.025 and 0.030 rad at stages 1–6. An additional
projection prevents an opposing residual rate from cancelling more than 20%
of the directed reference hip/knee velocity during extension. Larger rear
bounds, up to 0.18 rad at the final stage, apply only during stabilization.
Per-episode stage indices preserve a consistent action envelope across
curriculum promotions.

The existing completed-episode curriculum accounting is retained. Promotion
requires 90% success in two batches of at least 64 primary episodes. Stages
zero and one require verified simultaneous 30 mm rear flight; later stages
also require the configured forward pitch target with absolute pitch rate
below 2 rad/s. Standing-start reset poses remain unchanged. This is deliberately
more demanding than unloading or a contact-detector transition.

## Physical objective

The new reward set prioritizes supported geometric takeoff, clearance,
controlled forward rotation, and a low-speed terminal pitch target. Terminal
credit is gated on prior genuine takeoff and current front support. Body
contact and angular speed above 3 rad/s are penalized. Front drift has no
penalty below 4 cm; larger displacement is penalized quadratically. Both front
load requirements remain active. The curriculum's support-failure drift limit
starts at 5 cm and progressively tightens to 4 cm.

The first terminal pitch goal is 32 degrees and increases toward 85 degrees.
This is a progression toward a handstand, not a claim that a 32-degree state is
recoverable by the existing balance policy. No balancing-policy handoff or
hardware recovery has been validated. Those must be tested before claiming
handoff readiness, particularly with terminal pitch rate and front contact.

## Parity and checkpoint compatibility

Sixteen matched seeds 3000–3015 reproduce the independent native reference,
including its varied joint positions, orientation, mass, friction and gains.
The final zero-action trace differences are below 0.002 mm in minimum rear
clearance, 0.00001 rad in pitch, 0.00013 rad/s in pitch rate, and 0.003 N m in
motor torque. All 16 reproduce supported 30 mm takeoff. Contact loading and
geometric liftoff timings are explicitly compared with a 4 ms tolerance;
geometry is always required in addition to force release. Native contact
forces and backend contact-force buffers have a physics-sample timing offset,
so they are not claimed to be bit-identical.

The opt-in CPU batch-copy path leaves native MuJoCo stepping and contact solving
unchanged and is tested against the original backend's public buffers. It is
installed only on `go2frontyield` CPU instances. The original backend class and
other tasks are unchanged.

The warm start uses:

```
logs/go2kickup_sync_grf_1500_20261109_201802/Nov09_20-19-51_/model_1225.pt
SHA256 5b4876a3d0fb0c1dd30470da3ad4ea8384ad38499f89077cd35b0ef168711b63
```

This selects the newer sync/GRF run explicitly, rather than the other local
iteration-1225 checkpoint. Ordered actor/critic observation lists, action lists,
network architecture, normalization flags, scaling and motor settings are
checked. Actor normalization is disabled; critic normalization is enabled.
Strict state-dict loading preserves the critic's mean, variance and count.
The standard training burn-in subsequently adapts critic normalization to the
new rollout distribution; actor observations remain unnormalized.

`warm_start_verification.json` proves checkpoint 0 has identical actor/critic
weights and normalizer tensors, empty actor/critic optimizer states, iteration
zero and fresh curriculum counters. `--resume` is not used. The bounded old
actor passes 16/16 takeoffs before optimization; a fresh-actor comparison is
conditional on harmful warm-start behavior, rather than automatically
launching another long pilot.

## Molab Online commands

Run from the repository root on `features/kickup`, Python 3.11. Set the source
path explicitly if using a different copy of the selected checkpoint. Preserve
its adjacent `files/` directory and saved configuration.

```bash
set -euo pipefail
uv sync --frozen

# Gate: 16 matched zero-residual trials, numerical and contact timing assertions.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run \
  scripts/evaluate_frontyield.py --output-dir output/frontyield_zero_verified

# Actor/critic warm start; NEW optimizers, NEW curriculum, separate run.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2frontyield --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 200 --save_interval 25 --seed 7 --headless --disable_wandb \
  --experiment_name go2frontyield_warm_pilot \
  --warm_start logs/go2kickup_sync_grf_1500_20261109_201802/Nov09_20-19-51_/model_1225.pt \
  --warm_start_task go2kickup

# Evaluate the newest run in this explicit experiment (resolved once and printed).
# No --resume or --original_cfg
# is needed for the warm start above; those would have different semantics.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run \
  scripts/evaluate_frontyield_pilot.py \
  --experiment-name go2frontyield_warm_pilot \
  --output-dir output/frontyield_pilot_comparison --watch

# --watch may be omitted after model_200.pt exists. The evaluator loads each
# checkpoint's saved effective configuration, restores its curriculum level,
# and evaluates iterations 0, 50, 100 and 200 on identical starts.

# Only if bounded warm-start actions harm takeoff: independent fresh pilot.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2frontyield --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 200 --save_interval 25 --seed 7 --headless --disable_wandb \
  --experiment_name go2frontyield_fresh_pilot

# PPO playback defaults to saved config, stage and episode duration for this task.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/play.py \
  --task go2frontyield --device cpu --num_envs 1 --seed 7 --no-keyboard \
  --experiment_name go2frontyield_warm_pilot --checkpoint 200

# Original optimized reference-only playback remains independently available.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run \
  scripts/search_kickup_reference.py \
  --parameters gym/envs/go2/references/kickup_front_yield.json \
  --playback-only --viewer --repeats 10
```

Each PPO iteration collects 512 control steps in 16 environments: 8,192 samples.
Checkpoint 0 is saved before any update; 50, 100 and 200 count new pilot
iterations, not the source checkpoint's iteration number. No 1,500-iteration
experiment is part of this workflow.

## Artifacts and tests

`output/frontyield_zero_verified/evaluation.json` contains parity differences,
contact timings, configuration and per-trial results. Its CSVs include physical
clearance, front drift, pitch and pitch velocity, all motor torques, joint
positions, applied residuals and all foot normal forces at 500 Hz.

`output/frontyield_pilot_comparison/summary.json` compares 0/50/100/200, with
per-checkpoint configuration, checkpoint SHA256 and per-trial CSVs in the
corresponding iteration folders. Summary metrics include physical takeoff,
clearance, front contact/drift, pitch and rate, applied push residual,
curriculum success, nonfoot contacts and overrotation. Curriculum success is
reported separately from physical takeoff, since later stages require larger
pitch angles. It is not used as a substitute for physical evaluation.

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m pytest \
  tests/unit_tests/test_frontyield.py tests/unit_tests/test_kickup_reference.py \
  tests/unit_tests/test_go2_kickup.py tests/unit_tests/test_kickup_cpg.py \
  tests/unit_tests/test_play_inference.py -q
```

Tests cover the NumPy/Torch reference match, zero-action residuals, extreme
adversarial actor outputs, phase-dependent bounds, directed extension-rate
protection, actuator torque limits, batched CPU buffer equivalence, genuine
MuJoCo liftoff, and the pre-existing reference/curriculum/playback regressions.

Generate the matched trajectory comparison plot after evaluation:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/kickup-mpl XDG_CACHE_HOME=/tmp/kickup-cache \
  uv run scripts/report_frontyield_pilot.py \
  --pilot-dir output/frontyield_pilot_comparison \
  --zero-dir output/frontyield_zero_verified
```

The independent evaluator audits every native nonfoot-ground contact with
negative contact distance, even if too small to exceed the training detector's
1 N threshold. Such trials fail physical success. A reported handoff candidate
requires 75–105 degree pitch, absolute pitch rate at most 1.5 rad/s, front
support and at least 30 mm rear clearance for 60 ms after 1.05 s. This is only a
kinematic screening condition; it does not prove that the balancing policy can
recover the state.


## Files changed for this integration

- `gym/envs/go2/go2frontyield.py` and `go2frontyield_config.py`: opt-in task,
  physical objective, phase/rate residual limits and curriculum gates.
- `gym/envs/go2/kickup_reference.py`: batched Torch generator; the independently
  validated NumPy generator and the reference JSON remain unchanged.
- `gym/envs/go2/frontyield_cpu.py`: opt-in batched CPU state synchronization.
- `gym/envs/__init__.py`: new task/config registration.
- `scripts/train.py`: strict warm-start interface, scaling and motor checks.
- `scripts/play.py`: saved-config, stage-aware repeated playback for the new task.
- `scripts/evaluate_frontyield.py`: zero-action parity and checkpoint evaluation.
- `scripts/evaluate_frontyield_pilot.py`: pinned-run 0/50/100/200 evaluation.
- `scripts/report_frontyield_pilot.py`: matched trajectory plot and result table.
- `tests/unit_tests/test_frontyield.py`: reference, residual and backend parity tests.
- This document and links in the existing kick-up/reference documentation.

Pre-existing README and robot-metrics edits were preserved. The original
`go2kickup.py`, `go2kickup_config.py`, curriculum implementation files, reference
JSON and historical checkpoints were not modified by this integration.

## Completed local warm-start pilot and handoff

The completed run is
`logs/go2frontyield_warm_pilot_fast/Nov09_21-28-01_/`.
It finished 200 iterations / 1,638,400 PPO samples, with checkpoints
0/25/50/75/100/125/150/175/200. All logged numeric vitals are finite.
`pilot_verification.json` and `warm_start_verification.json` record the checks.
The source checkpoint SHA256 and original reference JSON are unchanged.

| Controller | Takeoff | Mean peak minimum rear clearance | Mean maximum front drift | Mean peak pitch | Mean terminal pitch | Mean peak absolute pitch rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-residual reference | 16/16 | 109.08 mm | 2.69 cm | 26.35° | -3.18° | 2.874 rad/s |
| Warm start, iteration 0 | 16/16 | 103.29 mm | 2.66 cm | 26.12° | -6.49° | 2.866 rad/s |
| Iteration 50 | 16/16 | 102.34 mm | 2.66 cm | 26.12° | -6.47° | 2.866 rad/s |
| Iteration 100 | 16/16 | 100.69 mm | 2.66 cm | 26.12° | -6.39° | 2.866 rad/s |
| Iteration 200, saved stage 2 | 16/16 | 84.77 mm | 2.61 cm | 25.66° | -9.72° | 2.826 rad/s |

All four matched checkpoint evaluations retain approximately 99.2% front load
retention, with zero native nonfoot-contact trials and zero overrotation trials.
Worst-case peak simultaneous rear clearance at iteration 200 is 65.82 mm;
worst front drift is 3.80 cm. Mean absolute applied rear push residual is zero
at 0/50/100, and 0.00909 rad at 200, below its stage-2 0.01 rad limit. The rate
projection remains active. This is a change in the saved stage and permitted
envelope as well as in weights; the table must not be interpreted as a pure
fixed-envelope learning curve.

The curriculum reached stage 2 late in the run. Stage-0 matched curriculum
success is 16/16 at 0/50/100; at stage 2 it is 0/16 because the 50° pitch goal is
not reached. No checkpoint yields a screened handoff candidate, and an actual
balancing-policy handoff was not tested.

**Conclusion:** the implementation preserves reliable physical takeoff, but the
warm-start pilot does not meet the rotation/recoverability improvement goal.
The wider envelope lowers clearance and worsens terminal pitch. Around 95% of
front push residual samples are already at their small cap at iterations 0/50,
which limits corrective variation. Do not continue this warm start for 1,500
iterations on this evidence. Compare a fresh actor before deciding whether the
old actor's saturated action output is worth retaining; keep the physical
push-protection gates in either case.

![Warm-start checkpoint comparison](../output/frontyield_pilot_comparison/comparison.png)

The requested fresh comparison was started in
`logs/go2frontyield_fresh_pilot/Nov09_22-05-31_/`, then **stopped at the user's
handoff request after two iterations**. Only `model_0.pt` was saved there; this
is not a completed fresh pilot and no fresh-versus-warm training conclusion is
claimed. The warm-start training and evaluation processes have completed, and
the fresh training process has stopped. No training remains running.

Start a separate fresh pilot on Molab without overwriting either run:

```bash
set -euo pipefail
uv sync --frozen

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run \
  scripts/evaluate_frontyield.py --output-dir output/frontyield_zero_handoff

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2frontyield --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 200 --save_interval 25 --seed 7 --headless --disable_wandb \
  --experiment_name go2frontyield_fresh_handoff

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run \
  scripts/evaluate_frontyield_pilot.py \
  --experiment-name go2frontyield_fresh_handoff \
  --output-dir output/frontyield_fresh_handoff_comparison
```

The integration changes are currently workspace changes, including new
untracked source/config/test files. Commit or upload them with the reference
JSON to Molab before running these commands; no commit or push was performed.
The targeted suite passed **35 tests**, and Ruff passed for the changed Python
implementation, scripts and tests. No 1,500-iteration experiment was launched.
