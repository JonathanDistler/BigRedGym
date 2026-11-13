# Earlier front-foot placement and coordinated rear-leg sweep

This opt-in extension continues `front_catch_parallel` on
`experiment/com_energy_phase_ppo`. Select `--suite anticipatory_v1`; the default
`--suite original` retains the previous A=fixed/B=slide/C=catch experiment.
Prior source bundles, checkpoints, preparations, plots and results remain intact.
The previous notebook is also copied to
`output/front_catch_anticipatory_preserved_20261112/`.

## Matched conditions

| Condition | Objective |
|---|---|
| A | Original fixed-front-support reward and phase, with no new reward |
| B | Contact-transition objective plus anticipatory front-foot placement |
| C | Same B objective plus coordinated rear-leg sweep |

B and C both use the existing C contact-transition gates underneath the new
terms. Thus unloading/sliding/recontact physics and legacy catch rewards are
identical between them; only the rear-sweep term differs. A is not welded to the
floor. All three retain the 73-input, 12-action CPG-free actor, 103-input critic,
100% Go2 geometry, original friction, collision detection, position bounds,
PD torques and actuator speed limits. No action filters or fixed joint schedules
are introduced. The unchanged preserved BC-initialized checkpoint is
`output/momentum_phase_diagnostic_20261112_v1/C_seed42/checkpoint_002.pt`, SHA256
`f20e26503b24bd5ed60ae43d983300f0cbf16d757ba2831ebce5a5ca9e22045c`.

## Causal phase and anticipatory placement

The one-shot maneuver phase starts phase 3 at certified rear-leg liftoff,
without waiting for a pitch maximum or front-foot recontact. The actor sees the
same phase representation in training and playback. Loading/push-off still use
the existing demonstration timing. Catch/braking/balance phases still require
measured support/orientation events. Unilateral front support is enough to earn
the new rising-swing objectives; bilateral support is required for the existing
stable catch/terminal objective. Brief unsupported motion is physically allowed
but earns neither new objective until valid support returns. Existing impact and
contact-gap limits remain in the legacy catch reward.

A short-horizon support forecast uses actual whole-body COM `c`, velocity `v`,
gravity magnitude `g`, foot positions in world coordinates, and tau=0.08 seconds:

```
c_future = c + tau*v - 0.5*g*tau^2*z_hat
h_future = max(c_future.z - foot_radius, 0.05)
c_support_predicted = c_future + sqrt(h_future/g)*v
```

This is a ballistic/LIP-inspired *reward heuristic*, not an exact capture basin
or a guarantee of recoverability. The same forecast is used to compare original
and current foot locations, so COM motion alone cannot improve placement credit.
For each foot, project onto the fixed initial task heading and request backward
travel only when the predicted support lies behind its original location. Cap
the requested travel at 8 cm. Credit the reduction in absolute error to that
requested location, divided by 8 cm and clipped to [0,1]. Average per-foot credit
allows staggered motion; both feet do not have to move. Multiply by a Gaussian
of the horizontal predicted-point distance to the nearest current front foot
(sigma=0.30 m), and normalized achieved pitch progress. Backward travel past a
useful location does not earn more reward.

Current rising-swing gating requires certified takeoff or a labeled valid
intermediate start, rear feet clear/unloaded, at least one front foot loaded,
positive COM vertical velocity, pitch velocity >0.2 rad/s, phase in [0.4,0.8),
positive head/calf clearance, collision/limit-safe history and no excessive
front impact. It never uses future trajectory information. Offline analysis
compares onset time against the eventual measured safe pitch maximum. The
existing useful-swing test also rejects trajectories that have fallen more than
5 degrees below their best swing pitch.

## Rear-leg sweep and momentum redistribution

At each control step, link COM Jacobians and world inertial tensors measure
rear-link momentum relative to rigid motion at the trunk angular velocity:

```
L_rear_relative = pitch_axis . sum_links[
    I_world * (omega_link - omega_trunk)
    + (x_link-c) cross m_link *
      (v_link-v_COM-omega_trunk cross (x_link-c))
]
```

Negative rear relative momentum, while trunk pitch is increasing in the desired
direction, can indicate useful transfer of rotation to the body. It is NOT new
whole-body angular momentum. Actual COM/support momentum and front/rear external
moments remain measured by native MuJoCo. A rigidly rotating articulated pose
with zero joint motion has zero relative rear momentum, verified by a test.

Rear hip means sagittal `thigh_joint`; knee means `calf_joint`. Both hips must
move with matching left/right signs, and both knees with matching signs. Hip and
knee signs need not match each other. The coordination factor rises with the
minimum of their four absolute rates up to 1 rad/s. The sweep score multiplies
this factor by normalized pitch progress, positive whole-body pitch momentum,
and clipped useful negative relative rear momentum (scale=0.4 kg m²/s).
A small factor, 0.75+0.25*inertia_contraction, favors reducing measured locked
whole-body inertia relative to takeoff, capped at 40% contraction. Contraction
alone never pays. The scale is configurable reward normalization, not a claim
that a successful full-swing demonstration exists.

Both new rewards pay only improvements in their episode best score:

```
r_early = 0.8 * max(0, early_score - best_early_score)
r_sweep = 0.6 * max(0, sweep_score - best_sweep_score)   # C only
```

Their total episode budgets are bounded by 0.8 and 0.6. Cycling the legs, waiting
at an elevated pose, or raising rear feet alone cannot repeatedly collect these
terms. Scores/credits and onset timers reset for standing and intermediate
starts and are included in full checkpoint serialization. The five-second
handstand definition and 89.4166 J target calibration are unchanged.

## BC and PPO

All conditions use exactly the same initial weights, observation dimensions,
seeds, valid reset bank, shared performance curriculum and PPO hyperparameters.
The demonstrated loading/push-off frames retain weight 1; post-takeoff imitation
weight is 0.025; independent balance-point demonstrations retain weight 0.15.
The PPO supervised coefficient remains 0.02. The current teacher is not treated
as a full-handstand demonstration. Actors control all twelve joints in every
phase; no reference controller or controller handoff is needed at deployment.

The environment count, horizon, update count and seeds determine the equal
interaction budget. Evaluation interactions are counted separately. No extended
local pilot or broad schedule/trajectory search was run for this extension.

## Validation and local smoke

Run all 35 relevant tests:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/unit_tests/test_front_catch_anticipatory.py \
  tests/unit_tests/test_front_catch.py tests/unit_tests/test_momentum_phase.py \
  tests/unit_tests/test_com_energy_phase.py \
  tests/unit_tests/test_physical_morphology_ppo.py -q
```

The two-update check used 16 worlds, four process workers, horizon 4, seeds 42/43,
and all A/B/C conditions: **768 training transitions**, plus 5,952 deterministic
evaluation transitions. It saved full checkpoints at updates 0/1/2 and evaluated
at each update. Inputs and original baseline are unchanged. Tests cover
unilateral support, wrong-direction placement, overshoot, rising/descent gates,
head/calf/impact rejection, event phase, exact A equivalence, B/C base equivalence,
relative momentum under rigid motion, reset state and exact resumed dynamics,
in addition to the previous COM/momentum/vector/PPO/serialization tests.

Outputs: `output/front_catch_anticipatory_smoke_20261112_v1`,
`output/front_catch_anticipatory_validation_20261112/tests.xml`,
`output/front_catch_anticipatory_report_20261112_v1`, and
`output/front_catch_anticipatory_timing_20261112_v1`.

The rising-swing gate opens before peak pitch in unmodified BC playback
(approximately 0.72--0.86 seconds across the four held-out standing episodes).
The short smoke collected zero anticipatory-placement reward in the training
batches. Held-out final standing playback did activate small placement rewards
(about 0.012--0.020 per episode averaged within each B/C training seed), at the
rising-swing gate before peak pitch. These can credit small placements already
achieved during loading; they do not establish a learned post-takeoff step. C
collected small nonzero rear-sweep reward in training, while B correctly
collected none. This is a
validation of gating/reward plumbing, not evidence of a learned catch. Do not
interpret two-update pitch fluctuations as a matched training conclusion.

## Molab launch, resumption and output

Upload `output/front_catch_anticipatory_molab_20261112_v3.zip` and open
`notebooks/molab_front_catch_parallel.ipynb`. It uses a fresh workspace and run
directory, verifies hashes, installs/validates dependencies, runs tests and
benchmarks the new C instrumentation at 8/16/32/64 environments. The existing
selector chooses 16 or 32 from measured throughput; there was no new local
benchmark sweep. W&B remains disabled. No cloud run has been launched. The notebook records a local Git snapshot of
the verified archive source because a legacy morphology validation test needs
Git HEAD. This snapshot does not claim to restore the original repository history;
the file-hash manifest identifies the actual packaged source.

After activating the validated Python 3.11 runtime, launch **all three
conditions**, seeds 42 and 43, with the following exact bounded command:

```bash
env -u PYTHONPATH WANDB_MODE=disabled WANDB_DISABLED=true \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/front-catch-mpl XDG_CACHE_HOME=/tmp/front-catch-cache \
  python scripts/run_front_catch.py \
  --suite anticipatory_v1 \
  --preparation output/front_catch_preparation_20261112_v3 \
  --output output/front_catch_anticipatory_molab_v1 \
  --num-envs 16 --workers 4 --updates 10 --horizon 32 \
  --seeds 42 43 --eval-every 2 --checkpoint-every 2 --max-seconds 1200
```

This is 5,120 training interactions per condition/seed and 30,720 total. Use
`--num-envs 32 --horizon 16` for **the same interaction budget** and physical
constraints. Alternatively use `--benchmark <benchmark>/selection.json` instead
of `--num-envs`; the notebook displays the resulting budget before launching.
Choose the worker count from the Molab host benchmark. The notebook's default
32-step horizon with a selected 32 worlds uses 61,440 total, still matched across
A/B/C. Wall-clock stopping happens only between complete matched update rounds;
resume to finish the planned equal budget.

Resume the latest saved generation in the same directory:

```bash
python scripts/run_front_catch.py \
  --output output/front_catch_anticipatory_molab_v1 \
  --resume output/front_catch_anticipatory_molab_v1/checkpoint_000002.pt \
  --workers 4 --max-seconds 1200
```

Use the actual latest generation. Source hashes, input hashes, MuJoCo and Torch
versions must match. Checkpoints contain all six networks, optimizers, per-run
Torch RNGs, global RNGs, independent environment RNGs, native integration states,
phase/reward-credit state and committed history. Old files are never overwritten.
The new suite cannot resume an old objective checkpoint as if it were the same
experiment. Use the old source bundle to resume the earlier experiment.

The notebook generates both existing reports and
`scripts/report_front_catch_timing.py`, including CSV and PNG/PDF plots of foot
motion, sagittal hip/knee rates, pitch/peak timing, COM motion, inertia/momentum,
front/rear contact moments and collisions. Timing rows keep start kind and seed;
intermediate-start outcomes are never counted as standing-start success. Raw
motion onset alone is not a successful or rewarded catch.

Periodic immutable recovery archives and a final full downloadable archive
include checkpoints, source bundle and reports. Download before ending the
session. The notebook supervises a separate process so an interrupt preserves
the latest complete update boundary; a runtime restart must restore the backup
and use the exact same source/runtime contract.
