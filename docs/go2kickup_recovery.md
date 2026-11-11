# CPG-guided kick-up recovery training

Updated frontier dataset, gated curriculum, interrupted 50-update diagnostic and
current commands: [recovery frontier report](go2kickup_recovery_frontier.md).
The measurements below describe the earlier small-bank experiment.

This is an opt-in training environment and reproduction procedure on
`ppo_kickup`. No 200-iteration run has been launched. Full-transition robustness
has not been demonstrated. Existing references, checkpoints, and prior results
were retained. The pre-existing README edits and `output/` contents were not
replaced.

## Verified controller and architecture

The successful near-vertical controller is `go2balance`, checkpoint
`logs/go2balance/Oct31_23-29-27_/model_1500.pt`. Its saved source/configuration
is loaded with `checkpoint_configs`; the documented CPG overrides are explicitly
restored: enabled, modes FL/FR stance and RL/RR swing. The old config alone says
CPG disabled, so silently loading only that config is insufficient.
`datasets/recovery_policy_audit.json` records its SHA256 and complete interface.

The CPG **does not generate periodic joint trajectories** in this implementation.
`ContactCPG` supplies force-coupled phase and desired-contact observations to PPO.
The nominal handstand posture supplies joint targets; PPO supplies residuals.
This distinction is preserved, including the original torque controller:

```
phase_dot = 2*pi*frequency - sigma*filtered_load*(cos(phase) + zeta)
q_target = handstand_reference + 0.25 * PPO_action
motor_torque = clip(kp*(q_target-q) + kd*(v_target-v) + tau_ff)
```

Phase integration runs every 2 ms physics step; control and policy run at 100 Hz.
At episode reset phases are `[0, pi, pi, 0]`, filtered loads zero, desired contacts
`[1, 1, 0, 0]`. Native trajectory resets restore phase and filtered load instead.
The fixed per-leg references are front `[0, -1.095796, -0.95]`, rear
`[0, 2.045796, -0.95]` radians.

Actor observations: **59**, critic observations: **63**, actions: **12**.
Actor input order is angular velocity, projected gravity, commands, joint
position observations, joint velocities, previous position residual, handstand
stage, CPG phase sin/cos, and CPG desired contacts. The critic additionally sees
base height and linear velocity. New COM reward information does not enlarge
the actor input. Both checkpoint networks load strictly.

Canonical actuator order: FL, FR, RL, RR; each hip, thigh, calf. Native joint
limits in radians: hips ±1.0472; front thighs −1.5708 to 3.4907; rear thighs
−0.5236 to 4.5379; calves −2.7227 to −0.83776. Torque limits per leg are
23.7, 23.7, 45.43 Nm. The 0.25-radian residual scale is a scale, not an action
clip. The original controller clips torque; it does not add a new target slew
limiter. Existing transition evaluation retains its own physical target limits.
The audit JSON includes native and soft joint limits and velocity limits.

## Real trajectory data and diagnosis

`scripts/build_recovery_dataset.py` runs actual standing starts with the frozen
kick actor, aggressive prefix, adaptive reference-first posture controller, and
frozen CPG balance actor. No artificial contact forces or projected handoff
poses are used. The aggressive prefix is explicitly selected; the default
kick reference is unchanged.

Samples are taken every 50 ms from completed blending through the following
second, while collision-free and base height exceeds 0.3 m. Thus failures are
saved **before** physical collapse, rather than resetting into penetrated poses.
A sample's outcome label is its source trajectory's subsequent outcome, not an
independent recovery guarantee for every snapshot. “Near” means the trajectory
achieved at least 0.1 s continuous valid hold; it is a broad screening category.

Saved separately from logs:

| Artifact | Seeds | States | Full five-second outcomes |
|---|---|---:|---:|
| `datasets/go2kickup_recovery_train_v2/` | 10000–10015 | 98 | 0/16 |
| `datasets/go2kickup_recovery_validation_v2/` | 6000–6015 | 80 | 1/16 |
| `datasets/go2kickup_recovery_heldout_v2/` | 7000–7015 | 73 | 0/16 |

Validation includes 20 snapshots from the recovered seed 6003, 36 from near
trajectories and 24 from failed trajectories. Training contains 46 near and
52 failed snapshots; it contains no independently successful full transition.
The successful validation seed is deliberately excluded from training. Older
versioned dataset directories are retained. These small, highly correlated
banks establish the pipeline; substantially broader training data is still
needed. Validation/held-out seeds were previously inspected in this branch;
they are fixed regression cohorts, not a new untouched generalization test.

The NPZ contains `mjSTATE_INTEGRATION` (including velocities, time, actuator
activation, warmstart and applied inputs), mass/inertia/COM and friction, plus
controller phase/load, applied residual, target velocity, feedforward torque,
three-frame command history, commands, stage and PD gains. New training captures
also preserve derived inverse-weight/inertia constants. Each reset restores
per-environment model properties, sets the integration state, calls `mj_forward`
to reconstruct contacts/forces, restores solver warmstart, and synchronizes
observation tensors. No foot-force arrays are assigned. Model copies prevent
randomized replay physics from leaking into other worlds or nominal resets.

Baseline validation exposed two subtle issues now covered by regression checks:
`mj_setConst` disturbs data state, and recomputing nominal constants changes
values retained by the original model after rotor-inertia setup. Nominal resets
therefore restore exact model copies; replay restores captured constants and
integration state. Existing backend behavior is unchanged globally.

`datasets/recovery_diagnosis_v2.json` compares one recovered and 29 failed
completed handoffs (two trials fail before handoff). Seed 6003 reproduces a
6.51-second hold, handoff pitch 87.70°, rate +0.406 rad/s, COM offset −5.7 mm,
and COM speed 0.0517 m/s. Failed completed handoffs average about 1.15 m/s COM
speed. Rear clearances are about 1.12 m in the success versus 0.67–0.72 m in
failures. Front support load, elevation and joint configuration also separate
these samples. Several hip/joint variables have large descriptive differences;
world foot coordinates are heading-dependent. The ranking is descriptive,
confounded and based on one positive seed, not a causal recovery classifier.

CSV traces include orientation, body angular velocity, full COM position and
velocity, centroidal angular and linear momentum, front support positions and
loads, rear-foot/joint positions and velocities, clearance, base height,
nonfoot collision flags, policy/applied targets and torques. Evaluation reports
maximum target step, saturation and support retention. The full-system bridge
checks native nonfoot contacts every physics step. Recovery training similarly
latches any nonfoot-ground penetration over 10 micrometres for the episode.

## Curriculum and rewards

All stages retain the original CPG/contact observation interface and joint
residual controller. Earlier stages remain available through 20% replay.

0. Original nominal near-vertical resets and original model constants.
1. Up to 2° tilt and 0.12 rad/s angular disturbance.
2. Up to 8° tilt (moving COM relative to support), 0.3 rad/s angular disturbance,
   and up to 0.25 m/s per horizontal base-velocity component.
3. Real trajectory contact variations within 0.3 rad of vertical and COM speed
   below 0.5 m/s (29 eligible training samples).
4. Actual near-successful transition trajectories (46 eligible training samples).
5. All physically feasible collected transition handoff samples (98 samples).

Promotion requires 128 completed primary-stage episodes, at least 80% achieving
five seconds continuous strict balance, for two consecutive batches. Earlier
replay episodes cannot promote; holds followed by terminal falls cannot promote.
Curriculum state uses the existing checkpoint serialization. There is no
iteration-count-driven advancement.

Original rewards retain support/load sharing, rear clearance, orientation,
angular and linear stability, sustained holds, slip, unloading, collisions,
torque, action-rate/second-difference and joint-limit terms. Added terms:

- COM speed: `-min(||v_COM||², 9)`, weight 3.
- Horizontal COM/support-midpoint displacement:
  `-min(||COM_xy-support_xy||² / 0.05², 25)`, weight 3.
- Valid front-supported hold ramp `min(hold_seconds/5, 1)`, weight 20.

Strict balance and the hold counter still require the existing multi-component
balance criterion, including support, clearance, posture and stability; pitch
alone cannot promote. Nonfoot contacts are latched, with the existing failure
timer and discrete terminal cost. The support midpoint is a shaping reference,
not a formal support-polygon stability certificate.

## Validation and commands

Run from repository root with the repository's Python 3.11 environment.
`env -u PYTHONPATH` avoids this machine's unrelated Python 3.12 ROS packages.
All output directories must be new. On Molab, copy the checkpoint **and its
adjacent saved `files/` source tree**, training NPZ, and current repository code.
Dataset regeneration and full-system evaluation additionally require the kick
checkpoint, `output/handoff_reference_search/candidate_reference.json`, and
`output/handoff_cpg_comparison/atlas.json` from this checkout.

```bash
# Rebuild independent training data if needed; never use validation as training.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/build_recovery_dataset.py --seed 10000 \
  --output-dir datasets/recovery_train_new

# Native replay, CPG/checkpoint interface, rewards, curriculum and regressions.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/unit_tests/test_kickup_recovery.py tests/unit_tests/test_go2_balance.py \
  tests/unit_tests/test_posture_transition.py tests/unit_tests/test_kickup_handoff.py \
  tests/unit_tests/test_kickup_cpg.py tests/unit_tests/test_frontyield.py \
  tests/unit_tests/test_handstand_curriculum.py -q

# Check original checkpoint baseline performance in both environments.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/validate_recovery_baseline.py \
  --dataset datasets/go2kickup_recovery_train_v2/train.npz \
  --output-dir output/recovery_baseline_new

# ONE PPO iteration on actual stage-4 transition resets; no full run.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/train_kickup_recovery.py --validate \
  --dataset datasets/go2kickup_recovery_train_v2/train.npz \
  --output-dir output/recovery_smoke_new

# MOLAB: explicitly launch 200 iterations only after validation passes.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/train_kickup_recovery.py \
  --dataset datasets/go2kickup_recovery_train_v2/train.npz \
  --output-dir output/recovery_molab_200_new

# After training: supply the exact new model path from configuration/result JSON.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_kickup_recovery.py \
  --fine-tuned logs/REPLACE_WITH_NEW_EXPERIMENT/REPLACE_WITH_RUN/model_200.pt \
  --output-dir output/recovery_paired_evaluation_new
```

The launcher makes and verifies a separate `warm_start.pt` copy, loads both
networks with fresh optimizers, disables W&B, selects MuJoCo CPU and 16 worlds,
starts the full curriculum at stage 0, uses 200 iterations and saves every 25.
Experiment names include UTC microseconds. It saves the effective configuration,
original checkpoint hash and actual runner geometry. The verified geometry is
256 policy/control steps per environment (2.56 s), 4,096 transitions per rollout,
and optimizer batch size 1,024. PPO2 uses its existing `max_gradient_steps`
generator, not an assumed fixed number of minibatches or epochs. The smoke mode
starts at stage 4 to exercise actual trajectory resets and records finite reward
calls, parameter/state checks and losses. It is not a training performance result.

Full-system evaluation generates each standing/kick cohort once, then reuses
its native states, randomized models, warmstarts and targets for original and
fine-tuned policies. Both use the same adaptive posture controller and target
rate limit. It writes per-controller validation/held-out CSVs and summaries:
five-second success, longest and total valid hold, handoff dynamics, support,
nonfoot collisions, torque saturation and target continuity. No fine-tuned
200-iteration checkpoint or resulting success rate exists yet.

Implementation files: `gym/envs/go2/recovery_state.py`,
`go2kickup_recovery.py`, `go2kickup_recovery_config.py`; task registration in
`gym/envs/__init__.py`; scripts `audit_recovery_policy.py`,
`build_recovery_dataset.py`, `diagnose_recovery_states.py`,
`train_kickup_recovery.py`, `validate_recovery_baseline.py`,
`evaluate_kickup_recovery.py`; tests `test_kickup_recovery.py`. Existing evaluation
scripts gained an optional snapshot callback, additional dynamics telemetry,
recovery-task loading and total stable-time reporting; default controllers and
reference JSONs were not modified.

## Recorded verification outcomes

- 47 targeted unit/regression tests passed; Ruff checks and `git diff --check`
  passed for the implementation.
- `output/recovery_baseline_validation_v4/comparison.json`: original and recovery
  environments both achieve 14/16 nominal five-second holds with identical
  successful trial IDs. Earlier failed validation artifacts are retained.
- `output/recovery_ppo_smoke_v2/result.json`: one actual stage-4 PPO iteration,
  513 finite reward calls, finite final actor/critic parameters and native state;
  value loss 147.2894, surrogate loss 0.3419. W&B disabled.
- The first nominal-only smoke completed with finite optimizer losses but had no
  completed episodes, so its episode-summary logger printed NaNs. It was replaced
  for validation by the instrumented stage-4 smoke above, whose episode summaries
  are finite. The earlier artifact remains available.
- Original balance checkpoint SHA256 was checked again after the smoke test and
  remained unchanged. Work remains on `ppo_kickup`; no old reference was edited.
- `output/recovery_paired_smoke_v1/`: paired full-system evaluation successfully
  reloads the saved recovery-task checkpoint. Original: validation 1/16, held-out
  0/16. One-iteration smoke checkpoint: validation 0/16, held-out 0/16; best holds
  0.70 s and 0.28 s respectively. This is a regression after the smoke update,
  not evidence of improvement. The smoke checkpoint is not promoted. The
  original baseline remains available unchanged.
- Task-registration lint also exposed four pre-existing identical duplicate
  kick-up dictionary entries; these were removed without changing their values
  or registration behavior.

See [the dataset/interface audit and checkpoint-series evaluation](go2kickup_recovery_audit.md)
for trajectory-level coverage, exact observation slices, checkpoint normalization,
CPG parity tests, and the Molab asset preflight. Remote availability has not been
verified; the user explicitly requested no connection to Molab.
