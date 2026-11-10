# Go2 rear-leg takeoff

The [reference-only optimization](go2kickup_reference.md) now demonstrates
16/16 supported 30 mm takeoffs with less than 4 cm front drift. Use its opt-in
reference playback and validation commands. The historical long PPO commands
below remain deferred. A separate [bounded-residual pilot](go2frontyield_pilot.md)
integrates this reference into the opt-in `go2frontyield` task.

The task trains a standing-start kick only. Observations and actions retain the
existing kick-up interface; there is no balancing policy or policy handoff.
Existing checkpoints and their saved configurations are never modified.

## Diagnosis and changes

The active kick generator is `Go2Kickup._standing_posture`, not the standalone
legacy `KickupCPG`. It uses a shared rear-leg clock. Previously the stage-zero
reference amplitude was only 0.12, the push lasted 0.30 s, and the PD velocity
target was zero even while the position reference moved. Damping consequently
opposed commanded extension. The final extended pose then remained static.
The front standing targets used full position stiffness with small residuals.
These are plausible mechanics bottlenecks, not proof of actuator saturation.
The saved older reward also tracked front posture and paid clearance outside
the kick window; its success test did not require contact release or timing.
The more recent pre-change implementation already had force/clearance event
checks, but no force hysteresis, explicit takeoff event bonus or flight ramp.

The new reference preserves standing, 0.15 s settling and 0.55 s crouching,
then extends over configurable `kick_motion.extend_seconds = 0.20` s.
Smoothstep position interpolation and its analytic velocity give continuous
position and velocity at phase boundaries. Stage amplitudes now range from
0.70 to 1.00 so early stages retain a substantial physical push. Maximum nominal
knee reference velocity at full amplitude is 10.875 rad/s, below the URDF's
15.70 rad/s limit. Velocity targets are clipped to joint velocity limits;
position targets are clipped to joint position limits. Existing torque clips
remain 23.7 Nm hip/thigh and 45.43 Nm calf, with the original PD gains 60/3.
A new penalty discourages measured speeds above 90% of URDF limits; actual
simulation velocity is not artificially clamped.

Front position gain is multiplied by configurable `front_gain_fraction = 0.65`;
damping remains active. Front joints retain independent residual actions.
Anchors are reward/success conditions, not welds or kinematic constraints.
Both front contacts, bounded horizontal velocity and displacement are required
for takeoff rewards. Support loss over 0.08 s or excessive drift disqualifies
an episode. Existing torso-contact and base-height terminations remain enabled;
foot unloading itself does not terminate an episode.

Rear contacts use a Schmitt trigger in body-weight units per foot: on at 0.03,
off below 0.015. A takeoff event requires previous loading, force below the off
threshold, clearance above 2 mm (world foot centre minus sphere radius), upward
world foot velocity, and occurrence inside the extension/lift window. Both
feet must produce events within the stage timing tolerance. Small force alone
cannot earn takeoff. The event bonus is paid once per episode and divided by
control dt because PPO integrates reward rates. Flight rewards ramp over 40 ms
of continuous supported flight and reset on recontact. Minimum-foot clearance
has only 20% shaping credit until verified coordinated takeoff.

Useful impulse integrates every physics sample: excess load above 0.25 body
weights per rear foot, clipped at 1.5 body weights, times actual positive knee
extension rate (capped at 1 rad/s), body weight and physics dt. Only the first
push window with valid front support earns credit. Per-foot credit saturates
at body weight times 0.18 s. New impulse is divided by body weight, control dt
and twice the 0.65 load target; a left/right sharing factor suppresses unilateral
pushes. Stationary support and repeated landings cannot farm this reward.

## Seven stages

| Stage | Milestone | Reference amplitude | Front drift limit | Timing tolerance |
| --- | --- | ---: | ---: | ---: |
| 0 | Both previously loaded rear feet unload below 0.10 body weights each | 0.70 | 7 cm | 120 ms |
| 1 | Brief genuine flight, both feet above 3 mm | 0.75 | 6 cm | 100 ms |
| 2 | Both feet above 1 cm | 0.80 | 5 cm | 80 ms |
| 3 | Both feet above 3 cm | 0.85 | 4 cm | 70 ms |
| 4 | Both feet above 6 cm | 0.90 | 3.5 cm | 60 ms |
| 5 | Both feet above 10 cm | 0.95 | 3 cm | 50 ms |
| 6 | Both feet above 16 cm with nose-down rotation | 1.00 | 2.5 cm | 40 ms |

Stage zero requires 60 ms unloading with valid front support. Every later stage
requires 60 ms continuous clearance and rear contact release, coordinated
takeoff and knee-extension onset, and rear height mismatch below 4 cm. Final
stage additionally requires projected gravity x above 0.15 and absolute pitch
rate below 3 rad/s. Promotion requires 40% success in two batches of at least
64 completed primary episodes. Warm starts restart this curriculum at zero.

## Reward weights

| Term | Weight |
| --- | ---: |
| clearance | 5.0 |
| takeoff event | 4.0 |
| sustained flight | 2.0 |
| upward velocity, release, push impulse, front support | 1.0 each |
| synchronization | 0.6 |
| pitch, positive pitch velocity | 0.5 each |
| unloading, takeoff timing penalty | 0.4 each |
| force spikes penalty | 0.3 |
| extension timing penalty, rear motion tracking | 0.15 each |
| rear slip penalty | 1.0 |
| front displacement penalty | 10.0 |
| front slip penalty (`handstand_slip`) | 4.0 |
| instability, action rate penalty | 0.05 each |
| residual motion, velocity limit penalty | 0.1 each |
| squared torque penalty | 0.000005 |
| second action difference penalty | 0.005 |
| joint position limit penalty, collision penalty | 1.0 each |
| foot contact force penalty | 0.2 |
| termination penalty | 0.5 |

## Molab commands

From the repository root on `features/kickup`, Python 3.11, CPU MuJoCo:

```bash
uv sync --frozen
# A: actor/critic warm start, NEW optimizers, NEW curriculum, separate run.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2kickup --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 1500 --save_interval 25 --seed 7 --headless --disable_wandb \
  --experiment_name go2kickup_takeoff_finetune \
  --warm_start logs/go2kickup_sync_grf_1500_20261109_201802/Nov09_20-19-51_/model_1225.pt \
  --warm_start_task go2kickup

# B: random initialization, independent experiment.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2kickup --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 1500 --save_interval 25 --seed 7 --headless --disable_wandb \
  --experiment_name go2kickup_takeoff_fresh
```

Replace the warm-start path if using the other 1225 checkpoint. Saved actor and
critic architecture, observation/action lists, scaling and control settings
are imported; new kick mechanics, rewards and curriculum are retained. Strict
state-dict loading rejects incompatible networks. Do not use `--original_cfg`
for training these experiments: that would restore old rewards. `--resume` is
for continuing one of the new experiments with its optimizer/curriculum state.
Both runs collect 512 steps per environment: 8,192 samples per iteration.
Use identical seeds and evaluate multiple training seeds for a robust comparison.
The 1500-iteration experiments are intentionally left for Molab; local validation
runs only one iteration each.

## Physical evaluation

```bash
# Reproduce a baseline with its saved task code and config (trusted local run).
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/evaluate_kickup.py \
  --checkpoint logs/go2kickup_sync_grf_1500_20261109_201802/Nov09_20-19-51_/model_1225.pt \
  --saved-config --episodes 16 --output-dir output/kickup_baseline
# Evaluate with current mechanics/rewards: omit --saved-config.
```

The evaluator performs repeated four-second deterministic episodes at stage
zero by default (`--level 0` through `--level 6`) with 16 CPU environments and checks every reward for finiteness. It saves
checkpoint SHA256, per-episode simultaneous takeoff, sustained supported flight,
maximum minimum clearance, maximum front drift, contact retention and impulse.
`success_percentage` requires at least 60 ms synchronized supported flight at the stage clearance target and
survival; `curriculum_success` is separate, since stage-zero unloading is not
kick-up success. CSV contains per-foot forces, contacts, heights, vertical
velocities, body pitch/rate, joint speeds, applied torques and saturation ratios.
Impulse in the independent evaluator uses control-step sampling; the task's
logged impulse integrates physics substeps. Training `kickup/*` vitals include
force, contact, flight, clearance/drift maxima, rotation, saturation and success.
These training vitals are sample averages, not independent episode success rates.

```bash
uv run python -m pytest tests/unit_tests/test_go2_kickup.py tests/unit_tests/test_kickup_cpg.py -q
```

Smoke tests demonstrate numerical stability and checkpoint compatibility. They
do not establish a trained policy's takeoff reliability or handoff readiness.

## Local validation, 2026-11-09

50 tests passed across kick-up, CPG, handstand curriculum, CPU physics/reset and
contact diagnostics. Fresh and actor/critic warm-start PPO smoke runs completed
one iteration each with 16 CPU environments and 8192 samples, W&B disabled.
Both optimizers and curriculum state were fresh. All measured rewards/losses
were finite. These are smoke runs, not completed 1500-iteration experiments.

Two distinct local checkpoints have iteration 1225. The newer sync/GRF run
(SHA256 `5b4876a3d0fb0c1dd30470da3ad4ea8384ad38499f89077cd35b0ef168711b63`)
reproduced the planted-feet failure with its saved implementation/config:
0/16 coordinated takeoffs, no sustained flight, per-episode maximum minimum
clearance 1.39–1.74 mm. Front contact retention averaged 99.1%; maximum front
drift ranged 3.63–4.96 cm. During 0.70–1.00 s, rear calf torque utilization peaked
at 26.2% left and 36.3% right, far below clipping. Mean rear knee speeds were
+0.52 and -0.017 rad/s. Mean rear forces were 51.1 and 66.6 N; pitch speed was
between -0.274 and +0.274 rad/s. This supports insufficient/coordinated extension
and angular momentum, rather than torque saturation, as the immediate failure.
See `output/kickup_baseline_sync/` for raw CSV and per-episode results.

The older `logs/go2kickup/Nov09_15-08-38_/model_1225.pt` has different saved task
code and behavior; its historical curriculum success is not a physical takeoff
metric. Always identify the run as well as the iteration. The commands above
select the newer checkpoint that reproduced the reported failure.

Evaluating that same newer policy with current mechanics produced coordinated
rear event pairs in 15/16 episodes and maximum minimum clearance up to 8.36 mm,
but **0/16 physically supported successes**: mean maximum front drift increased
to 8.05 cm and contact retention fell to 94.4%. The evaluator correctly rejects
rear release accompanied by invalid front support. These measurements do not
show that the trained objective has been achieved. Fine-tuning must learn to
retain front anchoring under the stronger push; compare this with fresh training.
Current-mechanics results are in `output/kickup_sync_modified/`. Local smoke
checkpoints live under `logs/go2kickup_takeoff_fresh_smoke/`,
`logs/go2kickup_takeoff_finetune_smoke/` (older checkpoint), and
`logs/go2kickup_takeoff_sync_finetune_smoke/` (newer checkpoint).

Changed source files: `gym/envs/go2/go2kickup.py` (controller, contacts, rewards,
curriculum gates and diagnostics), `gym/envs/go2/go2kickup_config.py` (parameters
and weights), `gym/utils/robot_metrics.py` (CSV telemetry),
`scripts/evaluate_kickup.py` (independent evaluation),
`tests/unit_tests/test_go2_kickup.py` (physical event and controller tests), and
this document. Pre-existing README edits and output artifacts were preserved.
