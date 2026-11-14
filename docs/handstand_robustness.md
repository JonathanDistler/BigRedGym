# Handstand robustness experiment

This is an incomplete research pipeline, not an autonomous handstand result.
The immutable nominal replay succeeds. The recovered balance actor succeeds with
recorded transition commands, but the learned full-maneuver student fails before
takeoff. A projected trajectory-LQR candidate also fails. Long PPO training is
therefore gated. No physical robot commands or SDK connections are used.

## Audit and reference identity

Work is isolated on `experiment/go2-handstand-robustness`, starting at
`3e1faf30981802e183764cfd71888ffcffdfc103`. Main refs and the upstream PR are untouched.
The centroidal and safety research branches point to
`05b8fe09f016a4736f5849ef9e1aece337434125`; cleanup points to
`55090411e37e9affa1e0856943f31d7a8c79991e`.

The complete command sequence is the hash-locked schema-1 `actions[0:320]` in
`artifacts/handstand_reference/trajectory.npz`, copied from the adjacent research
checkout without changing the original. The copy has write permission removed;
SHA256 validation, rather than filesystem WORM storage, establishes its identity.
`configs/handstand_reference.json` records hashes, initial state, joint order,
gains, timing and limits. The original reference artifact is available from the
[user-supplied release](https://github.com/JonathanDistler/BigRedGym/releases/tag/assignment6-handstand-data-v1).
The release page could not be independently fetched in this session; the local
artifact was verified by hash and exact continuous dynamics replay.

The validated construction preserves 66 control commands through 1.32 seconds.
It retains subsequent reference commands for 0.05048272 seconds relative to that
point: control steps 66, 67 and 68. The saved balance actor takes over at step 69
(1.38 seconds). The front thigh and calf normalized offsets are -0.0204364 and
+0.10256115, with a 0.64517335-second trapezoidal envelope, 0.10-second rise and
0.15-second fall. These values come from the preserved height-stabilization
manifest and source `scripts/stabilize_centroidal_capture.py`. They are not a new
optimizer result. Both replay and the recovered hybrid retain the correction
through the post-prefix pulse. Changing this short capture segment is not benign.

The local `output/retained_go2_preparation_v3/balance.pt` matches the original
`output/balance_feedback_v2/balance_actor.pt` hash
`bcd6a1f5b59ebe905b73be8080e7a21e1b714f22a7b239c9111d2ee4a8b94fcd`.
It is a 34-channel affine neural feedback actor, analytically initialized from
LQR, not a PPO-trained full transition. Its original near-vertical validation
reported three successes in four correlated resets. Those results are prior
research, not newly reproduced independent population statistics.

The legacy 93-channel ordering is gravity 0:3, original free-joint angular
velocity 3:6, joint offset 6:18, joint velocity divided by speed limit 18:30,
previous target 30:42, time/8 at 42, morphology scale at 43, old phase 44:47,
momentum 47:61, contact context 61:73, extra phase context 73:93. Only channels
0:30, 55 (COM height/0.6) and 56:59 (COM velocity) affect the affine actor.
There is no running normalization. Simulation COM quantities make this a
privileged teacher/baseline until a corresponding hardware estimator is validated.
Outputs are already clamped normalized position actions; do not apply tanh twice.

MuJoCo integration is 500 Hz and position control is 50 Hz, kp=60, kd=3.
Actions obey `q_desired = nominal + 2*action`, joint ranges, target slew at
0.7 times URDF joint-speed limits, torque limits, and the original overspeed
cutoff. Joint order is FL, FR, RL, RR, each hip/thigh/calf. This is MuJoCo native
order; it is not a validated mapping to the physical Go2 API.

The standing start is the exact recorded integration state at time zero, with
upright quaternion, nominal feet at 0.022 m, no external force, and base above
0.3 m. Reset perturbations are drawn once at episode start. Joint limits,
prohibited contacts and penetration over 3 mm are rejected. Small compliant foot
penetration is retained; there is no settling teleport or mid-episode reset.
The tested reset envelope is deliberately small and not a validated broad reset
distribution. Joint noise changes foot placement kinematically.

The existing 500 Hz takeoff certificate requires prior bilateral hind loading,
recent powered extension and positive work, synchronized release within 40 ms,
both rear feet clearing 15 mm, front support and limited front-foot drift.
Capture requires both front feet loaded above 5% bodyweight and rear clearance
above 0.10 m. The original strict balance gate is imported unchanged. It also
checks orientation, COM motion and support, heights, slip, forbidden collisions,
front-calf/head clearance and front load. Five continuous seconds remain mandatory.

## New actor and failed pilots

The student receives 43 channels: projected gravity, body-frame angular velocity,
12 joint offsets, 12 normalized joint velocities, 12 previous filtered targets,
and time/8. It does not receive COM, global position, contact forces, future
states or reference actions. The MLP is 43–128–128–12 with tanh outputs. Fixed
mean and standard deviation come from the demonstration, with std floor 0.05;
they are checkpoint buffers. Phase is permitted, but a successful student must
also demonstrate useful feedback under disturbances. Contact and base-velocity
estimators are still missing from this student.

The initial 500-epoch cloning pilot used only measured observation/action pairs
from the valid nominal rollout. Low nominal action MSE did not transfer to a
successful continuous standing-start rollout. Freezing observations and perturbing
gyro inputs measure dependency, not disturbance recovery. No successful DAgger
labels are fabricated. BC can optionally collect labels from the trajectory
feedback candidate, but accepts only complete successful continuous episodes;
that candidate has not passed its gate.

The model-based candidate linearizes actual bounded PD dynamics along the
reference, including the target state, then solves finite-horizon LQR. Offline
linearization uses privileged state; inference projects gains into the same
43 sensor channels and phase. It is explicitly not a neural student. The large
projection residual indicates that removing unobserved state loses important
feedback. Its nominal failure is retained. It is not selected as a controller or
recovery teacher.

## Evaluation definitions and pilot limits

See `../output/assignment6_metrics.md` for generated comparison tables and the
ignored `output/final_*/evaluate.json` files for individual failures. Tests use
matching seeds starting at 14110. Two trials per each of 25 conditions are diagnostic only;
fixed nominal and fixed pushes are deterministic duplicates. They do not support
population-level confidence claims. Wilson intervals are emitted for genuinely
sampled reset conditions; duplicates must not be interpreted as independent data.
A 100-trial final evaluation is deferred until an autonomous skill exists.

Pushes are world-frame 10 N forward/backward/lateral or 1 Nm about world y,
lasting 0.10 seconds, applied at the base COM. Times are 0.6 seconds (takeoff),
0.688 seconds (first capture gate), 1.32 seconds (height stabilization),
and 3 seconds (hold). These are documented fixed windows,
not online event detectors. The JSON includes full wrenches, frame and timing. The nominal first takeoff
certificate is at 0.638 seconds, first capture gate at 0.688 seconds, and
first strict interval starts at 1.324 seconds; the longest interval is 5.014 s.
The first strict interval may be brief and is not automatically the certified
five-second interval.
The simulator-native force convention is force then torque.

Recovery requires 0.2 continuous seconds of the unchanged strict gate after the
pulse, legal continuation, and an established pre-push hold. Takeoff/capture
cases without prior balance are called post-push acquisition. Recovery latency
is measured from pulse end to the beginning of that stable interval; control
steps are rounded upward at 20 ms. They are not foot-placement steps. Maximum
axis deviation is measured after pulse onset through episode end. Return of
fixed replay to the gate is not evidence of feedback disturbance recovery.
The episode remains 6.4 seconds: a late interruption can recover yet fail to
complete a new five-second hold within that horizon. The full-success gate is
never shortened to accommodate this.

Parameter probes are friction x0.9, uniform link mass/inertia x1.05, available
torque x0.95, and one control-step (20 ms) command delay. These are small
sensitivity tests, not calibrated hardware distributions. Recomputing MuJoCo
mass constants also changes nominal reference constants; the separate
`mass_recompute_control` test exposes that confound. Do not attribute its failure
to mass alone. Delay is a command queue, not an identified motor-response model.
Random sensor errors, COM velocity resets, a progressively advanced curriculum,
and mixed starts are not yet implemented/validated.

Assignment 6 hold metrics use the first five seconds of the longest valid strict
interval. Failed short intervals are labeled partial. Axis error is the angle
between -body-x (shoulder-to-hip direction) and world +z, separately from the
original full-rotation strict-gate error. Pitch uses the recorded heading axis.
Sign changes ignore |error| <=0.1 degrees; frequency is sign changes per second,
not full oscillation cycles per second. Front-foot slip integrates the mean of
both geom-center horizontal speed norms at 2 ms, in meters, excluding takeoff.
Front-leg kinetic energy sums all FL_/FR_ links, including feet. Linear and
angular velocities use COM Jacobians; world angular velocity is rotated into
the inertial principal frame before applying actual MuJoCo inertias. The reported
value is a time average in joules. Action smoothness is mean squared normalized
command difference per control step, not torque jerk. Perturbation sweeps have
not established a maximum recoverable disturbance.

## Training status and resource limits

The preserved BigRedGym PPO implementation was inspected. This isolated pilot
uses a small standard clipped-Gaussian PPO loop because the minimal native replay
has no task-registry vector environment/critic contract. It reuses the upstream
MuJoCo model and strict evaluator, without importing the research framework.
The Gaussian samples latent commands and tanh maps them into bounded actions;
the transform Jacobian cancels in the PPO likelihood ratio. GAE uses terminal
full episodes, gamma=0.99, lambda=0.95, clip=0.2, four epochs, Adam, gradient clip 1.

`--envs` is a count of serial complete episodes per update, not simultaneous
worker processes. `--rollout-steps` currently denotes optimizer minibatch size,
not truncated rollout length. This is a deliberate pilot limitation; the
requested 16–64 parallel/64–128 rollout-length benchmark is not implemented or
claimed. `configs/handstand_ppo_5000.json` is a gated proposal, not a measured
optimal configuration. No PPO skill pilot or long run has been launched on the
failed student. A two-update deterministic surrogate test verifies actual PPO
optimizer/RNG/normalization resume; it does not prove a healthy MuJoCo PPO pilot.

Checkpoints contain actor, critic, log std, optimizer, Python/NumPy/Torch RNGs,
observation statistics, update count, stage-A curriculum state and selection
history. Resume requires the same trajectory hash, seed, episode count and
minibatch size. Updates start fresh episodes, so no unsaved simulation state is
needed for exact continuation. Saves are atomic at evaluation boundaries and
milestones 50/250/500/1000/2500/5000. `latest.pt`, `best.pt`, and milestone snapshots
are separate. Selection prioritizes actual complete strict success and then hold
duration; reward alone never selects a checkpoint. Current stage-A validation is
nominal and deterministic, not sufficient for robust held-out checkpoint selection.
Plateau stopping defaults to five evaluations. Stage B–E advancement and protection
against forgetting are future work; the serialized curriculum never implies that
they have been trained. JSON curves include training reward/loss and validation.

The machine exposes eight CPUs, 5.6 GiB RAM and no available Torch CUDA device.
MuJoCo CPU is feasible; Warp/CUDA is not available. Licensed VSim was not evaluated.
No assumption of GPU acceleration or 4096 environments is made. The measured
native rollout benchmark includes strict 500 Hz scoring and Assignment 6 metrics;
see `output/final_cpu_profile/benchmark.json`: 50.27 control transitions/s and
737.1 MiB peak RSS for a 320-step successful rollout in the first benchmark;
the final repeated profile measured 54.88 transitions/s at about 100% of one
CPU core. Resource measurements are process
peak RSS, not total WSL memory. Candidate worker counts and
training throughput remain unmeasured because the functional student gate failed.
At 50 transitions/s, 5000 updates with 16 complete 320-step episodes would
require approximately 5.9 days of simulation alone; this is an optimistic
extrapolation, not measured PPO throughput. No uncontrolled background training
process is left running.

## Commands

From this checkout, the existing compatible local interpreter is:

```bash
export HANDSTAND_PYTHON=/home/jonathandistler/Distler_Gym/BigRedGym/.venv/bin/python
export GO2_HANDSTAND_TRAJECTORY="$PWD/artifacts/handstand_reference/trajectory.npz"
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92
```

On a new Python 3.11/Molab host, extract the portable bundle, install `uv`, and
run `uv sync --frozen --no-dev`; set `HANDSTAND_PYTHON=$PWD/.venv/bin/python`.
The existing lock pins MuJoCo 3.11.0, Torch 2.13.0, NumPy 2.4.6 and TensorDict
0.13.0. No remote runtime has been tested and no cloud training has been launched.
Do not replace the pinned MuJoCo version for this benchmark.

```bash
# Exact original replay regression:
env -u PYTHONPATH "$HANDSTAND_PYTHON" scripts/replay_go2_handstand.py \
  --trajectory "$GO2_HANDSTAND_TRAJECTORY" --trajectory-sha256 "$GO2_HANDSTAND_SHA256"

# New imitation pilot and measured CPU rollout:
scripts/run_handstand_robustness.sh bc --epochs 500 --output output/bc_new
scripts/run_handstand_robustness.sh benchmark --output output/benchmark_new

# Evaluate current student (not a successful controller):
scripts/run_handstand_robustness.sh evaluate --checkpoint output/bc_sensor_pilot/bc.pt \
  --trials 2 --output output/student_eval_new

# Replay baseline and replay-transition/live-balance hybrid:
scripts/run_handstand_robustness.sh evaluate --trials 2 --output output/replay_eval_new
scripts/run_handstand_robustness.sh evaluate \
  --teacher /home/jonathandistler/Distler_Gym/BigRedGym/output/retained_go2_preparation_v3/balance.pt \
  --trials 2 --output output/hybrid_eval_new

# Reproduce the failed model-based candidate:
scripts/run_handstand_robustness.sh build-feedback --output output/feedback_new
scripts/run_handstand_robustness.sh benchmark \
  --feedback output/feedback_new/controller.npz --output output/feedback_eval_new

# Infrastructure-only short PPO command; not recommended for the failed student:
scripts/run_handstand_robustness.sh train --checkpoint output/bc_sensor_pilot/bc.pt \
  --updates 2 --envs 1 --eval-every 1 --output output/ppo_smoke
# Exact continuation to cumulative update 3:
scripts/run_handstand_robustness.sh train --checkpoint output/ppo_smoke/latest.pt \
  --resume --updates 3 --envs 1 --eval-every 1 --output output/ppo_smoke

# 5000-update option: intentionally refuses the present failed initializer.
# Use only after a healthy nominal skill, simulator resume smoke and resource benchmark.
scripts/run_handstand_robustness.sh train --checkpoint output/ppo_smoke/latest.pt \
  --resume --updates 5000 --envs 1 --eval-every 10 --output output/ppo_smoke

# Evaluate a future selected PPO checkpoint; --trials 100 is the eventual final gate:
scripts/run_handstand_robustness.sh evaluate --checkpoint output/ppo_smoke/best.pt \
  --trials 100 --output output/ppo_best_heldout

# Regenerate the measured Assignment 6/comparative table:
env -u PYTHONPATH "$HANDSTAND_PYTHON" scripts/report_handstand_robustness.py \
  --evaluation Replay=output/final_replay/evaluate.json \
  --evaluation Hybrid=output/final_hybrid/evaluate.json \
  --evaluation Student=output/final_student/evaluate.json \
  --output output/assignment6_metrics.md

# Focused regression plus meaningful exact-resume test:
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$HANDSTAND_PYTHON" -m pytest \
  tests/unit_tests/test_handstand_replay.py tests/unit_tests/test_handstand_robustness.py -q

# Portable assets; never add the generated bundle/checkpoints to Git:
env -u PYTHONPATH "$HANDSTAND_PYTHON" scripts/package_handstand_robustness.py \
  --reference "$GO2_HANDSTAND_TRAJECTORY" --checkpoint output/bc_sensor_pilot/bc.pt \
  --teacher /home/jonathandistler/Distler_Gym/BigRedGym/output/retained_go2_preparation_v3/balance.pt \
  --output artifacts/handstand_robustness_bundle.tar.gz
```

Outputs and checkpoints remain ignored. Keep the generated manifest/checksums with
any transferred bundle. Load only trusted Torch checkpoints; they contain pickle.

## Hardware readiness — no deployment

| Item | Status | Evidence or missing work |
|---|---|---|
| Simulation action bounds | PASS | Original joint, effort, speed and target-slew constraints retained |
| Strict replay regression | PASS | 5.014 s, zero state error, no prohibited contacts |
| Autonomous full neural skill | FAIL | Student collides before certified takeoff |
| Robust complete maneuver | FAIL | Small perturbations defeat reference/hybrid; no robust autonomous result |
| Student observation availability | NOT EVALUATED | IMU/encoders plausible; estimator errors, synchronization and contact/velocity estimator absent |
| Legacy teacher hardware observations | FAIL | COM height/velocity lack validated hardware estimator |
| Go2 action API and ordering | NOT EVALUATED | MuJoCo native ordering and PD semantics not mapped or tested on hardware |
| Real torque/current limits | NOT EVALUATED | URDF caps do not establish real current, voltage or gearbox capability |
| Temperature and sustained loading | NOT EVALUATED | No thermal model or real motor data |
| Landing contact impulses | NOT EVALUATED | No hardware load/impact qualification |
| Frequency/end-to-end latency | NOT EVALUATED | 50 Hz simulated commands; no real timing measurements |
| Smoothness | NOT EVALUATED | Simulation command-change statistic only; physical jerk/impact limits absent |
| Sensor/actuator mismatch | FAIL | Limited parameter probes fail; no system identification |
| Out-of-distribution behavior | FAIL | No validated recovery region or deployable supervisor |
| Safe abort and supervision | FAIL | No deployable watchdog, fall handling or safe-abort controller |
| Controlled hardware test environment | NOT EVALUATED | No rig, spotting, exclusion zone or hardware test procedure validated |
| Overall sim-to-real readiness | FAIL | Simulation success cannot establish physical safety |

Next useful work is a valid transition/capture feedback teacher and an observable
support/velocity estimator, followed by verified correction trajectories and a
student that retains nominal takeoff/capture/balance. Only then should PPO tuning,
stage advancement, worker benchmarking and broad held-out testing consume a long
training budget. Hardware deployment remains outside this work's scope.
