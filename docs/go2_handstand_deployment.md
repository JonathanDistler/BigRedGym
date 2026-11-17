# Go2 handstand deployment integration candidate

**NOT READY FOR HARDWARE.** Software-only staff-review candidate based on research
`a38b193` and Cornell `fixdeploy` at
`8ca9bf98373a234de662f9fb8242a228b7efde21`. No SDK connection, robot command,
training, main-branch edit or change to the original controller was performed.

The provisional team file is
`go2_deploy/custom_implementation/handstand_provisional.py`. No established group
name was found in the reviewed handstand documentation. Staff must supply the
team filename and select it in Cornell's `utility/import_custom_impl.py`.
Renaming alone does **not** complete runtime integration.

## Startup and software-only reproduction

Use Python 3.11 and MuJoCo **3.11.0**, with CPU Torch, NumPy and pytest.
The recorded environment used Python 3.11.16 and Torch 2.13.0+cu130 on CPU.
The project lockfile supplies dependencies (`uv sync --frozen --no-dev`);
verify `mujoco.__version__` before running. `PYTHONPATH` must not point to another
Python environment. No SDK extra or network interface is needed for these tests.

The committed inference bundle is `artifacts/handstand_deployment/`:

- `capture_checkpoint.pt`: original trusted release checkpoint, SHA256
  `90117fc2ff80033663578e12ba664e4b1ff4c15c4c04f19619409a800458ba6c`.
- `trajectory.npz`: unchanged reference, SHA256
  `21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`.
- `policy.pt`: scripted **complete** 47→12 capture/balance controller, including
  320 actions, centers, scale, gains, bound and active-step boundary.
- `go2_fk.mjb`: nominal model masses, inertias, geometry and joint kinematics,
  serialized with MuJoCo 3.11.0. Used for FK/Jacobians only; never stepped by the
  deployable estimator. It is a version-specific binary, not a portable URDF.
- `deploy_meta.json`: per-file hashes, joint order, posture, initial target,
  configuration and required runner changes. Hashes provide integrity against
  accidental corruption; the release and source remain the provenance trust root.

```bash
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/go2-mpl
export PY="$PWD/.venv/bin/python"
mkdir -p output/capture_ppo_conservative artifacts/handstand_reference
cp artifacts/handstand_deployment/capture_checkpoint.pt output/capture_ppo_conservative/latest.pt
cp artifacts/handstand_deployment/trajectory.npz artifacts/handstand_reference/trajectory.npz
export GO2_HANDSTAND_TRAJECTORY="$PWD/artifacts/handstand_reference/trajectory.npz"
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92
"$PY" -m pytest tests/unit_tests/test_handstand_replay.py \
  tests/unit_tests/test_handstand_robustness.py tests/unit_tests/test_handstand_takeoff.py \
  tests/unit_tests/test_handstand_capture.py tests/unit_tests/test_handstand_deployment.py -q
"$PY" scripts/export_handstand_deployment.py --output output/new_review_bundle
"$PY" -c 'from go2_deploy.handstand.bundle import load_review_bundle; p,m,c=load_review_bundle("output/new_review_bundle"); print(c["status"])'
"$PY" scripts/validate_handstand_deployment.py --trials 100 --output output/adapter100
"$PY" scripts/measure_handstand_sensor_drift.py output/sensor_drift.json
"$PY" scripts/replay_handstand_sensor_noise.py \
  --recording output/adapter100/nominal_imu.npz --output output/sensor_replay.json
```

For the original baseline, use the exact command in `handstand_reproduce.md`
with `--checkpoint artifacts/handstand_deployment/capture_checkpoint.pt`.
That legacy CLI also hashes unused balance/feedback teachers; obtain those
from the reference release bundle as documented there, and pass their paths
with `--balance` and `--feedback`. These teachers are not needed for selected
controller inference or the new adapter regression.

Exports refuse to overwrite an existing output directory. `load_assets` validates
the immutable checkpoint/reference hashes before unpickling the trusted checkpoint,
then checks checkpoint kind, reference metadata, tensor shapes, finite values,
active steps, bound and reference actions. `load_review_bundle` validates the
bundle metadata/hashes, warms inference before activation, and loads the FK model.
Neither loader constructs a robot controller or transport.

## Observation contract

Canonical joint order: FL, FR, RL, RR, each hip/thigh/calf. All angles are radians,
angular velocities rad/s, linear velocities world xyz m/s. Float32, one frame,
no observation history or running mean/std. The exact nominal array is preserved
in reference and bundle metadata, including tiny signed hip offsets.

| Slice | Size | Piece |
|---|---:|---|
| 0:3 | 3 | body projected gravity `R.T @ [0,0,-1]` |
| 3:6 | 3 | body gyro, unscaled |
| 6:18 | 12 | encoder position minus exact nominal |
| 18:30 | 12 | encoder velocity / `[30.1,30.1,15.7] * 4` |
| 30:42 | 12 | `(previous applied absolute target - nominal) / 2` |
| 42:43 | 1 | completed command steps × 0.02 / 8 |
| 43:44 | 1 | estimated world COM height / 0.6 |
| 44:47 | 3 | estimated world COM velocity |

Every custom observation getter reads the same immutable snapshot and returns
already normalized values, so every observation divisor is 1. The custom
previous-target piece is deliberately named `handstand_previous_target`:
Cornell's separate `DeployScaling.dof_pos_target=2` is the action multiplier.
Reusing that name for an already normalized observation would scale it twice.
The existing hardware permutation is preserved:
`[3,4,5,0,1,2,9,10,11,6,7,8]`, in both directions, with no sign changes.

## Estimator, initialization and sample synchronization

`SensorEstimator` integrates every distinct timestamped calibrated body-origin
specific-force sample once: `a_world = R @ force + [0,0,-9.81]`, then velocity,
then height. At 2 ms dt this reproduces research semi-implicit Euler mathematics.
A separate FK state receives quaternion, body gyro, encoders and zero root
translation velocity. World COM height adds the FK COM offset to integrated root
height; world COM velocity adds subtree COM Jacobian × generalized velocity to
integrated root velocity. No root position/velocity, simulator COM, solver
contacts or force measurements enter this implementation.

Only the simulator test harness synthesizes ideal body-origin accelerometer data
from simulation acceleration. Its generated sensor recording is explicitly
simulation telemetry, not an SDK recording. The research and ported observers
consume the same samples. Diagnostic true states stay in the simulator scorer.

Explicit standing calibration supplies initial root height, zero root velocity,
world/body alignment and canonical encoders. `IMUCalibration` accepts measured
body-from-IMU/world rotation matrices, IMU lever arm and biases. It subtracts
`alpha × lever + omega × (omega × lever)` to refer acceleration to the body
origin. Gyro differentiation is unfiltered and can amplify noise; constants,
stationarity, timing and lever-arm derivative quality require physical validation.
The boolean `calibrated=True` is an explicit software contract, not evidence that
calibration has occurred. No physical calibration values are provided or assumed.

A 50 Hz control snapshot must follow ten expected 500 Hz samples. Getters never
integrate. Duplicate/out-of-order/nonfinite samples and gaps >3 ms are rejected;
estimates older than 6 ms or from the future are rejected. These software thresholds
are review settings, not qualified hardware tolerances. Sensor/encoder timestamps
must share a calibrated clock. Cornell exposes a latest-message slot rather than
a lossless timestamped queue: no claim of real 500 Hz delivery is made.

## Complete hybrid law and command path

The prefix uses phase `round(obs[42]*400)`, clipped to 0:109, and 46
reference-centered features (time excluded). Scales are `.02` for gravity,
`.1` for gyro, `.02` for the 36 joint/history channels, and `[.02,.1,.1,.1]`
for COM channels. Normalized deltas clamp to ±20. Prefix correction is
`bound*tanh(gain[k] @ delta)` with the saved bound .05.

At step **110 / 2.2 s**, use the saved 12×34 balance weight on raw reference
errors `[obs[:30],obs[43:47]]`. Clip correction to ±.05 and add the selected
reference action, then clip action to ±1. The late law intentionally does not
use saved legacy balance bias/lower/upper. After step **319**, keep its reference
row while continuing live feedback; elapsed time itself continues increasing.
TorchScript scripting preserves both branches; generic zero-vector tracing is
not used. Critic and optimizer are unnecessary for inference; original checkpoint
is retained intact in the bundle.

At 50 Hz, desired absolute target is `clip(nominal + 2*clip(action,-1,1), limits)`.
Applied increments are capped at `[.4214,.4214,.2198]` rad per leg each command.
`TargetPath` stores float64 absolute applied targets, reproducing simulation
arithmetic, and feeds the previous applied target back to the next observation.
There is no extra tanh or EMA. Research PD is Kp=60/Kd=3, evaluated at 500 Hz,
with `[23.7,23.7,45.43]` Nm effort limits and overspeed torque suppression.

`Session.activate` resets estimator, target and command count from an explicit
standing epoch; an initial epoch sensor snapshot integrates zero elapsed time.
`propose` returns normalized action and bounded absolute target, and does not
advance phase. `acknowledge` advances exactly once only if the applied target
matches. A pending command blocks another proposal. Control intervals differing
from 20 ms by >2 ms fault the session. `cancel` prohibits further proposals;
reactivation resets all state. These are offline operator semantics; physical
cancel/recovery response must be approved by staff, including loss of process.
Software refusal to produce another target is not a physical stop guarantee.

## Actual Cornell interface compatibility and blockers

The four hooks match exact template signatures. Their stock-runner behavior
is fail-closed when no reviewed Session/snapshot provider has been attached.
Pinned Cornell source is retained under `tests/fixtures/cornell_fixdeploy/`.
The test executes the actual `DeployConfig` and `policy_bundle.load_bundle`
without SDK imports and verifies the complete exported actor can load.

**The Cornell template loads the full inference law, but its unchanged runner
cannot execute the complete bounded sensor/control pipeline.** Required changes:

1. `_on_lowstate_msg` must timestamp/calibrate/enqueue every synchronized IMU and
   encoder sample with bounded queue overflow detection. `_control_loop` must
   consume all fresh samples once before building one atomic snapshot.
2. Apply actual 50 Hz, EMA off, exact nominal posture, kp=60/kd=3. The stock
   bundle loader retains **100 Hz / EMA on** even when metadata includes the
   correct settings; a regression test demonstrates this. Additional custom
   class attributes alone do not override those runtime fields.
3. Replace the action application with the bounded target path, multiply scale
   exactly once, store actual applied target, acknowledge after successful send,
   and reset phase on deliberate activation rather than absolute monotonic time.
   The custom `Session` is the reference software contract for this change.
4. Reconcile physical actuator semantics with staff: Cornell `action_to_lowcmd`
   sends position/kp/kd and zero feedforward torque/velocity. It does not implement
   research 500 Hz effort clipping or overspeed torque suppression. The four hooks
   cannot enforce those motor-side protections. Their equivalence is **FAIL**.
   No motor limit is bypassed or weakened to obtain parity.
5. Approve watchdog, nonfinite/stale-data handling, explicit operator activation,
   cancellation, process-death behavior and recovery. Stock emergency behavior
   includes recovery stand commands; suitability after a handstand fall is unknown.

The smallest next action is course-staff review of timestamped ingestion and
actuator protection capabilities at the pinned interface, followed by an approved
software runner patch. Do not attach the current four-hook candidate to a robot.

## Results and readiness checklist

See `docs/validation/go2_handstand/` for committed raw reports, exact counts,
channel errors, noise sensitivity and timing. The target remains **67/100**
held-out successes on seeds 301001–301100 plus a separate nominal success, using
unchanged 8-second horizon, perturbations and continuous five-second strict gate.
The new runner records all trial metrics and sensor-stream parity; synthetic
noise results are input-distribution effects, not code-parity failures.

| Check | Status |
|---|---|
| Immutable checkpoint/reference identities and complete inference export | PASS |
| Exact Cornell bundle loader can load complete inference law | PASS |
| Boundary phases 0,1,47,109,110,111,319,320,400 | PASS |
| Joint mapping round trip, units, normalization and slew tests | PASS |
| Stale/invalid sensors, corrupt/missing assets, wrong metadata/order rejection | PASS |
| Offline activation/reset/cancel and applied-target acknowledgement | PASS |
| Original/adapter full matched simulation regression | See measured report |
| 50 Hz/EMA semantics through unchanged Cornell runner | FAIL |
| Lossless synchronized 500 Hz physical telemetry | NOT EVALUATED / blocked |
| Physical effort/overspeed equivalence through supported interface | FAIL |
| Physical COM calibration, bias/noise/lever-arm qualification | NOT EVALUATED |
| Hard real-time deadlines on deployment computer | NOT EVALUATED |
| Front-contact loads, harness, exclusion area and failure recovery | NOT EVALUATED |
| Course-staff supervised procedure and operator approval | NOT EVALUATED |
| Overall hardware readiness | **NOT READY** |

No software test or simulated success authorizes a physical attempt.
