# Go2 trajectory optimization continuation and model-based tracking

22 November 2026. Branch: `experiment/trajectory_optimization_ta`, continued from `bfbe5c7`. This work preserves the original constrained optimizer and CapturePolicy. No PPO training or robot connection was performed. Hardware-test choices remain with course staff; the measurements below describe software experiments.

## What is available

1. **CapturePolicy**, unchanged: the existing 47-input controller and original checkpoint.
2. **Trajectory–LQR**, newly implemented: a 43-input, non-neural inference controller, fitted from native model linearizations and the existing full twelve-joint reference. It completed a nominal standing-to-handstand trial with five seconds of strict support. It is substantially less robust in the tested perturbation/noise cases.
3. **Crouch-start native optimization**, continued: quasistatic IK/inverse-dynamics preparation, endpoint/velocity state matching, and a terminal pulse basis. It has not achieved a complete crouch-to-handstand success. The strongest tested continuation reached 1.20 seconds of strict support.

These are different experiments. A nominal trajectory-tracker success is not a solution certificate for the old zero-cost constrained NLP or a reproduction of the old 67/100 robustness result.

The prior temporary worktree disappeared, but its committed source and the Downloads review ZIP survived. The failed mesh and collision-free warm start were restored into ignored `output/recovered_source` and `output/recovered_warm_start`. The source candidate hash is `d9304a374482c966cf1b86f507e8596c9da7adbe2d74906e57780d8eaf93c3f9`. Work and logs now live in a durable worktree under `.worktrees/trajectory-optimization-ta`.

## Why the formulation changed

The old full-window clearance optimization avoided calf contact but ended about 66 degrees from handstand. Its reduced sagittal warm-start fit averaged left/right commands and had 0.198225 rad RMS fitting error. The continuation therefore retains the full twelve-joint reference and models the actual bounded position-command/native-contact path.

The original zero-task-cost collocation formulation, Euler defects, unilateral/friction constraints, calf inequalities, speed-dependent planning curve and strict export gates remain unchanged. The new native experiments have explicitly nonzero endpoint/tracking costs, following the later instruction to use penalties where useful. They do not silently remove failed equations from the old feasibility certificate.

### Initialization is a state, not just a joint posture

The pinned reference starts at base-origin height 0.3413013268 m and has a nonzero **body pitch rate 0.00616877 rad/s**. Its first 0.4 seconds include a crouch; there is no long static standing prefix. Merely standing up and waiting does not recreate its initial conditions or momentum. Directly starting the reference near its crouch phase also failed in tested native rollouts.

A bug found while implementing the new IK path is covered by a test: `mj_setState` does not update derived geometry transforms. FK must be recomputed before reading foot positions. Otherwise stale terminal-handstand foot locations are mistakenly used as the standing target.

### Command precision matters

The exact original float64 reference replays bit-for-bit and reaches 5.014 s strict support. Nearest float32 bounded targets alone shortened nominal open-loop support to 3.086 s before a calf contact. The reference is dynamically sensitive; a planned pose diagram is insufficient.

All new controller evaluations use the same float32 target conversion as the deployment adapter. Rounding is inward at both joint and applied-target slew bounds:

\[
\ell_k=\max(q_{\min},q^T_{k-1}-0.7s\Delta t),\quad
u_k=\min(q_{\max},q^T_{k-1}+0.7s\Delta t).
\]

The command is clipped to this interval, quantized to float32, and advanced by one representable value inward if rounding crossed a bound. The actor outputs normalized actions; the runner applies the 2 rad/action scale exactly once.

## Native crouch preparation

Start height is 0.27 m, with zero generalized velocity. Quasistatic IK preserves the desired four-foot geometry along a quintic body-height path. A vertical-force least-squares initializer solves the floating-base gravity balance; joint PD preload is

\[
q^T_j=q_j+\bigl(h_j-J_j^Tf\bigr)/K_p.
\]

These forces initialize targets only; no ground force or external root wrench is injected during simulation.

The optimizer has 36 parameters: twelve endpoint offsets, twelve endpoint-slope directions, and twelve late pulse amplitudes. With \(\sigma(s)=10s^3-15s^4+6s^5\),

\[
u(t)=u_{IK}(t)+\sigma(t/T)p_0+
\sigma(t/T)(t/T-1)p_1+
\sigma(r)\sigma(1-r)p_2,
\]

where \(r\) spans the last min(0.45,T) seconds. Every basis vanishes initially. The pulse can excite a small unloading/momentum transition instead of requiring a static endpoint to recreate the moving reference start.

Native shooting minimizes half the squared residual norm. Final pose scales are 1 mm root Z, 2 mrad rotation and 3 mrad joint position; final velocity scales are 2 mm/s translation, 2 mrad/s rotation and 5 mrad/s joint speed. Global root XY is omitted from this preparation merit as a flat-ground translation gauge; its actual error is still reported. Clearance/contact/joint/speed residuals remain explicit. The reference velocity, including its small nonzero pitch rate, is the handoff target.

Finite differences use 0.1 mrad parameter probes; bounds are ±0.3 rad coefficients. The budget is checked between simulations and best candidates are checkpointed. This is local nonlinear least squares, not RL. A solver termination message does not establish task success. Continuous native replay retains all original collision, joint and strict five-second support checks.

Measured terminal velocity matching improved, but remains imperfect. The pulse solve reached merit 3.2293 and approximately 1.51 mm vertical / 0.88 mrad rotational error; its independent pure-reference continuation held only 0.332 s. Additional prefix-feedback variants reached at most 1.20 s. Direct phase-start searches over source steps 0,2,…,26 produced no five-second success. All failed outputs remain in ignored `output/`.

## Model-derived trajectory feedback

The new tracker uses no CapturePolicy checkpoint, critic or learned residual gains. Its reference comes from the disclosed hash-locked existing trajectory; it is not a claim of discovering the maneuver from scratch.

At reference samples, the native 50 Hz PD/contact transition is linearized:

\[
\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k,
\quad x=(\delta q_{18},\delta v_{18}).
\]

State probes are ±1e-4 in tangent-position/velocity coordinates; target probes are ±1e-3 rad. Each hypothetical rollout has its own reset to the linearization state. The actual validation trial is never reset after initialization. Linearization includes the original model, passive terms, native compliant contacts, effort clipping and overspeed protection.

Finite-horizon Riccati recursion uses

\[
K_k=(R+B_k^TP_{k+1}B_k)^{-1}B_k^TP_{k+1}A_k,
\quad P_k=Q+A_k^TP_{k+1}A_k-A_k^TP_{k+1}B_kK_k.
\]

The chosen model spacing is ten control steps after source step 110. The diagonal Q is `[400]*6 + [10]*12 + [25]*6 + [1]*12`; R is `100 I`. A terminal local Riccati iteration supplies the tail value. These are disclosed tracking/control-effort design weights, not RL rewards.

### Sensor-only reconstruction

A local map E converts sensor deviations to model-state deviations using calibrated projected gravity, body gyro, joint positions and velocities. For reference gravity \(g_B\), small observable tilt is

\[
\delta\theta_B\approx-[g_B]_\times\delta g_B.
\]

Front-foot material contact points provide the local no-slip position/velocity relation

\[
\delta p_B\approx-J_{c,q}\delta q,\qquad
v_B\approx-J_c[0,\omega_B,\dot q_j]^T.
\]

The material point is the sphere's instantaneous bottom point, not just its center. The map is linearized offline about the reference, giving output gain \(G_k=K_kE_k\). The deployed actor only multiplies sensor deviations by stored gains; it does not read simulator positions/velocities or integrate accelerometers.

Contact-consistent reconstruction is an approximation, especially under slip/transient contact. A reduced-observation Riccati variant and denser prefix feedback were also tested and did not improve the perturbation results. No global stability certificate is claimed from these local models.

### Complete inference law

A nominal-only gain-margin sweep selected multiplier 0.4:

\[
a_k=\operatorname{clip}_{[-1,1]}
\left(a^\mathrm{ref}_{j(k)}-
\tfrac12\operatorname{clip}_{[-0.1,0.1]}
\left(0.4G_{j(k)}(y_k-y^\mathrm{ref}_{j(k)})\right)\right),
\quad j(k)=\min(k,319).
\]

Gains are zero before source step 110. The 0.1 bound is in target radians, equivalent to 0.05 normalized action. Feedback continues after 319. No action EMA or additional tanh is applied.

The full artifact includes actions `(320,12)`, centers `(320,30)` and feedback gains `(320,12,30)`. A generic neural-weight export would omit essential control law data.

## TA calibration and timing feedback

This controller has **43 observations**, not CapturePolicy's 47:

| Slice | Channels |
| --- | --- |
| 0:3 | calibrated body projected gravity |
| 3:6 | calibrated body gyro, rad/s |
| 6:18 | canonical joint position minus nominal |
| 18:30 | joint velocity / `[30.1,30.1,15.7]` repeated four times |
| 30:42 | previous successfully published target offset / 2 |
| 42:43 | completed control steps / 400 |

There is no COM-height/velocity input, no dummy value and no double integration. Thus accelerometer scale/bias cannot directly change this actor. This architectural property does not imply gyro-noise robustness; the actual noise results below are poor.

The self-contained runtime script is `go2_deploy/custom_implementation/go2trajectory_lqr.py`. It implements all four requested hooks and the same optional prepare/initialize/raw-callback/target-bound/publication callbacks used in Cornell PR #26. It reads `main_controller.last_lowstate_msg` directly. No transport rejection or externally injected per-cycle sensor queue is required.

Full-rate samples are not integrated; the callback tracks unique-tick freshness. The latest synchronized gyro/encoders are sampled at 50 Hz. A 10.6 ms receipt age is accepted; 15 ms receipt and 20 ms pre-command snapshot budgets remain bounded. Frozen timestamps, out-of-order ticks, invalid sensors, wrong initialization, changed gains, missing publication and cancellation fail closed and require a mode reset. Phase advances only on successful DDS publication, not physical motor tracking.

The calibration preset supplies model IMU alignment and the TA gyro-bias estimate. Accelerometer scale, lever arm and initial COM height are not required by this actor. Staff-confirmed IMU calibration is an activation input; the generated preset loads for preflight with its confirmation flag false. Joint order, 50 Hz, Kp 60/Kd 3 and the hardware permutation are unchanged.

## Measured comparisons

All tests use Python 3.11.16, MuJoCo 3.11.0, SciPy 1.16.3 and CPU Torch. The original reference SHA256 remains `21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`.

| Controller / condition | Result |
| --- | --- |
| Original float64 reference, exact source start | 5.014 s; exact replay, state error 0 |
| Float32 open-loop reference, nominal | 3.086 s; calf contact at 5.172 s |
| New trajectory–LQR, gain margin 0.4, nominal | **5.014 s; complete success** |
| Same, joint perturbations ±0.003 rad and ±0.03 rad/s | **0/20 held-out successes** |
| Same perturbation batch plus gyro noise σ=0.007 rad/s | **0/20; nominal also fails (4.392 s)** |
| Crouch preparation + tested prefix tracker variants | Best strict hold **1.20 s**, no full success |

Nominal trajectory–LQR had zero illegal contacts and actual joint-limit excess, speed ratio 0.72748, and all-leg minimum calf clearance **69.56 mm**. Its success video and a failure video are generated by the CLI.

**Actuator qualification:** the new native path uses the original URDF effort clip and overspeed model, matching the CapturePolicy simulator. Its nominal run exceeds the old optimizer's provisional linear torque-speed envelope by **14.248 Nm**. It therefore does not satisfy that original hard-constrained planning certificate. The strict old mesh/export gate remains unchanged. This mismatch is explicitly recorded in the new bundle metadata rather than hidden by changing a motor limit.

The 0.4 gain margin was selected on the canonical nominal trial, not on the perturbation score. The reference data are a prior. Small-sample failed tests and discarded methods are retained; neither passing software tests nor a single nominal success establishes comparable robustness.

## Reproduce

From this research branch, provide the original release reference at `artifacts/handstand_reference/trajectory.npz`. No checkpoint is needed to fit or run the new tracker. Install the existing experiment requirements if SciPy 1.16.3 is absent.

```bash
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/optimize_go2_tracking.py --trials 20 --output output/tracker_NEW
python scripts/optimize_go2_tracking.py --gyro-noise .007 --trials 20 \
  --output output/tracker_noisy_NEW
MUJOCO_GL=egl python scripts/optimize_go2_tracking.py --trials 1 --video \
  --output output/tracker_video_NEW
# On a desktop, replace --video with --viewer and omit MUJOCO_GL=egl.

python scripts/optimize_go2_staged.py --optimize --iterations 80 --max-seconds 180 \
  --output output/preparation_NEW
python scripts/optimize_go2_tracking.py \
  --preparation output/preparation_NEW/preparation.npz --trials 3 \
  --output output/prepared_tracker_NEW
# Direct crouch phase-init experiment, without a preparation or state teleport:
python scripts/optimize_go2_tracking.py --direct-crouch --source-start-step 20 \
  --trials 0 --output output/direct_phase20_NEW

export GO2_HANDSTAND_TRAJECTORY="$PWD/artifacts/handstand_reference/trajectory.npz"
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92
python -m pytest -q
```

Use fresh output directories. `feedback.npz` preserves both raw `base_gains` and selected gains, so reloading it with the same gain scale does not scale twice. A run writes a complete bundle manifest only after independent nominal strict success; failed candidates retain diagnostics without that manifest. Videos require the optional imageio/imageio-ffmpeg packages already used by the deployment preview. No media or large arrays are committed to Git.

### Cornell staff loading

Use Cornell's PR #26 runner checkout (tested exact `4bd82bd` interface). **The inherited research branch's older runner is not interchangeable with that loader.** Copy only the new custom Python file into its `go2_deploy/custom_implementation/` directory and provide the generated bundle directory.

```bash
# Run in the Cornell PR #26 checkout after copying the custom file:
GO2_CUSTOM_IMPLEMENTATION=go2trajectory_lqr.py python -c \
  'from go2_deploy.policy_bundle import load_bundle; p,c=load_bundle("/path/to/tracker_bundle"); print(c.obs_dim(),c.ctrl_freq,c.kp,c.kd,c.exp_moving_avg)'
# Expected: 43 50 60.0 3.0 False. This is loader-only, not a robot connection.
```

`GO2_TRAJECTORY_CALIBRATION` selects a staff-confirmed IMU calibration JSON; otherwise the bundle's `calibration.json` is loaded for preflight. The complete scripted controller and reference/gain buffers load without SciPy, MuJoCo or research modules in the runtime custom file. Actual motor protections, initialization, loads and testing procedures remain staff inputs.

## Remaining work

The nominal model-based controller is a second implemented inference route, but its tested sensitivity is much higher than CapturePolicy. The crouch-start boundary problem is still unsolved. Next useful work is an explicit model/observer diagnostic for unobserved body velocity and changing support, followed by contact-aware feedback and robust native trajectory optimization. The old point-foot Euler feasibility claim should not be revived without independently passing its original constraints and replay.

## Final software validation and provenance

Implementation commit: `8621982`. A clean Git archive of that commit, with the
external release reference provided, passed **300 tests, zero failures, zero
skips**. The repository's default selection deselected 97 optional Warp/VSim/Unitree
backend tests. Fourteen existing TorchScript deprecation warnings were reported.
Ruff lint/format and Git whitespace checks passed for the new code.

The original CapturePolicy was separately rerun with the documented checkpoint,
student-only evaluator, seed 301000 and 100 held-out trials. It reproduced **67/100**.
All 101 complete-success flags and hold durations, including the nominal trial,
matched the archived original results exactly (maximum hold-duration difference 0).

The new controller was then replayed through the actual custom hooks with float32
quaternion, gyro and encoder packets, calibrated using a clearly identified
synthetic-only test configuration. It achieved the same **5.014 s** nominal hold.
Maximum errors across the 43 observations, all normalized actions and all final
commanded targets were **0, 0 and 0**, respectively, against independent calculations
on the same input packets. The command sink acknowledged the array actually
applied to the simulator; it was not a hardware acknowledgement. Tests explicitly
cover source steps 0, 1, 47, 109, 110, 111, 319, 320 and 400.

The exported actor and custom file were also loaded through the exact Cornell
PR #26 loader/configuration at `4bd82bd`, in an isolated checkout to avoid mixing
it with this research branch's older runner. The result was **43 observations,
50 Hz, Kp 60, Kd 3, EMA off**, with the complete `(320,12)` action buffer and
reference/feedback data present. The calibration preset loaded with confirmation
false. No SDK/controller constructor was invoked.

A 500-cycle synthetic callback/observation/JIT/target-conversion benchmark,
excluding 20 warmup cycles, measured **0.332 ms median, 0.553 ms p99 and 0.866 ms
maximum**. It excludes DDS, CRC, logging and real scheduling. This is a software
compute measurement, not a robot timing measurement.

To reproduce the full hook replay from this branch:

```bash
python scripts/optimize_go2_tracking.py --adapter-dry-run --trials 0 \
  --output output/adapter_replay_NEW
```

The local review archive includes the complete actor bundle, self-contained
custom file, source hashes, videos, numeric reports and archived preparation
parameters. It excludes the protected PPO checkpoint and training outputs. The
original release reference remains an external, hash-locked input. GUI operation
was not evaluated in the headless environment; MP4s were generated and inspected.
The source branch and preserved failure reports are part of the methodology,
not claims that the failed crouch/perturbation conditions passed.
