# Go2 handstand deployment gaps

Static audit of `experiment/go2-handstand-robustness` at `a545e5b`, compared with
[LampLighterLab/BigRedGym fixdeploy](https://github.com/LampLighterLab/BigRedGym/blob/8ca9bf98373a234de662f9fb8242a228b7efde21/go2_deploy/custom_implementation/go2trot_base.py).
GitHub branch tip verified as `8ca9bf98373a234de662f9fb8242a228b7efde21`.
All local line references below refer to a545e5b; deployment references refer to
that pinned upstream commit. No training, simulation, SDK connection or source edits.

## Controller identity and three priority fixes

The current successful controller is **CapturePolicy (`capture_residual_v1`),
47 inputs / 12 outputs**, not the failed 43-input BC student or the privileged
93-input replay-transition/balance hybrid. The documented evaluated checkpoint is
`output/capture_ppo_conservative/latest.pt` (update 2): nominal success and 67/100
held-out standing-start successes under ideal calibrated simulation sensors
(`docs/handstand_capture.md:95–119,135–161,176–180`). This is conditional simulation
success, not established hardware readiness.

1. **Build the exact sensor/COM estimator and observation contract.** Replace the
   template's 48-channel trot vector with the 47-channel table below; calibrate IMU
   frame, accelerometer origin/bias, initial height and stationary start, and
   integrate timestamped IMU samples at 500 Hz. The new controller uses estimated
   COM, not simulator COM. Sources: `capture.py:356–399`, `core.py:120–142`.
2. **Match actuator semantics before using policy outputs.** Set 50 Hz, kp=60,
   kd=3, scale=2 rad/unit, exact reference nominal posture; disable upstream EMA
   and implement joint clipping, target slew, effort and overspeed protection.
   Preserve previous *applied target* in observations. Sources: `core.py:144–180`,
   `handstand_replay.py:23–25,163–175`; upstream `deploy_config.py:10–18`,
   `main_controller.py:192–201`, `utility/deploy_utility.py:99–120`.
3. **Export the complete capture/hold law and phase buffers.** Preserve checkpoint
   actions, centers, gains, scale, bound, active_steps and balance buffers;
   advance phase from maneuver activation, switch at step 110 (2.2 s), and hold
   the last reference index after step 319 while retaining feedback. The generic
   trot loader/exporter cannot directly reconstruct this checkpoint. Sources:
   `capture.py:402–451`, `takeoff.py:20–52`, `scripts/handstand_takeoff.py:89–111`;
   upstream `rl_controller.py:10–46`, `policy_bundle.py:49–64`.

Here `core.py`, `capture.py`, `takeoff.py` mean files under
`experiments/handstand_robustness/`; `handstand_replay.py` is under `gym/envs/go2/`.

## Exact observation and normalization

Float32, one frame, no stack. Joint-valued entries use FL, FR, RL, RR, each
hip/thigh/calf. Let q0 be trajectory `nominal`, s=`[30.1,30.1,15.7]` repeated four
times, qT the previous clipped/slew-limited absolute target, and k the completed
control-step count. R maps body to world.

| Zero-based slice | Size | Value delivered to policy |
|---|---:|---|
| 0:3 | 3 | projected gravity `R.T @ [0,0,-1]` |
| 3:6 | 3 | body angular velocity, rad/s (`qvel[3:6]` in MuJoCo free joint) |
| 6:18 | 12 | `q - q0`, radians |
| 18:30 | 12 | `dq / s` |
| 30:42 | 12 | `(qT - q0) / 2`, normalized previous applied target |
| 42:43 | 1 | `k * 0.02 / 8` (time/8; no cyclic trot phase) |
| 43:44 | 1 | estimated world COM height / 0.6 |
| 44:47 | 3 | estimated world COM velocity xyz, m/s |

Sources: `core.py:120–129`, `capture.py:376–399,464–467`.
Capture normalization is **reference-centered**, not running mean/std:
`round(obs[42]*400)` selects a center (prefix index clipped to 0:109).
Subtract its first 42 and last four channels, divide by
`[.02]*3 + [.1]*3 + [.02]*36 + [.02,.1,.1,.1]`, clip to [-20,20].
Phase itself selects the row and is excluded from the 46 feedback features.
Prefix action is `clip(reference_action + .05*tanh(gain[k] @ delta), -1, 1)`.
From step 110, use 34 raw features `[obs[:30], obs[43:47]]` minus corresponding
reference centers, multiply by `balance_weight`, clip correction to ±bound
(default .05), add reference action and clamp to [-1,1]. The late law does not
use the prefix scales or balance_bias/lower/upper. Do not add another output tanh.
Sources: `capture.py:405–451`, `takeoff.py:45–52`.

The template instead orders gyro, gravity, commands, q, dq, previous target,
sin/cos phase and frequency (48 total). Its inherited divisors include gyro .3,
q `[1.0472,2.53075,.94247]*4`, dq `[2,2,4]*4`, and previous target
`[.2,.3,.3]*4`; none matches this contract. Upstream sources:
`custom_implementation/go2trot_base.py:5–24`, `gym/envs/go2/go2_config.py:146–155`,
`gym/envs/go2/go2trot_config.py:98–106`. Observation scales are **divisors**
(`utility/deploy_utility.py:86–93`).

## Actions and physical mapping

Twelve normalized position actions, ordered:
`FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf` (all `_joint`).
`q_des=clip(q0+2*clip(action,-1,1),joint_limits)`;
`qT += clip(q_des-qT, ±0.7*s*0.02)` each 20 ms.
Thus target increments are capped at `[.4214,.4214,.2198]` rad per leg per step.
q0 is the exact 12-value array in `configs/handstand_reference.json:10`
(approximately `[0,.73341596598,-1.46549031595]` per leg; preserve tiny hip signs).
Initialize qT from trajectory `initial_target`.

At each 2 ms simulator step, torque is `clip(60*(qT-q)-3*dq, ±effort)`;
effort=`[23.7,23.7,45.43]` Nm per leg. Set torque zero when `abs(dq)>=s`
and torque accelerates that motion. Physical motor-side PD and torque limiting
must be reconciled with these substep semantics; setting kp/kd alone does not
reproduce the protection. Sources: `core.py:144–180`,
`handstand_replay.py:163–175,197,350–372`, `configs/handstand_reference.json`.

Upstream already permutes canonical order to Unitree FR, FL, RR, RL with
`[3,4,5,0,1,2,9,10,11,6,7,8]` in both directions
(`utility/deploy_utility.py:6–10,30–46,112–120`); retain it and verify signs/offsets.
`RLController.act` multiplies normalized output by `DeployScaling.dof_pos_target`
(`rl_controller.py:56–66`), then adds default posture. Configure this factor as 2,
exactly once. Its built-in previous-target piece is an offset in radians, so
using divisor 2 works only after the command path actually stores the slew-limited
applied target offset (`utility/deploy_utility.py:57–59`).

## Simulator-only quantities and the four custom hooks

Simulator COM/velocity (`core.py:134–142`), exact global root state, solver
contacts/forces, geometric clearances, foot anchors, strict success gates and
reference integration states are unavailable as direct LowState observations.
They belong to the privileged hybrid, diagnostics/reward or replay validation;
the successful capture policy does not consume them directly. Its ideal body-origin
specific force is synthesized from `qacc` (`capture.py:36–42`), so physical IMU
acceleration needs mounting/lever-arm correction. Estimated COM needs model mass
and encoder FK/Jacobians plus integrated root motion; it is not an SDK COM field.
The current estimator constructs a MuJoCo FK model (`capture.py:314–330,369–388`);
a robot-side implementation needs equivalent kinematics/model assets without
stepping simulation. Ground-truth observer error checks are diagnostic only.

| Hook | Required implementation |
|---|---|
| `CustomDeployConfig` | Ordered pieces: gravity3, gyro3, q_offset12, dq12, applied_target12, elapsed1, estimated_COM4; custom sizes 1 and 4; total 47. Override inherited divisors: gravity/gyro/q/time/COM=1 (custom outputs already normalized), dq=s, applied_target=2 when returned in radians. Preserve policy-internal normalization. Set runtime posture, 50 Hz, kp/kd, EMA explicitly in actual DeployConfig/bundle configuration. |
| `get_custom_buffers_dict()` | Initialize activation step/epoch, last IMU timestamp, root world velocity zero, calibrated root height, orientation/frame calibration, kinematic constants, qT=`initial_target`, and initialized/stale flags. No trot frequency or dummy tensor. |
| `update_custom_buffers(controller,t)` | Reset phase/observer/history on deliberate maneuver activation; advance one step per issued 20 ms command. Consume every fresh IMU sample once with sample dt (500 Hz expected), integrate `a_world=R*f_body+[0,0,-9.81]`, velocity then height, and compute COM FK/Jacobian correction. Keep qT synchronized with the actual bounded command; reject stale data. `t` is absolute monotonic time, not maneuver time. |
| `get_custom_obs_pieces()` | Supply elapsed=`k*.02/8`, estimated COM4, and override previous-target getter if needed. Reuse correctly mapped gyro/gravity/encoders with calibrated frames and scales. Getters should read one synchronized snapshot, without integrating again. |

Upstream hook sites: `go2trot_base.py:26–57`, `main_controller.py:103–107,160–188`,
`utility/deploy_utility.py:72–96`. Extra `CustomDeployConfig` attributes alone do
not override frequency/gains/posture: upstream `deploy_config.py:10–24,45–56`
only imports its observation list and scaling class. The hooks also do not replace
the actuator path. Sensor ingestion and bounded command application therefore
require follow-up integration beyond implementing these four functions.

## Files to carry and export gaps

- **Selected checkpoint:** `output/capture_ppo_conservative/latest.pt`, present in
  the robustness worktree; SHA256
  `90117fc2ff80033663578e12ba664e4b1ff4c15c4c04f19619409a800458ba6c`.
  Use `scripts/handstand_takeoff.py:89–111` to reconstruct its kind and policy
  state; do not load it through generic `logs/go2trot` setup. The balance weights,
  action prior, reference observation centers and normalization are embedded.
- **Reference:** `artifacts/handstand_reference/trajectory.npz`, present; SHA256
  `21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`.
  Keep `configs/handstand_reference.json` and `configs/handstand_capture.json`.
  NPZ actions (320×12), nominal and initial_target support command/prior identity;
  initial_state/reference_states/pitch_axis support simulator regression, not
  physical state reset (`handstand_replay.py:43–78`).
- **Legacy balance artifact**, only for reconstruction/hybrid/teacher comparisons:
  `output/retained_go2_preparation_v3/balance.pt` in the adjacent BigRedGym checkout,
  optionally packaged as `artifacts/balance.pt`; expected SHA256
  `bcd6a1f5b59ebe905b73be8080e7a21e1b714f22a7b239c9111d2ee4a8b94fcd`.
  It is unnecessary for selected CapturePolicy inference because weights are embedded.
  `output/feedback_corrected/controller.npz` / `artifacts/takeoff_teacher.npz`
  supplies BC teacher construction, not selected policy inference
  (`scripts/handstand_capture.py:243–256,484–507`).
- **Still to produce:** inference-only `policy.pt` and `deploy_meta.json`, plus
  calibrated model/estimator assets and provenance hashes. Upstream bundle export
  assumes `actor.num_obs/num_actions`, traces an unbatched zero vector, and does
  not serialize ctrl_freq/EMA (`policy_bundle.py:49–83,91–115`). CapturePolicy
  lacks those dimension attributes: provide a 47→12
  adapter and preserve phase-dependent buffers, then validate boundary indices
  and recorded input/output parity offline. Explicitly persist 50 Hz/EMA-off;
  otherwise bundle loading keeps the 100 Hz/EMA-on defaults.

Validation here was source inspection, branch identity and file hashes only.
No claim of physical deployment parity is established.
