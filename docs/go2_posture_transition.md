# Short posture-transition reference: implementation and results

The reliable-transition success criterion has **not** been met. A frozen kick
actor plus an opt-in intermediate reference can now produce individual,
collision-free five-second handoffs, including one fresh validation seed.
The selected reference does not generalize reliably to held-out starts.
No PPO updates were run and no replacement reference was promoted.

The new code is isolated in `gym/envs/go2/posture_transition.py` and
`scripts/optimize_posture_transition.py`. Existing checkpoints, the 30 mm
reference, recovery atlas, and previous results remain unchanged. Completed
experiments verify protected input SHA256 values. Each search saves its
controller/source snapshot, effective policy configurations, parameters,
individual outcomes, and validation/held-out traces in a new directory.

Hold counters use 100 Hz control samples (10 ms resolution). The recorded
6.51 s successful hold spans approximately 6.5 s of physical observation;
this convention does not affect its five-second classification or explain
the historical 3.88 s failure.

## Exact reproduction of the 3.88-second trial

`scripts/reproduce_transition_hold.py` reproduces the old six-case bridge
experiment in its original order. The best case is seed **3009** (batch slot
9), with the aggressive reference and the fresh kick actor, switching at
global **1.2 s** and blending targets for **0.6 s** into the October 31 balance
controller. Both actors remain frozen. The native kick state matches the
previous `candidate_states.npz` exactly: maximum absolute qpos/qvel error is
zero. The hold reproduces as 3.8799999999999613 s.

The policy paths are:

- Kick: `logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt`
- Balance: `logs/go2balance/Oct31_23-29-27_/model_1500.pt`
- Aggressive prefix reference: `output/handoff_reference_search/candidate_reference.json`

The balance actor's saved source/configuration and documented CPG overrides
are restored, including stance/stance/swing/swing contact modes. Its action
space is translated into physical canonical joint targets around its own
handstand reference. Observations include the applied previous target rather
than an untranslated kick action. Deterministic inference, disabled pushes,
explicit CPG initialization and native model/state transfer are recorded.

Failure is physical, not an evaluation cutoff:

- At 3.8 s after switching, pitch is 90.56° and the COM is about 3 cm ahead
  of front support. At 4.1 s, pitch is still 89.64° but that offset is 10.5 cm
  and COM speed is 0.445 m/s. Pitch alone conceals the developing instability.
- At 4.2 s pitch rate reaches 0.366 rad/s; by 4.3 s it is 1.84 rad/s.
  Front support unloads as the robot overrotates. Native nonfoot contact
  appears at 4.566 s after switching, on `Head_upper`. Base world X moves
  from 0.043 m at 3.8 s to 0.565 m at contact; base height falls from
  0.566 m to 0.211 m.
- Rear-hip commands drift well away from the nominal 2.046 rad posture.
  The left rear hip target approaches 4.25 rad at 4.1 s and 4.52 rad at
  4.2 s, moving rear-leg mass while the front support/COM relationship fails.
- Before 4.2 s, peak torque is about 53.5% of its limit and peak joint speed
  about 28.4% of its limit. Saturation appears later in the fall; it did not
  initiate the instability.
- The initial position target is continuous at the switch. Policy updates
  during blending still cause target steps up to 0.232 rad, and later steps
  reach 0.322 rad. These are separate from the initial switch and motivate
  the new target-rate limiter.
- The configured episode is 12 s. Neither `terminated` nor `timed_out` ends
  this trial before the first native collision. The batch stops after all
  worlds have physically failed, at 4.57 s of recovery simulation.

[Detailed reproduction](../output/transition_reproduction_v2/reproduction.json)
includes native switch qpos/qvel, joint command, gains, policy configurations,
and exact state assertions. Its `seed_3009_physics.csv` records every 2 ms:
world base/COM coordinates, complete root state, support forces, joint states,
policy/applied targets, torques, limit ratios, and nonfoot contact body names.

## Opt-in controller and optimizer

The searched families are:

1. An early bridge from the **original** reference: front yielding/stiffness,
   rear hip/knee and retraction timing, launch coordination, takeover time,
   policy blend duration and small pitch feedback. These starts come from
   the original trained kick's actual standing trajectories.
2. A late bridge from the aggressive prefix: symmetric front/rear hip/knee
   waypoint offsets, takeover time and blend duration. This truncates the
   aggressive trajectory before its later collisions; it is not a standalone
   replacement for the protected reference.
3. A reference-first bridge: move toward the balance posture before enabling
   balance-policy blending, so the actor first sees a closer observation
   distribution.
4. An adaptive reference-first bridge: add bounded pitch-error, pitch-rate
   and COM/support-offset feedback during the intermediate phase.
5. A small local refinement with target-rate limits enabled, using new
   validation and held-out seeds.

**All positive examples use the aggressive prefix, not the untouched 26.3°
reference.** They use the same frozen kick checkpoint with an explicitly
selected reference. The original-reference family did not produce a valid
complete transition. The default reference/configuration is not changed.

Search uses a NumPy full-covariance CMA-ES with reflected box constraints.
No additional optimization package is required. The dense guidance is the
scaled distance to successful samples in the existing atlas: pitch, pitch
rate, COM relative to front support, base height, and all 12 joint positions.
Closest distance during the bridge supplies guidance; terminal states are
also saved for separate diagnosis. Complete five-second recovery is the
dominant objective, followed by shorter valid holds, collision costs, atlas
distance and excessive joint-speed costs. There is no clearance or increasing
angular-momentum objective.

Native qpos/qvel, solver warmstarts, model masses/inertias and friction are
carried from sampled kick trajectories. A failed prefix stays failed. Native
nonfoot contact is checked every physics substep, with no reset into a more
favorable posture. Both policies use their own observation/action interfaces.
Each evaluation runs through the completed blend plus 6.5 s of recovery,
unless all worlds fail first; automatic episode resets are never used.

The current opt-in controller clips position targets to native joint limits,
clips feedforward velocity targets, clips motor torques, and rate-limits the
applied position target to the existing joint velocity limits at every physics
step. The measured largest target step is approximately **0.0602 rad/2 ms**,
corresponding to the largest native joint velocity limit. This bounds changes
from policy updates as well as controller entry. Historical unslewed behavior
is available only with `--legacy-target-steps` for comparison. Actual velocity
ratios and saturation are logged separately from commanded limits.

Five-second success starts counting after blending finishes and requires
continuous 75–105° pitch, absolute pitch rate ≤1.5 rad/s, both front loads
≥6% body weight, rear loads ≤1.5%, both rear clearances ≥20 cm, and no prior
nonfoot contact/fall. A collision after a completed five-second endpoint is
logged separately; it does not erase that completed endpoint. Some early
search artifacts additionally required survival of the entire 6.5 s horizon;
the report retains those artifacts and adjudicates five-second holds without
overwriting them.

## Validation and comparison

The first four searches screened 274 parameter vectors on four starts each.
Each family's two finalists received 16 fresh validation starts; its selected
candidate then received 16 held-out starts. The rate-limited local search
added 19 vectors and used entirely new validation and held-out cohorts.

| Family | Validation success, each finalist | Selected held-out success |
|---|---:|---:|
| Original-reference early bridge | 0/16, 0/16 | 0/16 |
| Aggressive-prefix late bridge | 0/16, 0/16 | 0/16 |
| Reference-first | 0/16, 0/16 | 0/16 |
| Adaptive reference-first | 1/16, 1/16 | 0/16 |
| Rate-limited local refinement | 0/16, 0/16 | 0/16 |

The adaptive finalists succeed on the same validation seed, 6003; those are
not two independent successful seeds. That trajectory is reproduced both
with historical target steps and with the new rate limiter:

| Seed 6003, rate-limited candidate | Measurement |
|---|---:|
| Complete five-second outcome | Success |
| Observed continuous hold | 6.51 s |
| Maximum body pitch | 94.00° |
| Pitch at completed handoff | 87.70° |
| Pitch rate at completed handoff | +0.406 rad/s |
| COM ahead of front support at handoff | −5.7 mm |
| COM speed at handoff | 0.052 m/s |
| Front support fraction after takeover | 1.00 |
| Nonfoot contacts | 0 |
| Torque saturation | 0 |
| Maximum actual joint velocity / limit | 0.377 |
| Maximum physics target step | 0.0602 rad |

This improves on the old supervisor's 0/16 five-second outcomes and its best
3.88 s example, but **not in reliability**. The same rate-limited adaptive
candidate scores 0/16 on the original held-out cohort. Its local refinement
also scores 0/16 on a new held-out cohort. No reference is promoted.

As a control, the frozen balance policy recovers 15/16 nominal-height and
12/16 elevated near-vertical poses under the same randomized kick mass and
friction. The main failure is therefore entering a suitable dynamic/support
state, rather than mass variation alone making balance impossible.

The rate-limited refinement's final 16 held-out trials quantify the remaining
gap: zero successes, maximum hold 0.47 s, and 16 nonfoot-contact failures.
Fourteen survive to completed handoff. Their mean pitch is 80.0° (range
50.5–122.2°), mean absolute pitch rate 0.596 rad/s, base height 0.530 m, mean
absolute COM/support offset 0.227 m, and COM speed 0.799 m/s. Successful atlas
initializations have base height about 0.595–0.646 m and COM/support offsets
within roughly 6 mm. The closest terminal atlas distance is 2.46 scaled units
(mean 7.03); the best intermediate distance is 1.59. Atlas distance is a
guidance metric, not a hard certificate: seed 6003 succeeds outside the old
supervisor's conservative 0.75 neighborhood.

The remaining problem is dynamic support geometry, body elevation and COM
velocity, even when joint configuration and pitch are closer. More peak
rotation alone does not address it. A learned intermediate controller is
**not established as necessary**: valid reference-only bridges exist. The
searched parameterizations do not supply adequate state-dependent robustness.
A front-foot/task-space bridge with phase adaptation is a reasonable next
controller family; a learned bridge is an alternative, not a demonstrated
geometric necessity. No further PPO training is recommended or launched here.

[Full comparison, plots and dynamics](../output/posture_transition_final_report_v2/REPORT.md)
and [randomized balance control](../output/transition_randomized_balance_control/atlas.json).
Validation seeds are 6000–6015, held-out 7000–7015; local refinement uses
8000–8015 and 9000–9015. Screening seeds are 3000, 3004, 3009 and 3012.
0/16 has a Wilson 95% upper bound of approximately 19.4%; it is not proof
that every possible reference must fail.

## Exact MuJoCo commands

Run from the repository root. Output directories must not already exist.
Omit `--viewer` for headless runs. Viewer commands require a graphical session.

```bash
# Exact historical seed-3009 reproduction, including the original batch/order.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/reproduce_transition_hold.py \
  --output-dir output/reproduce_388_new --viewer

# New smooth controller; replay the exact validation batch and show seed 6003.
# Uses the separately named opt-in preset; does not change the default kick.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/optimize_posture_transition.py \
  --parameters gym/envs/go2/references/posture_transition_adaptive.json \
  --source aggressive --late-bridge --reference-first --adaptive \
  --replay-validation --view-slot 3 --viewer \
  --output-dir output/posture_validation_playback_new

# Reproduce the rate-limited local CMA experiment and independent cohorts.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/optimize_posture_transition.py \
  --source aggressive --late-bridge --reference-first --adaptive \
  --initial-parameters gym/envs/go2/references/posture_transition_adaptive.json \
  --sigma .05 --generations 3 --population 6 \
  --validation-seed 8000 --heldout-seed 9000 \
  --output-dir output/posture_smooth_search_new

# Historical successful screening example, including its original target steps.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/optimize_posture_transition.py \
  --parameters gym/envs/go2/references/posture_transition_screen_success.json \
  --source aggressive --late-bridge --legacy-target-steps --replay-screen \
  --output-dir output/posture_screen_replay_new
```

The adaptive preset uses takeover 1.12658 s (nearest control frame 1.13 s),
reference duration 0.37246 s, policy blend duration 0.18023 s, front hip/knee
offsets +0.10244/−0.03696 rad, rear hip/knee offsets −0.18357/−0.11454 rad,
pitch/rate feedback −0.00993/+0.17364, and COM-offset feedback −1.03611 rad/m.
Feedback is bounded to ±0.25 rad and fades out as policy mixing finishes.
Exact values are in the preset and experiment JSONs.

38 targeted tests pass, covering bounded-CMA convergence/covariance, immutable
reference inputs, timing validation, blend endpoints, native target-rate
limits, supervisor/state-copy invariants, and existing front-yield/reference/
balance regressions. Ruff checks pass on the new code.
