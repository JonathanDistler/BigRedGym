# Capture correction: diagnosis, evidence and continuation

The 1.88 s rear-calf collision is principally a **phase coverage failure**. The
previous takeoff actor deliberately clamps its nominal phase and feedback table
at step 47 (0.94 s command), so at 0.96 s it stops executing the optimized
rotation/capture sequence. It was a validated 0.96 s skill, not a complete
capture controller. The collision was not evidence that the nominal reference
itself cannot capture from standing.

The new controller extends the validated phase coverage through 2.2 s and uses
live sensor-error feedback throughout, then retains the near-vertical balance
feedback in reference-relative form. The nominal demonstration is disclosed as
a feedforward prior. This is not takeoff/handstand learning from scratch.

## What was measured

Four matched controllers: original offline reference, old phase-clamped prior,
original replay/live-balance hybrid, and update-10 takeoff PPO student. All start
from identical nominal or seeded perturbed standing states. All retain the
original geometry, physics, 50 Hz commands, 500 Hz integration, target slew,
joint ranges, torque limits and strict handstand gate.

At 0.96 s the old student's action diverges by about 0.10 normalized units.
At 0.962 s its filtered target differs by 53 mrad (prior-only: 41 mrad).
Orientation tracking exceeds 1 degree around 1.04 s; pitch velocity reverses
around 1.1 s and COM height subsequently falls. Front feet are also unloaded
in the *successful* reference around 0.94–0.96 s, so that flight interval alone
is not a student-specific loading failure. The old controller misses both
subsequent capture targets and the validated 1.32–1.97 s front-leg extension
pulse/handoff. No new rear-retraction waveform or collision exemption is needed
to correct this nominal failure.

The nominal PPO student's first illegal geometry contact is at 1.872 s:
MuJoCo geom 0 / geom 18, world / RL_calf. The following solver interval ends
at 1.874 s and supplies approximately [2.956, -2.313, 8.543] Ns world impulse on
the calf; its normal is world +z. First *geometric* contact and solver impulse
timestamps are distinguished. The prior-only case hits world / RR_calf
(geom 0 / geom 24) around 1.758 s geometry / 1.760 s solver end.

Restoring reference commands at 0.96 s preserves the nominal 5.014 s result.
Waiting until 1.00 s does not recover it; neither do later 1.04/1.10 s switches.
These are predeclared continuous diagnostic switches, not autonomous recovery
claims. They distinguish bad post-prefix commands from the later collision.

The actual final support transfer is also distinguished from the earlier
capture flag: the original gate first trips at 0.688 s, support is subsequently
lost around 0.780 s, both front feet load again around 1.056 s, and load/gate
flicker occurs through about 1.384 s. The longest strict hold begins around
1.386 s. The reference controller takeover at step 69 (1.38 s) and optimized
extension pulse are preserved.

Full 0.8–2.3 s traces are in `output/capture_diagnosis_tail/`, including actual
post-failure simulation tails. The original failure is never erased by those
tails. Plots cover pitch/rate, COM position/velocity and support-relative COM,
contacts, all calf clearances, joint positions/speeds/targets/torques and caps,
action differences, effective phase error, world contact points/normals and
solver impulses. Native solver forces are read immediately after `mj_step`,
before `mj_forward` recomputes constraints. Impulse equals that step's actual
solver force times 0.002 s. No contact/body forces enter the deployable actor.

## Local viability is not standing-start success

Twenty-three explicit 1.28 s snapshot probes were physically checked: pitch
±0.001/0.003 rad, roll ±0.001 rad, angular velocity ±0.01/0.05 rad/s,
structured joint perturbations ±0.0005/0.002 rad, contact-height gaps
±0.0002/0.0005 m, and command phase offsets ±20/40 ms. All 23 are valid and
collision-free through 2.2 s using reference commands. Only the unperturbed
probe gets five strict seconds within the original 6.4 s horizon; nearby
probes collide later or fail that hold gate.

This separates A/C (post-prefix actions and phase/handoff coverage cause the
immediate failure) from B (recorded-action long-term balance is very sensitive).
It does not certify a safe neighborhood radius. Probe results are explicitly
marked as snapshot interventions and never enter full-maneuver success counts.

## Corrective teacher, sensor contract and student

The bounded transition teacher reuses the preserved projected trajectory-LQR
feedback, advances its phase correctly and switches to the retained affine
balance feedback matrix after step 69. Corrections remain within ±0.05
normalized action (±0.10 rad before the existing target filter/range limits).
The nominal prior preserves all validated transition commands and extension
pulse. A small early-gain ablation (0 or 0.1 versus 1.0) did not improve the
checked training cases and was not adopted. A stationary-foot-only velocity
estimator was rejected because it drifted into nominal head impact; rejected
prototype source/results remain in ignored output artifacts.

The selected observer uses ideal body-origin specific force and orientation
at 500 Hz. It integrates root velocity/height from calibrated ordinary standing,
then combines them with IMU/encoder-only forward kinematics. The separate FK
data is never stepped and never resets the physical robot. Initial root height
is the nominal standing calibration; initial root translation velocity is zero.
The joints may move initially, and their COM contribution is included by FK.

Actor inputs are the previous 43 sensor/command/phase channels plus estimated
COM height / 0.6 and three estimated COM velocity channels. There is no actual
world base translation/velocity, simulator COM, contact force or future
measurement in the actor. The existing Go2 utility/logger exposes IMU
quaternion, gyro and accelerometer fields; no hardware connection was made.
Frame alignment, sensor offsets, noise, bias and real 500 Hz delivery remain
unvalidated. The body-origin accelerometer is a simulation-compatible ideal
measurement, not a qualified hardware sensor model. On the 67 successful
held-out trials, the maximum observer root-velocity error is 0.000154 m/s and
height error 0.0000701 m against diagnostic-only ground truth.

The capture actor preserves feedback derivatives by analytical initialization
from the validated feedback law, followed by BC on actual successful closed-loop
rollouts. Twelve teacher trials were attempted; seven meet current 0.1 s strict
capture stability at 2.2 s, yielding 770 accepted state/action pairs. The other
five are retained as failures and do not supply successful correction labels.
No fixed actions are copied onto arbitrary off-reference states. BC uses seed
4110, 100 epochs; demonstrations use 121000–121011. Capture PPO starts a fresh
optimizer because the actor changes from 43 to 47 channels; the old update-10
checkpoint remains unchanged.

At 2.2 s the policy hands over to the retained balance feedback matrix around
the nominal reference. Live sensor correction continues every control step.
It does not substitute recorded balance outputs alone. The phase-prior-only
ablation explicitly disables these corrections and is labeled as replay.

## Outcomes and limits

Generated report: `output/capture_report.md`; all failed trials are also in
`output/capture_report.failures.csv`. Root comparison plots are
`output/capture_diagnosis_tail/capture_comparison.png` and the multi-page
`capture_diagnostics.pdf`. Prefix tables include 1.26, 1.50, 1.76, 2.00 and
2.20 s (50 Hz sampling adapts the requested 1.25/1.75 boundaries).

A prefix success requires legal powered takeoff, bilateral front support,
rear clearance >0.1 m and orientation within 5 degrees of the successful
reference at the endpoint. A separate stable-capture field requires the current
unchanged strict gate continuously for 0.1 s. These partial-skill certificates
never replace the full five-second gate.

On matched prefix seeds 141000–141004, the student passes 4/5 at 1.26 s and
5/5 at every later boundary. Fixed full-phase reference passes only 1/5 by
2.0/2.2 s on those starts; the old frozen prior passes 0/5. On matched full
held-out seeds 201001–201008, full-phase replay is 0/8, frozen prior 0/8,
privileged-COM hybrid 2/8, new student 6/8. BC and two-update PPO both achieve
6/8 on that full comparison; no PPO improvement over BC is established.

The sealed additional seed set 301001–301100 gives **67/100** complete
standing-start strict successes, Wilson 95% interval **57.3–75.4%**, plus a
separate nominal success. The other 33 remain in the denominator: 28 illegal
contacts and five insufficient-hold episodes. Successful completion takes
6.4–7.16 s. The common full-evaluation budget is 8 s and terminates only after
actual five-second success or failure; no mid-trajectory reset is used.
The original reference benchmark still ends at 6.4 s with exactly 5.014 s hold
and zero state error. Any fixed-prior extrapolation beyond its recorded endpoint
is explicitly a held-last-command comparison, not the immutable benchmark.

Across the 67 successful held-out five-second intervals: orientation mean
0.8753 deg, pooled RMS 1.1493 deg, overall maximum 5.7115 deg; pooled pitch RMS
1.1419 deg; mean pitch zero-crossing rate 0.4179/s (0.1-degree noise exclusion);
mean front-foot horizontal path 0.017223 m; mean supporting-leg KE 0.004121 J.
These hold-only metrics condition on success. They never remove failures from
the probability denominator. Crossing rate is not automatically cycle frequency.

This is a genuine simulation feedback result with a disclosed planned prior
and ideal calibrated sensors. It is not robust arbitrary-state completion,
noise/domain/push robustness, a policy learned from scratch, or hardware readiness.
The sealed test set is diagnostic evaluation data; training on its failures
would require a new held-out set.

## PPO and resource decision

The first two-update capture smoke at learning rate 1e-4 had excessive KL
(about 0.69 in the first update) and is retained in `output/capture_ppo_smoke`.
The selected repeat uses 1e-5, minibatch KL stopping above 0.02 and an unhealthy
final-KL bound 0.05. Final update KL values are 0.0364 and 0.0212. Gradients are
finite and clipped at norm 1; recorded pre-clip norms are larger. The two
completed stochastic training episodes did not satisfy current terminal capture
stability, even though takeoff and deterministic nominal/partial validation
were retained. This and the lack of improvement over BC justify stopping PPO.

The evaluated checkpoint is `output/capture_ppo_conservative/latest.pt`, update
2. Verification continues it into a separate directory to update 3 and matches
uninterrupted update 1–3 exactly, including optimizer, RNGs, normalization,
curriculum, and selected best payload. It does not overwrite the evaluated
update-2 artifact. Worlds restart only at declared update boundaries; no live
mid-maneuver simulator continuation is claimed by a training resume.

Local capture benchmark: 65.3–66.2 simulation transitions/s for serial counts
1/4 and length 128, about 98% of one CPU core, 749–752 MiB peak RSS. PPO with
validation is about 18.6 training transitions/s and peaks around 762 MiB.
The 100-case evaluation took about 418 s. These are local measurements.
**Actual Molab throughput remains NOT MEASURED.** No remote runtime is connected.
The notebook measures it before permitting its optional capture pilot.

Do not run 250 or 5000 updates now. A bounded <=50 continuation is prepared,
but first verify real Molab physics/resume/throughput and require objective
progress. Beyond 50, the CLI additionally requires hash-checked evidence of
at least 20 matched full-phase-prior/student held-out trials, actual objective
improvement, a capture-task exact-resume receipt and a measured Molab benchmark.
Runtime location is declared provenance, not remote attestation.

## Exact commands

Run from this checkout or the extracted capture bundle. On a new Molab host,
upload the updated notebook and `handstand_capture_bundle.tar.gz`, paste the
adjacent SHA256 sidecar, click extract/install, then the original nominal
regression. Installation remains Python 3.11 with `uv sync --frozen --no-dev`;
all subprocesses use the extracted root and scrub inherited Python/UV selectors.
See the existing takeoff runbook for the equivalent terminal extraction/install.

```bash
# Local interpreter; on Molab use "$PWD/.venv/bin/python" instead.
export HANDSTAND_PYTHON=/home/jonathandistler/Distler_Gym/BigRedGym/.venv/bin/python
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/go2-mpl

# Four old-controller traces and plots:
"$HANDSTAND_PYTHON" scripts/handstand_capture.py compare --output output/capture_diag_new
"$HANDSTAND_PYTHON" scripts/plot_handstand_capture.py --input output/capture_diag_new
"$HANDSTAND_PYTHON" scripts/handstand_capture.py rescue --output output/capture_rescue_new
"$HANDSTAND_PYTHON" scripts/handstand_capture.py viability --output output/capture_viability_new

# Regenerate the valid corrective teacher demonstrations and capture BC:
"$HANDSTAND_PYTHON" scripts/handstand_capture.py bc --seed 4110 --demonstrations 12 \
  --epochs 100 --trials 4 --output output/capture_bc_new

# In the bundle: artifacts/capture_student.pt is BC;
# artifacts/capture_ppo_latest.pt is the evaluated update-2 policy.
"$HANDSTAND_PYTHON" scripts/handstand_capture.py prefix \
  --checkpoint artifacts/capture_ppo_latest.pt --controllers reference prior student \
  --trials 4 --output output/capture_prefix_new
"$HANDSTAND_PYTHON" scripts/handstand_capture.py full \
  --checkpoint artifacts/capture_ppo_latest.pt --controllers reference prior hybrid student \
  --evaluation-seed 201000 --trials 8 --output output/capture_full_new
"$HANDSTAND_PYTHON" scripts/handstand_capture.py full \
  --checkpoint artifacts/capture_ppo_latest.pt --controllers student \
  --evaluation-seed 301000 --trials 100 --output output/capture_100_new

# Two-update capture smoke, real process restart, exact-resume comparison:
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint artifacts/capture_student.pt --updates 2 --envs 1 \
  --rollout-steps 128 --trials 2 --output output/capture_smoke_new
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/capture_smoke_new/latest.pt --resume --updates 3 --envs 1 \
  --rollout-steps 128 --trials 2 --output output/capture_smoke_new
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint artifacts/capture_student.pt --updates 3 --envs 1 \
  --rollout-steps 128 --trials 2 --output output/capture_continuous_new
"$HANDSTAND_PYTHON" scripts/verify_handstand_checkpoint.py \
  --left output/capture_smoke_new/latest.pt --right output/capture_continuous_new/latest.pt \
  --output output/capture_smoke_new/resume_verification.json
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py benchmark \
  --checkpoint output/capture_smoke_new/latest.pt --env-counts 1 4 --lengths 128 \
  --output output/capture_benchmark_new

# Prepared only; do not launch until the Molab checks and objective gates pass.
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/capture_smoke_new/latest.pt --resume --updates 50 --envs 1 \
  --rollout-steps 128 --trials 2 --output output/capture_pilot_new

# Generate a NEW paired >=20-case report for a prospective selected checkpoint.
"$HANDSTAND_PYTHON" scripts/handstand_capture.py full \
  --checkpoint output/capture_pilot_new/latest.pt --controllers reference student \
  --evaluation-seed 501000 --trials 20 --output output/molab_capture_paired20

# 250 remains BLOCKED without actual evidence. Template is intentionally invalid.
"$HANDSTAND_PYTHON" scripts/handstand_takeoff.py train \
  --checkpoint output/capture_pilot_new/latest.pt --resume --updates 250 --envs 1 \
  --rollout-steps 128 --trials 2 --allow-long-takeoff-continuation \
  --capture-continuation-proof output/capture_continuation_proof.json \
  --output output/capture_250
```

Populate a copy of `configs/handstand_capture_continuation_proof.example.json`
with the actual paired report, Molab benchmark and Molab resume receipt paths
and SHA256s. The code checks matching starts/conditions/horizons, checkpoint
identity, reference identity, improvement and capture-task receipt. Do not
relocate local receipts and label them as measured Molab execution.

For local commands without extracting the bundle, substitute
`output/capture_bc_seed4110/bc.pt` and `output/capture_ppo_conservative/latest.pt`
for the two capture artifact paths. Packaged takeoff/balance/feedback defaults
are resolved automatically; local original artifacts remain available at their
existing paths.

Regenerate measured Assignment 6 tables and the complete failure list:

```bash
"$HANDSTAND_PYTHON" scripts/report_handstand_capture.py \
  --diagnostics output/capture_diagnosis_tail \
  --viability output/capture_viability/viability.json \
  --rescue output/capture_rescue/rescue.json \
  --prefix output/capture_prefix_final/prefix.json \
  --comparison output/capture_full_heldout/full.json \
  --final output/capture_final100/full.json \
  --ppo output/capture_ppo_conservative/train.json --output output/capture_report.md
```

All previous checkpoints, bundles, branches, and the successful reference remain
preserved. Generated diagnostics, source snapshots, model files and archives stay
outside Git history. No hardware commands, main pushes or upstream PR changes
are performed.
