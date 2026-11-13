# Phase-conditioned momentum and energy pumping: bounded diagnostic

Implemented on `experiment/com_energy_phase_ppo` in the new
`experiments/momentum_phase/` package. Previous experiments, models, controllers,
checkpoints and outputs are unchanged. No unrelated branches were merged.

**Outcome: the momentum objectives did not outperform the matched baseline in
this short diagnostic.** The measurements point to front-support mechanics and
poor retention of useful angular impulse, followed by collision, rather than a
missing reward for arbitrarily large velocity. This is not a proof that the task
is dynamically infeasible.

## Matched PPO results

Full 100% Go2 geometry, identical existing certified BC-phase checkpoint,
identical networks and physical constraints, training seeds 42/43. Each condition
received two PPO updates of 256 transitions per seed: **512 transitions per
condition/seed, 3,072 total training transitions**. Checkpoints were evaluated
before training and after every update on resets 94000/94001. Final means cover
four rollouts per condition; this is a diagnostic, not a statistical convergence
study. All conditions began at 32.06° certified peak pitch.

| Condition | Final certified collision-free pitch | Takeoff | 45° / 60° / 80° pass-through | Five-second handstand |
|---|---:|---:|---:|---:|
| A: energy/orientation/phase baseline | 33.23° | 4/4 | 0 / 0 / 0 | 0/4 |
| B: momentum tracking and energy pumping | 32.59° | 4/4 | 0 / 0 / 0 | 0/4 |
| C: B plus terminal objectives and balance BC | 33.19° | 4/4 | 0 / 0 / 0 | 0/4 |

Maximum individual PPO rollout: **33.91°**. All final rollouts eventually hit one
or both front calves. The peak metrics exclude collisions, self-contact and
actuator-state violations at physics rate. The historical 32.30° result used a
different, smaller budget; A above is the appropriate matched comparator.

At the rotation apex, A/B/C had respectively **48.75 / 48.06 / 48.02 J** potential
energy and **4.47 / 4.49 / 4.59 J** full-system kinetic energy. Apex pitch rates
were approximately zero. Mechanical energy peaked below 55 J, versus the measured
89.42 J handstand potential. B/C produced nonzero bounded upward-energy and
rotational-energy progress rewards, unlike the old standing-height-only progress
term, but no terminal reward. They never reached braking or stabilization phases.

Post-update PPO KL ranged approximately 0.0026–0.0141; clipping fraction remained
below 0.102. The inherited soft KL stopping rule stayed active. BC loss is finite;
C's normalized loss is larger because it includes unfamiliar balance states.
The held-out takeoff action MSE remained around 7–8e-6. Balance action MSE changed
from 0.12162 initially to 0.12117 in C: the short run did not learn the archived
balance controller to useful accuracy.

## Momentum, force and energy definitions

For all positive-mass links in the robot root subtree, excluding world/ground,
let c be whole-robot COM, x_i link inertial COM, v_i and omega_i their world
velocities from MuJoCo body-COM Jacobians, and I_i the rotated principal inertia:

- m = sum(m_i), c = sum(m_i x_i)/m, v_c = sum(m_i v_i)/m.
- H_c = sum[I_i omega_i + (x_i-c) cross m_i v_i].
- H_s = H_c + (c-s) cross m v_c, about current front support reference s.
- I_c = sum[I_i + m_i ((r_i dot r_i) identity - r_i r_i^T)].
- I_s = I_c + m (|c-s|² identity - (c-s)(c-s)^T).
- K_links = sum[0.5 m_i |v_i|² + 0.5 omega_i^T I_i omega_i].
- K_system = K_links + 0.5 sum(armature_j qdot_j²).
- U = m g c_z; E_mechanical = U + K_system.

The energy reference is world z=0 with native gravity (0,0,-9.81). Trunk COM and
whole-robot COM remain distinct; only whole-robot COM defines U. The new
`mechanical_energy` column includes armature, correcting the ambiguity of the
older link-only `total_energy` diagnostic. Energy deficit to examine is
U_target - mechanical_energy; `energy_deficit` retains the explicitly labelled
potential-only U_target - U quantity for compatibility.

The task pitch axis is fixed from initial horizontal heading:
e = world_up cross heading. All momentum and moment projections use e. Body pitch
and its derivative are measured in that heading's sagittal plane, with positive
pitch rotating the nose downward toward the handstand. A yaw-covariance test
rotates robot state and axis by 90° and obtains identical pitch, pitch rate,
projected momentum and inertia. COM pendulum angle remains a separate quantity.

MuJoCo contact-frame forces and intrinsic contact moments are transformed to
world coordinates with the correct geom-order sign. Front-foot, rear-foot and
nonfoot ground forces and moments are logged separately. Uniform gravity has zero
moment about COM; it contributes (c-s) cross m gravity about support. External
body forces are included. Internal self-contact and joint forces cannot create
net whole-robot angular momentum.

The active support reference is the load-weighted front-contact centroid. For a
moving reference, dH_s/dt = torque_s - v_s cross (m v_c). This transport correction
and its validity flag are logged. A fixed initial support point is also used for
an independent balance check; neither moving/sliding contact nor contact loss is
interpreted as a fixed pivot.

Independent link sums agree with MuJoCo subtree COM/velocity/angular momentum.
K_system and U agree with MuJoCo energy, including rotor armature. The native
2 ms trapezoidal external-moment integration has maximum cumulative COM-momentum
residual 0.0582 kg m²/s (5.16% of the observed envelope). On reset 93000, halving
the physics timestep reduces residual from 0.0510 to 0.0275 kg m²/s while retaining
31.97° takeoff. This is a finite-step/contact-switch approximation, not an exact
per-step torque certificate. Native 2 ms dynamics are used for all PPO conditions.
An airborne internal-actuation test verifies that moving the legs does not create
net COM angular momentum without an external moment.

Locked-shape inertia and H_c²/(2 I_c) are **proxies** for coherent pitching energy.
They are not additive to K_system and do not exactly describe an articulated robot.
The variable-length pendulum endpoint is the actual COM; support length, angle,
inertia, whole-system energy, actuator work and contact power are recorded.
Leg tucking may redistribute momentum or change inertia; it does not generate H.

## Calibrated targets, phases and rewards

The existing `recoverability_maps_20261112` archive supplies 54 checked snapshots
from successful balance continuations. Model mass and all joint coordinates were
revalidated against their recorded labels. Mass is **16.2764066211 kg**; selected
COM height **0.5600034658 m**; U_target **89.4166207975 J**, using precisely the same
height reference and model as standing rollouts. No maximal-height jumping pose
or invented transition teacher was used.

Independent terminal calibration measures U=87.51–92.56 J across these snapshots,
body angular speed 0.216–0.474 rad/s, COM speed 0.049–0.100 m/s, orientation error
relative to the selected pose <=0.041 rad, and front-calf clearance 28.8–33.3 mm.
COM projections lie inside the front support segment with foot radius. Desired
terminal angular velocity and momentum are zero, with velocity width 0.5 rad/s;
the moving snapshots themselves are not assumed to have exactly zero momentum.

A one-shot phase p supplies sin(pi p), cos(pi p), p. Loading and takeoff timing
start from the previous demonstration knots (0.02, 0.68, 0.70 s); new teacher
playback records the exact live loading, extension and certified release events.
After release the clock realigns to the measured event:

1. p<0.2: front loading, requiring observed forward weight transfer.
2. 0.2<=p<0.4: rear push-off; cannot leave without certified takeoff.
3. 0.4<=p<0.6: rotation/reconfiguration; elapsed time alone cannot start braking.
4. 0.6<=p<0.8: braking begins at >=60°, or >=45° with pitch rate >2.5 rad/s.
5. p>=0.8: >=80° with pitch rate <1 rad/s, followed by physical stabilization.

Phase completion is not a success certificate. Success requires five continuous
seconds measured at 500 Hz: certified rear release, bilateral front support,
full-orientation error <10°, body angular speed and pitch speed <0.5 rad/s,
COM speed <0.25 m/s, COM support-region distance <4 cm, low front-foot slip,
<10 cm front drift, positive calf clearance, native joint/velocity safety and no
invalid collision anywhere in the episode. Any bad physics sample resets dwell.

All A/B/C actors share the same 61-input, (256,256,128)-hidden, 12-output network;
critics have 91 inputs. The original 47/77 checkpoint weights are copied exactly,
with zero columns appended for 14 common estimates: COM/support momentum,
potential and kinetic energy, mechanical deficit, support inertia, COM support
position/height, COM velocity, pitch rate and support validity. A also receives
these common phase/observation updates, so the rewards are compared on the same
architecture and initialization. These are MuJoCo state estimates; a hardware
state estimator has not been implemented.

Observed safe prefixes bound the target envelope: COM momentum <=1.1278 and
support momentum <=3.9360 kg m²/s. These are reward-target bounds, not artificial
clamps on physical momentum. For support-pitch momentum:

L_target = min(L_cap, sqrt(2 I_s max(U_target-U,0))) * a(p) * b(body_pitch),

where a rises continuously from zero during loading to one during push-off,
and b tapers from one at 60° to zero at 90°. The square-root estimate is capped
by measured attainability; it does not assert that all kinetic energy converts
to potential. Sigma_L=0.9840 kg m²/s. Config fields expose caps, widths, weights
and filter time constants.

B/C add exp(-0.5 ((L_support-L_target)/sigma_L)²), with current support/safety
and useful-push/flight gating. Positive-impulse credit requires actual COM H
increase, positive measured rear and net ground moments, rear actuator work and
loaded rear contact. Its episode credit is capped. Early U progress is measured
from the actual upward-swing start after work and forward rotation begin, rather
than the higher initial standing COM. Upward U and useful rotational-energy
credits are bounded best-so-far increments, not income for holding a pose or
repeated pumping cycles. Excessive momentum is penalized; target momentum tapers
near upright. All native torque, joint, velocity and collision checks remain.

C adds

r_terminal_rate = 4 * exp(-0.5 ((U-U_target)/12 J)²)
                   * exp(-0.5 (SO3_orientation_error/0.35 rad)²)
                   * exp(-0.5 (body_angular_speed/0.5 rad/s)²)
                   * current_support_and_collision_validity.

A separate Gaussian COM support-region objective has width 4 cm. Reward rates
are integrated with control dt=20 ms. Contact gates are never filtered. B/C
replace the old pumping proxy and two-factor target term; A retains the previous
energy/orientation objective. The terminal term was zero throughout this run,
so this experiment does not demonstrate learning from that term.

## Supervision, targeted schedules and smoothing

Initialization is the same prior certified CPG-free BC-phase checkpoint in every
condition and seed. Six frozen teacher rollouts produce 326 real safe prefix
frames using the exact live phase/estimator; four seeds train and two are held out.
Strong early BC weight is 1; post-release weight is 0.1 in A and 0.025 in B/C.
Sixteen real safe post-release frames from the best bounded candidate receive weak
0.05 weight in B/C. C additionally uses 54 archived balance state/actions with
weight 0.15, after checking action representability. No teacher supplies a full
standing-to-handstand demonstration. All imitation occurs during training;
inference remains one actor producing twelve joint targets with no controller
handoff or mandatory joint-reference schedule.

Only eight fixed conditions were played on two resets each: unchanged control,
rear tuck, rear extension, front fold, front opening, front-target compliance,
combined leg adjustment, and phase-filter ablation. Adjustments begin only after
certified takeoff and ramp over 80 ms. Examples: front fold adds +0.12 rad to front
hip targets and -0.20 rad to front calf targets; rear tuck uses +0.15/-0.30 rad.
This was 16 rollouts, not an optimization search.

| Frozen schedule | Mean certified peak | Apex U | Apex calf gap | Outcome |
|---|---:|---:|---:|---|
| Unchanged | 31.92° | 47.19 J | 17.7 mm | Front-calf collision |
| Front fold | 34.71° | 39.95 J | 14.3 mm | Head-lower collision |
| Rear tuck | 31.10° | 47.48 J | 18.0 mm | Front-calf collision |
| Rear extension | 31.65° | 45.34 J | 19.4 mm | Collision / joint-limit termination |
| Front opening | 30.03° | 46.49 J | 20.4 mm | Rear-calf collision |
| Front compliance | 31.22° | 43.41 J | 19.2 mm | Front-calf collision |
| Combined | 29.98° | 47.42 J | 19.8 mm | Nonfoot collision |
| Phase filter | 31.90° | 47.14 J | 17.7 mm | Front-calf collision |

Front folding changes the failure mode and increases body pitch while lowering
the COM. It is not a successful energy-pumping solution. The rear-extension
limit violation is rejected, not hidden by projecting simulator state.

The old global filter is off by default in all PPO conditions. The ablation uses
tau=0 during push-off, 5 ms after release and 40 ms in stabilization. It preserves
push-off impulse and changes peak pitch by only -0.02°. Stabilization filtering
was not reached or validated. Native target slew limits, limited PD effort,
velocity protection and the existing target-acceleration penalty remain; no
unprovided hardware acceleration specification is invented. Actual acceleration
is governed by MuJoCo inertia/contact dynamics and actuator limits. The diagnostic
ramp is smooth, but no claim of improved terminal smoothness is made.

## Limiting mechanism and next step

At final PPO evaluation, positive rear-contact impulse is about 9.3–9.7 kg m²/s,
while the front-contact contribution is about -8.3 to -8.6 kg m²/s. Net useful
push impulse remains only 0.95–0.99 kg m²/s. Positive rear work of 26.8–28.7 J is
not retained as that much useful mechanical energy: signed work, opposing contact
moments, contact power, changing geometry and losses must be distinguished.
After rear release, COM behind the support creates a gravity moment opposing
rotation about support. Pitching momentum decays before the COM has climbed to
the handstand geometry, then calf contact ends the nominal path.

Thus the evidence supports **insufficient retained useful impulse/energy and
front-support geometry**, with poor later-leg timing sometimes worsening the
collision mode. It does not support excessive learned terminal braking: those
phases were never reached. Nor does it establish global infeasibility. The target
reward remains outside the visited region; tiny PPO updates cannot resolve that
reachability gap. No extended PPO or morphology experiment was launched.

Before larger training, test a narrowly scoped change to front-leg support
geometry/timing during the actual push, measuring net impulse, COM climb and
calf/head clearance together. Require a valid >=45° prefix with useful COM-energy
progress before escalating training. Raising L targets beyond observed capability,
rewarding rear clearance alone, or adding more uniform filtering is unsupported.

## Reproduction and artifacts

Runtime: Python 3.11.16, MuJoCo 3.11.0, PyTorch 2.13.0+cu130, NumPy 2.4.6.
Eighteen physics/new-plus-regression tests pass. Native checkpoint playback
reproduces saved joint trajectories bit-for-bit. Source/initialization hashes,
configurations, phase events, offline labels, model assets, optimizer/RNG/filter
and environment state are preserved. Output directories and ZIP files use
exclusive creation; choose fresh names when rerunning.

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/unit_tests/test_momentum_phase.py tests/unit_tests/test_com_energy_phase.py \
  tests/unit_tests/test_physical_morphology_ppo.py -q

env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/validate_momentum_phase.py \
  --output output/momentum_validation_new

env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/run_momentum_phase.py \
  --validation output/momentum_validation_new --output output/momentum_diagnostic_new \
  --updates 2 --steps 256 --seeds 42 43

env -u PYTHONPATH MPLCONFIGDIR=/tmp/momentum-mpl XDG_CACHE_HOME=/tmp/momentum-cache \
  .venv/bin/python scripts/report_momentum_phase.py output/momentum_diagnostic_new \
  --validation output/momentum_validation_new

env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/play_momentum_phase.py \
  output/momentum_diagnostic_new/C_seed42/checkpoint_002.pt \
  --output output/momentum_playback_new --seeds 94000 94001

env -u PYTHONPATH .venv/bin/python scripts/calibrate_momentum_terminal.py \
  --output output/momentum_terminal_calibration_new
```

Measured results are in `output/momentum_phase_diagnostic_20261112_v1`:
`summary.csv`, `comparison.csv`, `offline_imitation.csv`, per-checkpoint
`trace_*.csv` (50 Hz), `physics_*.csv` (500 Hz), PPO `training.csv`, and figures
`momentum_energy.png`, `phase_contacts_learning.png`, `joint_commands.png`,
`phase_portrait_moments.png`, `support_geometry.png`, `ppo_rewards.png`,
`targeted_schedules.png`. Validation is `momentum_phase_validation_20261112_v2`;
its development predecessor is retained. No PPO was run with the earlier gate.
Independent terminal calibration and deterministic playback have separate outputs.

Molab: upload/extract the provided `momentum_phase_molab_final_20261112.zip` in a fresh
Python 3.11 workspace. The included validation gate and exact offline data allow
reuse without rerunning the schedule study. Source or initialization changes
invalidate the gate and require validation again. The notebook
`notebooks/molab_momentum_phase.ipynb` provides the same bounded launch.

```bash
python -m pip install mujoco==3.11.0 numpy torch pandas matplotlib pytest tensordict wandb
OMP_NUM_THREADS=1 python scripts/run_momentum_phase.py \
  --validation output/momentum_phase_validation_20261112_v2 \
  --output output/momentum_molab_new --updates 2 --steps 256 --seeds 42 43
MPLCONFIGDIR=/tmp/momentum-mpl XDG_CACHE_HOME=/tmp/momentum-cache \
  python scripts/report_momentum_phase.py output/momentum_molab_new \
  --validation output/momentum_phase_validation_20261112_v2
# Exclusive backup; download the new archive from Molab's file browser.
python - <<'PY'
import tarfile
with tarfile.open('output/momentum_molab_backup_new.tar.gz', 'x:gz') as backup:
    backup.add('output/momentum_molab_new')
PY
sha256sum output/momentum_molab_backup_new.tar.gz
```

The runner deliberately has no extended-training mode and caps runs at four
updates. These commands were prepared; no remote Molab training was launched.
