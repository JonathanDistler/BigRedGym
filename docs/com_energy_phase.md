# COM energy and phase-conditioned Go2 diagnostic

This opt-in experiment lives on `experiment/com_energy_phase_ppo`, already based on
`059ac39` (recoverability), whose parent `cc0d7a8` contains the working takeoff and
rotation PPO implementation. No merges, old source edits, checkpoint replacements,
CPG changes, or morphology changes were made. New code is in
`experiments/com_energy_phase/`; entry points are `scripts/*com_energy_phase.py`.

## Measured outcome

The authoritative short run is `output/com_energy_phase_diagnostic_20261112_v2`.
Each arm received **256 PPO transitions per training seed**, two updates, seeds
42/43, full 100% Go2 geometry. Each checkpoint (initial, update 1, update 2) was
played deterministically from evaluation seeds 91000/91001. This is four final
rollouts per arm, insufficient for a statistical learning claim.

| Arm | Initial certified peak pitch | Final certified peak pitch | Final takeoff | Five-second handstand |
|---|---:|---:|---:|---:|
| A: BC takeoff PPO | 31.66° | 32.30° | 4/4 | 0/4 |
| B: energy + orientation | 31.66° | 32.28° | 4/4 | 0/4 |
| C: B + phase and phase-weighted BC | 31.99° | 32.28° | 4/4 | 0/4 |
| D: C + reward/action EMA | 25.51° | 24.73° | 4/4 | 0/4 |

**The final diagnostic shows no improvement over baseline.** All final rollouts
ultimately hit a front calf on the ground. Certified peak angles concern the
collision-free prefix, not the subsequent collision. The historical 30–35°
baseline is reproduced. Phase initialization changes the initial C actor slightly;
D's additional initial degradation comes from filtering that same actor's commands.

At the supported rotation apex, mean potential energies were about 47.76, 47.74,
48.21, and 42.31 J for A/B/C/D, respectively. Target potential is **89.42 J**.
Link kinetic energies there were about 4.08, 4.08, 4.19, and 1.88 J. These are
instantaneous articulated link energies, not all available as useful pitching
energy. Pitch velocity approached zero at the apex. Initial standing height was
already higher than the swing-apex COM height: the energy-progress reward received
no positive height credit. The gated multiplicative target reward remained zero.

For evaluation seed 91000 averaged over both training seeds, D reduced target
acceleration RMS from approximately 35.18 rad/s² (C) to 25.56 rad/s², but reduced
peak COM angular momentum from about 1.12 to 0.66 kg m²/s and rear clearance from
about 17 cm to 3.5 cm. Smoothing the existing rapid push-off hurts this configuration.
The `joint_tracking_rms` summary is position tracking error in radians; it is not a
torque error. Additional frozen-policy ablations log joint velocities, requested
PD torque and actual limited torque for a direct control-interface comparison.

A development run `..._v1` is retained. Its stronger command-acceleration cost
produced 33.78°/33.62° in B/C but was superseded when acceleration units and target
safety gates were corrected. It is not evidence for the final implementation.
There were two small diagnostic runs (2,048 training transitions each), no large
searches or extended Molab training. Initial validation-only runs are preserved.

## Articulated measurements and conventions

Use the positive-mass descendants of MuJoCo's free-base robot root, excluding
world and unrelated bodies. With link inertial positions x_i, masses m_i, body-COM
Jacobians J_vi/J_wi, principal inertias D_i and world inertial rotations R_i:

- m = Σ m_i; c = Σ m_i x_i / m; v_c = Σ m_i J_vi qdot / m.
- Trunk COM is `xipos[base]`; whole-robot COM is c. All energy rewards use c.
- I_i = R_i D_i R_iᵀ.
- H_c = Σ [I_i ω_i + (x_i−c) × m_i v_i].
- I_c = Σ [I_i + m_i((r_i·r_i)1 − r_i r_iᵀ)].
- K_links = Σ [½m_i |v_i|² + ½ω_iᵀ I_i ω_i].
- K_translation = ½m|v_c|²; internal kinetic energy is K_links−K_translation.
- MuJoCo rotor armature contributes K_rotor = ½Σ armature_j qdot_j² separately.
- E_p = m|g|c_z, with the same world z=0 reference for target and current pose.

Independent sums are checked against `subtree_com`, `subtree_linvel`,
`subtree_angmom`, `body_subtreemass` and MuJoCo `energy`. Tests account for rotor
armature when comparing kinetic energies. Playback errors were below 3e-16 m for
COM and about 4e-15 kg m²/s for momentum. Main training CSV `total_energy` is
K_links+E_p; rotor and MuJoCo-compatible kinetic energy are separate columns.

The support reference is the force-weighted centroid of currently loaded front
foot contact points. Plane height and normal come from those contacts. Both
front feet must carry >5% body weight for valid bilateral support. Contact force
normals are read in MuJoCo's contact frame. A support-line segment with foot radius
provides horizontal COM distance. Unsupported references have an explicit invalid
flag; numerical placeholder values are never interpreted as valid pivots.

`pivot_valid` additionally requires low foot sliding velocity, consecutive valid
support, and <0.15 m/s movement of the contact reference. This rejects contact
switches and moving centers of pressure for fixed-pivot interpretation. Sliding
contact can still be physical support: it does not become a stationary pendulum
pivot. This implementation uses the existing level-ground Go2 model; inclined
terrain has not been validated.

For support point s, r=c−s, L=|r|. The signed sagittal pendulum angle is
atan2(r_x,r_z), **zero directly above support**, distinct from body pitch.
The full angle to upward gravity is acos(r_z/L). Body pitch follows the existing
atan2(−R_20,R_00) convention; positive pitch rotates the forward body axis downward.
Do not interpret a large body pitch as a large COM pendulum angle.

I_support,y = I_c,yy + m(r_x²+r_z²).
H_support = H_c + (c−s)×m v_c is instantaneous momentum about the world point s;
with moving s it does not obey the stationary-pivot torque equation.
The coherent rigid pitching estimate H_c,y²/(2I_c,yy) is a diagnostic proxy,
not an additive kinetic-energy term. It must not be added to K_links.
The point-mass barrier m|g|(L−r_z) and required-momentum estimate
sqrt(2 I_support,y max(E_target−E_p,0)) ignore articulated redistribution,
actuator work and contact losses. Neither is an exact feasibility certificate.

A fixed L is captured at first valid front support and compared with variable
L(t) using m|g|L cos(theta_gravity), including lateral displacement. The exact potential is still computed from articulated
COM height. Before support, these model comparisons are marked invalid.
Training always uses full MuJoCo dynamics, native collisions and motor limits.

## Target provenance

The checked archive is `output/recoverability_maps_20261112`:
`recoverable_states.npz` and `validated_region.json` contain snapshots from
successful balance continuations in empirically validated cells. Fifty-four
near-vertical, supported, collision-free snapshots matched this model's COM and
all twelve joint coordinates within 1e-7. The median-height member supplies the
actual joint configuration and complete orientation matrix, rather than a
straightened jumping pose or maximal height.

Measured mass: **16.2764066211 kg**; selected COM height: **0.5600034658 m**;
E_target: **89.4166207975 J**. Full qpos, rotation, measurements, index and source
hashes are in `reference.json`. This establishes a balance target, not a reachable
standing-to-handstand trajectory or a broad recovery region. No IK fallback or
invented full-handstand teacher was needed.

## Reward and safety architecture

r_E = exp(−½((E_p−E_target)/sigma_E)²), sigma_E=12 J.
r_R = exp(−½(error_SO3/sigma_R)²), sigma_R=0.35 rad.
error_SO3 = acos(clamp((trace(R_targetᵀR)−1)/2,−1,1)).
The complete matrix comparison handles pitch sign, yaw and roll.

Target reward rate is 4 r_E r_R; integrated reward is rate × control dt.
The logged `r_target` is this integrated term. `r_energy` and `r_orientation`
are the instantaneous Gaussian signals. EMA traces are logged separately.

Target gating requires current certified rear flight, front support, clean
collision/limit history, <10 cm front drift, <6 cm COM distance from the support
region, COM speed <0.75 m/s, pitch speed <2 rad/s, and no contact loss/collision
anywhere in the current physics interval. A stale EMA is reset immediately when
this gate fails. Success uses tighter 10° full-orientation error, <0.5 rad/s pitch
speed, <0.25 m/s COM speed, <4 cm support-region distance for five continuous
seconds. These are physical events, independent of phase progress.

Dense terms are best-so-far, one-use increments of normalized useful potential,
orientation progress and capped rotational energy proxy while below target.
They require certified rear flight and valid front support: static leaning,
unsupported jumping, repeated squat cycles and stationary intermediate poses
cannot collect them indefinitely. The existing physics-rate takeoff preparation,
release, flight and rotation credits remain. Pumping credit is capped at 0.2 per
episode before curriculum scaling, tapers with energy deficit, and requires the
correct rotation direction. Near target, velocity braking and stable dwell
replace encouragement to add energy. Existing effort/target-change penalties
remain; new dimensionless target-acceleration and near-target power penalties
apply only to the new reward arms.

Pumping is an experimental reward, not proof of physical pumping: signed work
is integrated from actual limited generalized force times joint velocity at
500 Hz. New frozen-policy diagnostics also integrate positive actuator work.
Internal leg motion redistributes angular momentum; ground torque supplies
external angular impulse. Changes in approximate pendulum energy alone do not
establish actuator generation. A full contact-work/dissipation decomposition is
not implemented; the diagnostic cannot uniquely separate every energy loss.

EMA alpha=exp(−dt/tau), beta_action=1−exp(−dt/tau_action), with zero tau giving
instantaneous signals. D uses tau_E=tau_R=40 ms, tau_action=20 ms. Commands are
filtered before the existing speed-related slew limit and clipped PD effort.
All native joint, torque, velocity and collision checks remain active.
The frozen-policy ablation separates reward-only and action-only smoothing;
the PPO C/D comparison alone changes both and cannot isolate their effects.

## Phase, supervision and curriculum

The actor still emits twelve joint targets. C/D append sin(pi p), cos(pi p), p
for a one-shot p∈[0,1] to the existing 44 actor observations (47 total); the critic
gets the same three values (77 total). A half-circle distinguishes start/end.
A monotonic time-only option exists; event conditioning is the default.

The observed dataset is `output/multimorph_bc_20261111_v2/bc_dataset.npz` and its
actor hash must match. Forty-eight morphology/seed prefixes supply measured
forward loading, peak rear-calf extension and rear-foot unloading events.
Median full-geometry timing knots are **0.02, 0.68, 0.70 s**. Coincident events
are separated by one control frame for ordered interpolation; original event
measurements are retained. Later 0.90/1.30 s timing knots are configurable nominal
windows, not demonstrated full-handstand schedules. No joint reference is played.

Phase cannot pass 0.199 without rear loading, or 0.399 without takeoff.
After measured takeoff, the phase clock aligns to that event; it remains below
0.8 until 75° body pitch, and reaches 1 only at a physically stable target.
Demonstration alignment uses recorded unloading, while live takeoff uses the
stricter physics-rate certification with dwell/work/impulse checks; this small
event latency difference is an explicit approximation. Later training should
recapture demonstration phase through the identical live tracker.

Actor migration copies the 44-input actor exactly and initializes new phase
columns to zero. C/D receive eight small supervised updates restricted to those
columns. PPO uses a configurable shared BC objective with full push-off weight
and 0.1 after release in C/D; A/B retain uniform BC anchoring. Last four episode
seeds are excluded from supervised optimization. Actual imitation targets are
only observed prefix actions across the existing morphologies. Online control
and standalone checkpoint playback use the same Phase implementation.

Performance curriculum A–E uses 16 completed episodes and a 75% passing fraction:
certified takeoff; ≥45°; ≥60°; ≥80° plus controlled recovery; five-second success.
Early certificates concern the safe prefix even if a later fall occurs. Final
approach requires clean whole-episode safety. Milestones are pass-through, not
static holds. Dense shaping gains increase from .25 to .5 to 1 as stages advance.
The smoke budget did not advance the curriculum. Geometry remains 100%; a future
90→95→100 comparison is deliberately deferred and not implemented in this runner.
An optional `--intermediate-fraction` enables terminal-state training from the
checked balance archive after stage D is reached (default zero, unused in these
diagnostics). Every reset rechecks actual support, rear clearance, joints,
velocities, collisions and near-vertical orientation. These episodes explicitly
log `intermediate_start`; they never claim certified takeoff or standing-transition
success. `transition_success` requires an episode that started from standing.
The valid-reset mechanism and rejected-standing negative control are tested.
Earlier 45°/60° reset states remain unavailable; none are fabricated from the
kick-up teacher. Existing physical morphology/recoverability tools remain.

## Tests, artifacts and reproduction

Thirteen new-plus-regression tests pass: independent COM/trunk/mass/energy/momentum,
orientation/yaw, time coefficients, phase events/reset, grounded/unsupported
negative controls, EMA reset, action filter, finite actor/critic gradients,
reference matching, demo alignment and deterministic physical checkpoint replay.
Seven tests belong to the new module. The runner executes those before any PPO
update and stores `validation.txt`; source snapshots, hashes, model assets,
configurations, optimizer/network/RNG/environment/filter/curriculum state are
saved. Old checkpoints are not modified. Replay from a serialized C checkpoint
was verified against its saved trace.

The equations' exponential rewards have finite derivatives for finite errors;
PPO differentiates its surrogate/BC losses, not the MuJoCo reward implementation.
This is CPU MuJoCo research code, not a hardware-certified deployment policy.

```bash
# From the repository root; avoid the shell's incompatible Python 3.12 PYTHONPATH.
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/unit_tests/test_com_energy_phase.py \
  tests/unit_tests/test_physical_morphology_ppo.py -q

# Always choose a NEW output directory. The runner refuses existing directories.
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/run_com_energy_phase.py \
  --output output/com_energy_phase_new_run --updates 2 --steps 128 --seeds 42 43

env -u PYTHONPATH MPLCONFIGDIR=/tmp/com-energy-mpl XDG_CACHE_HOME=/tmp/com-energy-cache \
  .venv/bin/python scripts/report_com_energy_phase.py output/com_energy_phase_new_run

env -u PYTHONPATH .venv/bin/python scripts/play_com_energy_phase.py \
  output/com_energy_phase_new_run/C_seed42/checkpoint_002.pt \
  --output output/com_energy_phase_new_playback --seeds 91000 91001

env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/ablate_com_energy_phase.py \
  output/com_energy_phase_new_run/C_seed42/checkpoint_002.pt \
  --output output/com_energy_phase_new_ablation

# Prepare a portable source + exact offline data/target bundle (exclusive creation).
env -u PYTHONPATH .venv/bin/python scripts/package_com_energy_phase.py \
  --output output/com_energy_phase_molab_bundle_new.zip
```

Molab Online: upload and extract that bundle into an empty Python 3.11 workspace,
then run these cells/terminal commands. No remote training was launched here.

```bash
python -m pip install mujoco numpy torch pandas matplotlib pytest tensordict wandb
OMP_NUM_THREADS=1 python scripts/run_com_energy_phase.py \
  --output output/com_energy_phase_molab_diagnostic --updates 2 --steps 128 --seeds 42 43
MPLCONFIGDIR=/tmp/com-energy-mpl python scripts/report_com_energy_phase.py \
  output/com_energy_phase_molab_diagnostic
# Back up all checkpoints, RNG/filter state, CSVs and source snapshots together.
# Use a new archive filename; download it from the Molab file browser.
tar -czf output/com_energy_phase_molab_backup_new.tar.gz \
  output/com_energy_phase_molab_diagnostic
sha256sum output/com_energy_phase_molab_backup_new.tar.gz
```

The main artifact directory contains `comparison.csv`, `summary.csv`,
`physical_summary.csv`, per-checkpoint `trace_*.csv`, per-arm PPO `training.csv`,
`physics_rewards.png`, `phase_portrait_actions.png`, target provenance and event
alignment. Frozen-policy separate smoothing ablations are in
`output/com_energy_phase_ablation_20261112_v1`. Extended torque/work telemetry was
added after the final training run; the training source snapshots remain exact.

Frozen C-seed42 final-policy ablation (two evaluation seeds, zero PPO updates):
instantaneous and reward-only EMA both reached 33.56°; action-only EMA and both
filters reached 26.07°. Reward-only EMA produced bitwise identical actual joint
trajectories. Positive actuator work dropped from 26.92 to 18.50 J with action
filtering. Requested-versus-applied torque RMS was 0.65 versus 0.73 Nm; requested
PD torque is evaluated at the sampled final state, whereas actual torque is from
the preceding physics step, so this includes sampling lag as well as limits.
All four variants had zero target reward: this ablation establishes a control
filter effect, not a reward-EMA learning benefit.

Recreate its plots and summary with:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/com-energy-mpl XDG_CACHE_HOME=/tmp/com-energy-cache \
  .venv/bin/python scripts/report_com_energy_ablation.py \
  output/com_energy_phase_ablation_20261112_v1
```

## What the evidence supports next

The intermediate-reset option and additional diagnostics were added after the
short PPO run; default standing-reset behavior is unchanged and tests pass.

The immediate limits are inadequate upward COM energy accumulation and arrest of
pitching momentum, followed by front-calf collision. The experiment cannot claim
that energy injection alone solves the support geometry problem: actuator work,
front-leg mechanics and contact feasibility all matter. The narrow target reward
is inactive throughout these trajectories; phase features and 256 transitions
per seed do not establish learning benefit. Do not extrapolate these smoke runs
to PPO convergence or a reliable handstand.

Before a larger Molab run, retain C without action filtering, replay/alignment
through the exact live takeoff tracker, and verify that dense energy progress is
reachable from a valid intermediate state without calf contact. Then run a small
independently seeded diagnostic with stronger post-takeoff exploration and
separate reward-only smoothing. Do not launch extended training or a morphology
comparison on the current evidence. No broad trajectory optimization is needed.
