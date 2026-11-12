# Separate pendulum-guided rotation experiment

Branch: `ppo_kickup_e2e`. The old fixed-versus-morphology takeoff comparison,
CPG controllers, physical models, BC datasets, actors and checkpoints are not
overwritten. New source is opt-in through `RotationEnvironment`, `RotationRun`
and `protocol_rotation.json`. No offline actor fitting or repeated takeoff
initialization is performed. The same 44-input actor, 74-input privileged critic,
12 bounded joint-target actions, gains, slew limits, native torque/speed limits,
optimizer settings and PPO configuration are retained.

## Artifacts

- `output/rotation_diagnosis_20261111_v1`: twelve final-policy replays at 2 ms
  physics resolution, control-rate full states, liftoff snapshots, counterfactuals,
  replay-certified BC phase labels and executable validation results.
- `output/rotation_search_20261111_v2`: bounded action search, all candidate
  metrics, saved best action/state sequences, optional demonstration and replay
  validation. `..._v1` is a preserved failed reporting attempt described below.
- `output/rotation_pilot_20261111_v1`: global/phase × fixed/curriculum × seeds
  42/43, with periodic full-scale stochastic/deterministic probes and complete
  network/Adam/curriculum/RNG/simulator checkpoints.
- `output/rotation_report_20261111_v1`: measured mechanics and pilot CSVs, PNG/PDF
  figures covering pitch, velocity, phase portraits, energies, length, contact
  moments, BC/PPO statistics, full-scale learning, pendulum residuals, momentum,
  tracking/retraction and search feasibility.

The old checkpoints initialize this **new objective** at update 30. Both BC
conditions for each geometry/seed load the identical actor, critic, Adam and RNG.
The two geometry methods inherit their own different training histories; this
is a continuation comparison, not a new common-initialization causal test.
The critic is retained as a warm start although its old value targets differ
from the new objective. Physical/tracker/episode states and stage history are
retained. Episode reward accumulators are cleared at the objective boundary;
the first completed training episodes can be partial episodes. Fresh evaluation
episodes always start standing. There is no running normalization state to reset.
The inherited protocol fields `base_training_steps` and `initialization` describe
the parent experiment's budget and original BC ancestry. The new runner's hard
bounded `--updates` argument determines its actual budget; migration records
identify the actual distinct update-30 starting checkpoints.

## Measured rotation limit

Final policies from both methods/seeds were replayed on diagnostic resets
61000/61001 at full scale; B was also replayed at its actual 95% morphology.
After certified release, front contacts produce a mean opposing COM-y moment
around −4.9 to −5.3 N m at full scale (about −5.3 to −5.9 at 95%). Rear contact
moment is zero over these rear-unloaded intervals. COM pitch momentum decays
from roughly .66–.92 kg m²/s at full-scale liftoff to .025–.096 at peak pitch.
The direct contact-wrench integral and measured momentum-rate are compared in
`baseline_mechanics.csv`; the validation replay has an RMS instantaneous
momentum-rate residual of 1.35 N m, consistent in sign and scale but not exact.

Joint power is net positive between liftoff and peak. Front joints deliver
about 8–10 J positive work and .05–.38 J negative work; rear joints also deliver
net positive work. Thus a simple explanation that all motors actively brake
the maneuver is unsupported. Internal joint torques cannot themselves create
net centroidal angular momentum; they change shape and contact reactions.
Measured front ground reaction moments oppose rotation even while the motors
do positive work. Increasing actuator work alone need not increase useful
body rotation or COM-over-support energy.

Counterfactual post-certificate actions hold either all joint targets or rear
targets while leaving front actions learned. Holding all targets reduces peak
pitch, although these cases avoid the later nonfoot fall and settle without
achieving a handstand. Holding rear targets barely changes peak pitch and still
falls. These interventions change articulated and contact dynamics together;
they do not uniquely identify a single joint as causal. The existing learned
continuation helps rotation relative to freezing all joints, yet fails to
produce sufficient useful momentum and COM progress. Rear calf retraction begins
about 46–60 ms after certification in the full-scale replays; timing and target
tracking are recorded rather than assumed instantaneous.

## Model conventions and limits

Positive body pitch and world-y angular momentum use MuJoCo's measured base
orientation and centroidal momentum. For COM displacement `(dx,dz)` from front
support, the user's pendulum angle is `theta=atan2(-dx,-dz)`, measured from
**downward** vertical. COM-over-support is `theta=±pi`, not zero. Body pitch is
a different articulated quantity and is never substituted for theta.

Front support is the instantaneous vertical-force-weighted center of actual
front ground contacts. Contact-frame wrench components are transformed with
`contact.frame.T`, with the sign chosen for force on the robot. Contact moments
include both `(contact_position-COM) × force` and the contact couple. Ground
forces/moments are logged separately for front, rear and nonfoot contacts.
The contact-force center can move due to force redistribution as well as foot
motion; it is not a fixed material pivot.

The model computes measured COM position/velocity, support position/finite-
difference velocity, relative sagittal L and L_dot, relative COM angular rate,
and instantaneous articulated inertia. COM-y inertia sums every link's rotated
inertia and parallel-axis term. The support-axis approximation is
`I_support = I_COM_y + mass*L²`. Logged energy estimates are:

```
V_gravity = sum(link_mass * g * link_COM_z)
barrier = mass*g*max(L-relative_COM_z, 0)
K_pendulum = .5*I_support*theta_dot²
K_centroidal_axis = H_COM_y²/(2*I_COM_y)
deficit = max(barrier - useful_toward_upright_K_pendulum, 0)
theta_ddot_prediction = -(g/L)*sin(theta) - 2*(L_dot/L)*theta_dot
```

These kinetic estimates are alternatives, not additive full articulated kinetic
energy. They omit shape angular momentum decomposition and do not prove a
trajectory can reach any specified body orientation. The frozen COM barrier is
roughly 7.1–7.3 J at full-scale liftoff while the pendulum kinetic estimate is
only .09–.59 J in these replays. Motors can add work afterward, so this deficit
is not an impossibility proof. High energy can also cause overrotation, slip or
an unrecoverable terminal velocity.

The model is marked eligible only after at least 10 ms of rear-unloaded,
front-supported, collision-free, low-drift contact with support speed <.15 m/s.
Missing support uses a labeled geometric fallback for diagnostics and receives
no model-guided potential. Moving-support velocity is subtracted; rapid support
motion/switching is rejected. Support acceleration, finite-patch support moments,
and articulated shape torque are **not** in the supplied torque-free equation.
Consequently eligible full-scale acceleration residual RMS is approximately
7–10 rad/s². This is a substantial modeling limitation, not validation of exact
Go2 dynamics. Conditional predictions restart at each eligible interval and use
measured L/L_dot; they are not predictions from morphology alone. The CSV/plots
retain measured-vs-predicted values and eligibility masks.

The optional model only modifies a bounded desired-velocity potential. Its
estimate never bypasses contact certification, native limits or terminal checks.
Body-pitch velocity scheduling is the fallback when the model is ineligible.

## New reward protocol

All original takeoff terms and certification are preserved exactly. The new
configuration disables the old 35° goal termination so rotation and later failure
are observable. The new terms are logged individually:

- `rotation_progress`: `2*(gamma*Phi_next-Phi_previous)`. Phi is zero before
  certified takeoff, without useful front support, with rear reload, collision,
  or violated speed/joint bounds. Otherwise it contains bounded body-pitch
  progress, bounded orientation-appropriate speed matching, and front-support
  progress. Restoring support restores potential; loss removes it. Terminal
  potential is zero for failure and timeout, so lean cycles cannot accumulate
  discounted progress income.
- `rotation_milestone`: one .5 credit each at 45/60/75/85°, only with certified
  rear-unloaded front support, native state validity, no collision and
  `abs(pitch_velocity)<=3 rad/s`. Credits cannot repeat during an episode.
- `rotation_arrival`: a small hold term only within 10° of upright, speed below
  .5 rad/s, useful front support and COM within .10 m sagittally of support.
- `rotation_overrun`: penalizes pitch beyond 100° or speed beyond 5 rad/s.
- `rotation_command`: additional bounded command effort cost, alongside the old
  torque and command-change costs.
- `rotation_collision`: adds a unit penalty for nonfoot/self collision or
  actuator-state termination, in addition to the original failure penalty.

The velocity envelope falls toward zero near upright. The approximate energy
deficit can inform the envelope only on eligible model samples, and its desired
speed is capped at 2 rad/s. No unbounded energy bonus is paid. The potential
telescoping and pre-takeoff negative controls are unit tested. Milestones and a
controlled near-upright hold are the new task income, not rear-foot clearance.

## Phase-aware BC ablation

The original global BC schedule has already reached its floor of 2. Both
conditions continue from that floor; the global condition applies it to every
demo frame. Phase labels come from replaying all 48 original demonstrated
episodes through the unchanged certificate, never from a clock-only guess.
No-takeoff replays would remain fully anchored; all 48 certify here.

For a demonstrated observation after certification, its eventual multiplier is
`.1 + .9*exp(-time_since_certificate/.1)`. A further four-update ramp gradually
interpolates from multiplier 1 to that value. Pre-certificate observations retain
coefficient 2. Weighted frame losses are averaged over the original sampled
batch without renormalizing by the weight sum. The loss logs both weighted and
unweighted BC loss, mean frame multiplier, and actual BC objective contribution.

Only 147 of 2,835 BC training frames are post-certificate. The dataset ends at
.78 s; minimum actual multiplier is .6459 and average multiplier at full phase
decay is .9910. Thus the minimum observed-frame coefficient is about 1.292,
not .2, and total anchor relaxation is small. It does not free a large set of
late-rotation demonstrations that do not exist. Demo sample weighting also
does not isolate separate actor parameters; the network remains shared.
`bc_action_rms` in diagnostic replays is action deviation from the original BC
actor on current observations. The separately named on-policy latent penalty is
a diagnostic reference, **not an extra loss applied to rollout observations**.
Actual BC loss/contribution is the offline-demo term logged by PPO.

## Bounded feasibility result

Six contexts cover all final policy/seed combinations at full scale plus B's
95% geometry. Search starts at the policy's own certified liftoff snapshot and
uses the same action interface, contact geometry and native actuator limits.
Three action knots with grouped front/rear joint-type residuals give 18 search
variables. A seeded CEM search uses three generations of 24 candidates and
25 control steps (.5 s) per candidate. No runtime controller is installed.
Candidate validity requires no nonfoot/self collision, joint violation <.001
and speed ratio <=1.001 throughout the evaluated horizon. Search score and
best valid pitch are reported separately; a high pre-failure pitch is not a
valid sequence.

The 432 completed candidates use 8,456 simulator transitions, separately from
PPO. A first attempt completed the first 72 candidates, then failed JSON output
on a NumPy boolean; it is preserved in `rotation_search_20261111_v1`, adding
1,436 search transitions. The repaired complete run repeats that context;
total search cost including the failed attempt is 9,892 transitions. No search
transitions are counted as policy learning.

Best valid peaks: fixed 42 at 100%, **45.19°**; fixed 43, 40.75°;
curriculum 42 at 95%/100%, 28.76°/37.35°; curriculum 43 at 95%/100%,
31.51°/44.71°. No valid 60° trajectory was found. This bounded search is not
a proof of physical impossibility or a globally optimal action search.

The 45.19° prefix replays exactly on its development reset, but ends rotating
backward at −2.26 rad/s. Three nearby reset checks (61002–61004) fail validity and
peak at 43.76–44.81°. It is fragile, not a recoverable handstand. The optional
NPZ contains the standing actor prefix followed by the optimized actions,
44-dimensional observations, qpos/qvel, targets, times, phase labels and scale.
It remains separate from the original BC dataset and is **not used in this
pilot**. No deployed controller or inference CPG is introduced.

The 95% 35°/.04-second progression threshold was not relaxed. These warm-start
95% search contexts did not reach 35°, but the finite search cannot distinguish
an inherent morphology limit from a limited policy/search parameterization.
It therefore does not justify retroactively weakening the original criterion.

## Exact reproduction and Molab commands

Run from the repository root with W&B disabled. Every output path must be new.
Local Python is `.venv/bin/python`; remove exported ROS/Conda PYTHONPATH.

```bash
export WANDB_MODE=disabled
export MPLCONFIGDIR=/tmp/rotation-mpl
export XDG_CACHE_HOME=/tmp/rotation-cache
unset PYTHONPATH
.venv/bin/python scripts/rotation_learning_tools.py diagnose \
  --source output/multimorph_bc_ppo_continuation_20261111_v2 \
  --output output/rotation_diagnosis_NEW
.venv/bin/python scripts/validate_rotation_learning.py --diagnosis output/rotation_diagnosis_NEW
.venv/bin/python -m pytest tests/unit_tests/test_rotation_learning.py \
  tests/unit_tests/test_takeoff_learning.py tests/unit_tests/test_takeoff_bc_regularization.py \
  tests/unit_tests/test_physical_morphology_ppo.py -q
.venv/bin/python scripts/rotation_learning_tools.py search \
  --source output/rotation_diagnosis_NEW --output output/rotation_search_NEW
.venv/bin/python scripts/export_rotation_demonstrations.py \
  --diagnosis output/rotation_diagnosis_NEW --search output/rotation_search_NEW
.venv/bin/python -u scripts/run_rotation_learning.py \
  --source output/multimorph_bc_ppo_continuation_20261111_v2 \
  --diagnosis output/rotation_diagnosis_NEW --output output/rotation_pilot_NEW --updates 6 \
  > output/rotation_pilot_NEW.log 2>&1
.venv/bin/python scripts/report_rotation_learning.py \
  --diagnosis output/rotation_diagnosis_NEW --search output/rotation_search_NEW \
  --pilot output/rotation_pilot_NEW --output output/rotation_report_NEW
```

For Molab, upload `rotation_learning_molab_bundle.zip`. Use a manually executed
terminal/cell; no training launches on notebook import:

```bash
git clone --branch ppo_kickup_e2e --single-branch https://github.com/JonathanDistler/BigRedGym.git BigRedGym_rotation
cd BigRedGym_rotation
python - <<'PY'
import hashlib,json,pathlib,subprocess,zipfile
root=pathlib.Path.cwd()
with zipfile.ZipFile('../rotation_learning_molab_bundle.zip') as z:
    m=json.loads(z.read('molab_manifest.json'))
    subprocess.run(['git','checkout','--detach',m['base_commit']],check=True)
    for name,digest in m['files'].items():
        f=(root/name).resolve();assert f.is_relative_to(root)
        data=z.read(name);assert hashlib.sha256(data).hexdigest()==digest,name
        f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
print('Verified source/checkpoint/data bundle')
PY
python -m venv .venv
.venv/bin/python -m pip install torch==2.13.0 mujoco==3.11.0 numpy==2.4.6 pandas==3.0.5 matplotlib==3.11.1 pytest ruff
export WANDB_MODE=disabled MPLCONFIGDIR=/tmp/rotation-mpl XDG_CACHE_HOME=/tmp/rotation-cache
unset PYTHONPATH
```

The full reproduction commands above work after extraction. To continue the
**new rotation** checkpoints for another bounded diagnostic, only after reviewing
the recommendation below:

```bash
.venv/bin/python scripts/validate_rotation_learning.py --diagnosis output/rotation_diagnosis_20261111_v1
.venv/bin/python -u scripts/run_rotation_learning.py \
  --source output/rotation_pilot_20261111_v1 --resume-update 36 \
  --diagnosis output/rotation_diagnosis_20261111_v1 \
  --output output/rotation_molab_bounded_next --updates 6 \
  > output/rotation_molab_bounded_next.log 2>&1 &
echo $!
tail -f output/rotation_molab_bounded_next.log
```

The runner allows only 1–10 new updates per invocation and has no million-step
launch switch. Resume loads each checkpoint's new protocol/phase schedule,
actor/critic, Adam, curriculum, observation arrays, simulator state and RNG.
Do not use the old takeoff evaluator for rotation checkpoints: it would omit
the opt-in environment/model and new reward configuration.

Monitoring, plot regeneration and full-geometry deterministic/stochastic
checkpoint evaluation use the saved probes and runner evaluator. The command
below evaluates checkpoints without PPO updates (see its CLI in the source):

```bash
cat output/rotation_pilot_20261111_v1/outcome.json
tail -n 4 output/rotation_pilot_20261111_v1/phase/B_seed42/training.csv
.venv/bin/python scripts/evaluate_rotation_checkpoint.py \
  --run output/rotation_pilot_20261111_v1 --update 36 \
  --output output/rotation_fresh_evaluation_molab
.venv/bin/python scripts/report_rotation_learning.py \
  --diagnosis output/rotation_diagnosis_20261111_v1 --search output/rotation_search_20261111_v2 \
  --pilot output/rotation_pilot_20261111_v1 --output output/rotation_report_molab
```

Complete source/results backup and browser download from a manually executed
Molab Python cell (choose a fresh filename):

```python
import pathlib,zipfile
import marimo as mo
root=pathlib.Path('BigRedGym_rotation')  # Path('.') if already inside
backup=root.parent/'rotation_full_backup.zip'
with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('output','experiments','scripts','docs','notebooks','resources/robots/go2'):
        for f in (root/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:
                z.write(f,f.relative_to(root))
mo.download(backup.read_bytes(),filename=backup.name)
```

CPU MuJoCo is used; GPU availability does not accelerate these Python simulation
loops. Local source/bundle checks do not establish that Molab dependency
installation or execution has been tested in the cloud.

## Completed PPO pilot and recommendation

**No-go for extended PPO under the current rotation protocol.** Preserve this
pilot as a negative short-budget result. Takeoff remains reliable, but useful
rotation does not improve, the phase ablation has no supported advantage, and
the physical morphology curriculum remains blocked at 95%. This does not prove
that a single actor cannot learn rotation; the six-update budget, narrow .01
exploration and sparse post-certificate BC data are important limits.

All eight runs complete updates 31–36 without a divergence or retention stop.
Each receives 6,144 new training transitions, 49,152 total. Each A condition
spends all new transitions at true 100% geometry; each B condition spends all
6,144 at 95%. No new B full-scale training is claimed. The original 90% → 95%
events and all curriculum history are retained, with no new promotion or reset
of actor, critic or Adam. The pilot's safety gate stops a matched quartet if
KL>.03, clipping>.30 or either takeoff mode loses more than 25 percentage points
at a periodic probe. Evaluations are at 30/32/34/36 on resets 62000–62007,
separate from training, diagnosis and search. All probes give 8/8 deterministic
and stochastic takeoffs at current and full geometry. Five-second success is
zero; all full-horizon policy episodes eventually have nonfoot collisions.

Full-scale deterministic peak pitch on the **same periodic reset family**,
mean ± one sample standard deviation across two training seeds:

| BC condition | Geometry method | Update 30 | Update 36 |
|---|---|---:|---:|
| Global floor 2 | Fixed | 34.15 ± .90° | 33.68 ± .85° |
| Phase-aware | Fixed | 34.15 ± .90° | 33.50 ± .85° |
| Global floor 2 | Curriculum transfer | 31.09 ± 1.25° | 29.95 ± .98° |
| Phase-aware | Curriculum transfer | 31.09 ± 1.25° | 29.81 ± .72° |

These new-family update-30 values differ slightly from the earlier 58000-family
report despite identical loaded actors. Do not compare different reset families
as if they measured a training change. Stochastic final pitch is 31.68/31.61°
fixed global/phase, and 28.67/28.51° curriculum global/phase.

Final fixed global/phase liftoff COM-y momentum is .835/.828 kg m²/s versus
.857 initially; curriculum global/phase is .650/.651 versus .711 initially.
The estimated liftoff energy deficit does not shrink: about 6.81/6.84 J fixed
and 7.14/7.16 J curriculum. These unfiltered point estimates remain approximate;
raw diagnostic trajectories contain the eligibility masks. Mean front-moment
learning curves average the whole certified-post-takeoff interval until failure,
whereas the baseline mechanics table isolates liftoff through peak pitch.

Across the pilot, KL is .01305–.01953 and clipping 4.00–10.94%. BC loss is
.00435–.00609 and BC objective contribution .00870–.01218. Original-demo action
RMS deviation is .00188–.00277 normalized action units. Phase-aware batch mean
weights reach about .990, and the original takeoff prefix remains anchored.
The small differences between global and phase conditions do not establish a
mechanism effect or a curriculum advantage.

An independent final family, 64000–64007, is evaluated only after training:
**192/192 certified takeoffs** across all eight policies and applicable scales,
with zero handstands. Full-scale peak pitch is:

| BC condition | Method | Deterministic | Stochastic |
|---|---|---:|---:|
| Global | Fixed | 33.64 ± .94° | 31.99 ± .59° |
| Phase | Fixed | 33.46 ± 1.02° | 31.94 ± .67° |
| Global | Curriculum | 29.96 ± 1.09° | 29.29 ± .96° |
| Phase | Curriculum | 29.88 ± .83° | 29.18 ± .65° |

`heldout_per_seed.csv` and `heldout_summary.csv` retain all scale/mode results.
The eight final checkpoint restore audits verify all network/optimizer tensors,
parameter groups, curriculum, Torch/NumPy RNG, observations and MuJoCo integration
state before evaluation; they perform zero optimizer updates.

The next bounded investigation should improve **recoverable trajectory
feasibility**, not buy a large unchanged PPO budget. Search with a terminal
state constraint (front support, low COM drift/speed and manageable angular
velocity), validate 45° sequences on multiple new resets, and expand toward
60° only after robust intermediate behavior exists. Measure contact lever arms,
front-support motion and articulated retraction timing rather than maximizing
clearance or kinetic energy. Include a crossed policy-at-95% feasibility probe
before attributing the 95% gate failure to geometry itself. Refine the moving-
support/shape-momentum approximation before increasing model influence.

If robust safe prefixes are obtained, fit optional post-takeoff demonstrations
to a **copy of the same actor**, keeping original prefix data and held-out
certification gates. Then repeat a matched small-budget global/phase ablation
with actual post-takeoff coverage and identical augmentation for A/B. The current
fragile demo is an optional development artifact, not suitable evidence for
deployment or a reason to erase the original BC actor. None of this follow-up
training has been launched.

## Validation and accounting

All 26 targeted tests pass, including disabled-instrumentation physics/reward
parity, angle/energy conventions, potential telescoping, contact-wrench sign,
BC weighting, new auxiliary-state restore, and original PPO resume invariants.
Ruff and Python compilation pass for the changed scripts/modules. The executable
preflight gate verifies exact original takeoff reward/certificate preservation,
finite rotation rewards and the rotation-source hash before optimization.

`preservation_audit.json` verifies 244 historical artifacts against the previous
bundle manifest. PPO training uses 49,152 transitions; periodic probes use
40,261, ordinary checkpoint evaluations 15,089 and fresh final tests 10,032.
Search uses 9,892 including the preserved failed output attempt. Demonstration
and diagnostic replay costs are separate in their raw files; no offline BC
optimization was performed. `interaction_accounting.json` records the budgets.
All figure horizontal learning axes use actual cumulative per-policy training
interactions (30,720 → 36,864), not nominal updates or pooled evaluation steps.

The source/checkpoint bundle includes both old and new experiment states,
dataset/phase hashes, optional demonstrations, exact physical assets, measured
tables, figures and code. W&B stays disabled. Source availability and local
resume verification do not imply Molab cloud execution has been tested.

Final audit found and repaired a logging-only issue: the original takeoff step
replaced its component dictionary each step, so the added rotation **episode
component columns** in raw training/evaluation episode logs recorded only the
last control step. Per-step reward outputs, total episode reward, PPO updates,
behavior metrics and go/no-go results are unaffected. The affected raw pilot
tables are preserved and those component columns must not be interpreted as
episode sums. `reward_components_reconstructed.csv` sums all components from
the original saved per-step evaluation traces for 288 evaluation episodes;
each sum matches the actual returned episode reward within numerical precision.
No training reward components were invented or retroactively reconstructed.

The source correction retains extra component accumulators before calling the
base step. A regression test verifies identical physical motion and returned
rewards with historical versus corrected logging, and correct cumulative
component sums. `source/rotation_pilot_original.py` and
`validation_pilot_original.json` preserve the exact pilot-era source/hash and
pre-training gate; the current `validation.json` revalidates the corrected
source. Reproduction with the correction changes logging, not the objective.

Bundle reproduction (new filename required):

```bash
.venv/bin/python scripts/package_takeoff_molab.py \
  --output output/rotation_learning_molab_bundle_NEW.zip \
  --extra-root output/multimorph_bc_20261111_v1 \
  --extra-root output/multimorph_bc_20261111_v2 \
  --extra-root output/multimorph_bc_ppo_retention_20261111_v1 \
  --extra-root output/multimorph_bc_ppo_continuation_20261111_v2 \
  --extra-root output/rotation_diagnosis_20261111_v1 \
  --extra-root output/rotation_search_20261111_v1 \
  --extra-root output/rotation_search_20261111_v2 \
  --extra-root output/rotation_pilot_20261111_v1 \
  --extra-root output/rotation_report_20261111_v1 \
  --extra-root output/rotation_heldout_20261111_seed42 \
  --extra-root output/rotation_heldout_20261111_seed43
```

To regenerate final held-out tables alongside the figures, add
`--heldout output/rotation_heldout_20261111_seed42 output/rotation_heldout_20261111_seed43`
to the reporting command. The archive SHA256 and post-packaging verification
record are delivered separately, avoiding a circular self-hash dependency.
