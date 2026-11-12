# Go2 leg-length experiment

This opt-in experiment lives on `ppo_kickup_e2e`. It creates new assets and run
folders; it does not change the original URDF, CPGs, balance policies, or saved
checkpoints. No extended PPO experiments have been run. The scientific question
remains open until the paired policy experiments are complete.

## Methodology and scope

The source is [Chakraborty et al., Actuator Dynamics Curricula for
Narrow-Viability Tasks in Legged Robot Learning](https://arxiv.org/html/2609.09492),
particularly Sections 3, 4.2–4.4. The paper changes stiffness on Spot, maintains
critical damping, drives progress with an EMA of completed episode lengths,
compares ten training seeds, and evaluates ablations under deployment dynamics.
It does **not** study leg-length changes. Its cart-pole monotonicity result does
not imply morphology monotonicity on Go2.

Here the changed transition kernel comes from simulated leg morphology. Gains
remain 60 Nm/rad and 3 Nm s/rad. A single fresh actor–critic commands bounded joint
position offsets around standing throughout each episode. No CPG, reference
motion, balance-policy initialization, controller switch, or reset-pose
curriculum is used in PPO. Scripted commands are used only in the feasibility
screen. The performance schedule uses actual five-second successes in two
nonoverlapping windows of 32 completed episodes, each with success >=70%, rather
than survival alone, which could reward remaining on four feet. The schedule
only moves forward; a new morphology is selected on the next episode reset.
Actor, critic, Adam state, and exploration parameters persist across stages.

`experiments/morphology/protocol.json` specifies A–D, ten matched training seeds,
1000 matched held-out evaluation episodes per condition/seed, and equal training
budgets. These are planned experiments, not measured results. The default CLI
uses ten evaluation seeds for development; supply the protocol's full seed list
for final experiments. Evaluation costs are additional and must be reported
separately from PPO interaction costs. D is optional because four methods times
ten seeds, each with repeated evaluation, is expensive on this serial CPU runner.
The 2,048,000-control-step budget is a starting protocol, not demonstrated to be
adequate; decide any extension identically for all conditions before comparison.

## Physical transformation

`gym/envs/go2/leg_morphology.py` supports exactly .80, .85, .90, .95, and 1.00.
Original thigh and calf kinematic lengths are each .213 m. The map in each
segment frame is `A = diag(1, 1, s)`. The torso, hips, joint axes and ranges,
rotor shapes/masses/inertias, foot spheres/masses/inertias, and transverse leg
widths are unchanged. The calf and foot joint translations become -.213*s m.
Mount translations on a shortened segment follow A; mounted motors and feet do
not themselves shrink. Fixed lower-calf collision links inherit A expressed in
their own rotated local frame. Their rotations are retained, with translations
transformed through the parent's frame. Hip-to-thigh lateral offsets are fixed.

Density is held constant in the deforming thigh/calf material: `m' = s*m`.
The local central COM becomes `c' = A*c`. For the central inertia tensor I,
`C = trace(I)/2 * identity - I`, `C' = s*A*C*A.T`, and
`I' = trace(C')*identity - C'`. Thus transverse and axial inertia do not use
one uniform scale factor. Explicit inertials also preserve the source loader's
inferred density/mass semantics for the two lower-calf collision links.

Collision boxes are mapped exactly to convex meshes. Tilted cylinders become
closed 128-sided convex meshes under the same affine map. Their maximum radial
polygon error before mapping is `r*(1-cos(pi/128))`, under 5 micrometres for the
largest leg cylinder. This small representation change at shortened scales is
a documented numerical confound, not an exact cylinder equivalence claim.
Original DAE visuals have their axial mesh scale updated; the current CPU loader
discards unsupported DAE visuals and uses the collision shapes.

At 1.00 the generator bypasses every transformation and copies the source URDF
byte for byte. Tests compare compiled geometry, body COMs, inertias, joint limits,
friction, damping, and armatures, then compare positions, velocities and
accelerations exactly over 500 matched physics steps with contacts and applied
forces. The generator refuses to overwrite any existing destination.

Whole-body mass decreases as segment material shortens; motor and trunk mass
remain fixed. This is a length/material-removal experiment. A fixed-mass
redistribution ablation would be needed to separate the contribution of mass
from length before attributing any improvement solely to kinematic reach.
Neither model should be described as a verified manufactured Go2 variant.

URDF effort limits (23.7 Nm hip/thigh, 45.43 Nm calf), no-load velocity limits
(30.1 and 15.7 rad/s), position constraints and rotor armatures stay unchanged.
A common PD torque clamp is used by all A–D methods. Above the URDF speed limit,
positive-power torque is suppressed; external impacts can still exceed a motor's
no-load speed. This is a conservative motor envelope, not a calibrated Go2
current/voltage/thermal model. No gains or action scales are annealed.

Initial stance uses the same joint angles and matched seeded joint perturbations.
Forward kinematics places the feet .003 m above the ground rather than blindly
scaling root height. Near-vertical probes adjust the front thigh angle to keep
front legs vertical in world coordinates and use the same clearance rule.
PPO always resets from standing.

## Feasibility screen and interpretation

Run the short screen before deciding whether to train:

```bash
env -u PYTHONPATH .venv/bin/python scripts/morphology_feasibility.py \
  --output output/morphology_axial_validation --seeds 3
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache \
  .venv/bin/python scripts/report_morphology.py \
  --feasibility output/morphology_axial_validation \
  --output output/morphology_axial_validation/figures
```

Use a new output directory on each run. `PYTHONPATH` is cleared because the
interactive shell can inject incompatible Python 3.12 packages into this
project's Python 3.11 environment.

The pilot pairs three seeds, two push amplitudes and two extension durations,
and a near-vertical grid of 80/85/90 degrees times -.3/0/.3 rad/s, at all five
scales: 195 attempts total. It records COM, actual COM-to-front-support distance,
whole-body pitch inertia about COM and support, body pitch and pitch rate,
rear-foot clearance, all four foot loads, front drift/support retention, illegal
body-ground and self contacts, rear torque, normal impulse, speed-envelope
activation and torque saturation. The full rear normal impulse includes static
support; `push_impulse` restricts it to the extension window, and
`excess_push_impulse` subtracts a nominal half-body-weight rear load. The latter
is only a proxy: load sharing changes during crouching. Initial static geometry
and transient dynamic quantities are saved separately.

Physical leg length is `.213*s` per segment. Effective pendulum length is
`norm(COM - front_support_midpoint_on_ground)` and includes the unscaled trunk
and hip offsets. It is posture dependent and is not `.426*s`.

The greatest observed nose-down rotation is a bound for these particular
commands, **not** the maximum rotation achievable under all admissible controls.
Passive/nominal-PD stability near vertical is not a learned-policy viability
kernel. Failures cannot establish impossibility. Screening retains the same
commands for all scales and terminates on any nonfoot ground penetration.
This avoids counting a knee-supported crouch as a successful different maneuver.
The front-load/drift and rear-clearance success tests exclude many contact
shortcuts; remaining maneuver classifications should be checked using saved
traces and rendered trials before a publication claim.

A curriculum candidate must improve standing-transition success over the
full-size matched probe. With only three seeds this is a screening flag, not
statistical certification. Any candidate needs independent validation with more
seeds and an initial-state sweep, a broader bounded-action search or trajectory
optimization, and policy-controlled perturbation tests before a claim that the
learnable set of states expanded. If no candidate improves success, C/D refuse
to launch from that screening file. Do not substitute improved survival or a
smaller pendulum length for demonstrated transition feasibility.

## PPO experiments and reporting

A/B can be launched explicitly, e.g.:

```bash
env -u PYTHONPATH .venv/bin/python scripts/train_morphology.py \
  --method A --seed 0 --output output/morphology_A_seed0
env -u PYTHONPATH .venv/bin/python scripts/train_morphology.py \
  --method B --seed 0 --output output/morphology_B_seed0
```

For C/D also pass `--screening path/to/screening.json`; the shortest demonstrated
candidate is selected. B defaults to .85 as requested. D advances after equal fixed iteration blocks, reserving a final full-scale
block (250 iterations from .85 or 200 from .80 under the default budget). C is allowed to stall at a stage; its final full-scale evaluation
still occurs, and stalled progress is an experimental outcome. Every method is
evaluated at full geometry regardless of its training morphology. Do not give
C extra optimization steps if its progression is slow.

Actor and critic both use [256,256,128] ELU networks, a common Gaussian action
head, and PPO with GAE, clipped ratios, Adam and gradient clipping. PPO uses the
same reward at all scales: pitch alignment, rear lift, front support and strict
stability, with torque, action-change and fall costs. Reward formulas are in
`experiments/morphology/task.py`. The success criterion requires a contiguous
five seconds with pitch within ten degrees of vertical, |pitch rate|<.5 rad/s,
COM speed<.25 m/s, both front feet loaded above 5% body weight, rear clearance
>.10 m, front drift<.10 m, and no illegal ground contacts. Episodes last at most
eight seconds. It is a full standing-to-handstand endpoint, not a reset-to-balance
success. A second evaluation repeats seeds with a .3 rad/s pitch-velocity
increment at two seconds; logs distinguish trials that die before the impulse.
This is a limited recovery test, not a general fallen-robot recovery claim.

Each new run records source snapshots, git commit/diff/status, original URDF hash,
configuration, seeds, hyperparameters, actual control/physics steps, stage and
advancement events, per-episode reward components and termination causes,
full-scale evaluation attempts and traces, and optimizer diagnostics. Checkpoints
contain both networks, optimizer, schedule and RNG states. They are saved in new
folders, and no previous checkpoint is loaded or modified. The runner currently
does not implement exact process-restart continuation of simulator state.

Regenerate all available figures/tables with one command:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache \
  .venv/bin/python scripts/report_morphology.py \
  --feasibility output/morphology_axial_validation \
  --runs output/morphology_A_seed0 output/morphology_B_seed0 \
  --output output/morphology_report
```

Add all seeds and methods to `--runs`. PNG and vector PDF are emitted alongside
CSV tables. Figures include training reward/survival, full-size five-second
success, morphology schedule/events, physical dynamics, matched trajectories,
final success/survival/stability ablations, and PPO optimization diagnostics.
Variability is calculated across training seeds after averaging evaluation trials
within each seed. Bands/bars show one standard deviation when multiple seeds
exist. The pilot does not supply enough seed diversity for a publication claim.
Training figures are omitted and listed in `report.json` when training data do
not exist. Missing training curves are never synthesized. A 64-step PPO smoke
check is software validation, not a learning comparison.


## Measured pilot outcome

The canonical saved run is `output/morphology_axial_validation`. All 195 paired
attempts completed: 60 standing push attempts and 135 near-vertical hold probes.
No attempt met the five-second success criterion. All standing probes terminated
with rear calf ground contact; all near-vertical probes eventually contacted the
floor with a front calf. The longest valid handstand interval across the grid was
.64 s. No self contacts were observed in this probe set.

| Scale | Segment length (m) | COM-support distance (m) | COM pitch inertia (kg m²) | Mean push survival (s) | Standing successes |
| --- | --- | --- | --- | --- | --- |
| .80 | .17040 | .34654 | .45845 | .7533 | 0/12 |
| .85 | .18105 | .36058 | .47907 | .7025 | 0/12 |
| .90 | .19170 | .37477 | .50084 | .6725 | 0/12 |
| .95 | .20235 | .38908 | .52383 | .6550 | 0/12 |
| 1.00 | .21300 | .40348 | .54809 | .6400 | 0/12 |

These are reset-geometry quantities and command-specific survival times, not
learned-policy metrics. In this screen shortening reduced inertia and effective
pendulum length, but did not improve successful transition coverage. Near-vertical
mean survival was slightly **shorter** at .80 (.8926 s) than at 1.00 (.9174 s).
Mean extension-window normal impulse was 14.47 Ns at .80 versus 14.69 Ns at 1.00;
there is no observed push-impulse advantage. Rear clearance never exceeded .009 m
in the standing probes. Mean motor saturation fractions were between zero and
.000206, so widespread actuator clipping does not explain these failures.

The screen therefore does not justify selecting any shortened curriculum start.
A broader feasibility search should first find a valid maneuver with front-foot
support and calf clearance, then test paired coverage across initial states.
This is not evidence that a morphology curriculum cannot work, and does not
answer whether PPO learns more reliably. Extended PPO A–D has not been run.
PPO executions were separate one-iteration, 64-control-step software smoke
checks; their saved plots are not research learning curves.

Generated PNG/PDF figures and underlying CSV tables are in
`output/morphology_axial_validation/figures`. Missing PPO figures are explicitly
listed in that folder's `report.json`.
