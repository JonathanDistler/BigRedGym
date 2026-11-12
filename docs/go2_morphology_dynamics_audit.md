# Contact-corrected Go2 morphology dynamics audit

## Conclusion and next experiment

**Physical shortening did not improve kick-up feasibility in this bounded command
and feedback-controller audit.** This conclusion uses supported takeoff, impulse,
pitch, clearance and actuator constraints, not five-second balance success alone.
The original rear-calf failure was a reference/posture failure, not a penetrating
reset or an unavoidable property of shortened collision geometry. Correcting
that failure produces supported takeoffs at all three priority morphologies.
The full-size morphology then performs better under both matching schemes.

**Recommend C as a limited next actor–critic ablation:** keep full-size geometry
and use feasible standing postures to vary COM-to-support length. Retain a normal
full-size standing PPO baseline with the same architecture, rewards, interaction
budget and held-out seeds. Progress initialization posture back toward the normal
stance only after actual takeoff/transition thresholds, and evaluate both from
the normal full-size standing state. A short ablation is warranted by increased
supported lift duration, not by demonstrated handstand success. No new PPO was
run in this audit, and extended training remains premature.

Do not start physical-link curriculum A on the strength of lower inertia alone.
B remains a useful full-size baseline, but shortened geometry has not produced
better demonstrations in these tests. Do not adopt D as a universal impossibility
claim: these controllers leave actuator authority unused, and the command grid
is not trajectory optimization or an optimal-control bound. Preserve the
physical-shortening hypothesis as an unproven ablation; a tuned controller or
bounded optimization could change the outcome.

## Saved experiments and reproduction

Canonical measured data:

- `output/morphology_dynamics_validation`: 20 legacy contact replays, 144 new
  takeoff attempts and 54 independent balance probes.
- `output/morphology_posture_validation`: 36 additional full-size posture
  takeoff attempts.
- `output/morphology_dynamics_validation/figures`: eleven PNG/PDF figures,
  CSV summary tables, representative trajectories and pendulum-model comparison.

Earlier `morphology_contact_audit*` folders are development runs. They are
preserved, including interrupted runs; they are not pooled into the final results.
The prior 195-probe experiment is also preserved. All final figures use only the
two canonical runs listed above.

From the repository root, these commands reproduce the final experiment into
**new** folders (the runners refuse existing output folders):

```bash
env -u PYTHONPATH .venv/bin/python scripts/audit_morphology_dynamics.py \
  --output output/morphology_dynamics_repro --seeds 3

env -u PYTHONPATH .venv/bin/python scripts/audit_morphology_posture.py \
  --output output/morphology_posture_repro --seeds 3

env -u PYTHONPATH MPLCONFIGDIR=/tmp/morphology-mpl \
  XDG_CACHE_HOME=/tmp/morphology-cache \
  .venv/bin/python scripts/report_morphology_dynamics.py \
  --input output/morphology_dynamics_repro \
  --posture output/morphology_posture_repro \
  --output output/morphology_dynamics_repro/figures
```

To regenerate the canonical figures without simulation, use the same reporting
command with `morphology_dynamics_validation` and `morphology_posture_validation`.
No network download or new dependency is needed. IK uses damped least squares
with MuJoCo Jacobians and NumPy. The environment variable adjustment avoids
Python 3.12 packages injected into this project's Python 3.11 interpreter.

Validation:

```bash
env -u PYTHONPATH .venv/bin/python -m pytest \
  tests/unit_tests/test_morphology_dynamics_audit.py \
  tests/unit_tests/test_leg_morphology.py \
  tests/unit_tests/test_go2_standing.py \
  tests/unit_tests/test_go2_kickup.py -q
```

Result: **43 passed**, plus Ruff checks on the added files. Existing URDFs,
controllers, models, checkpoints and earlier results were not changed.

## Why the rear calves hit the floor

The replay retains the old controller's 10 ms target update period and samples
contacts at the 2 ms physics period. `legacy_contact_diagnosis.csv` records all
five morphologies and four old command settings. `first_contacts.json` records
the first penetrating nonfoot contact, geometry ID, world contact position,
root position, joint positions, body pitch and force. Contact time is resolved
to one physics step, not a continuous-time impact localization.

| Scale | Rear-calf reset clearance (mm) | First contact, full-amplitude crouch (s) | First contact, 60% amplitude (s) |
| --- | ---: | ---: | ---: |
| 1.00 | 36.21 | .462 | .802–.808 |
| .95 | 35.32 | .460 | .824–.842 |
| .90 | 34.42 | .460 | .856–.886 |
| .85 | 33.52 | .464 | .902–.954 |
| .80 | 32.65 | .468 | .982–1.080 |

There is no nonfoot penetration at reset. Crouching runs from .20 to .45 s;
full-amplitude contact first appears immediately after the push command begins.
However, the body has already tipped **rearward** by ten degrees during the
crouch: at .366/.368/.370/.374/.378 s for scales 1/.95/.90/.85/.80. Reduced-amplitude
commands tip rearward later and hit a rear calf during the post-push rotation
phase. Thus the contact is the consequence of rear-only folding and lost body
attitude, not proof that the reset or the shortened model is geometrically
infeasible. The commanded phase and actual loss-of-balance timing are distinct.

For example, at scale 1 and full amplitude, the first `RL_calf` contact is at
.462 s with body pitch -.45727 rad (-26.20 degrees), root position approximately
[-.09369, .000676, .26873] m, rear-left thigh 1.11233 rad and calf -2.27282 rad.
The actual joint values are within their constraints. The calf is almost
horizontal in world coordinates after the rearward rotation. The original
reference leaves the front legs standing while shortening the rear stance,
which creates the wrong initial rotation for a nose-down kick-up.

The audit checks lower support points of the actual compiled collision shapes:
analytic support functions for boxes/cylinders/spheres and transformed vertices
for generated convex meshes. It includes the lower-calf fixed collision links,
not just the foot endpoints. Assets retain all collision settings and use the
previously validated axial geometry generator. There are no collision masks,
ground exceptions, extra support forces or changed actuator gains.

## Feasible poses and matched conditions

Foot sphere centers are IK targets at world positions
FL=(.1917,.142,.022), FR=(.1917,-.142,.022),
RL=(-.1951,.142,.022), RR=(-.1951,-.142,.022) m. The solver uses morphology-specific
joint positions, preserves root attitude for stance IK, clamps to native joint
ranges and checks forward-kinematic residuals. Standing and crouching are solved
for **all four** feet; the handstand target pins only the front feet and iterates
root x until whole-body COM x matches the support midpoint. Rear legs in the
handstand target retain a defined posture; they are not extra supports.

Two matching families prevent equating identical joint angles with identical
foot configurations:

1. `relative_extension`: root height .34*s m standing and .27*s m crouching.
   This permits effective pendulum length to change naturally with morphology.
2. `world_geometry`: root height .30 m standing and .24 m crouching at every
   morphology, with the same four foot locations. Joint flexion compensates
   for the different link lengths.

Across both families and scales 1/.95/.90, the full 51-sample IK crouch sweeps have
<1e-7 m foot residual, at least 23.02 mm nonfoot clearance, and command rates
below .169 of the URDF joint-speed limits for a .25 s crouch. Quasistatic contact
wrench solutions have positive normal forces, feasible friction and free-base
residual below 3e-13. Required joint torques are below .155 of effort limits.
These prove that the **standing/crouching portion** need not use calf contact or
exceed joint/torque/velocity limits.

The separate front-pinned standing-to-target geometric sweep is collision-free,
but is **not quasistatically balanced** throughout: free-base wrench residuals
reach about 29 in combined force/moment residual norm. Its units mix forces and
moments, so it is a feasibility diagnostic, not a scalar physical wrench unit.
It supplies an articulated path energy profile, not a dynamically feasible
trajectory certificate or minimum-energy path. See the separately labeled
`four_foot_crouch` and `front_pinned_rotation` rows in `ik_path_summary.csv`.

## Independent takeoff and balance evaluations

Takeoff uses the same four commanded push settings (.12/.24 s extension,
front-knee targets -1.8/-2.2 rad), three paired initial pitch-velocity seeds
(uniform ±.02 rad/s), and both bounded open-loop PD and pitch-feedback rollouts.
The IK crouch is coordinated across front and rear legs. Front flexion and rear
extension then produce the intended nose-down rotation. Audit commands are evaluated at the 2 ms physics period and slew-limited
to .7 times the unchanged native speed limits. A future PPO comparison must
retain a common action period for both training conditions. Gains remain 60/3. Pitch feedback
is a deliberately simple bounded front-thigh correction; no learned controller
or controller handoff is used.

A **supported takeoff** requires at least 20 contiguous milliseconds with both
rear feet >15 mm above the ground, rear load below 4% body weight, both front
feet loaded above 5% body weight, positive body pitch, no nonfoot contact, no
speed-limit exceedance and joint violations below .001 rad. This is explicitly
not a complete standing-to-handstand success criterion. Maximum rotation is an
observed controller-specific quantity, not maximum achievable rotation.

All rollouts stop at the first penetrating nonfoot contact. The rear normal
impulse is integrated only in the commanded push window. Rear angular impulse
and total external angular impulse are integrated about the fixed initial front
support midpoint; the latter includes gravity and support forces. The first
illegal-contact sample is excluded from impulse accumulation. Positive actuator
work integrates the positive part of torque times joint velocity; signed net
work is logged separately. Neither is equal to gravitational potential rise.
Impulses use 2 ms force samples and therefore retain numerical quadrature error.

Of 144 takeoff attempts, **102 achieved supported takeoff**. Every attempt
avoided rear-calf penetration, and each eventually stopped on `FL_calf` contact.
The front-leg support/rotation controller is now the limiting component. Across
this takeoff suite, maximum actual speed was .6501 of the URDF limit, maximum
applied torque .6549 of effort limit, joint-limit violation zero, and torque
saturation fraction zero. No self contacts were observed. Therefore these are
valid partial takeoffs, not maneuvers made possible by ignoring contacts or
exceeding actuator limits.

For the `relative_extension` family with pitch feedback, values below are means
of per-attempt maxima/impulses over the four command settings and three seeds:

| Scale | Supported takeoffs | Mean peak pitch (deg) | Mean rear clearance (m) | Mean supported lift (s) | Push impulse (Ns) | Rear angular impulse (N m s) | Positive work (J) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 12/12 | 34.46 | .1636 | .1760 | 28.22 | 25.73 | 42.69 |
| .95 | 9/12 | 30.21 | .1179 | .1397 | 26.25 | 24.42 | 37.38 |
| .90 | 9/12 | 26.39 | .0781 | .1115 | 24.35 | 23.82 | 32.66 |

The same ordering appears in the open-loop and world-geometry comparisons.
For world geometry with feedback, mean peak pitch is 35.87/31.27/26.60 degrees
at scales 1/.95/.90. Consequently, the negative finding does not rest on the old
rear-calf failure or on the inability to hold a five-second handstand. The
controllers do not exhaust torque authority, so this also does not demonstrate
that shorter legs have a lower optimal performance ceiling.

Balance is tested independently from COM-aligned front-foot IK poses at
80/85/90 degrees and angular velocities -.3/0/.3 rad/s, under nominal joint hold
and gravity-compensated pitch feedback. These are 54 deterministic sensitivity
tests, not 54 successful transitions. No probe holds a valid handstand for five
seconds. The best intervals with gravity/pitch feedback are 1.282/1.276/1.270 s
at scales 1/.95/.90. Joint hold alone has a slightly longer best interval when
shortened (.968/1.014/1.058 s), showing controller dependence rather than a uniform
stability advantage. The heatmaps and saved COM-speed/contact traces expose this
separately from takeoff. A stronger balance controller could change these results;
the existing saved balance policies were neither loaded nor altered.

## The variable-length pendulum hypothesis

Use the supplied no-ankle-torque model:

```
theta_ddot = -(g/L)*sin(theta) - 2*(L_dot/L)*theta_dot
```

Here theta is the COM angle from the **downward** vertical. With an angle from
upright, the gravity sign reverses. It is not the robot's body-pitch angle.
For sagittal COM displacement dx from a fixed front pivot and height z,
`theta=atan2(-dx,-z)` and `L=sqrt(dx²+z²)`. No ankle torque is added.

Four quantities must remain distinct:

- Physical thigh/calf segment length: .213*s m each.
- Effective COM-to-front-support length: posture-dependent measured L, including
  the unchanged trunk and hip dimensions.
- Frozen-posture gravitational barrier to COM-over-support:
  `m*g*(L-z) = m*g*L*(1+cos(theta_initial))`.
- Articulated target/path potential energy, actuator work and torque: measured
  from the full model, not inferred from that scalar barrier.

At a **fixed initial angle** and mass, reducing L reduces the frozen barrier.
It does not do so automatically when a morphology change also lowers the initial
COM and changes its angle. In the relative-extension family:

| Scale | Effective L (m) | Pitch inertia at COM (kg m²) | COM z (m) | Frozen pendulum prediction (J) | Articulated IK path barrier (J) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.00 | .37002 | .53322 | .31615 | 8.600 | 23.436 |
| .95 | .35657 | .51052 | .30087 | 8.746 | 23.425 |
| .90 | .34325 | .48898 | .28553 | 8.909 | 23.406 |

Thus shortening decreases L and inertia but **increases the frozen barrier**
under this particular matching convention; the articulated barrier barely
changes. The unscaled trunk still has to be raised over the front support.
Per-unit-mass barrier data are also saved so that the modest mass reduction from
constant-density link shortening is not mistaken for a pure kinematic benefit.

For matched world geometry the articulated barrier does decrease:
29.434 → 26.823 → 24.279 J at scales 1 → .95 → .90. Yet takeoff pitch, impulse and
clearance still worsen. Lower energy demand alone therefore does not predict
better kick-up dynamics here. The shorter stroke and different geometry also
change how the same bounded joint commands deliver work and angular impulse.

The term `-2*(L_dot/L)*theta_dot` can amplify an existing angular velocity during
shortening. A smaller **static** morphology has L_dot=0; it does not itself pump
angular momentum or supply energy. Articulated motion during the push is needed.
The full robot can also store angular momentum in segment rotation, which the
point-mass equation omits.

`pendulum_validation.csv` compares differentiated COM motion with the supplied
equation only on front-loaded, rear-unloaded, noncolliding, low-drift intervals.
Samples within two physics ticks of interval boundaries are excluded from RMS
residuals. The report also integrates the equation conditionally using the
**measured** L(t), with state initialization at each eligible interval. This is
not a prediction from morphology alone or proof of equation validity for Go2.
The brief takeoff intervals yield mean conditional angle errors around
.007–.030 rad for pitch-feedback cases, with acceleration residual RMS around
.90–1.67 rad/s². Finite contact patches, residual pivot motion, multibody inertia,
shape momentum and numerical differentiation limit the comparison. Rear push-off
is excluded because its external moment is absent from the torque-free model.

## Full-size posture ablation and curriculum choice

Holding physical geometry at 100%, standing/crouching IK shifts the torso
forward by 0/.03/.06 m while retaining the same foot targets. All three paths
have valid contacts and actuator requirements. The .06 m configuration has
minimum crouch clearance .0307 m, quasistatic torque ratio <.162 and positive
normal forces. It is an attainable four-foot posture, not a near-handstand reset.

| Forward shift (m) | Effective L (m) | Frozen barrier (J) | Articulated target delta U (J) | Mean peak pitch (deg) | Supported lift (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| .00 | .37002 | 8.600 | 23.436 | 34.46 | .176 |
| .03 | .35702 | 6.385 | 23.297 | 33.03 | .171 |
| .06 | .34568 | 4.448 | 23.170 | 33.09 | .222 |

The .06 m posture reduces the frozen barrier by about 48% and extends supported
lift by about 26%, but reduces peak pitch and clearance slightly. The final
articulated target energy changes only about 1%. This is a **partial** benefit,
not proof of easier full transitions. It motivates testing posture as an
initialization curriculum (C) against standard full-size PPO before investing in
a physical-link curriculum. It does not justify claiming actor–critic learning
improvement without actual paired training.

Any next short PPO pilot should retain one actor–critic across posture stages,
keep posture fixed per episode, retain optimizer state, and use a joint takeoff
metric (positive pitch, rear lift and front support) for early progression.
Later progression must require transition/balance performance. Evaluate at the
same normal standing posture, not merely at full geometry; otherwise the posture
curriculum receives an easier evaluation reset. Start with a small fixed budget
and inspect return, support, saturation and optimization diagnostics before any
extended A–D run.

## Figure and statistical limits

All plotted trajectories and geometry values come from saved MuJoCo states;
analytical predictions use dashed lines and explicit labels. Energy measured
along IK configurations is labeled as an articulated **geometric path**, not a
successful dynamic trajectory. There is no successful complete handstand to
plot: the representative figure explicitly compares successful *supported
takeoffs* and failed takeoffs, all eventually unsuccessful as full transitions.

The command grid was specified before the final three-seed audit. The additional
posture test is a post-hoc exploratory ablation and needs independent validation.
Bands show standard deviation across three paired initial-condition seeds after
averaging the common command grid. They do not quantify training-seed variability,
policy-learning uncertainty, or optimal feasible-state coverage. Only sagittal perturbations were tested; lateral robustness remains unmeasured.
No claim about an expanded viability kernel, real hardware, or maximum achievable rotation follows
from these probes. Collision polygonization and constant-density mass changes
from the original morphology generator remain documented modeling confounds.
