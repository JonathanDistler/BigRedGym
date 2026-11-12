# Takeoff-first physical morphology learning

## Result and scope

The reward/action audit and a bounded behavior-cloning investigation are complete
on `ppo_kickup_e2e`. The physical curriculum remains **90% → 95% → 100%**. There
is no posture curriculum, runtime reference, CPG, or controller handoff in the
new actor. No extended PPO experiment was launched.

The important findings are:

- The v1 actors earn most of their positive return from leaning forward with
  four feet grounded. This is a reward shortcut, not rear-leg kick-up learning.
- The old action interface can already produce takeoff under appropriate
  commands. Its smoothing/range nevertheless distort fast reference targets.
  Direct target commands remove this distortion while keeping physical limits.
- The new reward pays for a measured active, synchronized rear push and supported
  flight. Grounded pitch and standing support have no ongoing positive income.
- A CPG-free actor cloned offline from the preserved kick-up teacher achieves
  **4/4 genuine takeoffs at 100%, 1/4 at 95%, and 0/4 at 90%** in short closed-loop
  tests. A low supervised loss alone does not establish morphology transfer.
- A larger matched takeoff PPO comparison is implemented and gated, but remains
  unrun. These validation results do not answer whether shortening improves PPO
  discovery. They establish a better-defined task and a feasible action interface.

## Canonical artifacts

- `output/takeoff_learning_certified`: strict reward/authority validation,
  preserved-policy motor traces, and BC revalidation.
- `output/takeoff_bc_investigation_v1`: teacher provenance, native frames,
  converted BC dataset, fitted actor, train/validation loss, and original student
  evaluation. The strict certificate was subsequently rechecked in the canonical
  validation folder without changing the fitted actor.
- `output/takeoff_ppo_small_noise_smoke`: one 128-transition PPO update per condition,
  seed 42, initialized from the same BC actor. This is software validation, not
  a learning comparison or a trained curriculum.
- `output/takeoff_exploration_calibration`: frozen-actor action-noise probes;
  no optimization is performed by this script.
- `output/takeoff_learning_certified/figures`: measured PNG/PDF figures and CSV
  tables. No illustrative learning curves or nonexistent stage promotions.

Earlier `takeoff_learning_validation*` and `takeoff_ppo_smoke*` directories are
preserved development/calibration runs. They are not pooled as extended PPO
results. The first BC run finished fitting/evaluating but failed while writing
its summary because a NumPy seed ID was not JSON serializable. Its summary was
reconstructed from the saved data and actor; neither the data nor weights were
changed. The exporter now converts those IDs correctly.

## 1. What v1 learned

The authoritative saved full-scale evaluations over both training seeds show:

| v1 episode reward component | Fixed 100% | 90%-start actor at 100% |
| --- | ---: | ---: |
| Grounded pitch | 3.606 | 4.659 |
| Front support | .388 | .394 |
| Rear lift | 0 | .000022 |
| Takeoff bonus | 0 | 0 |
| Near-vertical recovery / stability | 0 | 0 |
| Potential difference | -.407 | -.520 |

The new audit replays all four final checkpoints at full geometry and each
curriculum actor at its 90% training geometry, for two reset seeds each: twelve
8-second episodes. It logs at the 2 ms physics period:

- All 12 actor actions, filtered motor targets, actual joint positions/velocities,
  pre-clamp requested torques, and applied torques.
- Four foot normal forces, rear-foot clearance, body pitch/rate, front support,
  anchor drift, and nonfoot contacts.
- Recent loaded extension, useful impulse and takeoff certification.

After two seconds the rear knees are nearly stationary: maximum rear extension
velocities in these steady portions are about .007–.071 rad/s for A and
.014–.035 rad/s for B, depending on side/geometry. Rear normal forces remain
roughly 15–30 N per foot on average. Clearance is a small contact penetration,
not flight. The policies apply nonzero holding torque but do not exhaust motor
authority. Initial transients perform some positive knee work; that is not a
coordinated push-off. All twelve replayed episodes fail the genuine-takeoff test.

The old whole-episode front-support criterion is also unsuitable for qualifying
short kicks under noisy exploration. The new curriculum uses a certified local
flight interval, rather than demanding 60% support across an entire episode.

## 2. New takeoff certificate and common reward

`experiments/morphology/learning/takeoff.py` adds a separate environment and
tracker. The prior environment/reward/checkpoints are unchanged. The actor still
has 44 proprioceptive/context inputs and outputs 12 joint targets; the asymmetric
critic retains its 74-input privileged state. The tracker is used for rewards and
metrics, not as an actor reference or an extra controller.

A takeoff must have the following evidence in the same recent push:

1. Both rear feet have carried at least 8% of body weight individually.
2. Within a rolling .4-second window, **each** rear knee actually extends at
   least .03 rad, each rear leg supplies at least .05 J of positive actuator
   work while loaded, and each accumulates useful normalized normal impulse
   >=.002 s. Useful load subtracts .15 body weight per foot, caps at 1.5 body
   weights, and is gated by positive knee extension velocity.
3. Each rear foot unloads below 2% body weight, clears 8 mm while moving upward,
   and the two release times differ by at most 40 ms.
4. For at least 20 contiguous milliseconds, both rear feet clear 15 mm, both
   remain unloaded, both front feet carry >5% body weight, positive body pitch
   exceeds .12 rad, front drift stays below .10 m, and no collision or physical
   state-limit violation occurs.

Once a flight is certified, its support/clearance must remain valid to credit
additional rotation. An invalid prefix cannot later earn takeoff by bouncing off
an illegal contact. Pitch credited to the flight is sampled during genuine
supported flight, not from an unrelated later roll. Five-second handstand
success is not a prerequisite for learning this partial maneuver.

Every morphology uses the same reward:

- Bounded preparation credit, maximum .5 per episode, for useful loaded rear
  extension, additionally requiring actual positive rear-leg work.
- One-time genuine simultaneous-liftoff bonus of 3.
- One-use additional credit up to 1 for increasing supported airborne rotation.
- Supported-flight reward density, proportional to clearance and useful pitch,
  only during a certified active push-off flight.
- Common torque, command-change and failure costs.

There is **no** grounded pitch, standing support, or old terminal-potential
income. Preparation is bounded, so repeated squat cycles cannot pay indefinitely.
Real old actors can earn a small preparation credit for their initial loaded
extension, but cannot turn sustained four-foot lean into repeated reward. Genuine
reference takeoffs have substantially greater return. The scoring comparison
labels the different probe horizons and is a reward audit, not a learning curve.

Unit-level negative controls reject standing, lean, a single lifted rear foot,
folding without pushing, flight without front support, passive rear stretching,
asynchronous release, and stale push history. Actual trained-actor replay also
produces zero genuine takeoff events. Synthetic unit controls are not plotted as
measured experimental trajectories.

The takeoff-phase goal is 35 degrees of genuine supported rotation plus at least
40 ms of flight. It is the same at every morphology. It is a **partial-maneuver
endpoint**, not a handstand success claim. Takeoff training can terminate on this
goal; separate evaluations can disable that termination to inspect later motion.
Do not interpret the inherited handstand fields from short, goal-terminated
rollouts as an assessment of five-second balance.

## 3. Command authority

The following interfaces were tested with the same bounded reference/feedback
probe grid, two reset seeds, scales 1/.95/.90, extension durations .12/.24 s and
front-knee targets -1.8/-2.2 rad. These reference commands are validation probes
only; the learned actor has no runtime reference.

| Interface | Residual target range | Command filter | Maximum target error across probes |
| --- | ---: | ---: | ---: |
| Legacy | ±1.2 rad | .25 | .755 rad |
| Wider filtered | ±1.6 rad | .50 | .093 rad |
| Direct targets | ±2.0 rad | 1.00 | numerical roundoff |

The filter is inverted when checking representability. Projection and target
tracking errors are logged rather than hiding infeasible labels by clipping.
For these probes the direct interface reproduces desired targets to <2e-16 rad,
with no action projection. All native joint bounds still apply; commanded slew
remains limited to .7 times the unchanged URDF no-load speed. Motor gains remain
60/3. Effort limits, velocity envelope, rotor inertia, collision masks, leg
geometry and ground contacts are not modified.

The direct interface produces strict active takeoff in **6/8 full-scale probes,
4/8 at 95%, and 4/8 at 90%**. Certified minimum positive loaded work per rear leg
is approximately 9.4–12.2 J for successful probes, well above the .05 J check.
Maximum genuine flight pitch is approximately 44.4/37.6/31.8 degrees at
100/95/90%. The 20/35-degree curriculum boundaries have feasible witnesses.

The legacy interface produces the same takeoff counts on this small grid,
though smaller rotation/clearance and greater command distortion. Consequently,
**the old interface is not fundamentally incapable of takeoff**, and widening it
alone is not evidence of a PPO solution. The direct interface is selected to
make action authority explicit and demonstration conversion more faithful.
Both future A and B use it identically.

## 4. Behavior-cloning investigation

The preserved expert is the historical `go2frontyield` checkpoint named in
`scripts/reproduce_transition_hold.py`, with the preserved candidate kick
reference. Its saved contact-CPG setting is **disabled**; it is a reference-guided
kick-up controller. It is used strictly as an offline data source. No balance
policy or handoff controller is loaded into the new student.

The existing recovery dataset begins around 70 degrees and cannot by itself teach
the standing push-off. The exporter therefore replays the historical standing
kick prefix for source seeds 3000–3011, captures native states and actual applied
torques, and verifies the source checkpoint's hash before/after. All twelve
source episodes produce supported rear lift before any later invalid contact.
This is evidence of kick-up prefixes, not twelve complete five-second handstands.

Collision-free prefixes are retained; later colliding frames are excluded.
Labels are torque-equivalent position targets under the new fixed PD controller:

```
q_target_equivalent = q_actual + (tau_teacher + kd*qdot_actual)/kp
```

The command filter/slew/range map is then inverted and checked. Frames with
> .5 Nm representability error or invalid joint state are rejected. This avoids
copying teacher gains, velocity feedforward, reference residuals or hidden
controller memory into the actor contract. Student actor inputs contain only
its normal schema, including previous student target and elapsed time. Native
CPG/teacher phase and future trajectory states are never actor inputs.

The resulting dataset has **510 training frames and 250 validation frames**.
Train/validation splitting is by complete source episode: 3000–3007 train,
3008–3011 validation. The actor alone receives 1000 bounded offline gradient
updates; the critic and exploration parameters remain unchanged. Held-out bounded
action MSE falls to about **2.07e-5**. Source-model mass/friction randomization and
the new reset family can still cause a distribution gap; exact supervised fit
is not exact dynamics replay.

Fresh student-only closed-loop tests use seeds 2000–2003 and a two-second horizon:

| Evaluation morphology | Genuine active takeoffs |
| --- | ---: |
| 100% | 4/4 |
| 95% | 1/4 |
| 90% | 0/4 |

These counts were revalidated with the stricter positive-work certificate.
The historical teacher is not invoked during student rollout. This is promising
initialization for full-size takeoff, but **not a demonstrated shortened-leg
initialization**. The dataset is full-size only; treating it as balanced evidence
for a shortened curriculum would hide a real domain gap.

The primary prepared experiment starts A and B from matched random weights.
A separate optional BC-initialized A/B ablation loads exactly the same actor and
hashed dataset in both conditions, leaving critics seeded identically. Offline
updates and environment interactions are accounted for separately. Do not compare
a BC-initialized curriculum only against an uninitialized fixed-size baseline.

## 5. PPO validation, exploration and prepared larger comparison

Only software-scale PPO updates were run: 128 transitions per condition in each
named smoke check. They exercise actual takeoff rewards, asymmetric value input,
GAE, optimizer updates, logging and full/current-morphology evaluation. They do
not establish learning rates or a successful trained curriculum.

The final small-noise BC-initialized smoke has post-update KL **.00684/.00757**
and clipping fraction **.0313/.0234** for A/B. The target KL .01 is a **soft
minibatch stopping criterion**, not a hard post-update bound. Earlier smoke
checks reached KL .030/.034 at learning rate 1e-4, and .0111/.0152 at 5e-5;
those calibration artifacts are preserved. The prepared larger protocol now uses
1e-5 and latent std .05 for both conditions. The larger 1024-sample rollout
remains untested by these 128-sample software checks.

In that final smoke there is only one completed training episode per condition:
A has no certified takeoff, while B at 90% has a certified takeoff, about 38.2
degrees of genuine flight rotation and .125 m clearance. This is a sample from
the BC-initialized stochastic policy, collected before its one PPO update. It
is **not** evidence that PPO learned a new behavior or that B outperforms A.
After the update the deterministic B mean still does not take off at 90%.
Both arms retain a full-size deterministic takeoff. Stage promotion remains
unearned in this software-scale run.

Independent high-frequency target noise can destroy front support even when the
mean BC action sequence works. Frozen-policy exploration probes examine this
without optimizing the policy; their results are saved separately. Exploration
settings are common within each A/B comparison, and changes from the v1 task must
not be attributed solely to morphology.

The exploratory four-seed frozen-actor sample gave 3/4 takeoffs at 90% versus
1/4 at 100% with latent std .05; larger stds mostly broke support. Fresh seeds
2010–2017 did **not** confirm a shortened-leg advantage: both 90% and 100% gave
2/8, and 95% gave 0/8. Front support averaged about .55–.60 with std .05, compared
with about .20 at std .223 in the exploratory sample. The smallest tested
nonzero noise that retained support was selected as a common calibration
setting, not as a morphology-specific advantage. These are frozen-policy noise
probes, not learning curves; they keep the physical-learning hypothesis open
without presenting the initial favorable fluctuation as a result.

The prepared protocol is
`experiments/morphology/learning/protocol_takeoff.json`:

- A fixed 100% versus B performance-driven [.90,.95,1.00].
- One 12-target actor and privileged critic, common reward/network/hyperparameters.
- Four training seeds 42–45 and matched periodic evaluation seeds 1000–1015.
  Fresh final test seeds 20000–20031 are saved separately from development curves.
- Base budget 1,048,576 transitions per condition/seed; both conditions receive
  the same extension if needed to retain 262,144 **actual full-scale** curriculum
  transitions after promotion. Hard cap 1,318,912 leaves room for old episodes
  and rollout rounding. Evaluation steps are separate.
- Two nonoverlapping 32-episode windows with >=30% qualified takeoffs are required
  for promotion, after at least 8192 stage steps. At 90% require >=20 degrees of
  genuine flight; at 95% require >=35 degrees and >=40 ms of genuine flight.
- Support quality is certified locally during takeoff. There is no whole-episode
  .60 front-support gate and no mandatory five-second balance test in this phase.
- Morphology changes only at episode reset. Actor, critic, Adam, RNG and simulator
  state persist across stages. No forced promotion on timeout or stalled progress.

This larger study is **prepared, not executed**. The primary question remains
whether physical shortening changes discovery under the same takeoff-oriented
learning setup. Neither the reference witnesses nor BC transfer rates answer
that learning question.

## Reproduction commands

Use new output directories. From the repository root:

```bash
# Offline demonstration generation and bounded actor-only BC investigation.
env -u PYTHONPATH .venv/bin/python scripts/prepare_takeoff_bc.py \
  --output output/takeoff_bc_repro --updates 1000

# Strict reward, motor and action-authority validation, including BC revalidation.
env -u PYTHONPATH .venv/bin/python scripts/validate_takeoff_learning.py \
  --output output/takeoff_validation_repro \
  --bc-actor output/takeoff_bc_repro/bc_actor.pt

# Software-scale PPO only: one matched update, no extended training.
env -u PYTHONPATH .venv/bin/python scripts/run_takeoff_ppo.py \
  --output output/takeoff_smoke_repro \
  --validation output/takeoff_validation_repro/validation.json \
  --bc-actor output/takeoff_bc_repro/bc_actor.pt --smoke

# Regenerate measured-data figures and CSVs.
env -u PYTHONPATH MPLCONFIGDIR=/tmp/morphology-mpl \
  XDG_CACHE_HOME=/tmp/morphology-cache \
  .venv/bin/python scripts/report_takeoff_learning.py \
  --validation output/takeoff_validation_repro --bc output/takeoff_bc_repro \
  --ppo output/takeoff_smoke_repro --output output/takeoff_validation_repro/figures
```

The canonical report, including the exploration probes, regenerates with:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/morphology-mpl \
  XDG_CACHE_HOME=/tmp/morphology-cache \
  .venv/bin/python scripts/report_takeoff_learning.py \
  --validation output/takeoff_learning_certified \
  --bc output/takeoff_bc_investigation_v1 \
  --ppo output/takeoff_ppo_small_noise_smoke \
  --exploration output/takeoff_exploration_calibration \
  --fresh-exploration output/takeoff_exploration_fresh_seeds \
  --output output/takeoff_learning_certified/figures
```

Prepared larger launch, **not executed here**:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_takeoff_ppo.py \
  --output output/takeoff_large_random_90 \
  --protocol experiments/morphology/learning/protocol_takeoff.json \
  --validation output/takeoff_learning_certified/validation.json \
  --execute-extended
```

The larger launch requires the explicit `--execute-extended` flag, the prepared
protocol, and a successful validation file whose semantic reward-code hash still
matches. It also checks the validated command/observation contract. It is not the
default and was not invoked in this work. Omit `--bc-actor` for the primary random
initialization experiment; use it identically in both arms only for the separate
BC ablation. Earlier v1 checkpoints and dataset files are never overwritten.

Unit tests cover powered synchronized liftoff, negative controls, stale history,
release asymmetry, filter inversion, double-count prevention, local support gates,
stale validation rejection and checkpoint replay including tracker state. The
full validation command includes existing morphology/standing/kick-up tests.
Two reset seeds per reference condition and four BC test seeds are diagnostic,
not sufficient statistical power for a robotics-paper learning-advantage claim.

The combined targeted regression run passed **57 tests**; Ruff checks passed on
all added/changed takeoff files. Original model/checkpoint files were preserved.
