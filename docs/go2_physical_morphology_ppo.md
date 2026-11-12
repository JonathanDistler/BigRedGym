# Physical leg-length actor–critic PPO: first matched learning diagnostic

## Outcome

Actual CPG-free PPO training was run on `ppo_kickup_e2e`. This experiment keeps
**physical leg-length** as the curriculum variable. No posture curriculum,
scripted kick reference, CPG, pretrained balance actor, or controller handoff was
used. Original models, checkpoints, controllers and earlier experiment folders
remain unchanged.

The measured diagnostic completed **98,304 training control steps**: methods
A/fixed 100% and B/90%-start, each using seeds 42 and 43, each with 24 iterations
of 1,024 transitions. Physics runs at 500 Hz and policy actions at 50 Hz, giving
983,040 training physics steps. Additional evaluation interactions are logged
separately (145,745 control steps total) and never used in PPO gradients.

Both policies learned actions and supported forward lean. At full geometry the
90%-trained policy produced more forward pitch on average, with greater variation
between seeds. Neither deterministic policy lifted the rear feet meaningfully.
No policy achieved a five-second handstand or the near-vertical recovery metric.
The curriculum stayed at **90% for both seeds**. No stages were skipped and no
unearned 95%/100% training curves are shown.

The predeclared continuation gate failed for both seeds, so the extended budget
was not spent. This is a completed **learning diagnostic with a stalled
curriculum**, not a completed 90→95→100 training experiment or evidence against
the morphology hypothesis. In particular, the apparent improvement in lean is
not a demonstrated improvement in standing-to-handstand success.

## Measured full-size evaluation

Each saved policy is evaluated deterministically on the original-size geometry,
with the same eight held-out seeds 1000–1007 and the same standing family.
The final comparison averages trials within each training seed, then averages
the two training seeds. Values are mean ± sample standard deviation across
training seeds:

| Method | Mean maximum supported body pitch | Rear clearance maximum | Supported takeoff | Five-second handstand | Episode reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: fixed 100% | 14.30 ± .14 deg | .985 ± .028 mm | 0/16 | 0/16 | 3.580 ± .052 |
| B: trained at 90%, curriculum stalled | 18.84 ± 5.04 deg | .897 ± .009 mm | 0/16 | 0/16 | 4.526 ± .844 |

The approximately 1 mm maximum rear clearance is the reset clearance, not
rear-foot takeoff. All these final deterministic evaluations survived eight
seconds without nonfoot collisions. That is safe leaning, not handstanding.
The separate .3 rad/s pitch-velocity perturbation at two seconds was applied
in every final recovery trial; it also produced zero handstand successes.

Across stochastic training episodes there were 17 supported takeoffs in A
(9 and 8 by seed) and 16 in B (4 and 12). Episode counts differ because episodes
terminate at different times, but the interaction budgets are identical.
The takeoffs are genuine intermediate events under the specified contact/limit
checks; they did not consolidate into deterministic rear-foot lift. No training
episode met the near-vertical recovery condition. Per-episode results are saved,
including collisions and unsuccessful takeoffs.

Post-update approximate KL remained below .020 in all four runs. Mean PPO
clipping fractions were approximately .046/.059 for A and .034/.052 for B.
The actor weights changed in all runs; mean action magnitude increased and
reward rose. This rules out a disabled actor or a no-op optimizer, while not
proving that the reward learned the desired maneuver.

## Files and exact reproduction

Canonical run: `output/physical_morphology_ppo_90_v1`.

- Each `A_seed*`/`B_seed*` directory contains source snapshots, configuration,
  actor/critic/Adam/RNG/simulator checkpoints, training and episode CSVs,
  full-scale evaluation CSVs and traces, curriculum windows/events and outcome.
- `diagnostic_decisions.json` preserves the measured stop decisions.
- `software_versions.json` identifies the Python, Torch, MuJoCo and NumPy versions.
- `figures` contains ten measured-data PNG/PDF comparisons and underlying CSVs.
  The reporting script performs no simulation or training.
- `experiments/morphology/learning/protocol_80.json` is the separate prepared
  five-stage ablation. It has **not** been run.

From the repository root, run into a **new** output folder:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_physical_morphology_ppo.py \
  --output output/physical_morphology_ppo_90_repro

env -u PYTHONPATH MPLCONFIGDIR=/tmp/morphology-mpl \
  XDG_CACHE_HOME=/tmp/morphology-cache \
  .venv/bin/python scripts/report_physical_morphology_ppo.py \
  --input output/physical_morphology_ppo_90_repro \
  --output output/physical_morphology_ppo_90_repro/figures
```

The first command includes the predeclared conditional continuation policy;
it does not extend training merely because reward increases. Use
`--diagnostic-only` to impose the diagnostic budget regardless of the gate, or
`--smoke` for a separate 64-step-per-method software check. A normal reproduction
of the measured v1 result stops at the diagnostic gate. The runner refuses an
existing output root, preserving earlier results.

To regenerate canonical figures without training:

```bash
env -u PYTHONPATH MPLCONFIGDIR=/tmp/morphology-mpl \
  XDG_CACHE_HOME=/tmp/morphology-cache \
  .venv/bin/python scripts/report_physical_morphology_ppo.py \
  --input output/physical_morphology_ppo_90_v1 \
  --output output/physical_morphology_ppo_90_v1/figures
```

The prepared five-stage ablation has its own entry point/configuration:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_physical_morphology_ppo.py \
  --protocol experiments/morphology/learning/protocol_80.json \
  --output output/physical_morphology_ppo_80_ablation
```

That command is supplied for reproducibility, **not recorded as an executed
experiment**. Its 90% and 95% advancement conditions match the corresponding
conditions in the 90%-start experiment. It inherits the same reward, networks,
PPO hyperparameters, seeds, training budget and full-scale evaluation rules.
The .80/.85 stages add 10/12-degree progression thresholds.

## Environment and policy

New files are separate from the earlier smoke runner and feasibility controllers:

- `experiments/morphology/learning/environment.py`: physical environments and
  fixed standing-family initialization.
- `experiments/morphology/learning/ppo.py`: actor, privileged critic, PPO/GAE and
  performance curriculum.
- `scripts/run_physical_morphology_ppo.py`: matched-budget experiment runner.
- `scripts/report_physical_morphology_ppo.py`: measured-data figures and tables.

The old `scripts/train_morphology.py` and its old A–D naming/feasibility gate are
preserved. The new runner uses the A/B meanings in this request and does not
require the old scripted-controller screening to establish learning feasibility.

The actor has 44 inputs: projected gravity, base angular velocity, joint position
offsets, joint velocities normalized by unchanged speed limits, current command
offsets, elapsed episode fraction, and physical morphology scale. It outputs 12
Gaussian latent coordinates, transformed by tanh into bounded joint position
offsets around the fixed standing configuration. There is one actor across all
scales. The critic has 74 inputs: the actor observations plus base velocity and
height, COM position/velocity, relative foot positions, loads and anchor drift.
This privileged state is not fed into the actor.

Both networks have [256,256,128] ELU hidden layers. PPO uses Adam at 1e-4,
gamma .995, GAE lambda .95, four epochs, minibatches of 256, clipping .2,
entropy coefficient .005, value coefficient .5 and gradient clipping at 1.
Approximate-KL early stopping uses target .02. Initial Gaussian log std is -.7.
The log probability includes the tanh Jacobian; the base Gaussian entropy is
used as the common exploration regularizer. Finite eight-second episodes do
not bootstrap through termination or timeout. The exact saved configuration is
the authority for every run.

The same initial network parameters are used for A and B within a seed.
Reset perturbations use per-world, per-episode seed streams, preventing one
world's reset frequency from changing another world's perturbations. Each
method keeps its own Torch RNG state. The comparison matches training control
steps, not episode counts or wall-clock duration. All saved policy checkpoints
are evaluated at 100%; B additionally receives own-stage held-out diagnostic
rollouts for the continuation decision. Evaluation interactions are accounted
for separately from the equal learning budgets.

## Physics and initialization

All five previously implemented physical morphologies are reused without
editing their generator or original URDF. Scale 1 remains the byte-identical
original URDF. Models are immutable and shared; each world has independent
MuJoCo data. Morphology is selected only when an episode resets. Other episodes
retain their old geometry when the global schedule advances.

The initial-condition family is fixed throughout training: root height .34*s,
fixed front/rear world foot targets from the audit, and morphology-specific
four-foot IK joint positions. Small matched joint and pitch-velocity
perturbations are applied, and minimum foot clearance is set to 1 mm.
This is a single relative-extension standing family, **not a posture curriculum**.
There are no near-vertical training resets or reference trajectories. At full-size
evaluation both methods use exactly the same full-size reset family and seeds.

All native joint constraints, torque/speed limits, rotor armatures, masses,
inertias and collision masks retain their established morphology semantics.
PD gains are fixed at 60/3, with an identical .25 command filter and target slew
limit .7 times the native speed bound in both methods. Actuation is integrated
at 2 ms, with positive-power torque suppressed beyond the no-load speed bound.
Invalid joint/speed states are logged and terminate episodes; contacts are not
suppressed to manufacture success. Self collisions are also terminal.

Across the completed stochastic training runs, applied torques stayed within
native effort limits, joint-limit violation was zero, and maximum measured
speed stayed below .707 of the native speed limit. Some torques saturated during
exploration; saturation fractions are logged rather than hidden. Nonfoot
contacts, including head and rear-calf contacts, remain common exploration
failures and terminate the episode.

## Common reward and success criteria

Reward version `supported_kickup_v1_identical_all_morphologies` was frozen before
the measured runs. It is a new common learning-task reward; the older feasibility
and PPO-smoke reward implementations are preserved. Every scale and both methods
use exactly the same terms and coefficients:

- `2*sin(positive_pitch)*front_support` per second;
- rear clearance / .15 m, clipped to [0,1], multiplied by positive-pitch sine,
  front support and weight 2 per second;
- front support .05 per second;
- near-vertical recovery 2 per second and strict stability 8 per second;
- a one-time supported-takeoff bonus of 1 per episode;
- potential difference `gamma*Phi_next-Phi`, where
  `Phi=2*clip(sin(pitch),0,1)*front_support`; non-timeout termination sets
  next potential to zero, while timeout retains the terminal pose potential;
- normalized torque cost .02 per second, command-change cost .02 and a .5
  failure cost.

Every component is logged separately, including the terminal costs. There is
no morphology-dependent coefficient or reward switch between stages.

Supported takeoff requires 20 contiguous milliseconds of both rear feet >15 mm
clear, rear normal load <4% body weight, both front loads >5% body weight,
positive pitch >.12 rad, front drift <.10 m and valid joint/contact/speed state.
Near-vertical recovery requires pitch within 15 degrees of upright, rear
clearance >.10 m, both front feet loaded, drift <.10 m, |pitch rate|<2 rad/s
and COM speed <.75 m/s. Strict stability narrows the angle to ten degrees,
|pitch rate|<.5 and COM speed<.25; five contiguous seconds defines success.
The measured maximum pitch is an orientation diagnostic, not proof of a valid
handstand; transient rotations past vertical can also give large maxima.

## Curriculum and full-scale budget

The 90%-start schedule is [.90,.95,1.00]. At .90 an episode qualifies if it
achieves supported takeoff, maximum supported pitch >=15 degrees, episode front
support fraction >=.60, and valid measured speed/joint limits. At .95 it must
also reach >=45 degrees and recover near vertical for at least .10 s (or achieve
strict success). Thus early promotion does not require five-second success,
while later promotion still asks for progress toward balance.

Promotion requires two consecutive nonoverlapping windows of 32 completed
current-stage episodes, each with qualifying fraction >=.30, plus at least 4096
stage interactions. Old-stage episodes do not count toward the new stage.
Policy parameters, Adam state, exploration parameters and RNG streams persist;
there is no network reconstruction on promotion.

The diagnostic has 24,576 training transitions per method/seed. Continuation is
allowed when same-stage held-out deterministic evaluations improve takeoff rate
by >=.10, or improve supported pitch by >=.05236 rad **and** rear clearance by
>=.005 m, with front support >=.50. Actual stage advancement also demonstrates
sufficient intermediate learning. Reward or survival alone cannot pass.
The final two diagnostic evaluations are compared to iteration zero.

If that gate passes, base continuation is 196,608 transitions per method/seed.
After full-scale entry the target is extended, identically for A and B, to allow
at least 65,536 **actual full-scale transitions**. A hard cap of 270,336 includes
room for old-morphology episodes to finish. The runner records actual full-scale
steps and marks the curriculum incomplete if the requirement is not met.
A stalled shortened stage is not forcibly promoted to spend the reserved budget.
No continuation was run here because neither seed passed the diagnostic gate.

## Why the curriculum stalled

The failure has several measured components:

1. The deterministic policy improves pitch and reward primarily by leaning with
   four feet down. Final rear-foot clearance never exceeds its ~1 mm reset value,
   far below the 15 mm takeoff criterion.
2. Stochastic takeoffs are infrequent and unstable: 4/90 and 12/102 completed B
   episodes, with zero near-vertical recovery. They are not consolidated into a
   deterministic maneuver.
3. The whole-episode front-support quality gate is stringent under the fixed
   exploration distribution. B's maximum episode front-support fractions are
   .528 and .518, so no episode reaches .60. All five completed stage windows
   therefore have zero qualifying fraction. Even ignoring this quality check,
   aggregate takeoff rates are below the .30 progression threshold.
4. Exploration remains broad (Gaussian std around .5), and many noisy episodes
   end in head/calf collisions or lose front-foot load. Meanwhile the dense
   pitch/support reward offers a safer four-foot-lean solution.
5. PPO is updating finite parameters with bounded KL. This is a task/reward,
   exploration and stage-qualification bottleneck in this diagnostic, not an
   observed numerical optimizer failure.

The results do **not** justify silently lowering gates, skipping to 100%, or
claiming that a physical curriculum has been fully evaluated. A next physical-only
iteration could predeclare a takeoff-local support quality statistic and a common
reward/exploration adjustment, then rerun matched A/B diagnostics from fresh
seeds or explicitly preserved checkpoints. This report does not substitute a
posture curriculum or quietly retune the measured v1 experiment.

## Validation and limitations

The six new tests cover actor/critic asymmetry, finite-horizon GAE, intermediate
promotion without five-second success, recovery gating, morphology latching,
physical actuator/collision preservation, and checkpoint replay. Replay restores
actor/critic, Adam, RNGs and MuJoCo integration/warmstart state; the next update
matches an uninterrupted update within numerical tolerances.

```bash
env -u PYTHONPATH .venv/bin/python -m pytest \
  tests/unit_tests/test_physical_morphology_ppo.py \
  tests/unit_tests/test_morphology_dynamics_audit.py \
  tests/unit_tests/test_leg_morphology.py \
  tests/unit_tests/test_go2_standing.py \
  tests/unit_tests/test_go2_kickup.py -q
```

There are only two training seeds and eight evaluation initial conditions per
checkpoint, with no stochastic-policy held-out success estimate. Figures show
measured means and one sample standard deviation across training seeds, not a
confidence claim or publication-grade statistical power. All learning results
use the new common task, rather than being pooled with previous smoke checks.
The five-stage ablation is prepared but unexecuted, and the 95%/100% curriculum
training stages remain untested. The physical-morphology research question is
still open.
