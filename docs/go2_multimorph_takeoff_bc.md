# Shared morphology-adapted kick-up BC and anchored PPO

## Scope and experimental control

This continues `ppo_kickup_e2e` without changing the actor architecture, rewards,
physical morphology models, old datasets/checkpoints, CPG controllers or prior
results. The actor remains a 44-input MLP (256/256/128), with 12 bounded joint
target outputs. Its final observation coordinate is physical scale. No CPG,
reference phase, contact state or teacher controller is an actor input/controller
at inference. The privileged critic and actuator interface are unchanged.

The focus is still whether physical leg-length progression improves learning.
The new fixed and curriculum arms both start from the exact same shared BC actor,
seeded critic and optimizer, and both receive the exact same offline BC anchor.
Neither receives stronger initialization or extra online full-scale interactions.
The new initialization, exploration setting and regularizer together differ from
the old retention study; this is not an ablation isolating any one repair.

## Offline teacher and demonstrations

`prepare_multimorph_takeoff_bc.py` reads the preserved successful
`output/handoff_reference_search/candidate_reference.json`, hashes it, and uses
its settle/thrust/tuck parameters to seed an offline feedback-reference family.
It does not invoke a balance policy or modify the historical controller. The
standing and crouch poses are solved by IK separately on each physical model,
with unchanged front-foot support locations, body dimensions and sphere radii.
Crouch body height is .27 times scale; standing height is .34 times scale.
The body/COM/mass/contact and actuator dynamics are evaluated in each actual
MuJoCo model, rather than transferring joint angles between models.

Thrust duration/front-knee/rear-hip candidates are searched independently per
scale. The first reference-seeded candidate passed at all three geometries:
.1881007 s extension, front knee -2.2, rear thigh .6118921 and rear knee -.9854441
rad. The chosen high-level parameters are the same, but the IK-derived joint
trajectories differ with scale. Front-thigh pitch feedback adjusts the commands
against actual motion. Direct targets remain within native joint bounds and the
common .7-native-speed command slew limit, with fixed 60/3 PD gains and native
motor effort/velocity limits. No gain or torque authority was increased.

Search resets 50000–50003 are separate from certification resets 51000–51015.
Each morphology passed **16/16** held-out teacher-prefix tests. A prefix requires
the existing strict active-push certificate, positive rear actuator work,
synchronized release, valid front support, >=20 degrees of supported rotation,
>=40 ms certified flight, no nonfoot/self contact and no motor/joint state limit
violation. The intended capture endpoint is 25 degrees and 60 ms of certified
flight. Later falls are excluded from demonstrations, not relabeled as success.
These are kick prefixes, not complete handstands.

The initial combined dataset and results live in
`output/multimorph_bc_20261111_v1`. A first bounded fit completed 1,250 updates;
closed-loop selection chose update 250 over lower-MSE but worse actors. On fresh
resets 53000–53015 it gave deterministic 12/16, 16/16, 16/16 takeoffs at
90/95/100%, but stochastic 4/16, 0/16, 0/16 at latent std .05. It failed the BC
launch gate and was not used for PPO.

A refinement collected another 16/16 accepted prefixes per morphology while
injecting bounded-action Gaussian noise with std .01 into the offline teacher's
actual commands. Labels are the noise-free feedback-teacher actions at those
actual perturbed states and target histories. This is offline state-distribution
coverage, not a runtime teacher or online controller handoff. The same episode
split is maintained: 51000–51011 train, 51012–51015 validation across all scales
and perturbations. Rejected trajectories cannot enter BC.

`output/multimorph_bc_20261111_v2` combines these frames with the original
adapted prefixes and fits the same actor for another 1,750 updates, stopping at
the closed-loop/MSE criterion. Total newly attempted fitting is 3,000 updates;
the selected actor lineage uses 250 + 1,750 updates, in addition to the preserved
old full-scale actor's original 1,000-update warm start. No critic or exploration
parameters were fitted by BC. Model selection uses resets 56000–56003 and fresh
final evaluation uses 57000–57015. The selected actor is `bc_actor.pt`.

| Geometry | Deterministic certified takeoff | Sampled, latent std .01 |
| --- | ---: | ---: |
| 90% | 16/16 | 15/16 |
| 95% | 16/16 | 16/16 |
| 100% | 16/16 | 16/16 |

The gate requires >=14/16 deterministic and >=12/16 sampled takeoffs at every
geometry. Its SHA256 binds the actor and recorded exploration std to the short
PPO launch. The v2 summary writer initially failed after saving weights and all
rollouts; `bc_gate.json` was reconstructed from those unchanged actual files.
The exporter bug is fixed. No weights or measurements were altered.

`bc_dataset.npz` saves observation, action label, actual pre-command qpos/qvel,
previous target, normal contact forces, COM, time, scale, source episode seed,
post-step actual commanded target and applied torque. For perturbed data,
`actions` means the noiseless teacher label; `commanded_target` records the
actual executed target. v1 also saves 2-ms teacher physics traces. Metadata
records morphology hashes/mass, dimensions and observation/action schema.

## Why the old PPO run regressed

`output/takeoff_regression_analysis_20261111_v2` analyzes every old checkpoint
on identical fixed demonstration states and replays updates 0/4/8 with paired
noise and reset seed 54000. This is mechanism evidence, not a causal proof that
one modification alone fixes learning. Its one-reset replays are not success-rate
estimates; use the original eight-reset retention experiment for rates.

- Old latent exploration stayed essentially .05 at every joint. The old hard
  lower std clamp exp(-3) prevents meaningful reduction below .0498. With action
  scale 2, noise near the linear part of tanh can perturb targets by about .1 rad
  independently every 20 ms. Holding PD gains fixed, that can request roughly
  6 Nm of noisy correction before physical clipping.
- Fixed-state joint-averaged bounded mean RMS drift after eight updates was
  .0058–.0080. Maximum individual-joint latent drift was .0093–.0149. Small
  action changes can alter the rear-release/front-load sequence in a contact
  system. Average per-update on-policy KL is not a bound on that sequence.
- Cumulative 12-joint Gaussian KL on the fixed demo states reached about
  .099–.187, while the logged per-update KL remained moderate. The state
  distribution on which KL is measured also changes with the policy.
- In the old full-scale initial replay's .4–.8 s push interval, mean rear-knee
  absolute target error was about .048/.049 rad for deterministic control versus
  .129/.116 sampled. Across these replays, mean front support was about .99 for
  deterministic versus .51–.59 sampled. Motor limits were respected; sampled
  actions repeatedly reached the torque envelope, although overall saturation
  occupied less than 1% of control samples. This supports jitter/tracking/contact
  sensitivity rather than a blanket lack of motor authority.

`ActorCritic` now has a configurable `log_std_min`; the default remains -3 for
backward compatibility. The new matched protocol explicitly uses initial std
.01 and floor -6 for both arms. It does not claim .05 exploration became robust.
Fresh frozen-final-actor noise probes are stored separately in
`output/multimorph_bc_noise_20261111_v1`.

## Optional BC regularization and short matched experiment

The PPO objective optionally adds normalized latent-action MSE on **training**
demonstration frames only. No validation episodes enter the anchor. Dataset hash
and initialization dataset identity are checked. Initial coefficient 10 decays
linearly to 2 over 20 updates in this bounded study; both coefficients, duration,
batch size and normalization are configurable, including a zero final weight.
An absent/zero anchor preserves old PPO behavior and RNG sampling. PPO is free
to improve beyond the teacher; regularization is a tunable early retention aid,
not an inference controller or a replacement for the reward.

The optimizer logs BC loss/coefficient, per-joint mean shifts, cumulative Gaussian
KL and bounded-action drift on fixed training demo states. Standard on-policy KL,
clipping, physical metrics and full checkpoints remain available. The learning
rate stays at 1e-5; the repair does not rely on lowering it alone.

`protocol_takeoff_bc_anchor.json` specifies ten updates, 1,024 transitions/update,
eight worlds, four PPO epochs, identical networks/rewards/limits and training seeds
42/43. Both arms receive the combined offline anchor. A runs at 100%; B follows
90 → 95 → 100 with unchanged performance gates and no bypass. Evaluation every
two updates uses fresh matched reset seeds 58000–58007, both deterministic and
sampled, at current and original geometry. These full-horizon probes disable goal
termination. Periodic checkpoint evaluations retain their usual training endpoint.
The original persistent-regression and numerical/divergence stops remain active.

This ten-update budget cannot establish a full-scale-trained curriculum: a stage
requires at least 8,192 stage transitions, two disjoint passing 32-episode windows,
and useful rotation/flight. Promotion to 100% needs additional earned time at 95%.
Actor, critic and Adam state transfer through episode-reset morphology changes.
No controller substitution or optimizer reset occurs at promotion.

## Commands and Molab

Use new output directories for every rerun. From the repository root:

```bash
# Generate/certify teacher prefixes and initial bounded BC fitting.
env -u PYTHONPATH .venv/bin/python scripts/prepare_multimorph_takeoff_bc.py \
  --output output/multimorph_teacher_NEW --updates 3000

# Refinement on noisy but certified teacher rollouts.
env -u PYTHONPATH .venv/bin/python scripts/refine_multimorph_takeoff_bc.py \
  --source output/multimorph_bc_20261111_v1 \
  --output output/multimorph_refined_NEW --updates 2000 --latent-std .01

# Actual short matched launch; never a million-transition run.
env -u PYTHONPATH .venv/bin/python scripts/run_takeoff_retention.py \
  --output output/multimorph_retention_NEW --updates 10 --training-seeds 42 43 \
  --bc-actor output/multimorph_bc_20261111_v2/bc_actor.pt \
  --bc-gate output/multimorph_bc_20261111_v2/bc_gate.json \
  --protocol experiments/morphology/learning/protocol_takeoff_bc_anchor.json \
  --validation output/takeoff_learning_certified/validation.json

# Monitor (replace NEW with the selected run).
tail -f /tmp/multimorph_bc_ppo_retention_20261111_v1.log
cat output/multimorph_bc_ppo_retention_20261111_v1/B_seed42/curriculum_events.json

env -u PYTHONPATH MPLCONFIGDIR=/tmp/takeoff-mpl XDG_CACHE_HOME=/tmp/takeoff-cache \
  .venv/bin/python scripts/report_takeoff_retention.py \
  --run output/multimorph_bc_ppo_retention_20261111_v1

env -u PYTHONPATH .venv/bin/python scripts/assess_takeoff_retention.py \
  --run output/multimorph_bc_ppo_retention_20261111_v1

# Fresh full-horizon checkpoint replay at training and original geometry.
env -u PYTHONPATH .venv/bin/python scripts/evaluate_takeoff_checkpoint.py \
  --checkpoint output/multimorph_bc_ppo_retention_20261111_v1/B_seed42/checkpoint_000010.pt \
  --output output/multimorph_heldout_NEW --seeds 60000 60001 60002 60003 60004 60005 60006 60007

# BC and mechanism figures, with measured CSVs.
env -u PYTHONPATH MPLCONFIGDIR=/tmp/takeoff-mpl XDG_CACHE_HOME=/tmp/takeoff-cache \
  .venv/bin/python scripts/report_multimorph_takeoff.py \
  --bc output/multimorph_bc_20261111_v2 --teacher output/multimorph_bc_20261111_v1 \
  --regression output/takeoff_regression_analysis_20261111_v2 \
  --output output/multimorph_bc_20261111_v2/figures
```

Open `notebooks/molab_multimorph_takeoff.py` in Molab and upload
`multimorph_takeoff_molab_bundle.zip` beside it. The cells clone the correct branch,
pin its base commit and verify every overlaid file hash. This is required because
much of the local work and BC artifacts is still untracked. They install the
locally used dependency versions, verify reward/BC assets, launch only the bounded
pair without W&B, show logs/curriculum events, plot measurements, evaluate saved
policies and offer a complete checkpoint/source/results ZIP download. No extended
launch cell is included. Cloud execution/dependency installation is not claimed
tested here; local Python syntax and source/asset integrity are checked.
The legacy task registry imports TensorDict and W&B transitively, so their library
versions are included in setup. The new trainer does not create a W&B run or
logger, and `WANDB_MODE=disabled` prevents online logging/authentication.

The equivalent complete backup shell command, after training finishes, is:

```bash
python3 -m zipfile -c output/multimorph_takeoff_backup_NEW.zip \
  output/multimorph_bc_20261111_v1 output/multimorph_bc_20261111_v2 \
  output/multimorph_bc_ppo_retention_20261111_v1 \
  output/multimorph_bc_noise_20261111_v1 \
  output/takeoff_regression_analysis_20261111_v2 \
  output/multimorph_takeoff_molab_bundle.zip docs/go2_multimorph_takeoff_bc.md
```

All reported angles are radians unless converted explicitly to degrees; work is
joules, clearance metres, durations seconds. Whole-episode maxima are distinct
from `genuine_flight_pitch` and certified duration. Later collisions remain in
full-horizon evaluation and must not be hidden by calling a certified kick a
handstand. Learning-curve variability is sample standard deviation across the two
training seeds; BC reset variation is a different source of uncertainty.

## Actual short PPO result and readiness

Both paired seeds completed **10 updates**, 10,240 training transitions per
arm/seed (40,960 total). No numerical, divergence or behavioral-regression stop
triggered. All six periodic checkpoints (0/2/4/6/8/10) retained 8/8 deterministic
and 8/8 sampled certified takeoffs at each evaluated current/full geometry for
both seeds. Those reused periodic resets are development curves, not 576
independent test samples. Fresh final deterministic replays on seeds 60000–60007
also gave 8/8 in every arm/seed/geometry evaluated (A at 100%; B at 95% and 100%).

Both curricula earned 95%: seed 42 at transition 8,638 with a 96.875% qualifying
window, seed 43 at transition 8,445 with 100%. All previous windows also passed.
Actor/critic/Adam were preserved. Neither reached 100%. Only roughly 1,600–1,800
transitions remain after entering 95%, below the 8,192 minimum stage duration;
the 35-degree promotion requirement was not established either. This budget
cannot test a full-scale-trained curriculum, and no stage was bypassed.

Final deterministic development-probe statistics below first average reset
families within each training seed, then report mean ± sample SD across seeds:

| At update ten | Fixed A, 100% | Curriculum B, 95% | Curriculum B, 100% transfer |
| --- | ---: | ---: | ---: |
| Certified takeoff | 100% ± 0 | 100% ± 0 | 100% ± 0 |
| Mean peak pitch | 32.95° ± 1.78° | 30.14° ± .99° | 33.12° ± 1.29° |
| Mean peak rear clearance | 16.95 ± 2.32 cm | 13.67 ± 1.04 cm | 18.57 ± 1.35 cm |
| Front-support fraction | .857 ± .011 | .857 ± .002 | .843 ± .008 |
| Certified flight duration | .229 ± .004 s | .218 ± .003 s | .229 ± .004 s |
| Minimum rear-leg work at certification | 10.61 ± .74 J | 10.23 ± .26 J | 11.49 ± .35 J |
| Later nonfoot collisions | 100% | 100% | 100% |
| Five-second handstand | 0% | 0% | 0% |

The collision row is deliberate: the maneuver is certified while contact-valid,
then full-horizon evaluation continues and the robot falls. Demonstrations contain
no colliding frames. Reliable takeoff is established; sustained handstand is not.
Initial full-scale mean peak pitch was about 32° and clearance 15.8 cm, so there
are small positive changes, but seed variability overlaps. These two seeds and
short budgets do not establish a progressive-leg-length learning advantage.

Across updates/seeds, A post-update KL was .01137–.02429 and B .01265–.02312.
Average minibatch clip fractions were 0–9.80% for A and 0–10.16% for B. Maximum
bounded-action RMS drift from initialization on fixed demonstration states was
.00191 for A and .00147 for B; cumulative 12-joint Gaussian KL reached .230/.135.
The conservative retention gate passed. Its assessment was fixed to follow the
current training geometry after promotion; it reports the actual scales used
and does not pretend a newly entered stage has three repeated evaluations.

The same frozen final actor's independent noise probe at std .01 gave 8/8 at
all geometries; std .05 gave 4/8, 5/8, 4/8 at 90/95/100%. Front-support fractions
fell from approximately .85 to .43–.45. This independently supports reducing
high-frequency exploration jitter; it is not evidence that BC regularization
alone caused the retention improvement.

**Molab readiness:** the shared actor and bounded matched retention workflow are
ready to reproduce. The million-transition comparison remains unlaunched. A
reasonable next authorization is **20 total updates per arm/seed**, retaining
matched budgets, frequent mean/sample evaluation and the same stage gates. That
is 20,480 transitions per arm/seed (81,920 total for two paired seeds), and tests
more learning at 95% plus the scheduled anchor decay. Do not claim full-scale
curriculum training or a curriculum advantage until 100% is earned and receives
substantial actual training time. A longer scientific comparison should retain
at least four paired seeds and the prepared 262,144 full-scale-step reservation.
Broader reset/physics robustness and late-flight/balance learning remain open.

Final checkpoint paths:

- `output/multimorph_bc_20261111_v2/bc_actor.pt` — one shared actor.
- `output/multimorph_bc_ppo_retention_20261111_v1/A_seed42/checkpoint_000010.pt`
- `output/multimorph_bc_ppo_retention_20261111_v1/B_seed42/checkpoint_000010.pt`
- `output/multimorph_bc_ppo_retention_20261111_v1/A_seed43/checkpoint_000010.pt`
- `output/multimorph_bc_ppo_retention_20261111_v1/B_seed43/checkpoint_000010.pt`

All earlier periodic checkpoints remain. `comparison_per_seed.csv` and
`final_comparison.csv` preserve reset-aggregated metrics and across-seed
variability. BC/mechanism figures are in `multimorph_bc_20261111_v2/figures`;
PPO retention/stability/curriculum/anchor-drift PNG/PDF and CSVs are in
`multimorph_bc_ppo_retention_20261111_v1/figures`. The final targeted run passed
**17 tests**, including anchoring gradients, backward-compatible noise floors,
checkpoint continuation and promotion-aware assessment. Ruff and Python
compilation passed for the new/modified modules and Molab notebook.
