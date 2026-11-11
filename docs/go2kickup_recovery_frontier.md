# Expanded recovery frontier and interrupted diagnostic

Work is on `ppo_kickup`. This report supersedes the small-bank curriculum in
`go2kickup_recovery.md` for the frontier experiment. Original CPG architecture,
59/63 observations, 12 residual actions, checkpoints, atlas, kick reference and
seed-6003 source bank are preserved. No 200-iteration experiment is authorized
by the results reported here.

## Dataset and feasibility

`datasets/recovery_frontier_v4/` contains 285 valid native snapshots from 96
source rollouts: training 162/48, validation 64/24, held-out 59/24 (states/rollouts).
Training frozen-controller five-second recoveries are 64/162; validation 16/64;
held-out 14/59. Each snapshot is independently replay-labelled for 6.5 seconds,
not labelled from its parent trajectory's eventual outcome. Frozen success
requires a five-second continuous valid hold without a latched body collision.

The training kinds are 55 nominal, 36 disturbed, 36 atlas-boundary, 19 posture
transition and 16 standing-kick snapshots. Frames are sparsely captured across
0.08–4.4 seconds, rather than at every control step. Sampling chooses a stratum,
then a source uniformly, then a snapshot; long rollouts cannot dominate simply
by contributing more frames. States from a single rollout remain correlated.

Pose families vary heading, pitch, pitch rate, horizontal velocity, joint pose,
front-thigh asymmetry and initial clearance. Controlled velocity impulses are
followed by forward simulation; these are disturbance-recovery rollouts, not
claims of standing-start kick-up feasibility. Contact arrays are reconstructed
by MuJoCo and never assigned synthetic loads. Native solver state and CPG memory
are saved. Standing rollouts vary transition timing and include seed 6003 as a
training anchor, near-successful cases and atlas-boundary configurations.

Whole source rollouts and parameter families are disjoint. Training uses axial
velocity with heading families 0 and 0.3 radians; validation uses diagonal
velocity near heading 0.6; held-out uses lateral velocity near heading -1.1.
Standing timing families are separate. The historical seed-6003 validation is
now an explicit training anchor, so historical 6000-series full-system scores
are regression measurements, not an untouched generalization test. All current
cohorts have been inspected; further tuning requires a new final test cohort.

The closest measured training success/failure pair has pitch 1.58365/1.58380 rad,
pitch rate 0.5451/0.5481 rad/s, COM speed 0.0952/0.0939 m/s, support offset
-12.28/-12.13 mm and base height 0.59004/0.59016 m. The failed state requests
20.3% of the torque limit, has no joint-limit violation and reaches 11.1% of
joint speed limits. Thus these five coarse features cannot alone classify the
boundary. Full joint positions/velocities, contact positions and force vectors,
friction utilization, COM velocities, angular momentum and CPG phase/load are
retained in `output/recovery_frontier_report_v5.json` and
`output/recovery_frontier_authority_v4.json`.

Successful training states have median COM speed 0.0377 m/s (p10–p90
0.0204–0.0881); frontier failures 0.1578 (0.0390–0.3580); difficult failures
0.8569 (0.3108–1.6876). High speed and support loss are useful warnings, not
physical-impossibility certificates. Torque clipping, absent front contacts and
static support margins do not prove that future contact or dynamic recovery is
impossible. No viability proof or optimal-control search has been performed;
failed states remain **physical feasibility unknown**.

## Performance-gated six-stage curriculum

Two consecutive batches of 64 primary-stage completed episodes must each attain
at least 80% five-second recovery. Replay episodes cannot promote, and a terminal
fall invalidates an earlier hold. There is no fixed-iteration advancement.
Earlier-level replay remains 20%. Each primary-stage mixture is
recoverable/frontier/difficult; source-uniform selection applies within strata.

| Stage | States | Sources | Frozen raw success | Primary mixture | Expected frozen success |
|---|---:|---:|---:|---|---:|
| 0 | 44 | 7 | 100% | 100/0/0 | 100% |
| 1 | 64 | 11 | 100% | 100/0/0 | 100% |
| 2 | 131 | 42 | 48.9% | 70/30/0 | 70% |
| 3 | 162 | 48 | 39.5% | 60/30/10 | 60% |
| 4 | 162 | 48 | 39.5% | 50/35/15 | 50% |
| 5 | 162 | 48 | 39.5% | 40/40/20 | 40% |

The final three stages differ by sampling weights, not eligible membership.
Raw success counts states equally; expected success reflects source-uniform
mixtures. Neither is an independently estimated generalization rate.

The following are **primary-sampler-weighted p10–p90** distributions, including
front support probabilities (both / one / none), with 6% body weight per foot.
Full position, speed, pitch, rate and elevation distributions are in the report.

| Stage | COM/support x (m) | COM speed (m/s) | Both / one / none |
|---|---|---|---|
| 0 | -0.00632 to 0.00096 | 0.0183–0.0882 | 100 / 0 / 0% |
| 1 | -0.00649 to 0.00110 | 0.0209–0.0881 | 100 / 0 / 0% |
| 2 | -0.04580 to 0.00386 | 0.0244–0.2899 | 96.29 / 3.71 / 0% |
| 3 | -0.04943 to 0.02426 | 0.0251–0.3600 | 90.87 / 6.63 / 2.50% |
| 4 | -0.06720 to 0.05288 | 0.0260–0.7258 | 87.54 / 8.71 / 3.75% |
| 5 | -0.16995 to 0.06147 | 0.0278–0.8569 | 84.21 / 10.79 / 5.00% |

## Diagnostic provenance

The interrupted run saved iterations 0 and 25 under
`logs/go2kickup_recovery_20261110T193925_541524/Nov10_14-39-25_/`.
It used `recovery_frontier_v3/train.npz`, byte-identical to v4 training.
The continuation restores both optimizers, learning rate and curriculum into a
new log directory; native worlds, episode histories and RNG restart. It is a
50-update diagnostic with an interruption, not an exact uninterrupted replay.
W&B is disabled, MuJoCo runs on CPU with 16 environments, each rollout has
4096 transitions and the optimizer batch is 1024. The launcher defaults to 50 iterations; 200 requires an explicit override.
Checkpoints are saved at 0, 25 and 50. `output/recovery_frontier_50_resumed_v4/` records the continuation.

Resume exposed inference-only environment buffers. The launcher now wraps
normalization burn-in in inference mode after restoring curriculum, matching
rollout/reset semantics while leaving gradient updates unchanged. Early failed
resume attempts are retained separately. No checkpoint was overwritten.

## Exact local / Molab Online commands

Run in the repository with Python 3.11 and all gitignored assets transferred at
their original relative paths. Cloning the branch alone does not transfer these
untracked scripts, datasets or checkpoints. No remote connection was made.
Every output directory below must be new. These commands also run locally.

```bash
# Confirm branch, original assets and checkpoint interface before remote work.
git branch --show-current
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/verify_recovery_assets.py \
  --manifest datasets/recovery_frontier_assets_manifest_v2.json \
  --environment molab --verify-interface \
  --output output/recovery_molab_preflight_new.json

# Fresh 50-update diagnostic: original checkpoint copied, never overwritten.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/train_kickup_recovery.py --frontier-curriculum --iterations 50 \
  --dataset datasets/recovery_frontier_v4/train.npz \
  --output-dir output/recovery_frontier_molab_50_new

# Resume the interrupted local diagnostic, preserving its exact training path.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/train_kickup_recovery.py --frontier-curriculum --iterations 50 \
  --dataset datasets/recovery_frontier_v3/train.npz \
  --resume-checkpoint logs/go2kickup_recovery_20261110T193925_541524/Nov10_14-39-25_/model_25.pt \
  --output-dir output/recovery_frontier_resume_new

# RUN_DIR is the train.log_dir in the chosen output's configuration.json.
RUN_DIR=logs/REPLACE_WITH_EXACT_EXPERIMENT/REPLACE_WITH_RUN
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_recovery_frontier.py \
  --dataset-dir datasets/recovery_frontier_v4 --run-dir "$RUN_DIR" \
  --output-dir output/recovery_frontier_molab_evaluation_new

# Full standing-to-handstand regression on identical generated prefixes.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_kickup_recovery.py --run-dir "$RUN_DIR" \
  --dataset datasets/recovery_frontier_v4/train.npz \
  --output-dir output/recovery_frontier_molab_full_system_new

# Reproduce stage distributions and actuator/contact inspection.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/report_recovery_frontier.py --dataset-dir datasets/recovery_frontier_v4 \
  --output output/recovery_frontier_report_new.json
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/inspect_recovery_authority.py --dataset-dir datasets/recovery_frontier_v4 \
  --output output/recovery_frontier_authority_new.json
```

The new `datasets/recovery_frontier_assets_manifest_v2.json` validates current
implementation and all frontier datasets, in addition to original dependencies.
The original manifest is retained as historical evidence. Frontier NPZ checksums
are also recorded in `datasets/recovery_frontier_v4/provenance.json`;
transfer all three NPZs and labels, trajectories and provenance JSONs, plus the
v3 training bank if resuming this historical run. Never regenerate expected
checksums on the destination in place of verifying the transferred assets.

To regenerate independent copies of the banks (CPU, no training), run:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/build_recovery_frontier.py --output-dir datasets/frontier_base_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/extend_recovery_frontier.py --source datasets/frontier_base_new \
  --output-dir datasets/frontier_extended_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/strengthen_frontier_evaluation.py --source datasets/frontier_extended_new \
  --output-dir datasets/frontier_final_new
```

Regeneration additionally needs the protected historical seed bank listed in
`PROTECTED` in `build_recovery_frontier.py`; frozen-policy replay labels can be
runtime-sensitive and must be measured again. These commands never replace the
saved v4 banks. Use the verified transferred banks for the exact recorded test.

## Completed 50-update results and decision

**Do not launch 200 iterations. Retain the original balance checkpoint.**
The dataset/curriculum problem is substantially improved, but this PPO recipe
loses known recoveries before it can learn the frontier.

Exact run:
`logs/go2kickup_recovery_20261110T213910_891120/Nov10_16-39-10_/`.
State-replay results: `output/recovery_frontier_learning_resumed_v1/summary.json`.
Full-system results: `output/recovery_frontier_full_resumed_v1/summary.json`.
Decision and aggregated metrics: `output/recovery_frontier_learning_decision_v1.json`.

| Metric | Original / iteration 0 | Iteration 25 | Iteration 50 |
|---|---:|---:|---:|
| Near-vertical retention | 14/16 | 3/16 | 1/16 |
| Established recoverable validation + held-out | 30/30 | 6/30 | 2/30 |
| Frontier validation | 0/16 | 1/16 | 0/16 |
| Frontier held-out | 0/18 | 0/18 | 0/18 |
| Difficult handoff validation + held-out | 0/59 | 0/59 | 0/59 |
| All validation states | 16/64 | 3/64 | 1/64 |
| All held-out states | 14/59 | 4/59 | 1/59 |
| Combined five-second success | 24.39% | 5.69% | 1.63% |
| Mean longest strict hold | 1.889 s | 1.207 s | 0.862 s |
| Mean front support retention while collision-free | 51.48% | 50.10% | 48.85% |
| Mean COM speed while collision-free | 0.876 m/s | 0.931 m/s | 0.965 m/s |
| Mean absolute body pitch rate while collision-free | 1.655 rad/s | 1.747 rad/s | 1.771 rad/s |
| Mean residual RMS | 0.575 rad | 0.580 rad | 0.584 rad |
| Mean motor saturation fraction | 3.81% | 3.82% | 3.97% |
| Standing transition, historical validation | 1/16 | 1/16 | 0/16 |
| Standing transition, historical held-out | 0/16 | 0/16 | 0/16 |

Each aggregate weights snapshots equally; these correlated snapshots are not
123 independent trials. Collision-free telemetry is censored at each trial's
first collision and remains descriptive. Per-split and per-eligible-stage values
and complete outcomes are retained in the evaluation JSONs. Stages 3–5 share
eligibility, so their unweighted replay rates equal the all-state rates above;
their actual training mixtures differ as reported earlier.

Paired checkpoint-25 wins/losses versus original are 1/24; checkpoint 50 has
0/28. Five states improve longest hold by more than 0.1 s at checkpoint 25,
and one at checkpoint 50. These isolated partial gains do not offset forgetting.
Full-system mean hold differences for validation/held-out are -0.0569/+0.0013 s
at checkpoint 25 and -0.2038/-0.0044 s at checkpoint 50. These historical full
transition cohorts overlap the training anchor family and are regression-only.
All policies receive identical native replay inputs; initial measurements agree
exactly. Every iteration-0 replay outcome matches the original, and full-system
iteration 0 has exact paired hold/outcome parity on both cohorts.

The final checkpoint records curriculum level 0, zero passes, 0% success in its
last completed 64-episode batch, and 359 cumulative stage-0/stratum-0 reset
samples. No frontier or difficult state was sampled. The expanded coverage is
available but was **not exercised by this run**. This is not evidence that the
frontier cannot be learned. It does demonstrate that extending this recipe to
200 updates is unjustified.

A four-state, two-second CPG probe (`output/recovery_frontier_cpg_probe_v1.json`)
shows phase progression of 10.93–11.36 rad for the original and 10.95–11.37 rad
for iteration 50, with desired contacts `[1,1,0,0]` retained and no collisions
during that short probe. Residual RMS does not collapse. This verifies active
clocks, not that their behavior is useful for long recovery. The policy does not
have an action that disables the CPG. Fixed evaluation cohorts prevent hiding
failure by reset selection. No success/reward-hacking claim is supported by
these results: measured recovery, hold and stabilization deteriorate.

The immediate limiting factor is **retention under PPO updates on demonstrated
states**, with reward/critic adaptation and training-time observation noise as
unresolved causes. The inherited runner adds observation noise during rollouts
even when `actor.add_noise` is false. A previously saved 16-reset noisy baseline
achieved 13/16 five-second holds (12 collision-free over the full horizon),
versus 16/16 with action exploration alone; this is a small exploratory ablation,
not a causal isolation. New COM/hold rewards also change the critic's target;
the saved critic normalizer remains interface-compatible but some normalized
inputs exceed 13 in magnitude. Neither observation parity nor finite losses
establishes a well-calibrated value function for the new objective.

Before another recovery run, preserve the original again and test conservative
policy retention (teacher/KL anchoring or tighter update limits), reward/value
calibration and observation-noise ablations on the same demonstrated resets.
Require frozen-policy-level retention in the actual eight-second noisy training
loop and deterministic validation before admitting frontier states. A physically
unsuitable handoff may still limit difficult cases, but cannot explain losing
near-vertical balance without ever sampling those handoffs. More failed
advanced states alone will not fix this result.

Validation: 60 targeted tests passed, Ruff and whitespace checks passed. The
new resume burn-in path has a regression test. All 162 training snapshots have
zero actor/critic observation difference, zero action difference and zero CPG
feature error versus the saved original environment. Both networks load strictly;
evaluation does not update saved normalization. Final parameters, native state
and PPO losses are finite (value loss 3.9901, surrogate loss 0.0003423).
`finite_reward_calls: 0` in the full-run result means per-call reward instrumentation
was disabled outside smoke mode; it is not a claim that rewards were individually
audited. All 25 resumed vitals records are finite. The interrupted run's earliest
empty-episode logging NaNs are retained; they are not optimizer-loss NaNs. Its
log contains ten unsaved updates beyond checkpoint 25; those updates were
lost and are not counted as part of this 25+25 continuation.

All seven original protected assets still match their pre-expansion SHA256s,
including the original checkpoint and seed-6003 source bank. Original checkpoint
SHA256: `6a56bebf31c26f025274ac18b995b68485fa3fb87e07c011e73872f7b36d755f`.
No model was promoted and no 200-update run was started.

A portable snapshot is saved as `output/recovery_frontier_resume_bundle_v1.tar.gz`.
It includes the current implementation, tests, documentation, verified original
assets, v4 banks, v3 resume training bank, interrupted and resumed run files,
checkpoints and completed evaluations. The adjacent `.sha256` file records its
checksum. Extract into a dedicated `ppo_kickup` checkout on Molab, then run the
manifest preflight above before training. The bundle contains no virtualenv;
create the repository's Python 3.11 environment using `uv sync --frozen` first.
