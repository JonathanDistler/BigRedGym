# Parallel Go2 swing-up and active front-foot catching

**The corrected 16-environment pilot preserved takeoff but did not learn a useful
front-foot catch or full handstand.** The primary result is
`output/front_catch_parallel_pilot_20261112_v2`. All previous models, checkpoints,
CPG controllers and experiment outputs remain unchanged on
`experiment/com_energy_phase_ppo`.

## Measured results

Ten PPO updates, 16 independent environments per condition/seed, 32 control steps
per rollout, seeds 42/43: **5,120 transitions per condition/seed, 30,720 total**.
Rollouts continue across updates; they are not restarted at a rollout boundary.
Held-out standing evaluation uses four deterministic resets per training seed,
before training and every two updates. Intermediate starts are scored separately.
All conditions start from the same preserved BC-initialized actor at 33.42°.

| Condition | Final standing-start peak pitch, mean | Certified takeoff | 45°/60°/80° | Five-second handstand |
|---|---:|---:|---:|---:|
| A: fixed-support objective | 35.63° | 8/8 | 0/0/0 | 0/8 |
| B: loaded backward yielding | 32.83° | 8/8 | 0/0/0 | 0/8 |
| C: unload, reposition, catch | 35.78° | 8/8 | 0/0/0 | 0/8 |

Maximum individual final standing rollout: 36.67° (A). C's mean advantage over A
is only 0.15°, with two training seeds and small reset perturbations; this is not
convincing evidence of an active-catching benefit. Every final standing episode
ended in a front-calf collision. None had a qualified catch reward or stable
handstand dwell. Final intermediate evaluations also had zero success and zero
stable dwell, including the independently validated near-vertical resets.
Initializing near vertical is not counted as achieving near vertical from standing.

A phase-aware audit is more informative than final foot displacement:

| Standing metric, mean | A | B | C |
|---|---:|---:|---:|
| COM potential at apex | 48.82 J | 45.99 J | 48.92 J |
| Full kinetic energy at apex | 4.93 J | 4.58 J | 5.23 J |
| Backward foot displacement at apex | -3.8 mm | -6.7 mm | -4.4 mm |
| Backward displacement at episode end | 79.0 mm | 77.6 mm | 78.4 mm |
| COM support-region distance at apex | 173 mm | 173 mm | 180 mm |
| Capture-point distance at apex | 283 mm | 280 mm | 294 mm |
| Mean first recontact time | 1.118 s | 1.273 s | 1.088 s |
| Mean apex time | 0.983 s | 1.128 s | 0.965 s |
| Front impulse, release to apex | -0.754 | -0.629 | -0.803 kg m²/s |
| COM momentum change, release to apex | -0.768 | -0.641 | -0.817 kg m²/s |

Negative backward displacement means actual **forward** movement in world
coordinates. The apparently promising 8 cm retreat occurs after the apex, during
the fall. In C the front feet therefore did not relocate backward at the useful
time, and opposing front moments were slightly worse than A during swing-up.
The zero rear-contact impulse after release is consistent with the measured COM
momentum decrease being dominated by front contact, rather than internal leg
motion creating momentum.

Restricting the push window to loaded rear feet, positive rear actuator power
and positive body rotation excludes most static-stance moment cancellation.
A/B/C front impulses in that window were approximately -1.34/-1.81/-1.15 kg m²/s,
rear impulses +2.18/+2.56/+1.98, and net COM momentum gains 0.819/0.734/0.819.
C retained essentially the same useful kick impulse as A. It did not turn that
impulse into a better support transition. Potential energy remains about 40 J
below the correctly calibrated 89.42 J handstand target.

Eight C training episodes received some physically qualified intermediate catch
credit, but no held-out episode did. These are transient training events, not
successful handstands or evidence of deterministic recovery generalization.

## Parallelism and CPU benchmark

The previous momentum runner was serial. The older takeoff runner already used
batched inference and independent MjData objects; that time-major collection
pattern is reused. The new vector implementation adds persistent process workers.
Each worker owns independent per-world MjData, contact/tracker/phase objects,
configuration copies, episode counters and NumPy generators. Read-only compiled
models can be reused within a worker. Policy inference and PPO remain centralized:
one shared actor–critic per condition/seed receives all its worlds' experience.

An initial profile found repeated link-Jacobian instrumentation expensive. The new
opt-in task uses native MuJoCo subtree COM/momentum and full-system energy at
500 Hz, cross-checked against the independent link sums to 1e-10. Contact forces,
world-frame moments and locked-shape inertia still use actual model geometry.
No simplified simulator, contact model, friction or actuator setting replaces
MuJoCo. The old experiments' measurement code is untouched.

Isolated fast-path benchmark, four workers; controller dt=20 ms with ten native
2 ms physics substeps per transition:

| Backend / worlds | Transitions/s | Aggregate RSS | PPO update time |
|---|---:|---:|---:|
| Serial batch / 8 | 57.1 | 789 MiB | 37.8 ms |
| Four processes / 8 | 147.0 | 3,754 MiB | 10.0 ms |
| Four processes / 16 | 152.5 | 3,761 MiB | 14.0 ms |
| Four processes / 32 | 155.1 | 3,773 MiB | 27.4 ms |
| Four processes / 64 | 159.7 | 3,805 MiB | 49.6 ms |

The selector chose **16**, the smaller 16/32 count within 5% of the fastest rate.
64 was benchmarked but not selected for training. RSS sums parent and workers and
can double-count shared library pages; notebook/kernel memory is additional.
Worker startup was about 17 seconds and excluded from collection throughput.
Pilot collection averaged about 145 transitions/s. More worlds did not provide a
large gain with four workers. Older diagnostic benchmarks are preserved; an
eight-worker development run overlapped another job and is excluded from the
primary throughput comparison. MJX/GPU contact parity has not been validated and
no GPU backend is selected automatically.

## Contact objectives and actor contract

All A/B/C conditions use identical 100% geometry, ground friction, contact solver,
joint ranges, target slew limits, PD gains, effort clipping and velocity protection.
**A's feet are not artificially welded.** A retains a fixed-support objective and
foot-motion penalty. B only credits loaded yielding without a post-release unload.
C permits measured unload/reload and short loss of front support. Differences are
objectives and contact-event eligibility, not different physical environments.
Actual motion is measured even when it violates a condition's intended behavior.

Each policy has 73 actor inputs, 103 critic inputs, hidden layers (256,256,128),
and its own twelve joint-target outputs. The preserved 61/91 checkpoint is copied
exactly, with zero weights appended for twelve per-foot contact/motion features.
All six initial networks are identical; paired training seeds control independent
exploration streams. Strong BC applies during loading/push-off (frame weight 1),
with .025 after release and .15 for the validated balance state/action labels.
There is no reference controller or controller handoff at inference. Exported
actor-only policies use the same phase and estimator implementation as training.
The estimates use simulator state; hardware state estimation is not implemented.

World backward displacement of each front foot is

b_i = -(foot_world_i - initial_foot_world_i) dot initial_heading.

This cannot be faked by measuring a foot relative to a translating trunk. Per-foot
world positions, normal loads, unload durations, reloads and velocities are logged.
A recontact requires at least 20 ms of unloading **after** the maneuver begins and
40 ms of stable bilateral loading. Preparatory contact changes are not catches.

One-shot phase observations are sin(pi p), cos(pi p), p. Existing certified
loading/push timing is retained; rear release is event-certified using prior
loading, joint extension, positive work, impulse and airborne dwell. Phase 3
permits front-foot relocation. Actual recontact plus rotation can initiate catch
phase 4. Near-vertical orientation, low angular speed and current support initiate
stabilization; phase completion requires the real five-second dwell.

## Reward, safety and reset curriculum

The existing takeoff preparation, release, airborne rotation and effort terms
remain. New increments for post-release pitch, capture geometry, useful placement
and stable recatch are bounded best-so-far episode credits. They do not pay for a
static intermediate posture or repeated foot cycling. Contact-loss rotation
progress is withheld until valid support returns. No uniform action filter is on.

Placement/catch credit additionally requires **current** rear-foot clearance and
unloading, pitch close to the attained rotation frontier, continued useful
rotation, COM descent slower than 0.25 m/s and capture-point support distance
below 12 cm. Actual placement must improve the support geometry compared with
the original foot locations evaluated at the same COM state. Backward motion by
itself earns no reward. A front impact over four body weights latches the episode
invalid for placement credit. All nonfoot collisions and actuator-state failures
invalidate positive progress and success. The constant-height capture point
c + sqrt(h/g) v is an approximate shaping signal, not a recoverability certificate.

Terminal reward combines exp(-0.5 ((U-U_target)/12 J)^2), full SO(3) orientation
error (width .35 rad), and low body angular speed (width .5 rad/s), with current
contact/collision gating. U_target=89.4166207975 J uses measured mass
16.2764066211 kg, native gravity and the validated whole-robot COM height
0.5600034658 m above world z=0. Full kinetic and mechanical energy include rotor
armature. COM/support momentum and front/rear external angular impulses remain
separate. A moving contact centroid is not treated as a fixed pendulum pivot;
COM or fixed-reference balances are used when interpreting moment changes.
`support_valid` in the inherited feature contract means bilateral loading;
raw support momentum is still calculated at the active force centroid during
unilateral contact, and plots retain it when either front foot is loaded.

Five-second success runs at native physics rate and requires current bilateral
support, clear/unloaded rear feet, correct full orientation, calibrated potential,
low body/COM motion, COM within the front support region, low slip and positive
head/calf clearance. Every bad sample resets dwell. Intermediate-start success is
never combined with genuine standing-to-handstand success.

The bank contains normal standing resets plus validated native states:

| Start | Training states | Held-out states |
|---|---:|---:|
| Forward loaded | 4 | 2 |
| Before certified release | 4 | 2 |
| After certified release | 4 | 2 |
| Partial rotation | 7 | 2 |
| Near-vertical recovery | 12 | 6 |

Initial states come from certified actor rollouts and previous saved trajectories;
near-vertical states come from independently successful balance continuations.
Identical states are deduplicated. Imported trajectory frames stay entirely in
training; evaluation uses separate source episodes. Acceptance rejects nonfoot
contacts, self-contact, joint/velocity violations and excessive penetration.
Up to 2.5 mm of native soft **foot** compression is allowed, not body penetration.
No contact force is assigned by hand. Two selected archived partial trajectories
provided no valid 45–60° reset bridge under these checks; none was invented.

Initial standing-start probability is .35, with the remainder across the five
other stages. A common performance curriculum preserves matching across A/B/C:
standing fraction increases to .50/.65/.80/1.0 only after shared measured dynamic
45°/60°/80°/85° gates. These are pass-through milestones, not static holds.
The pilot never passed its first gate, so the mixture stayed fixed.

A development pilot `..._v1` is preserved. Its gate could credit recovery after
rear-foot regrounding because it used historical pitch progress. A reward audit
identified the missing current-state condition; the corrected gate above has a
new grounded/falling negative-control test. The authoritative `..._v2` repeats
the same matched budget from the same initialization. Each pilot used 30,720
training transitions; neither is an extended run. Timing benchmarks, preparation
rollouts and software smoke checks are accounted separately.

## Interpretation

Active relocation has **not** demonstrated reduced front counter-torque, improved
COM recoverability, fewer collisions or near-vertical entry. The feet retreat too
late, and the learned mean policy still follows the original swing-up failure
mode. B also delays release and reduces COM height. C's small pitch change is
within ordinary PPO variation and does not correspond to a learned catch.

The limiting evidence is a combination of front-support mechanics, late foot
placement and failure to learn recovery from the supplied state distribution.
Useful rear impulse remains close to A, so merely generating more rear motion
is not a demonstrated solution. The gap between roughly 32° partial resets and
near-vertical resets is large; the single actor also fails nominal intermediate
balance evaluation. This does not prove a dynamically infeasible path.

The next bounded study should mine actual safe pre-catch states from the rare
qualified training events, strengthen late-phase balance distillation while
protecting early BC, and test phase-specific post-release exploration. Require
useful early relocation and improved COM geometry before expanding the budget.
More parallel worlds alone are not evidence of a better skill. No million-step,
remote Molab or MJX run was launched.

## Reproduction, Molab and resumption

28 experiment-plus-regression tests pass, including native/Jacobian measurement
agreement, independent world/config/RNG state, serial/process equivalence,
contact-event/impact negative controls, grounded/falling reward rejection, and
bitwise optimizer/RNG/world-state continuation. The completed runner was also
restored through its CLI without changing older checkpoints.

```bash
# New folders are required; old artifacts are preserved.
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/prepare_front_catch.py --output output/front_catch_preparation_new

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_front_catch.py --preparation output/front_catch_preparation_new \
  --output output/front_catch_benchmark_new --workers 4 --counts 8 16 32 64

env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/run_front_catch.py --preparation output/front_catch_preparation_new \
  --benchmark output/front_catch_benchmark_new/selection.json \
  --output output/front_catch_pilot_new --updates 10 --horizon 32 --seeds 42 43 \
  --max-seconds 1200

# Resume the latest complete checkpoint in the SAME run directory.
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/run_front_catch.py --output output/front_catch_pilot_new \
  --resume output/front_catch_pilot_new/checkpoint_000010.pt --workers 4

env -u PYTHONPATH MPLCONFIGDIR=/tmp/catch-mpl XDG_CACHE_HOME=/tmp/catch-cache \
  .venv/bin/python scripts/report_front_catch.py output/front_catch_pilot_new \
  --output output/front_catch_report_new --benchmark output/front_catch_benchmark_new

env -u PYTHONPATH MPLCONFIGDIR=/tmp/catch-mpl .venv/bin/python \
  scripts/analyze_front_catch.py output/front_catch_pilot_new \
  --output output/front_catch_mechanism_new
```

Checkpoint generations are exclusive and atomic. They contain all condition/seed
networks, optimizers, Torch/NumPy RNG states, per-lane generators/counters, full
MuJoCo integration/warmstart state, contacts, phases and observations. Resumption
checks source/data hashes and runtime versions. It retains world count, seeds,
horizon and total budget; worker grouping may change. Updates run as matched
rounds. A crash can discard an uncommitted round, but never overwrite an older
checkpoint. Separate append-only session folders preserve crash logs; reports use
only the committed history inside the selected checkpoint, avoiding duplicate
metrics after replay. Actor-only exports and standalone playback are included.

`notebooks/molab_front_catch_parallel.ipynb` is the executable Molab entry point.
Upload/extract `front_catch_molab_20261112.zip`, install/validate dependencies,
benchmark the actual host, and launch the bounded A/B/C pilot. W&B is disabled.
The notebook supports restart from the latest checkpoint, preserves incomplete
attempts, generates PNG/PDF/CSV reports, and creates downloadable recovery archives
as checkpoints appear plus a final full backup. A soft training deadline leaves
backup time. Hard session termination can lose uncommitted work, so download a
recovery archive before the session ends. All earlier checkpoint files remain.

Outputs: `front_catch_report_20261112_v3` contains separate standing/intermediate
CSVs, PNG/PDF figures, milestones, PPO diagnostics and exported single actors.
`front_catch_mechanism_20261112_v3` contains the phase-aware impulse/timing audit.
The fast isolated benchmark and the prepared bank are saved separately. The
notebook code cells parse, and its underlying commands/bundle are locally tested;
no connected Molab execution integration was available in this session, so no
remote execution is claimed. GPU MJX remains a separate, unvalidated option.


## Opt-in anticipatory extension

The next validated configuration is documented in [front_catch_anticipatory.md](front_catch_anticipatory.md). Select `--suite anticipatory_v1` for A=original fixed support, B=early placement, C=early placement plus rear sweep. Previous results and the default original comparison remain as reported above.
