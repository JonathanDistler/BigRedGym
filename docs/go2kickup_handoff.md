# Kick-up to frozen balance: measured gap

No PPO training was launched. All existing configurations, reference JSONs and
checkpoints were preserved. These are MuJoCo CPU evaluations on
`features/kickup`, not hardware results.

The fresh kick controller is
`logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt`.
Both available balance controllers were tested without updating their weights:
`logs/go2balance_longhold/Nov01_22-09-05_/model_1500.pt` and
`logs/go2balance/Oct31_23-29-27_/model_1500.pt`.

## Recoverable states

The older October 31 CPG controller is the stronger handoff target. With small
independent pose/velocity perturbations and random heading, its five-second
hold success is 14/16 at nominal 90° and zero pitch velocity, 13/16 with an
additional 2 cm initial height, and 16/16 with an additional 5 cm height.
The newer long-hold checkpoint achieves 3/16, 4/16 and 1/16 respectively on
those same probes. Checkpoint names and later training dates are insufficient
evidence of recovery reliability.

The CPG controller's basin is narrow: 80°, 100° and ±1 rad/s at 90° all produce
0/16 successes. Additional probes at 85°, 87.5°, 92.5° and 95° also produce
0/16; at 90°, -0.25 rad/s gives 11/16 and +0.25 rad/s gives 6/16. Front hip
offsets of ±0.15 rad fail 16/16. This does not establish a continuous boundary
between tested states. The 14/16 Wilson 95% interval is approximately
0.64–0.97; 0/16 has an upper confidence limit of approximately 0.19.

Each main atlas contains 25 regions × 16 trials. The refinement adds six
regions × 16 trials. Probes preserve valid joint geometry and put the lowest
foot above the floor; none of the main probe initial states has nonfoot ground
penetration. These controlled initial conditions are supplemented by six
unmodified physical trajectory states per controller × 16 trials. The latter
retain native qpos/qvel and the kick world's randomized mass, inertia and
friction rather than rotating or projecting the trajectory into a favorable
reset pose.

Success requires five continuous seconds within 75–105° pitch, pitch speed
at most 1.5 rad/s, both front loads at least 6% body weight, rear loads at most
1.5% body weight, both rear clearances at least 20 cm, and no fall/nonfoot
ground contact. Recovery is observed for 6.5 seconds. This physical hold test
is separate from the training reward and curriculum success predicate.

Artifacts:

- [CPG atlas and comparison](../output/handoff_cpg_comparison/summary.json)
- [Near-vertical refinement](../output/handoff_region_refinement/atlas.json)
- [Long-hold atlas](../output/handoff_gap_v5/atlas.json)
- [Long-hold trajectory replay comparison](../output/handoff_comparison/REPORT.md)
- [Final detailed comparison](../output/handoff_final/REPORT.md)

## Why the current kick falls short

The reference and fresh residual policy both peak at approximately 26.3°.
The fresh residual changes essentially none of the rotational trajectory:
peak sagittal momentum is 0.98 N m s and rear ground impulse over 0.7–1.05 s
is 14.60 N s in both cases. At maximum pitch, momentum has already fallen to
approximately 0.034 N m s and the COM remains about 6.9 cm behind the front
support midpoint. The robot reverses rather than approaching balance.

Over 0.7–1.2 s, the measured front contact angular impulse about the COM is
approximately -5.48 N m s, opposing the rear contact's +4.80 N m s. This makes
front geometry and support moment central to the shortfall. Increasing rear
clearance alone cannot fix it. Actual rear-knee extension begins near 0.788 s
and sustained retraction near 1.016 s. The target reference retains front
hips near +1.04 rad at peak pitch, far from the balance controller's roughly
-1.10 rad front reference; rear hips are roughly +0.44 rad versus +2.05 rad.
There is a large posture gap as well as an orientation gap.

The independent small search changes front yielding/gains, push coordination
and retraction timing. Its matched-seed candidate reaches 81.9° without a
residual and 80.7° with the existing frozen residual actor, despite *less* rear
ground impulse (about 9.0 N s). Retraction starts near 0.972 s and momentum
peaks near 2.45 N m s. The posture still misses the balance basin: front hips
are near +0.35 rad and rear hips near +1.02 rad at peak pitch. Contact angular
impulse, COM location and final joint configuration matter more than an
unbounded momentum objective.

This candidate is rejected as a replacement. It produces nonfoot ground
contact in 11/16 reference-only and 12/16 residual matched trials. On the
disjoint held-out search seeds it reaches 76.2° mean peak pitch but has 5/16
body contacts and about 17.7 cm mean maximum front displacement. Peak pitch
includes failed trajectories. The source reference remains unchanged.

The scaled distance to successful CPG atlas samples drops from approximately
14.33 to 11.78, still far outside the gate's 0.75 neighborhood. This metric
uses pitch/0.2 rad, pitch velocity/1 rad/s, COM-support offset/5 cm,
base height/5 cm, and joint positions/0.3 rad. It is a diagnostic distance,
not a reachability certificate or learned recovery probability.

## Handoff and intermediate transition investigation

The evaluation supervisor starts with the matched standing reset and checks
each control frame. It requires a neighborhood of a region with at least
75% measured success, 30 ms of front support, admissible orientation/rate,
and no body contact. Unknown states cause it to withhold switching.
All four tested controllers yield 0/16 full transitions and zero gated
switches. No overlap was demonstrated. Six forced-switch times from 0.9 to
1.6 s also fail to produce a five-second hold for either balance checkpoint.

State-based selection is causal; selected switch states are replayed in
independent worlds for balance evaluation. This is an evaluation-only replay
supervisor, not an online hardware controller. Native solver caches are
reinitialized at replay; physical positions, velocities and model parameters
are preserved. Each balance controller uses its own saved observations,
scales, default joint reference, deterministic inference and CPG initialization.
The old balance run requires documented CLI overrides `--cpg --contact_modes
stance stance swing swing`; these were restored explicitly. Saved controller
source is loaded because current balance source has incompatible additions.
State layout and strict actor state-dict loading are checked.

Joint targets blend in physical canonical joint radians using a smoothstep
over 0.2 s, initialized from the kick command. This avoids an immediate
position-target jump and does not impose a permanent slew limiter on the
balance policy. Gains change to the balance controller's saved gains; torque
continuity is not guaranteed. A hardware handoff would require further work.

An additional intermediate-controller experiment varies this blend to 0.1,
0.4 and 0.6 s, switching the candidate residual trajectory at 1.1 and 1.2 s.
Each combination has 16 trials. None achieves five seconds, but the 0.6 s
bridge at 1.2 s reaches a best individual hold of 3.88 s. This investigates a
short deterministic bridge without training another policy; it does not
establish a successful intermediate learned controller.
[Bridge trials](../output/handoff_bridge_probe/summary.json).

## Smallest next experiment

First search a short two-phase transition reference, keeping the existing
push-off and both policy weights frozen. Optimize only front/rear posture
blend start times and durations over approximately 0.4–0.8 s after launch.
Target the successful atlas's actual joint configuration and COM-support
offset, near-90° orientation and approximately -0.25 to 0 rad/s pitch rate.
Use the frozen balance controller's subsequent five-second hold as the
primary score, with failure/contact penalties and a 12 cm front travel
preference. Do not score larger momentum or clearance as progress.

A reasonable first budget is 32 bridge candidates on four screening starts,
then the best two on 16 disjoint starts each. The existing six bridge tests
justify this smaller trajectory experiment before another PPO run. If a
smooth reference still cannot connect the states, use a separate 25–50
iteration bridge residual pilot initialized from actual 70–90° trajectory
states, retaining frozen kick and balance actors. Do not spend 1,500 PPO
iterations on the current clearance-focused objective.

## Reproduction

Run from the repository root. Every evaluator requires a new output directory.

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_kickup_handoff.py \
  --kick-checkpoint logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt \
  --balance-checkpoint logs/go2balance/Oct31_23-29-27_/model_1500.pt \
  --balance-cpg --output-dir output/handoff_reproduction

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/search_handoff_reference.py \
  --parameters gym/envs/go2/references/kickup_front_yield.json \
  --candidates 64 --output-dir output/handoff_search_reproduction
```

Add `--candidate-reference PATH/candidate_reference.json` to compare a separate
candidate. Add `--bridge-durations 0.1 0.4 0.6` to repeat the intermediate
controller investigation at 1.1 and 1.2 seconds. Add
`--atlas-path PATH/atlas.json` to reuse a measured atlas from
the same balance checkpoint; provenance and trial counts are checked.
`scripts/report_kickup_handoff.py --evaluation OUTPUT_DIR` generates the
comparison table, physical trace plot and timing/impulse diagnosis. Trace CSVs
include every physics step, body pitch/rate, whole-robot angular momentum,
COM motion, joint states/reference/residuals, front drift and all foot forces.
The evaluator checks checkpoint SHA256 values again after evaluation.

Directories `handoff_gap_v1` through `v4` are superseded development diagnostics
and should not be used as evidence for the final recovery basin.

Validation: 30 targeted tests pass across handoff, front-yield, reference and
balance tests. Ruff checks pass for the three new scripts and handoff tests.
Tests include physical target blending, gate debounce, withholding unverified
switches, five-second outcome accounting and independent state snapshots.
