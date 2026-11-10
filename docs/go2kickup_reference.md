# Reference-only MuJoCo kick-up

The subsequent [bounded-residual integration and pilot](go2frontyield_pilot.md)
uses this reference in a separate PPO task. Results below are reference-only.

The new opt-in reference passes both 10 mm and 30 mm rear takeoff checks in
**16/16 fresh standing starts**. Both rear feet remain above 30 mm with front
support for at least **230 ms** in every trial. Maximum front-foot drift is
**2.35–3.56 cm** (mean 2.69 cm). No PPO training was run.

The existing `Go2Kickup` controller, its curriculum implementation, historical
saved configurations, and all checkpoints are unchanged. The experimental
trajectory is in `gym/envs/go2/kickup_reference.py`; selected parameters are in
`gym/envs/go2/references/kickup_front_yield.json`. Playback and evaluation use
`scripts/search_kickup_reference.py`, with zero policy residual and no actor.
It does not change the normal training task's reference by default.

## Comparison

These are new matched comparisons, with more varied starting conditions than
the preceding diagnosis. The baseline exactly uses the original stage-3
reference (amplitude 0.85, 0.15/0.55/0.20 s phases, front gain fraction 0.65,
velocity feedforward, no policy residual). Both references use identical native
MuJoCo models, initial states, randomized gains, mass and friction for each seed.
Each trial runs for 1.6 seconds with no reset and no kinematic foot constraints.

| Metric, 16 trials | Stage-3 reference | Improved reference |
| --- | ---: | ---: |
| Valid 10 mm takeoff | 0/16 | 16/16 |
| Valid 30 mm takeoff | 0/16 | 16/16 |
| Maximum simultaneous rear clearance, range | 1.7–2.0 mm | 87.1–138.8 mm |
| Mean maximum front drift | 16.68 cm | 2.69 cm |
| Worst maximum front drift | 18.53 cm | 3.56 cm |
| Mean peak absolute pitch rate | 3.16 rad/s | 2.87 rad/s |
| Mean peak absolute pitch angular momentum about COM | 1.24 N m s | 0.98 N m s |
| Rear normal impulse during extension, L/R | 16.05 / 15.61 N s | 5.27 / 4.94 N s |
| Rear excess impulse above 0.25 body weight/foot, L/R | 8.87 / 8.53 N s | 1.25 / 1.05 N s |
| Mean both-front-load retention after settling | 81.1% | 99.2% |
| Longest front-load gap | 256 ms | 4 ms |
| Nonfoot-ground-contact trials | 16/16 | 0/16 |
| Rear knee tracking RMSE during push, L/R | 0.254 / 0.250 rad | 0.029 / 0.027 rad |
| Peak rear knee torque, L/R | 28.16 / 27.10 N m | 11.53 / 11.16 N m |

The stricter contact audit also exposes a problem hidden by foot-only force
logging: in baseline seed 3000, both rear calves contact the ground from
0.572 s, during crouching. All 16 baseline trials have nonfoot-ground contact.
These contacts change the support geometry without necessarily triggering the
task's torso-only termination checks. The shallower, coordinated new loading
pose avoids nonfoot ground contact in all 16 trials.

The improved reference does **not** work by maximizing impulse, absolute angular
momentum, or torque. The baseline generates larger values but drives the body
backward during loading, then loses front support and slips. The new reference
moves the COM forward before launch, directs the push into nose-down rotation,
and retracts the rear legs after takeoff. Whole-trial peak absolute pitch rates
and angular momenta include unsuccessful motion; they are not standalone
success measures.

For seed 3000, the improved COM x moves from -0.013 m at 0.20 s to +0.061 m just
before push. COP x moves to +0.157 m at 0.90 s, near the front feet at +0.174 m;
the COM is +0.084 m and body nose-down pitch is 0.385 rad at that instant.
The baseline COM instead moves to -0.177 m and COP to -0.244 m, with the front
feet sliding backward. The CSVs include COM/COP positions, pitch angular
momentum, and separate front/rear contact pitch moments about COM.
COP is computed from actual world contact positions weighted by positive
vertical ground force, not from foot centers or a desired load distribution.

![Matched baseline and improved reference traces](../output/reference_validation_final/comparison.png)

## Trajectory and controller

Joint interpolation uses cubic smoothstep with analytic velocity. Left and
right legs share timing. Sagittal hip angles below refer to the URDF's thigh
joint; abduction targets remain zero.

| Parameter | Selected value |
| --- | --- |
| Settling | 0.200 s |
| Loading/crouch duration | 0.577 s |
| Nominal extension duration | 0.190 s |
| Rear hip crouch → extension | 1.090 → 0.435 rad |
| Rear knee crouch → extension | -2.058 → -1.011 rad |
| Hip push start / end | 0.779 / 0.947 s |
| Knee push start / end | 0.786 / 0.978 s |
| Front hip loading → push posture | 1.500 → 0.932 rad |
| Front knee loading → push posture | -1.855 → -1.750 rad |
| Front stiffness multiplier | 0.481 (nominal 28.88 N m/rad) |
| Front damping multiplier | 0.912 (nominal 2.735 N m s/rad) |
| Velocity feedforward multiplier | 1.0 |
| Rear retraction start / duration | 1.011 s / 0.202 s |
| Rear retracted hip / knee | 1.354 / -2.017 rad |

The torque law is position PD plus reference velocity feedforward. Rear gains
remain 60/3. Joint targets and velocity feedforward are clipped to the task's
URDF limits; torques retain its 23.7 N m hip/thigh and 45.43 N m calf clips.
Actual velocity is never clamped. No motor saturates in the selected 16 trials;
maximum actual joint velocity is 57.8% of its URDF limit. Front knee tracking
errors of about 0.38–0.39 rad reflect the intended compliance under load, while
rear knees track closely. The reference has ample torque headroom.

## Front-controller and trajectory ablations

Every ablation uses the same 16 validation seeds and the same selected rear
trajectory unless explicitly disabled. Success requires no nonfoot ground
contact over the full trial and no more than 4 cm maximum front drift.

| Ablation | Valid 30 mm trials | Mean maximum front drift | Interpretation |
| --- | ---: | ---: | --- |
| Selected reference | 16/16 | 2.69 cm | Coordinated loading, yielding, push and retraction |
| Front gain fraction 1.0 | 0/16 | 4.32 cm | Excess rotation/overshoot; all trials have nonfoot contact |
| Front gain fraction 0.1 | 0/16 | 2.20 cm | Insufficient support; all trials have nonfoot contact |
| Front targets fixed at original standing pose | 0/16 | 10.87 cm | Opposes the coordinated forward shift; no useful rear clearance |
| No velocity feedforward | 16/16 | 2.38 cm | Still works; mean supported peak falls from 109 to 83 mm |
| No post-push rear retraction | 0/16 | 2.48 cm | Brief real release remains, but only 23–39 mm peak clearance; 14 trials later have nonfoot contact |

Thus front posture coordination is essential here. Neither making the front
controller arbitrarily soft nor adding feedforward alone solves the problem.
Retraction matters for clearance and the subsequent supported motion, but it
is **not the cause of initial takeoff**: both rear feet cross 10 mm at
0.946–0.958 s, before retraction starts at 1.011 s. At that crossing both feet
are rising at 0.387–0.470 m/s, have previously carried load, and exert less than
0.015 robot body weights each while both front feet carry at least 0.06 each.

## Search and physical validation

The constrained search evaluated 240 candidates on seeds 100/101, with broad
random samples followed by local perturbations of the incumbent. It varied
crouch depth, front loading and push poses, extension duration, hip/knee delay
and duration, front stiffness/damping, and rear retraction. Joint and torque
limits were enforced throughout. The score rewards supported minimum rear
clearance and penalizes drift beyond 3.5 cm, sustained front-load loss, and
nonfoot-ground contact. The highest two-seed candidate proved too aggressive:
only 13/16 passed the first validation (seeds 1000–1015), so it was rejected.

A further 130-candidate local search used six different seeds (200–205) and a
worst-case objective capped at 60 mm, penalizing drift beyond 3 cm, front-load
gaps beyond 20 ms, nonfoot contact and excess joint speed. It retained the less
aggressive candidate 35 from the broad search. The final 16 validation seeds
3000–3015 were not used by either search. Trials vary:

- Joint positions: independent Gaussian noise, standard deviation 0.008 rad.
- Initial roll/pitch: ±0.012 rad; yaw: ±0.15 rad.
- Initial standing height: adjusted so the lowest foot has 1 mm clearance.
- Total mass/inertia: ±5% around the task's fixed seeded startup-randomized model.
- Sliding friction: uniform 0.6–1.0.
- Individual proportional and derivative gains: ±5%.

A valid trial needs both rear sphere surfaces at least 10 mm or 30 mm above the
world plane for 20 ms continuously, after previous rear loading, with both rear
forces below 0.015 body weights, both front forces above 0.06 body weights,
front drift at most 4 cm, and no nonfoot ground contact during the full trial.
The 30 mm holds actually last at least 230 ms. Clearance is sphere-center z
minus the URDF sphere radius. A detector transition or zero force with a foot
still on the ground cannot pass. Independent finite differences of the recorded
clearances confirm upward foot motion at the 10 mm crossing in all 16 trials.

The simulator is the existing task's CPU MuJoCo model, stepped directly at
500 Hz to make the parameter search inexpensive. Contact friction, soft-contact
settings (`solref = [0.02, 1.0]`), geometry and integration configuration are
retained. The harness applies bounded motor torques through `qfrc_applied`, as
the existing CPU backend does. It never welds, teleports or pins feet during a
trial. `provenance.json` records the effective configuration, model masses,
inertias, joint ordering and limits, timestep and URDF hash. No checkpoint is
loaded, rewritten, or used to initialize the reference.

These tests validate standing-start takeoff on the modeled flat plane with
mild variations. They do not validate hardware execution, arbitrary terrain,
long-term handstand balance, or policy handoff after the 1.6-second window.

## Reproduction and playback

Reference-only playback, repeating ten standing starts:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/search_kickup_reference.py \
  --parameters gym/envs/go2/references/kickup_front_yield.json \
  --playback-only --viewer --repeats 10 \
  --output-dir output/reference_playback
```

Independent evaluation and plots:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/search_kickup_reference.py \
  --parameters gym/envs/go2/references/kickup_front_yield.json \
  --validation-seed 3000 --trials 16 --output-dir output/reference_validation_repeat

env -u PYTHONPATH MPLCONFIGDIR=/tmp/kickup-mpl XDG_CACHE_HOME=/tmp/kickup-cache \
  .venv/bin/python scripts/report_kickup_reference.py output/reference_validation_repeat

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/unit_tests/test_kickup_reference.py tests/unit_tests/test_go2_kickup.py \
  tests/unit_tests/test_kickup_cpg.py tests/unit_tests/test_play_inference.py -q
```

The 30 tests pass, including analytic velocity checks, rejection of force-only
false liftoff, and a CPU MuJoCo regression requiring supported 30 mm clearance,
no body contact and motor/velocity limits. Headless playback execution also
passes. The graphical viewer command is provided above; the measurements and
plots were generated headlessly.

Raw broad-search, robustness-search and final trial artifacts are under
`output/reference_search/`, `output/reference_refinement/` and
`output/reference_validation_final/`. The final folder includes `summary.json`,
`validation.json`, `provenance.json`, `reference_parameters.json`,
`liftoff_events.json`, and per-trial 500 Hz CSVs for all seven ablations.
Searches are reproducible with `--candidates 240 --search-seed 482`; robust
refinement accepts `--robust --search-seed 9127 --candidates 130
--initial-parameters <candidate-35.json> <candidate-39.json>`. Those candidate
parameter dictionaries are preserved in the broad search's `search.json`.

## Next PPO step: retain the existing policy

Prefer **fine-tuning the existing actor**, with fresh optimizer state and a new
run, rather than discarding its useful front stabilization. First integrate this
opt-in reference into a separate training configuration and require zero-action
MuJoCo parity with these traces. Keep old runs and configs untouched. The old
actor's unbounded outputs must not be allowed to cancel the new trajectory.

Proposed curriculum, not implemented or trained in this change:

1. Verify the reference baseline, then begin with rear residuals zero through
   loading/push and front residuals bounded to ±0.01 rad. Unlock small rear
   residuals after retraction. Blend phase gates smoothly to avoid torque jumps.
2. After at least 90% supported 30 mm success over two independent batches of
   at least 32 completed episodes, allow rear ±0.02 rad and front ±0.03 rad.
3. Expand rear bounds to ±0.05 and ±0.10 rad only when the same physical success
   rate and motor margins persist. During extension, constrain residual rate so
   at least 80% of the reference's directed hip/knee motion is preserved.
4. Keep both front contact requirements throughout. Start drift allowance at
   4 cm, then tighten to 3.5 and 3 cm after reliable takeoff. Regress the bounds
   if physical takeoff or front support deteriorates.

Validate a bounded warm start against the reference before any training, then
use short pilot fine-tuning and held-out evaluation to test whether residuals
help. A new residual actor is a fallback if the warm start remains harmful
even within those bounds. No 1,500-iteration run is recommended or launched by
this work; the reference validation now supplies the prerequisite for a small,
controlled follow-up experiment.
