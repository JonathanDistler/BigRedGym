# Go2 slip-permissive swing-up — 12 November 2026

The opt-in exploratory study removes front-foot slip and drift from acceptance and optimization while keeping the original full Go2 physics. It has **not established a complete standing-to-handstand transition**. Angle crossings, upright arrival, successful frozen-controller recovery, and a five-second hold are reported separately.

All prior strict implementations, checkpoints and output directories are preserved. New implementation: `experiments/recoverability/slip_permissive/`. No BC/PPO updates were run. Original leg scale is 1.00 throughout this study.

## Two observers, one physical trajectory

`DualMetrics` inherits the unchanged strict observer. Both observers see exactly the same 500 Hz integration. There is no separate physics run for strict versus exploratory scoring.

- **Strict:** original instantaneous loaded-contact slip-speed requirement (0.08 m/s) and maximum foot drift (0.04 m) remain in the balance predicate. Entry also retains the previous 0.02 m accumulated loaded-slip cap and 0.04 m drift cap. The 0.02 m entry cap is **not** applied to the whole five-second balance interval; that would incorrectly change the original balance criterion.
- **Exploratory:** cumulative loaded slip and drift are recorded but never reject entry or balance. The instantaneous loaded foot-slip-speed predicate is also removed from this observer. No cost fixes the front-foot x/y location.
- **Both:** original native joint bounds (0.001 rad numerical screening tolerance), motor-speed/torque screening, illegal ground and self-contact checks, gravity, contact/friction solver and original model. Entry retains the previous 5 mm head/calf planning margin, 80 ms longest bilateral load interruption, 120 ms cumulative interruption and 0.75 m/s landing-impact bound. A load interruption is not automatically classified as flight; geometric contact is logged separately.
- A five-second hold still requires bilateral front loading above 5% body weight per foot, tilt within 10°, body angular speed at most 1.5 rad/s, COM speed at most 0.35 m/s, rear clearance above 0.10 m and rear load below 3% body weight. It does not require stationary feet in exploratory mode.

Takeoff is additionally reported from real rear-foot clearance/load for 20 ms, without the old takeoff certificate's drift restriction. The old certificate is still recorded and unchanged. Crossings are only recorded before the first hard failure. A 45° crossing is not credited as a handstand.

A successful five-second prefix is latched at its actual achievement time. To compare both observers fairly, simulation can continue until the strict observer also reaches five seconds, a hard failure occurs, or 6.5 seconds elapse. A later failure during this optional continuation remains recorded; it does not erase the already executed valid five-second prefix. Success timestamps distinguish this case.

## Archived trajectories

`output/slip_permissive_archive_20261112_v3/` is the final paired balance-state analysis. Earlier v1/v2 observer-development outputs are retained, but v1 stopped at the exploratory hold before giving strict its full horizon, and v2 did not latch completed prefixes. They are not the final comparison.

The stored-summary audit covers 1,088 standing-start branches: 960 earlier search branches plus 64 contact-transition and 64 configuration-planning held-out branches. None establishes near-upright arrival or a complete handstand. Zero complete failures can be explained solely by accumulated slip/drift. Eight records have a collision-free diagnostic 45° crossing, which does not establish terminal recovery or contact-mode validity. Stored summaries are not supplemented with invented high-rate measurements.

The stronger replay check, `output/slip_permissive_exact_replay_20261112_v2/`, executes all 72 saved native action arrays from their original perturbed standing resets, and checks **every saved qpos and qvel**. Maximum state error is exactly 0.0; strict and exploratory full-handstand successes are both 0/72. The failed v1 replay used the evaluation perturbation amplitude for training-seed baselines; it is retained as a failed reproducibility check, and v2 restores the original 0.012 rad versus 0.018 rad distinction.

The 24 balance-start continuations comprise three original bank states and nine action-integrated predecessors, each tested deterministically and stochastically. They are explicitly labeled **balance-start**, never standing-to-handstand. These tests preserve full MuJoCo integration state, actual last target, original front-foot anchors and saved balance CPG phase, filtered loads, clock and previous action. Final paired results are in the measured table below.

## Slip-permissive full-motion search

`search.py` subclasses the existing multiple-shooting framework. Three structured full-motion initializations use the 61-node configuration path and the actual validated near-upright bank posture: standing-fast, kick-prefix-fast and standing-slow. Intermediate configurations are optimizer hypotheses only. The terminal target is approximately 90°, the actual balance-bank body height/joint shape, COM over current moving front support, and regulated velocity. Each uses six outer least-squares evaluations; the artifact records the larger actual residual-call count and physical-unit continuity defects.

The new residual removes original foot x/y anchors, slip budget, drift and friction-exposure costs. Body x/y is free to translate. It retains collision clearance, native limits, actual dynamics continuity and bounded transient unloading. Absolute targets are restricted to the intersection of joint limits and the original nominal ±2 rad action range. Targets are delivered through the original PD torque clipping, speed guard and target-slew interface. No forces are injected and no optimized floating-base node is projected into validation.

All three plans are separately replayed continuously from standing. Their hypotheses do not close into successful executable motions: actual replays collide with front calves and fail far below the target. This is optimizer/search failure, not proof that a sliding handstand is physically impossible.

`refine.py` supplies an additional uninterrupted single-shooting search over a 12-dimensional, three-knot residual to the frozen kick policy. It modifies front/rear thigh/calf commands during both launch and continuation. Initializations are neutral and the earlier collision-free ~47° training candidate; **all old anchoring coefficients are forced to zero**. Deterministic Powell optimization uses 80 actual simulator evaluations per initialization. The objective has exactly zero slip/drift weight; it rewards supported pitch progression, terminal pitch, COM placement and angular regulation while retaining collision, clearance, impact and unloading costs. This is a finite structured local search, not another random population search.

A final controlled front-yielding variant adds three compliance-gain variables, seeded from the selected native-action reference. Its separate cohort and measured outcome are reported below. Native action scaling and torque limits remain unchanged when compliance changes a desired joint target.

Each multiple-shooting plan and each native-action candidate gets 16 development rollouts (8 reset seeds × deterministic/stochastic), with development-only selection. Each selected method then gets 64 held-out rollouts (32 new seeds × deterministic/stochastic). The yielding extension uses disjoint new development and held-out seeds. Evaluation cohort sizes are equal. The native-action methods use a 1.6 s control horizon, the path continuation uses 1.8 s, and shooting durations are saved per plan; unsafe runs stop early. Optimizer budgets differ and do not support algorithm-efficiency comparisons. The compliance extension uses different random numbers.

## Exact-state balance handoff

On a loaded, hard-safe 85–95° arrival, runtime creates a scratch clone of the entire numerical integration state and the balance adapter memory, then compares the original frozen balance rollout with the live handoff. The live floating base is untouched. The adapter receives the actual previous target rather than a fabricated zero previous action. CPG initialization is explicit and identical in scratch and live; the kick policy itself has no CPG state. Prior bank continuations restore their saved CPG state instead.

No standing-start candidate in the final cohorts reached this arrival gate. Therefore this study does not claim a successful online handoff or policy fusion. Passing balance-start counterfactuals does not close the forward reachability gap.

## Measured results

The automatically generated tables, representative peak mechanics and artifact links follow. `summary.json` contains the full values, including separate deterministic/stochastic results. Raw traces include actual contact forces, friction utilization, momentum, motor torque and native targets. Playback uses actual simulated states, not IK or shooting-node animations.

| Standing-start method | Trials | Strict / exploratory 5 s handstands | 45° / 60° / 75° crossings | Max safe supported pitch | Whole-trial hard-safe | Collisions | Native-limit failures | Mean slip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh_baseline | 64 | 0 / 0 | 0 / 0 / 0 | 36.36° | 0 | 64 | 0 | 0.0639 m |
| multiple_shooting | 64 | 0 / 0 | 0 / 0 / 0 | 2.54° | 0 | 64 | 0 | 0.1087 m |
| native_refinement | 64 | 0 / 0 | 11 / 0 / 0 | 46.11° | 36 | 9 | 19 | 0.0769 m |
| yield_refinement | 64 | 0 / 0 | 3 / 0 / 0 | 46.55° | 23 | 4 | 37 | 0.1952 m |
| geometric_continuation | 64 | 0 / 0 | 57 / 4 / 0 | 62.35° | 0 | 64 | 0 | 0.0624 m |

Every standing-start cohort has **0/64 upright arrivals, 0 frozen-balance handoffs and 0 five-second handstands**. A safe supported peak is measured before the first failure; it does not certify the remainder of the trial. In particular, **all 64 geometric-continuation trials eventually collide with a front calf**, including the four 60° crossings. The largest peak in an entirely hard-safe native-action trial is 46.55° (yield variant), still subject to the retained contact-mode admission criteria. Native-limit failures in the native-action refinements are joint-limit excursions; actual torques remain within the native clipped effort limits.

| Selected native refinement | Trials | Whole-trial hard-safe | Collisions | Native-limit failures | Mean supported peak | Mean slip |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 32 | 30 | 0 | 2 | 44.76° | 0.0924 m |
| stochastic | 32 | 6 | 9 | 17 | 32.70° | 0.0614 m |

The balance-start comparison yields **5/24 strict and 6/24 exploratory five-second prefixes**. The one additional exploratory prefix is a stochastic integrated-predecessor continuation, not a new standing-start success; its later optional strict-observer continuation encounters a joint violation. The original bank deterministic continuations pass both observers when both receive their full comparison horizon.

### Mechanical interpretation

In the representative native-action held-out trial, the supported peak is 44.65° at 0.740 s. The COM is still **0.1280 m behind the moving front support**, while pitch rate is only **0.0188 rad/s** and whole-body pitch angular momentum is **0.00955 kg m²/s**. Front normal forces are 74.40/77.24 N and tangential forces are 19.90/20.35 N. Friction utilization is 0.268/0.263, far below saturation at this instant. Applied torque utilization is 0.238 of native effort. Head and front-calf clearances at this peak are 48.3/23.5 mm.

These observations support a specific limitation: this trajectory exhausts useful rotational progression while its COM remains behind support. Sliding permission helps it reach a larger angle, but does not automatically supply sustained angular impulse or put it into the balance basin. They do **not** prove that torque saturation or friction is never limiting elsewhere in the motion.

This particular full trajectory has 88 mm accumulated loaded slip, minimum head/calf clearances of 12.5/10.3 mm, 74 ms maximum bilateral-load interruption, 146 ms cumulative interruption, 52 ms maximum geometric front-foot flight and 0.160 m/s maximum measured touchdown vertical speed. It violates the retained 120 ms cumulative-load-interruption criterion even though it avoids illegal contacts and native limit failures. It therefore remains excluded as a validated reference. The final COM moving ahead of support occurs after the robot has dropped back to approximately 12°; it is not evidence of upright recoverability.

The path-continuation variant uses a feed-forward whole-body joint path with pitch feedback, interpolating its final 15° into the actual balance-bank joint shape. Seven bounded variables control onset, blend duration, front/rear blending, front-knee and front-thigh offsets, and pitch feedback. Sixty deterministic Powell evaluations are followed by 16 development and 64 disjoint held-out rollouts. It reaches larger supported angles, but front-calf collisions still disqualify every complete trial. This local search failure does not establish physical infeasibility.

The physically meaningful next search is event-based front-leg extension/body-height and rear-leg timing planning that carries angular impulse through 45–60° while retaining calf clearance and controlled recontact. The evidence does not support restoring a foot-position anchor or reducing the friction coefficient. More peak-pitch reward alone would favor already rejected colliding trajectories.

### Artifacts

Final figures and actual MuJoCo playback: `output/slip_permissive_report_20261112_v3/`:

- `pitch_COM_slip_clearance.png`: pitch, COM relative to **moving** front support, slip, drift and collision clearances.
- `forces_torques_momentum.png`: measured contact forces/friction, applied torques, native joint commands, COM trajectory and pitch momentum.
- `body_height_foot_translation.png`, `contact_modes.png`, `motor_envelopes.png`.
- `actual_native_refinement.gif`, `actual_yield_refinement.gif`, `actual_geometric_continuation.gif`, `actual_multiple_shooting.gif`: actual physics replays; none shows a complete handstand. The geometric-continuation playback is deliberately retained despite rejection.
- `summary.json`, `representative_peak_mechanics.json`, `preserved_inputs_sha256.json`.

Raw optimized action sequences and full-rate safety summaries are retained in the corresponding `slip_permissive_{refine,yield,path}_20261112_v1/` directories. `actions_0.json` and `actions_1.json` contain native normalized actions and motor targets; the CSV traces contain applied torques. Multiple-shooting commands, durations, continuity defects and node hypotheses are explicitly separated in `plan_*.json`.

`output/slip_permissive_physics_20261112_v1/physics.json` confirms identical native/saved-model simulator options, including gravity `[0,0,-9.81]`, 0.002 s timestep, solver 2, integrator 0, cone 0, 100 iterations, all original solver flags and contact parameters. The existing model audit reports zero differences in geometry, mass, inertia, friction, joint ranges and damping/armature. The old physical contact-transition core still matches its archived source hash. Kick action scale remains 2 rad; the frozen balance residual scale remains 0.25 rad.

## Learning decision

**No-go for launching the diagnostic smoke or pilot from this study.** This does not require a complete 90° demonstration: the diagnostic gate accepts repeatable, physically valid supported intermediate progression at 45° with retained non-slip contact conditions. Only **1/64** selected native-action held-out trajectories meets that admission gate; the yielding variant supplies **0/64**. The geometric-path variant has 0/64 hard-safe complete trajectories. Intermediate prefixes of trajectories that later collide or violate native limits are not exported as validated demonstrations.

`pilot.py` produces an opt-in configuration with a two-update smoke, a separately gated ten-update pilot, dense pitch/COM/angular/clearance/recoverable-arrival rewards, **zero slip/drift reward weight**, and no extended-PPO path. The generated `output/slip_permissive_pilot_gate_20261112_v1/bounded_pilot.json` is disabled. No demonstration dataset is represented as training-ready, and no BC/PPO update ran.

## Reproduction and Molab

From the extracted bundle root, install the pinned `slip_permissive_requirements.txt`, then run the following. In this workspace substitute `.venv/bin/python` for `python`. `env -u PYTHONPATH` avoids contamination by an unrelated system Python environment. Each output path must be new; runners refuse overwrites.

```bash
python -m pip install -r slip_permissive_requirements.txt
env -u PYTHONPATH python -m pytest experiments/recoverability -q
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.audit --output output/molab_sliding_physics
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.archive --output output/molab_sliding_archive
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.replay_archive --output output/molab_sliding_saved_replay
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.run --output output/molab_sliding_search --max-nfev 6
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.refine --output output/molab_sliding_refine --max-evaluations 80
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.refine --output output/molab_sliding_yield --max-evaluations 100 --warm-plan output/slip_permissive_refine_20261112_v1/selected_plan.json --front-yield
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.path_follow --output output/molab_sliding_path --max-evaluations 60
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.baseline --output output/molab_sliding_baseline
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.report --output output/molab_sliding_report
env -u PYTHONPATH python -m experiments.recoverability.slip_permissive.pilot --evaluation output/slip_permissive_refine_20261112_v1/heldout.json --output output/molab_sliding_pilot_gate
```

The report command reads the bundled measured directories; reruns under custom names remain separate. The warm-start/path commands intentionally reference the bundled selected native reference. To study a newly optimized reference instead, change the explicit warm-start path; `path_follow.py` records its bundled source reference in the source code.

Portable notebook: `notebooks/molab_slip_permissive.ipynb`. Archive: `output/slip_permissive_molab_20261112_v1.zip`. The bundle includes frozen checkpoints and saved-source encoders, original robot assets, required earlier results, optimized controls, raw evaluations, plots, playback and a per-file SHA-256 manifest. It contains no automatic learning launch.

Validation: **51 tests pass**, including original strict/controller/physics tests and new observer-only relaxation, unchanged native dynamics, exact-state clone, slip-independent objective, native action envelope, retained collision/contact/limit rejection, success-prefix accounting and learning-gate tests. Saved-action replay independently reproduces 72 original rollouts with zero state error. Bundle hash verification and extracted-source test results are recorded alongside the ZIP.
