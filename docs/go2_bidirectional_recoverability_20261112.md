# Bidirectional Recoverability Expansion for Unitree Go2

## Outcome

There is no validated overlap between the forward kick-up region and the frozen balance recovery region. The original policies and all previous results are preserved on `experiment/recoverability_morphology_v1`. A separate two-update **upright-balance** smoke test was permitted after compatibility/reset/metric verification; it regressed and **blocked the ten-update balance pilot**. No transition PPO or extended PPO ran. No complete standing-to-handstand sequence or morphology curriculum improvement is claimed.

Canonical outputs:

- `output/bidirectional_recoverability_20261112_v1/`: recovery maps, forward interventions, exact-state handoffs and paired morphology rollouts.
- `output/bidirectional_rates_20261112_v1/`: signed-rate action search and independent recovery tests.
- `output/bidirectional_model_verification_20261112_v2/`: physical model verification.
- `output/bidirectional_balance_smoke_20261112_v1/`: separate learned checkpoint, update statistics and held-out baseline comparison.
- `output/bidirectional_balance_pilot_prepared_20261112_v1/`: opt-in pilot configuration and the tested rejection of its failed smoke gate.

[Machine-readable summary](../output/bidirectional_recoverability_20261112_v1/report/summary.json) and [measured tables](../output/bidirectional_recoverability_20261112_v1/report/measured_report.md) accompany this report.

## Methods and physical validity

The saved balance checkpoint is `logs/go2balance/Oct31_23-29-27_/model_1500.pt`; the kick checkpoint is `output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt`. Both remain frozen throughout offline evaluation. The full 1.00 model passes the saved-controller model/control audit. Tests verify saved-source observation/action/PD/CPG parity and exact-state physical/observation restoration.

Each recovery reset begins with the existing front-foot IK proposal at the requested pitch and zero initial velocity, then undergoes 40–60 ms of **motor-driven MuJoCo integration**. The reported start state is the integrated result, not an edited velocity or interpolated pose. Contact forces, friction, collisions, mass, gravity and actuator constraints are never assigned or suppressed. These are locally physical reset proposals; they are explicitly **not claimed to be reachable from standing**. Rejected preparations would count as failures, rather than being silently replaced. All 220 preparations in the initial map passed the screen.

Eleven families cover nominal, paired pitch drives, roll drives, asymmetric front-foot contact, rear tuck/extension, and COM motion through front-leg commands. Training seeds are 202611120–121; held-out seeds are 202611520–521. Training motor offsets use approximately 0.025 rad for 40 ms; held-out offsets use approximately 0.040 rad for 60 ms, with independent jitter. Thus held-out conditions differ in both seeds and disturbance strength/duration. Actual rates and COM position/velocity are saved; intent labels are not treated as measured signs.

The 54-entry historical recovery bank has only 18 unique physical states. The new bank deduplicates pitch/joint/velocity state at 1e-9 precision, excluding time and including morphology identity; exact full integration state and controller memory are retained for replay. It contains 220 unique valid starts and 37 unique starts that recovered deterministically. CPG phase/load, previous action, applied target, front anchors, accumulated slip and prior safety failure state are preserved across branches. No empirical envelope is used to reject recovery tests.

All recovery conditions retain the original continuous five-second upright hold, contact loading, speed, COM velocity, drift, slip-speed, rear clearance, joint and collision requirements. Temporary unloading resets the hold timer and is measured as a contact interruption; it is not automatically an irreversible failure. True collision, joint-limit or actuator-state failures remain irreversible.

Dynamics and safety/slip integration run at 500 Hz. Force, friction, momentum, configuration and clearance trajectories are logged at 100 Hz. Whole-body and rear-leg angular momenta use body-COM Jacobians and world-frame inertias, and the total is checked against MuJoCo's subtree angular momentum.

## Phase 1: recovery maps at 70–90 degrees

The initial map has 440 trials: five pitch bands × eleven perturbation families × four seeds × deterministic/stochastic continuation. Each cell has two independent seeds per split, so plotted fractions and Wilson intervals are descriptive estimates with limited statistical power.

| Pitch proposal | Train deterministic | Held-out deterministic | Train stochastic | Held-out stochastic |
|---|---:|---:|---:|---:|
| 90 degrees | 20/22 | 17/22 | 0/22 | 0/22 |
| 85 degrees | 0/22 | 0/22 | 0/22 | 0/22 |
| 80 degrees | 0/22 | 0/22 | 0/22 | 0/22 |
| 75 degrees | 0/22 | 0/22 | 0/22 | 0/22 |
| 70 degrees | 0/22 | 0/22 | 0/22 | 0/22 |

Nominal 90-degree deterministic cases recover 4/4. Perturbed 90-degree deterministic cases recover 37/44 overall. The frozen checkpoint's native stochastic action noise fails all initial-map tests; deterministic reliability must not be generalized to stochastic robustness.

The first small motor pulses all produced positive pitch rates. To genuinely test both signs, a supplemental search uses 48 bounded sagittal motor pulse profiles per band (240 search branches), lasting 40, 80 or 120 ms with offsets up to 0.35 rad. Selection uses training seed 202617120, requires no physical failure, loaded slip ≤0.02 m, drift ≤0.04 m and resulting pitch within 3 degrees of the requested band. No velocities are edited. Both signs were attained at **every band**; independent seeds 202617520–521 with additional command jitter confirm the signs. Across 40 deterministic/stochastic supplemental recovery tests, none recovered. This extreme-rate supplement is kept separate from the small-disturbance map rather than pooled into a misleading probability estimate.

[Recovery probability maps](../output/bidirectional_recoverability_20261112_v1/report/recovery_probability_maps.png). Full raw outcomes include support fraction, contact-gap duration, loaded slip/drift, joint/actuator ratios, COM quantities, first physical failure and minimum calf clearance.

## Phase 3: contact-aware forward reachability

The frozen kick policy is evaluated with fresh train seeds 202612120–121 and held-out seeds 202612520–521. Every method has the same seeds, initial morphology, stochastic noise draws and 2.5 s maximum rollout cap. Seven motor-space interventions compare baseline, front-foot position feedback anchoring, compliant front yielding, early/late rear retraction, anchoring plus retraction, and yielding plus retraction. Joint targets respect the existing target-rate and motor limits. No foot pinning or body forces are applied.

Held-out means over two seeds × deterministic/stochastic action modes:

| Intervention | Rear takeoffs | Calf/other physical collisions | Loaded front slip (m) | Safe supported peak (degrees) |
|---|---:|---:|---:|---:|
| Baseline | 4/4 | 4/4 | 0.0636 | 32.40 |
| Anchoring | 0/4 | 4/4 | 0.0180 | 28.85 |
| Compliant yielding | 2/4 | 4/4 | 0.0279 | 27.83 |
| Early rear retraction | 4/4 | 4/4 | 0.1027 | 38.20 |
| Late rear retraction | 4/4 | 4/4 | 0.2208 | 32.50 |
| Anchoring + retraction | 0/4 | 4/4 | 0.0180 | 28.85 |
| Yielding + retraction | 2/4 | 4/4 | 0.0259 | 31.08 |

The interventions do not solve contact feasibility. Anchoring cuts slip but removes certified takeoff; yielding cuts slip with lower rotation and reduced takeoff reliability. Early retraction increases rotation while worsening slip. The identical anchoring/anchoring+retraction outcomes mean the delayed rear intervention did not become useful before those branches terminated. All methods ultimately collide; none is admitted as a transition demonstration.

A representative held-out stochastic baseline shows the pre-peak process:

| Time | Pitch | Cumulative loaded slip | FL normal / tangent force | COM x from front midpoint | Whole-body / rear pitch momentum |
|---|---:|---:|---:|---:|---:|
| 0.30 s | 0.22 degrees | 0.0063 m | 28.3 / 9.4 N | −0.2149 m | −0.026 / −0.184 kg m²/s |
| 0.50 s | 1.52 degrees | 0.0093 m | 34.4 / 4.2 N | −0.2205 m | 0.328 / 0.070 kg m²/s |
| 0.70 s | 21.83 degrees | 0.0207 m | 66.7 / 39.7 N | −0.1714 m | 0.859 / 0.597 kg m²/s |
| 0.85 s | 30.23 degrees | 0.0344 m | 92.6 / 45.6 N | −0.1874 m | 0.142 / 0.149 kg m²/s |

Pitch peaks at approximately 0.86 s, after substantial loaded slip has already accumulated. COM velocity reverses from approximately +0.40 m/s at 0.70 s to −0.51 m/s at 0.85 s; forward rotation loses momentum while COM remains behind support. FL/FR thigh angles move from approximately 0.80 rad at 0.70 s to 0.55 rad at 0.85 s. Anchoring reduces tangential demand and slip but interferes with takeoff/rotation. This supports investigating coordinated contact geometry and momentum timing, not maximizing pitch.

The sampled FL friction-utilization maximum before peak is 0.746 for this baseline; it reaches 1.0 for early retraction. Baseline pre-peak requested torque is below 0.64 of the actuator limit. These measurements do **not** establish torque clipping or exceeding the friction cone as the sole cause of slip: contact compliance, changing joint targets/normal loads and unsampled force extrema can matter. The initial brief slip at 0.01 s is settling contact; the important additional loaded slip occurs during the later kick-up progression.

[Before/after normal/tangential force, friction, slip, COM, momentum and clearance plots](../output/bidirectional_recoverability_20261112_v1/report/front_contact_before_after.png). [Front-joint configuration and torque plots](../output/bidirectional_recoverability_20261112_v1/report/configuration_rotation_torque.png).

## Phase 4: forward/backward gap and exact handoffs

The best safe control-sampled forward state is retained from every original-morphology trial with no prior physical failure, loaded slip ≤0.02 m and drift ≤0.04 m. **All 56 candidates** receive exact-state deterministic balance continuations, regardless of empirical envelope membership. **0/56 recover.** These are tested candidates, not an exhaustive reachability certificate.

| Physical variable | Safe forward candidates | Deterministically recovered start states |
|---|---:|---:|
| Pitch | 19.28–35.30 degrees | 90.02–91.10 degrees |
| COM x relative to front support | −0.1819 to −0.1158 m | −0.00309 to −0.00139 m |
| Pitch velocity | 0.252–2.188 rad/s | 0.259–0.495 rad/s |

The marginal pitch separation is at least **54.72 degrees** and the marginal COM x separation is at least **0.1127 m**. These minima need not belong to the same pair of states. Angular-rate ranges overlap, so rate matching alone does not establish a bridge. Leg configuration, contact continuity, calf clearance and the unbridged pitch/COM geometry still prevent tested handoffs. No normalized L-infinity threshold is used as a sufficient or necessary recovery condition.

[Forward/backward state-region plots](../output/bidirectional_recoverability_20261112_v1/report/forward_backward_regions.png). `handoff_candidates.json`, `forward_candidates.pt` and `bridge_gate.json` retain features, exact branch states, outcomes and the closed transition gate. With no overlap, no fused standing-start sequence or transition PPO is launched.

## Phase 5: morphology feasibility and trade-offs

The existing constant-density axial leg-scaling implementation generates scales 0.80, 0.85, 0.90, 0.95 and 1.00. Independent verification checks topology, collision masks/friction, unchanged torso mass/inertia and foot geometry, positive inertias for massive bodies, inertia triangle inequalities, finite COM/mass, native joint limits and actuator effort/speed constraints. Massless sensor frames have no joints and correctly have zero inertia; the initial blanket positive-inertia check mistakenly included them. Its failed audit is preserved in `bidirectional_model_verification_20261112_v1`; corrected physical verification is in `v2` and passes every scale. All dynamic success/safety conditions are unchanged.

| Leg scale | Total model mass (kg) | Held-out baseline slip (m) | Supported peak (degrees) | Held-out recovery successes / cases |
|---|---:|---:|---:|---:|
| 0.80 | 15.194 | 0.0982 | 21.14 | 0/8 |
| 0.85 | 15.464 | 0.0923 | 23.72 | 0/8 |
| 0.90 | 15.735 | 0.0605 | 26.60 | 0/8 |
| 0.95 | 16.006 | 0.0597 | 29.50 | 0/8 |
| 1.00 | 16.276 | 0.0636 | 32.40 | 2/8 |

Forward morphology comparisons match baseline/yield interventions and deterministic/stochastic budgets at all five scales (120 total forward trials, including the original seven-method study). Recovery morphology probes match nominal 90/85-degree starts across both seed splits and action modes (80 trials). The recovery denominator includes both angles and action modes; original deterministic 90-degree recovery is 2/2 in the held-out subset. Every morphology still encounters collision in the forward tests. Small slip reductions at 0.90/0.95 trade against rotation and zero demonstrated recovery; no multi-objective advantage justifies a morphology schedule. Altered-model transfer failures do not prove those models are intrinsically harder, since both frozen policies were trained for the original morphology. Comparative morphology learning difficulty was not measured.

[Morphology trade-off plots](../output/bidirectional_recoverability_20261112_v1/report/morphology_tradeoffs.png). Geometry, COM offsets, masses, inertias, limits and per-trial contact/collision outcomes are retained. Morphology scheduling remains disabled; all learned-policy final evaluations use scale 1.00.

## Phase 2: bounded balance fine-tuning, curriculum and gates

Compatibility/reset/metric tests passed and nominal deterministic balance recovered 4/4 before the two-update smoke. The fine-tuning implementation clones the saved actor, freezes its observation-normalization buffers and preserves its unbounded residual-action/CPG/100 Hz controller interface. The baseline actor remains frozen and hash-checked. A fresh critic and Gaussian exploration standard deviation 0.03 initialize training. This exploration is deliberately smaller than the saved checkpoint's native evaluation noise; deterministic initial means are identical. Final stochastic comparisons use the same native saved-policy noise for baseline and clone, not an implied improvement from reducing evaluation noise.

Rewards cover stable loaded support, upright balance, angular-rate regulation, COM positioning, five-second hold, loaded slip and actuator effort. Original physics safety failures terminate the episode. Temporary unloading loses support reward/resets the hold but does not terminate the episode. Ablations are `none`, `no_slip`, `no_com` and `fixed_band`.

The adaptive band schedule is 90 → 85 → 80 → 75 degrees. It requires 20 distinct validation seeds, ≥80% recovery, no physical failures and slip ≤0.02 m before expanding. Four new validation seeds per update are separate from training and final evaluation seeds. After expansion, sampling retains 20% 90-degree states, 60% previous-band states and 20% new-band proposals. Initial 90-degree training starts must already have empirical deterministic recovery. All lower-band candidates must be valid integrated reset states. The two-update smoke did not advance the schedule.

The smoke used exactly **2 × 256 control samples**, capped at two PPO iterations, with actor learning rate 1e-5 and strict KL stopping. Only one minibatch optimizer step occurred in each iteration before KL stopping. Validation was **0/4 after each update**. On independent final deterministic 90-degree starts, the frozen baseline recovered **2/2**, while the learned clone recovered **0/2**. All lower-angle final tests also failed. The observed smoke validation failures exceeded the original 0.001 rad joint-limit tolerance, despite low speed and applied torque ratios. The baseline hash remained unchanged.

The learned checkpoint is retained as a failed smoke result. `smoke_gate.json` has `pilot_eligible=false`; the actual ten-update command was checked and rejects this gate before creating an output directory or training. No automatic rollback can be reported as learning improvement. A more conservative constrained/anchored update method would need separate validation; there is no evidence here to launch this ten-update pilot or transition PPO.

## Validation, budgets and reproducibility

- 25 unit/integration tests cover exact reset replay and controller memory, angular momentum against MuJoCo, action interventions without physical-state edits, baseline/clone parity and frozen normalization, adaptive gate independence, unloaded-contact handling, physical model scaling and existing safety/adapter invariants.
- Full model/control compatibility, original-policy hashes and per-source/checkpoint hashes are saved. Training, disturbance-search, held-out map, forward, curriculum-validation and final fine-tuning seeds are separated explicitly in source/results.
- Balance branches share a 6.5 s cap; forward interventions share a 2.5 s cap. Physical failures stop rollouts and successful balance stops at five seconds. Realized step counts and continuation durations are saved; this is matched **maximum** interaction budget, not a claim that early-failing branches consumed equal interactions. Preparation counters are retained separately in snapshot history; derive continuation steps from continuation duration/0.002 when reading the first-run schema.
- Each output directory and ZIP is created fresh. Existing experiment files and checkpoints are not overwritten. The current source adds only schema/gate fixes that do not alter completed offline trajectories or the non-advancing smoke.

## Exact local and Molab commands

Upload [the notebook](../notebooks/molab_bidirectional_recoverability.ipynb) and `output/bidirectional_recoverability_molab_20261112_v1.zip` to Molab using Python 3.11. Its setup extracts a fresh workspace, verifies all SHA-256 entries and installs the recorded package versions. Frozen validation runs by default; balance smoke/pilot cells are separately opt-in. The local pilot gate is closed.

From the repository root, the module commands are:

```bash
python -m pytest -q experiments/recoverability/test_core.py experiments/recoverability/test_transition.py experiments/recoverability/bidirectional/test_bidirectional.py
python -m experiments.recoverability.bidirectional.verify_models --output output/molab_bidir_models
python -m experiments.recoverability.bidirectional.run all --output output/molab_bidir_validation --seeds-per-split 2
python -m experiments.recoverability.bidirectional.rate_coverage --output output/molab_bidir_rates --budget 48
python -m experiments.recoverability.bidirectional.fine_tune prepare --validation output/molab_bidir_validation --output output/molab_bidir_balance_prepared
python -m experiments.recoverability.bidirectional.report --results output/molab_bidir_validation --rates output/molab_bidir_rates
```

Optional two-update **balance-only** reproduction:

```bash
python -m experiments.recoverability.bidirectional.fine_tune smoke --validation output/molab_bidir_validation --output output/molab_bidir_balance_smoke
```

The separately gated ten-update command is provided for reproducibility; **it must reject the measured failed smoke**:

```bash
python -m experiments.recoverability.bidirectional.fine_tune pilot --validation output/molab_bidir_validation --smoke-gate output/molab_bidir_balance_smoke/smoke_gate.json --output output/molab_bidir_balance_pilot
```

Local commands use `env -u PYTHONPATH .venv/bin/python` instead of `python` because the shell injects an incompatible Python 3.12 package path. The portable runner does not depend on that local environment. Export a new bundle with `python scripts/package_bidirectional_recoverability.py --output output/NEW_NAME.zip` and optional repeatable `--results OUTPUT_DIRECTORY` arguments. The packager refuses to overwrite an existing ZIP.
