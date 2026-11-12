# Go2 contact-mechanics transition study: 30–35 to 45–55 degrees

## Outcome

**No repeatable reference passed the physical/contact acceptance gate. No BC or PPO ran.** The study identifies more specific mechanical obstacles and produces launch/continuation action sequences for inspection, not valid training demonstrations. All earlier experiments and checkpoints remain preserved on `experiment/recoverability_morphology_v1`.

The optimizer can reproducibly increase **pre-collision, loaded-support progression** into the requested angle band. The best maintained-support/no-yield candidate averages 54.41 degrees on eight fresh deterministic resets and reduces loaded slip relative to the baseline, but every rollout subsequently contacts the front calf. Front yielding improves COM position but instead exposes lower-head ground contact. Higher pitch therefore does not certify a usable transition.

Canonical artifacts:

- `output/contact_transition_search_20261112_v1/`: 480 CEM candidate evaluations, 960 paired-training-reset simulator branches, spline parameters and optimization history.
- `output/contact_transition_validation_20261112_v2/`: 64 independent original-morphology validations, conservative takeoff-to-end contact-gap measurement, action/observation/state sequences and closed reference gate.
- `output/contact_transition_validation_20261112_v1/`: completed 64 mechanism ablations and 80 morphology comparisons.
- `output/contact_transition_mechanics_20261112_v1/`: 28 matched terminal-configuration probes and actual contact-moment measurements.
- `output/contact_transition_extension_20261112_v1/`: 54 branches testing front extension and launch timing.
- `output/contact_transition_catch_20261112_v1/`: 72 catch-refinement search branches plus eight independent validations.
- `output/contact_transition_report_20261112_v2/`: [measured tables](../output/contact_transition_report_20261112_v2/measured_report.md), [machine-readable summary](../output/contact_transition_report_20261112_v2/summary.json), figures and reconstructed geometric clearances.

## Physics and precommitted reference conditions

Every candidate begins standing in the full MuJoCo Go2. The existing native standing reset is perturbed by **40 ms of simulated joint-motor commands**, with per-joint target disturbances up to 0.012 rad for optimization and 0.018 rad for held-out evaluation. Preparation slip and the original foot anchors remain in the safety history. No intermediate pose/velocity reset, contact assignment, foot pinning, gravity change or external body force is used.

The frozen kick checkpoint remains `output/rotation_pilot_20261111_v1/global/A_seed42/checkpoint_000036.pt`. The source controller's 50 Hz target interface, target slew limit, PD gains, native joint limits, torque clipping and speed guards are preserved. Physics and safety/slip monitoring run at 500 Hz; diagnostic forces/momentum/torques are logged at 100 Hz. Original model/scaling audits pass, including collision masks, friction, mass/inertial consistency, joint limits and actuator constraints.

The acceptance conditions are intentionally stronger than crossing an angle once:

| Condition | Required value |
|---|---|
| Transition target | Loaded front support in the 45–55 degree band for at least 0.10 s |
| Complete reference | No illegal/self collision or joint/speed/torque violation through the 1.6 s launch/continuation horizon; terminal pitch remains 45–55 degrees with front support |
| Rear takeoff | Original rear-leg push-off certificate; rear clearance >0.015 m and rear load <3% body weight during the goal interval |
| Loaded front slip | ≤0.020 m, including preparation |
| Front-foot drift | ≤0.040 m from original anchors |
| COM placement | x from front-foot midpoint ≥−0.160 m and at least 0.020 m closer than at the first certified state at/above 30 degrees |
| Goal motion | Body y angular speed ≤2 rad/s, body x angular speed ≤1 rad/s, COM speed ≤0.75 m/s |
| Native calf-body clearance | ≥0.005 m from the ≥30-degree certified transition onward |
| Actuators | Actual torque ≤native limit, speed ratio ≤1.001, joint violation <0.001 rad; additional requested-torque ratio gate ≤1.5 |
| Maintained support | Bilateral loaded-support interruption ≤0.004 s |
| Transient unloading | Longest interruption ≤0.080 s, cumulative interruption ≤0.120 s, restored loaded support and measured vertical foot velocity at reacquisition ≤0.75 m/s |

Bilateral support means each front foot carries more than 5% body weight; an interruption can represent unilateral unloading rather than full flight. Final validation counts interruptions from **certified rear takeoff**, including events before 30 degrees. An unloading event alone is not a physical failure. Successful reloading and all other conditions remain mandatory. Reacquisition velocity is the measured contact-state velocity, not a separately certified pre-impact speed; no trajectory was accepted on this basis.

The original calf-body metric is retained in the gate. The report additionally reconstructs **all front-calf geometry including fixed lower-calf children**, head clearance and base height from saved full qpos/qvel. Those purely geometric reconstructions are not used to invent contact forces. All-body collision rejection always comes from the live simulator.

## 1. Why the existing kick terminates poorly

The baseline establishes a useful rear push-off but does not leave a favorable state for continued rotation. Representative fresh deterministic measurements:

| Rising pitch | COM x relative to front support | COM x velocity | Whole-body / rear-leg pitch angular momentum | Loaded slip |
|---|---:|---:|---:|---:|
| 24.95 degrees | −0.1646 m | +0.3899 m/s | 0.938 / 0.655 kg m²/s | 0.0167 m |
| 29.87 degrees | −0.1563 m | +0.0230 m/s | 0.661 / 0.513 kg m²/s | 0.0210 m |
| 33.97 degrees | −0.1781 m | −0.4647 m/s | 0.315 / 0.299 kg m²/s | 0.0267 m |

At approximately 30 degrees, almost the entire slip allowance is already spent. Whole-body rotational momentum falls by about two thirds between 25 and 34 degrees. Rear-leg momentum accounts for much of the remaining momentum; COM translation reverses while the COM is still behind support.

Front thigh configuration changes rapidly over this interval: the representative FL thigh moves from about 0.79 rad at 25 degrees to 0.49 rad at 34 degrees, while its calf stays deeply folded near −2.25 rad. The combination shrinks usable body-height/clearance margin and does not carry the COM forward to the recoverable region. This is a poor **combined contact, body-height and momentum configuration**, rather than simply insufficient peak pitch.

Actual signed contact wrenches support this explanation. In a separate matched baseline near 34 degrees, front ground force is approximately −84 N in x and +168 N vertically, front-contact pitch moment about the COM is approximately −4.8 N m, and the gravitational pitch moment about the front-foot midpoint is about −28 N m. These quantities use different moment origins and must not be added directly. The braking front wrench and COM-behind-support geometry explain why adding rear retraction alone is not enough. Whole-body/rear momentum is computed with world-frame inertias and body-COM Jacobians; the underlying telemetry has been checked against MuJoCo subtree momentum.

Friction utilization and torque saturation are **not sufficient single explanations**. The representative rising baseline has FL friction utilization about 0.51 near 30 degrees and 0.46 near 34 degrees, although substantial loaded slip exists. Finite contact compliance, changing normal loads and moving joint targets matter, and 100 Hz force logs can miss extrema. Some initial settling slip is also present. The optimizer's rejected high-pitch collisions often occur at low applied torque and joint-speed ratios, so these failures are not evidence that native actuator strength alone is insufficient.

[Momentum and contact-moment plots](../output/contact_transition_report_20261112_v2/momentum_contact_moments.png).

## 2. Controlled probes of the terminal policy configuration

Seven motor-space conditions were tested on four fresh matched standing resets: unchanged policy, hold all targets after certified 30 degrees, hold front targets only, rear counterrotation, rear retraction, front folding and front extension. All preserve the physical state and native controller.

Holding all targets avoids collision through the 1.6 s horizon in all four probes, but peaks near 32.6 degrees and still accumulates roughly 42.7 mm of loaded slip. Holding only the front targets also stalls near 32.5 degrees and still collides. Front folding reaches approximately 42.4 degrees but accumulates about 45 mm slip and fails physical safety. Rear counterrotation/retraction do not create the requested clean transition.

These interventions show that continuing the original policy's terminal target sequence contributes to the later collision. They also show that freezing its terminal configuration does not solve rotation or contact mechanics. Avoiding collision by stalling or returning to rear support is not an accepted transition.

## 3. Contact-aware launch and continuation optimization

The new runner optimizes **both launch and continuation** with seven time knots at 0.15, 0.35, 0.55, 0.75, 0.95, 1.15 and 1.40 s. Each knot controls:

- Front thigh/calf target offsets.
- Rear thigh/calf target offsets, allowing changes in retraction/extension timing.
- Front yielding gain, with load-aware and no-yield comparisons.
- Soft front-foot position feedback through joint targets, not an imposed contact constraint.
- Mirrored hip spread as a body-configuration variable.

Commands interpolate continuously between knots and retain native target bounds/rate limiting. There are two separately searched contact modes: maintained support and bounded transient unloading. CEM uses 24 candidates × five iterations × two front-control strategies × two contact modes. Every candidate is evaluated on two training resets; the worse score controls selection. The score includes loaded supported progression, terminal pitch, COM positioning, goal dwell, slip, drift, support-gap, torque-request and physical-failure terms. Pitch is not the sole objective, and **the score never overrides the independent acceptance gate**.

There are 480 candidate evaluations / 960 simulator branches, including repeated priors and elites. These are not 960 independent test conditions. Optimization seeds are 202620120–121; fresh validation seeds are 202620520–527. Held-out cases use larger reset disturbances and both deterministic and native stochastic frozen-policy actions. The common maximum budget is 1.6 s plus 40 ms preparation; physical failures terminate candidates, so realized interactions differ and are saved.

A targeted front-extension sweep adds 27 profiles on two separate resets (54 branches). A mechanics-directed catch refinement adds 36 profiles on two training resets (72 branches), followed by eight new reset seeds 202623520–527. Catch source parameters are saved separately so an evolving search file cannot silently change the experiment.

## 4. Held-out results: progress is not yet a valid reference

Eight deterministic resets per row below; another eight native-stochastic resets per row were evaluated separately. **All 64 held-out validations were rejected.**

| Search mode / front strategy | Mean pre-failure loaded progression | Loaded slip | Full-horizon physical safety | Accepted |
|---|---:|---:|---:|---:|
| Maintained / load-aware yielding | 45.64 degrees | 0.0300 m | 0/8 | 0/8 |
| Maintained / no yielding | 54.41 degrees | 0.0289 m | 0/8 | 0/8 |
| Transient / load-aware yielding | 45.88 degrees | 0.0273 m | 0/8 | 0/8 |
| Transient / no yielding | 40.72 degrees | 0.0716 m | 8/8 | 0/8 |

The yielding candidates move COM toward support but are too low: every deterministic yielding candidate contacts `Head_lower`. In a representative maintained-yielding case, COM x improves from −0.1599 m near 30 degrees to −0.1253 m at 45 degrees, but COM height is only about 0.283 m and loaded slip is already 28.2 mm. The calf's native body clearance remains around 76 mm, illustrating why calf clearance alone is not a sufficient geometric objective.

The maintained/no-yield candidate reaches 45 degrees with COM x around −0.1411 m and about 21.7 mm slip in the representative reset—close to, but already outside, the slip gate. It then peaks higher, COM falls back, and every deterministic validation hits `FL_calf`. Final mean COM x is approximately −0.202 m. The trajectory does not preserve its favorable intermediate COM placement.

The transient/no-yield candidate remains free of illegal contact and actuator-state failure in all eight deterministic tests, but this is **not contact feasibility**: slip averages 71.6 mm, progression is below the target, and the final COM is about −0.283 m behind support. Its average longest bilateral-support interruption is about 7.8 ms, so brief unloading itself is feasible; it does not cure the translation/slip problem. Native-stochastic validation also has zero accepted references, with one additional physical failure in this cohort.

The catch refinement is collision-free on all eight new reset conditions but stalls around 27–29 degrees with approximately 49–51 mm slip. It is rejected. Neither safety by stalling nor pitch through an eventual collision is a reference.

[Contact, force, friction and progression plots](../output/contact_transition_report_20261112_v2/contact_progression.png).
[COM trajectories](../output/contact_transition_report_20261112_v2/com_trajectories.png).
[All-calf/head clearance and base height](../output/contact_transition_report_20261112_v2/body_height_head_calf_clearance.png).
[Actual joint torque limits](../output/contact_transition_report_20261112_v2/joint_torque_limits.png).
[Requested versus applied torque ratios](../output/contact_transition_report_20261112_v2/requested_applied_torque.png).
[Optimized joint target sequences](../output/contact_transition_report_20261112_v2/optimized_joint_sequences.png).

## 5. What launch/continuation and leg ablations reveal

On the four matched baseline/ablation reset seeds, the baseline's certified loaded progression averages 35.00 degrees with 59.2 mm loaded slip. The maintained/no-yield optimized sequence averages 54.40 degrees with 28.7 mm slip—about a **51.5% reduction**—but still collides in all four cases and exceeds the 20 mm allowance.

With the same optimized sequence, launch-only modifications average 46.17 degrees but 91.2 mm slip; continuation-only modifications average 40.58 degrees and 34.6 mm slip. Front-only modifications reach 52.72 degrees with 39.1 mm slip; rear-only modifications reach 39.52 degrees with 69.0 mm slip. Removing soft anchoring reduces mean progression to 44.52 degrees. None passes acceptance. These are fixed-sequence ablations, not independently reoptimized methods.

Launch and catch must therefore be coordinated. A stronger rear impulse can reach a higher angle while worsening the contact state. More front extension can provide height but changes the support wrench and frequently loses rotational progression. Folding can improve pitch while placing the lower body/head or calf on a collision path. The unresolved task is steering **body height, COM translation, leg shape and contact wrench together**, not merely raising an angle target.

## 6. Controlled morphology feasibility study

The native-optimized sequence is transferred without retraining across the existing physical leg scales. Each scale has matched baseline/optimized, contact-mode and reset cohorts: 80 simulator branches total. Baselines repeat across contact-mode cohorts, so this is not 80 independent reset conditions. Geometry, mass, inertia, COM offsets, collision masks/friction, joint limits and actuator constraints pass the existing physical audit. No state is copied between morphologies.

Representative maintained-mode optimized results, four resets per scale:

| Leg scale | Safe supported peak | Certified takeoffs | Loaded slip | Physical collisions | Accepted |
|---|---:|---:|---:|---:|---:|
| 0.80 | 25.73 degrees | 4/4 | 0.0224 m | 4/4 | 0/4 |
| 0.85 | 30.78 degrees | 4/4 | 0.0223 m | 4/4 | 0/4 |
| 0.90 | 35.71 degrees | 4/4 | 0.0227 m | 4/4 | 0/4 |
| 0.95 | 41.57 degrees | 4/4 | 0.0225 m | 4/4 | 0/4 |
| 1.00 | 54.40 degrees | 4/4 | 0.0287 m | 4/4 | 0/4 |

Shorter legs modestly reduce this sequence's slip but lower useful progression and do not eliminate collision. This is a controlled transfer/feasibility comparison, not proof that a separately optimized shortened model could never help. No adaptive morphology schedule is activated. Final reference eligibility remains on the original 1.00 model.

## Gates, limitations and reproducibility

`reference_gate.json` has `repeatable_reference_found=false`, `bc_enabled=false`, `ppo_enabled=false`, `training_authorized=false`, and zero updates. Reference eligibility requires at least seven of eight deterministic held-out trials to pass **all** trajectory constraints, with no full-horizon physical failures in any of the eight. No candidate meets that criterion. BC/PPO preparation is therefore blocked; failed high-pitch trajectories must not be relabeled as successful demonstrations.

The action exports include the 44-element observation, normalized commanded action, actual rate-limited target and full qpos/qvel after each control step. Spline NPZs and per-control CSV/JSON sequences are supplied for inspection and exact parameter replay. They are not an admitted BC dataset.

**31 unit/integration tests pass.** New tests cover bounded continuous splines, motor commands without physical-state edits, rejection of peak-only and unsafe references, separate maintained/transient gates, contact loss counted before 30 degrees, disjoint seed cohorts and exact deterministic frozen-policy replay. Existing observation/PD/CPG, momentum, morphology and recovery tests also pass.

The initial search finished all 480 candidate evaluations but hit a result-assembly error at validation export. The error was fixed and the saved splines were replayed in fresh directories; optimization did not need to be rerun. First-pass comparisons in validation `v1` measured support gaps from the certified ≥30-degree state. Final validation `v2` additionally counts all interruptions from certified rear takeoff. Physical trajectories are unchanged by that measurement correction, and all candidates fail independently on slip/progression/collision. Report `v1` was interrupted by empty pre-probe moment fields; the corrected complete report is `v2`. All intermediate artifacts remain preserved.

This finite search does **not** prove that a feasible transition is impossible. It establishes that the tested action families, finite budgets and native hardware constraints did not yield an acceptable repeatable reference. Any next optimizer should address front-leg extension/body-height and COM placement with explicit all-body clearance constraints, while preserving the slip limit, rather than treating the observed 54-degree peak as success. The simple post-30 hold and extension probes show that repairing continuation alone is insufficient.

### Exact commands

From the repository root, use Python 3.11. Locally replace `python` with `env -u PYTHONPATH .venv/bin/python` to avoid the shell's incompatible Python 3.12 package path. All output paths must be new.

```bash
python -m pytest -q experiments/recoverability/test_core.py experiments/recoverability/test_transition.py experiments/recoverability/bidirectional/test_bidirectional.py experiments/recoverability/contact_transition/test_contact_transition.py

# Replay frozen optimized parameters, including full mechanism/morphology comparisons.
python -m experiments.recoverability.contact_transition.run --output output/molab_contact_validation --search-dir output/contact_transition_search_20261112_v1 --population 24 --iterations 5 --strategies load_yield no_yield

python -m experiments.recoverability.contact_transition.mechanics --output output/molab_contact_mechanics
python -m experiments.recoverability.contact_transition.extension_sweep --output output/molab_contact_extension
python -m experiments.recoverability.contact_transition.catch_refine --source-candidate output/contact_transition_catch_20261112_v1/source_candidate.json --output output/molab_contact_catch

python -m experiments.recoverability.contact_transition.report --search output/contact_transition_search_20261112_v1 --validation output/molab_contact_validation --mechanics output/molab_contact_mechanics --extension output/molab_contact_extension --catch output/molab_contact_catch --output output/molab_contact_report

# Optional NEW offline search; no BC/PPO code runs.
python -m experiments.recoverability.contact_transition.run --output output/molab_contact_new_search --population 24 --iterations 5 --strategies load_yield no_yield
```

The archived CEM and auxiliary sweeps used their saved original measurement schema. Current-source replay applies the stricter takeoff-gap measurement and corrected preparation-time labels; exact physical commands/states are the reproducibility target. Frozen source copies, checkpoint/source hashes and all raw records are retained. Fresh optimization with the stricter measurement may choose different elites and is a separate experiment.

Upload [the portable notebook](../notebooks/molab_contact_transition.ipynb) with `output/contact_transition_molab_20261112_v1.zip` to Molab. Its setup checks every SHA-256 entry and installs recorded dependency versions. The default cells replay archived references and run offline tests only. No training command is exposed by this experiment runner.
