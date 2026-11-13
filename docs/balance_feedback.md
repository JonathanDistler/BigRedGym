# Observable-feedback balance pilot — 2026-11-13

The local balance objective improved: three of four correlated near-vertical trials achieved a genuine five-second hold, versus zero for checkpoint50 B3. This is an analytically initialized neural balance actor, **not a standing-start handstand or a PPO-trained result**. No long training, parameter sweep, gain change, collider modification or external force was used.

## Targeted diagnosis

From the saved checkpoint50 B3 physics replay:

| Time (s) | Pitch (deg) | Pitch rate (rad/s) | COM-support x (m) | COM vx (m/s) | Front moment about COM (Nm) |
|---|---:|---:|---:|---:|---:|
| 0.32 | 86.32 | -0.448 | -0.0059 | 0.0050 | -0.833 |
| 0.60 | 81.44 | -0.002 | -0.0151 | -0.0702 | 0.583 |
| 0.80 | 84.25 | 0.598 | -0.0355 | -0.1841 | 0.467 |
| 1.00 | 86.67 | -1.105 | -0.0734 | -0.2621 | -9.052 |

Apparent pitch recovery hides worsening COM translation relative to front support. Foot slip reaches0.081m/s at0.80s; front support is lost at1.086s. Front knee torque is about19.43Nm at1.0s, without saturation at that sample. At1.20s pitch rate is−9.40rad/s with no front force; rear calves collide at1.44s. Dominant measured failure: inadequate coordination of COM motion, pitch velocity and leg geometry, followed by contact-moment reversal and support loss. Persistent torque saturation and an initially impossible pose do not explain this interval. This is a measured failure sequence, not a unique causal decomposition.

The saved full-state LQR teacher already maintained over6s of legal support on this narrow reset family. Its native equilibrium has base height0.57m with nominal kp60/kd3 and original actuator/contact constraints. Reset repair was unnecessary. Small supervised action MSE had not ensured that B3 retained stabilizing feedback derivatives outside its narrow demonstrations.

## One stabilization intervention

`experiments/balance_feedback/actor.py` analytically initializes a linear neural actor from the saved native LQR. Central differences map local physical perturbations into34 existing observation channels: gravity, angular velocity, twelve joint positions/velocities, COM height and COM velocity. A pseudoinverse projects LQR feedback into observable coordinates. Rank is32 of33; an unobservable heading direction is explicitly removed. The legacy teacher's omitted local yaw coordinate is not world yaw at vertical, so this is a local projection rather than exact global reconstruction.

Inference requires only the existing observation tensor, with no MuJoCo-state access, clock, teacher call or controller switch. Native contact forces informed the linearized plant; explicit force feedback was not added. All12 joints remain actuated normally. Outputs are bounded normalized position actions, `q_target = nominal + 2*action`, followed by existing joint, torque and speed constraints. Control50Hz, MuJoCo500Hz, kp60/kd3. No smoothing or contact locking. Float64 affine coefficients reduce numerical cancellation; output dtype matches the observation.

## Controlled results

Baseline: B3 from `output/multi_impulse_study_v2/checkpoint_000050.pt`. Four fixed, physically checked resets: one original held-out near89° state with nonzero pitch velocity, two body-y angular-rate perturbations ±0.08rad/s, and one body-y orientation perturbation0.003rad. Same joint/foot geometry and original contact model for both actors. These are **correlated perturbations of one held-out family**, not broad independent trials. Identical6.2s horizons and unchanged continuous500Hz success checks.

| Reset | B3 valid hold (s) | Feedback valid hold (s) | Feedback result |
|---|---:|---:|---|
| Original held-out | 0.324 | 6.176 | Timeout, successful hold |
| Angular rate +0.08 | 0.624 | 6.176 | Timeout, successful hold |
| Angular rate -0.08 | 0.614 | 3.692 | Genuine self-collision |
| Orientation +0.003 | 0.748 | 6.180 | Timeout, successful hold |

Five-second success: **0/4 → 3/4**. Illegal-collision episodes:4/4 →1/4. The failed trial is retained. Applied torque ratios never exceed1.0. Successful feedback trials have maximum torque ratios0.599/0.459/0.428, maximum joint-speed ratios0.086/0.062/0.052 and zero saturation. The failed trial reaches the real torque bound; limits were not relaxed.

Original matched reset: COM/support RMS53.6mm →0.91mm; COM-speed RMS0.294 →0.00966m/s; angular-speed RMS3.401 →0.0451rad/s; front-support fraction78.06% →99.81%. These are whole collision-free episode metrics and the actual episode lengths differ. Maximum initial slip is still0.502m/s for feedback: the continuous hold gate excludes the brief initial transient before certifying6.176s of low-slip, low-motion, front-supported balance with rear feet clear and no illegal impacts.

`output/balance_feedback_v2/` preserves the actor, exact evaluation resets, results.json, original-reset control/physics traces, full integration states and actual MuJoCo comparison.gif. Rendering uses the real model and saved integration states, a shared fixed camera, COM and native contacts. After B3 terminates its final frame is explicitly marked held; no episode reset is concealed. Last GIF frame decoded successfully. Initialization-only v1 rejected the unobservable state dimension and is preserved as a failed attempt, not evaluated success.

## Compatibility and later fusion

The93-channel ordering is unchanged and its first73 channels are the ADC prefix. `adapt_adc_observation()` only pads that prefix with20 zeros. Selected feedback channels are all within the original73 inputs. Original checkpoints are unchanged.

**Action adapter:** `BalanceActor.forward(obs)` already returns bounded normalized position actions: do not tanh again. Existing ADC/B3 networks still use `net.actor(obs).tanh()`. Distillation must compare final actions, or form student latent labels with `atanh(clamp(action))` explicitly. This prevents applying a second squash to the balance teacher.

1. Preserve ADC `output/multi_impulse_diagnosis_v1/B_best.pt`, recovered multi-impulse checkpoint52 and the new balance_actor.pt independently. Collect kick-up labels from ADC and balance/catch labels from the observable actor, with varied COM/joint/angular velocities.
2. Train one shared93-input/12-action network using state-conditioned kick-up and balance supervision. Preserve feedback sensitivity, not just static action MSE. Teachers provide training labels; no manually timed switch is deployed.
3. Fine-tune that one actor with mixed-start PPO, retaining certified standing takeoff and the five-second balance gate. Evaluate standing and intermediate starts separately under nominal dynamics; require evidence that fusion preserves both skills.
4. Expand a transition curriculum through native-valid, demonstrably recoverable catches toward75°,60°,45° and standing. Current60°/45° reset tiers remain empty. Do not infer their recoverability from this result.

No complete fusion, arbitrary-heading robustness, broad recovery, reliable single-foot support or hardware transfer was validated here.

## Reproduction and tests

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/validate_balance_feedback.py --output output/balance_feedback_NEW --render
env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/unit_tests/test_balance_feedback.py -q
```

Run serially with other heavy jobs stopped. Three focused tests passed: bounds/clock independence and ADC adapter; checkpoint action equivalence; real-contact balance exceeding1s with legal torques/speeds and no illegal collision. Only the changed adapter test was repeated after adding padding. Existing16 ADC/multi-impulse tests had passed during recovery and were not repeated. Original training sources remain unchanged, preserving checkpoint52's exact-resume contract. No PPO run was launched during this pilot.
