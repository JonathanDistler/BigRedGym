# Contact-aware full-body configuration and transition planning

## Main result

A **sampled collision-free, contact-consistent configuration connection exists from 30 to 90 degrees**, and the tested connection's nodes admit quasi-static contact forces within native motor/friction limits. **An executable, repeatable dynamic entry from the current kick-up motion was not demonstrated.** No BC or PPO ran, and no forward 55–60-degree extension was attempted.

This distinguishes three different findings:

1. **Configuration feasibility:** 447 of 975 tested pitch/height/COM/profile cells meet the explicit geometric and cell-target tolerances.
2. **Quasi-static equilibrium:** 69 cells also admit a feasible two-front-point-contact force/torque allocation. Dense continuation finds 61 such nodes from 30 through 90 degrees at 1-degree spacing.
3. **Dynamic reachability:** deterministic multiple shooting and uninterrupted MuJoCo replay do not find a transition satisfying progression, contact/slip and terminal recovery requirements. The final independent cohort has **0/64 accepted trajectories**.

All previous experiments/checkpoints remain preserved on `experiment/recoverability_morphology_v1`.

## Canonical artifacts

| Output directory | Contents |
|---|---|
| `output/configuration_atlas_20261112_v1/` | 975 constrained IK attempts, geometric/static results, full qpos/joint/force/torque values |
| `output/configuration_connectivity_20261112_v1/` | Dense configuration continuation; lateral load-redistribution tests |
| `output/configuration_shooting_20261112_v1/` | Four native-dynamics multiple-shooting plans; 16 development cases per mode; 64 initial held-out cases |
| `output/configuration_shooting_refined_20261112_v1/` | Slower launch timing and dynamics-defect penalty refinement; separate 16+64 cohort |
| `output/configuration_shooting_contact_refined_20261112_v1/` | Along-segment contact/slip/clearance refinement; separate 16+64 cohort |
| `output/configuration_final_replay_20261112_v1/` | Frozen-command replay with geometric contact/impact audit; **final independent 16+64 cohort**, model compatibility and learning gate |
| `output/configuration_reverse_20261112_v1/` | Successful-source checks and nine physically integrated predecessor probes |
| `output/configuration_audit_20261112_v1/` | Dynamics defects in physical units and reverse features recomputed from full integration snapshots |
| `output/configuration_planning_report_20261112_v2/` | Figures, CSV atlas and measured summary, including reverse-slip qualification |
| `output/configuration_visualizations_20261112_v1/` | Actual MuJoCo candidate replays and separately labeled geometry-only animation |

[Measured report](../output/configuration_planning_report_20261112_v2/measured_report.md), [summary JSON](../output/configuration_planning_report_20261112_v2/summary.json), [full physical-unit atlas CSV](../output/configuration_planning_report_20261112_v2/atlas_physical_values.csv).

## 1. Configuration feasibility atlas

The native full Go2 model is unchanged. The atlas covers:

- Pitch: 30, 35, …, 90 degrees.
- Body-root height: 0.300, 0.375, 0.450, 0.525, 0.600 m.
- COM x targets relative to the front-foot midpoint: −0.20, −0.10, −0.05, 0, +0.05 m.
- Three profiles spanning front-foot widths 0.28–0.36 m, fore-aft stagger 0–0.04 m, roll/yaw −2 to +2 degrees, lateral COM offsets −0.04 to +0.04 m and different rear-posture initializations.

At every cell, all **12 native joints** and root x/y translation are free IK variables. Pitch, height and the profile's orientation/contact specification define the cell. Analytic MuJoCo foot and subtree-COM Jacobians drive bounded nonlinear least squares. Collision-distance Jacobians use closest geometric points. Head, complete front-calf geometry including fixed lower-calf children, all non-foot ground clearance and native self-collision pairs are evaluated. Rear feet must clear the ground by at least 15 mm for the front-support atlas.

A successful geometric cell requires front-foot position error ≤0.1 mm, COM-target error ≤0.2 mm, non-foot/head/calf clearance approximately ≥5 mm, no self penetration and native joint limits. These are rigid-contact IK tolerances, **not a membership filter on moving MuJoCo states**: real simulated compliant foot contact can penetrate slightly and is validated with the native contact model. No geometric atlas state is assumed reachable from standing.

| Pitch | Geometry + target cells / 75 | Also static / 75 |
|---|---:|---:|
| 30 | 41 | 5 |
| 35 | 44 | 6 |
| 40 | 48 | 6 |
| 45 | 36 | 4 |
| 50 | 41 | 5 |
| 55 | 42 | 6 |
| 60 | 41 | 6 |
| 65 | 38 | 6 |
| 70 | 36 | 6 |
| 75 | 29 | 6 |
| 80 | 24 | 6 |
| 85 | 17 | 3 |
| 90 | 10 | 4 |

Local unsuccessful attempts include front-contact IK error (427), COM-target error (421), self-collision (131), head clearance (150), other-body clearance (155) and front-calf clearance (27). Counts overlap. They identify the constraints active in these searches; **local solver failures are not global geometric infeasibility proofs**. The initial narrow-foot smoke attempt also failed to reach its constraints; widening/continuing the deterministic profiles found feasible solutions. That failed smoke remains preserved.

[Pitch–height–COM atlas](../output/configuration_planning_report_20261112_v2/pitch_height_COM_atlas.png). Gray denotes no local solution, blue geometric feasibility and green geometry plus quasi-static feasibility; profiles are aggregated within each cell.

### Representative full-body shapes

Full twelve-joint vectors are in the CSV/JSON. Native order is FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf. Examples below use centered front feet at width 0.32 m and COM x ≈0:

| Pitch / root height | Front thigh angles | Front knee angles | Rear thigh angles | Rear knee angles | Maximum torque / limit |
|---|---|---|---|---|---:|
| 30 degrees / 0.300 m | 1.145, 1.145 rad | −1.877, −1.877 rad | 3.373, 2.598 rad | −1.143, −0.838 rad | 0.536 |
| 45 degrees / 0.375 m | 0.703, 0.703 rad | −1.793, −1.793 rad | 1.899, 2.283 rad | −0.964, −0.869 rad | 0.463 |
| 90 degrees / 0.525 m | −0.642, −0.632 rad | −1.484, −1.478 rad | 2.342, 2.200 rad | −1.018, −1.071 rad | 0.191 |

The low 30-degree equilibrium uses a substantially different rear posture from the kick policy, with one rear knee near its extension bound. It is not a robust demonstration simply because an IK/force solution exists. The 90-degree shape also differs from the learned balance posture, including front abduction and body height, so static feasibility does not imply balance-policy recovery.

### Static force and torque calculation

The compiled model has spherical feet and three-dimensional point-contact friction; the solver rejects unsupported rolling-moment contact models. For every successful geometric cell, it computes actual contact-point Jacobians, solves the full six unactuated equilibrium equations, and explores the force nullspace while checking:

- Positive normal forces.
- Exact Coulomb cones, `norm(f_tangent) <= mu * f_normal`, with native friction.
- Required motor torques `tau = gravity_bias_joints - J_joints.T * contact_force` within native effort bounds.
- Free-base force residual <0.05 N and moment residual <0.01 N m.

Static forces are **estimated**, never injected into dynamic simulation. Among feasible cells the largest required motor utilization is approximately 0.645. Some selected force allocations approach the friction boundary. The allocator minimizes peak motor utilization subject to the cones; it is not a maximum-contact-margin allocator. A low torque estimate alone must not be interpreted as robust contact feasibility. Every joint torque, cone margin, normal force, residual and clearance is retained.

[Static torque and friction plots](../output/configuration_planning_report_20261112_v2/static_torque_friction.png).

For two coplanar point feet with no rolling contact moments, static equilibrium necessarily requires `COM_xy = sum(fz_i * contact_xy_i) / (m*g)`. With unilateral normals, the COM projection lies on the segment between the two contact points. Thus many off-support cells are **statically infeasible for that specified contact mode** even though collision-free configurations exist. This theorem does not exclude dynamic motion outside the segment: contact moments can change angular momentum, and ballistic intervals need no static equilibrium.

## 2. Dense geometric connection and contact modes

Deterministic continuation at root height 0.45 m and centered fixed front contacts finds **61 geometric/static nodes** over 30–90 degrees at 1-degree spacing. The largest joint change between neighboring nodes is about 0.0734 rad. Another 180 interior geometric samples are collision-free; maximum interpolated front-contact error is about 0.0785 mm. This is strong evidence of a sampled kinematic connection, not a continuous-time dynamics certificate. Only nodes receive the exact static force test; no physical timing or static hold is imposed on the subsequently planned motion.

[Geometric joint path](../output/configuration_planning_report_20261112_v2/geometric_joint_path.png).
[Geometry-only MuJoCo animation](../output/configuration_visualizations_20261112_v1/geometry_only_path.gif) is explicitly labeled **not executed**.

The four dynamic strategies are enumerated rather than presumed feasible:

- **A, continuous bilateral support:** actual front loading must remain continuous within the specified monitoring tolerance.
- **B, load redistribution:** lateral COM and joint asymmetry alter left/right loading. Dedicated IK/force tests at FL load fractions 0.2, 0.5 and 0.8 find 8/12 geometric/static solutions. These show that some load redistribution is physically supported by the geometry/forces; they do not establish a dynamic transition. Final mode checking excludes planned complete unloading for B.
- **C, brief unload/recontact:** a 60 ms shooting interval proposes raised front contacts, followed by recontact. Real simulated contact states, not the proposed schedule, determine whether unloading occurred.
- **D, joint launch/arrival planning:** all joint commands may change from approximately 0.34 s, before the original takeoff/rotation sequence completes. This is a different entry condition from switching only after 30 degrees.

Measured contact modes distinguish **geometric contact** from **loaded support above 5% body weight per foot**. A low-load interruption is not automatically a ballistic flight. Final replay measures both at 500 Hz. Landing velocity conservatively retains the maximum foot vertical speed immediately before/after geometric first touch and loaded reacquisition. No velocity reset, contact pinning, friction change or impact-force assignment is used.

## 3. Whole-body dynamic optimizer

`shooting.py` implements deterministic bounded hybrid multiple shooting with native MuJoCo integration. It optimizes free-base position/orientation/velocity and all 12 joint positions/velocities at shooting nodes, together with full twelve-joint native motor commands. There are two shooting intervals for A/B/C and three for D. The optimizer uses feasible atlas configurations as guesses; **shooting node resets are numerical hypotheses only**.

The native 50 Hz target ramps, target slew bounds, 500 Hz PD/torque clipping, joint-speed guards, friction/contact solver and full collision geometry are retained. Actuator commands generate joint forces only; the floating base receives no artificial wrench. Endpoint contact-mode costs use actual forces produced by the engine. Later refinement adds 100 Hz quadrature penalties for loaded slip, friction exposure, head/calf clearance and support deficits within shooting intervals. Final physical validation remains denser at 500 Hz.

The first D schedule spans 0.58 s; slower refinement spans 0.94 s. State-defect penalties are increased in a homotopy. This is a **penalty multiple-shooting prototype**, not a converged hard-constrained collocation certificate. State continuity must satisfy explicit tolerances, and every motor sequence is independently replayed in one uninterrupted simulation from a physical standing-start prefix. No shooting node is copied into that replay. No intermediate static hold is required. Forward targets remain at 30–45 degrees; no 55–60-degree forward extension is enabled.

The original plans reduce weighted cost substantially but retain unacceptable state-matching defects. For example:

| Plan | Largest root-position defect | Largest root-orientation defect | Nominal uninterrupted terminal pitch |
|---|---:|---:|---:|
| A | 0.0160 m | 0.0903 rad | 18.20 degrees |
| B | 0.0132 m | 0.0959 rad | 17.11 degrees |
| C | 0.0195 m | 0.0361 rad | 16.36 degrees |
| D | 0.00468 m | 0.0153 rad | 38.28 degrees |
| Slower/contact-refined D | about 0.00020 m | about 0.00160 rad | 34.36 degrees |

The refined first interval still has joint-velocity and state defects above the acceptance tolerance. More importantly, enforcing dynamics more strongly moves the proposed terminal node from roughly 43.8 degrees toward 37.1 degrees; uninterrupted simulation ends lower still. A reduced defect or optimizer cost is not evidence of achieving 45 degrees. The along-segment contact refinement stalls near the same solution under the finite evaluation budget. This is a search/formulation limitation, not proof of dynamic impossibility.

[Defects in physical units](../output/configuration_planning_report_20261112_v2/shooting_defects.png), [full audit](../output/configuration_audit_20261112_v1/dynamics_defects_physical_units.json).

## 4. Development, independent held-out results and impacts

Each initial mode has **16 development perturbations**. Selection uses development only; the selected D plan receives 64 independent held-out resets. The slower refinement, contact refinement and final fixed-command audit each use separate 16+64 seed cohorts. These cohorts are not pooled into a single success-rate estimate.

| Initial mode | Development count | Physical stage passes | Original balance-policy recovery | Mean supported peak | Mean loaded slip |
|---|---:|---:|---:|---:|---:|
| A | 16 | 0 | 0 | 31.30 degrees | 0.0661 m |
| B | 16 | 0 | 0 | 31.27 degrees | 0.0697 m |
| C | 16 | 0 | 0 | 31.02 degrees | 0.0610 m |
| D | 16 | 0 | 0 | 37.81 degrees | 0.0629 m |

The final frozen-command cohort, with a new set of 16 development and 64 held-out perturbations, yields:

| Metric | Final 64 held-out results |
|---|---|
| Complete accepted / balance recoveries | **0/64 / 0/64** |
| Mean reachable peak / supported peak | 36.38 / 36.38 degrees |
| Terminal pitch range | 30.93–39.58 degrees |
| Mean loaded front-foot slip | **0.0689 m** |
| Head/calf collisions | 0/64 |
| Native joint/actuator-state violations during planned motion | 0/64 |
| Minimum head / full front-calf clearance | 0.00159 / 0.0221 m |
| Maximum pre/post contact-reacquisition vertical velocity | 0.221 m/s |
| Maximum loaded bilateral-support interruption | 0.006 s |
| Maximum full geometric-flight interval / total | 0.002 / 0.010 s |

The physical stage gate retains loaded slip ≤20 mm and drift ≤40 mm, all-body collision and actuator/joint constraints, ≥5 mm head/calf planning margin, a supported 40–50-degree arrival with rear clearance, and appropriate contact-mode behavior. It does **not** require a static hold. The frozen original balance policy must then recover for five seconds from the exact arrival state. Model, target memory and physical state remain unchanged during that counterfactual; no envelope filter is used.

The latest trajectories avoid the old head/calf collisions, but they still fail progression and slip constraints. In many arrivals COM x velocity has reversed backwards, and the direct balance handoff from this low-pitch/body-shape distribution fails rapidly. No 45-degree arrival is admitted merely because it is close in a normalized feature metric. Native actuator limits are respected during the planned segments; actuator failures in a subsequent failed balance counterfactual are reported separately in the recovery result.

[Contact, momentum, COM and clearance comparisons](../output/configuration_planning_report_20261112_v2/dynamic_contact_COM_clearance.png).
[Friction, motor torques and geometric contact history](../output/configuration_planning_report_20261112_v2/friction_torques_geometric_contacts.png).
[Actual final MuJoCo replay](../output/configuration_visualizations_20261112_v1/final_frozen_D.gif), [A](../output/configuration_visualizations_20261112_v1/actual_mode_A.gif), [B](../output/configuration_visualizations_20261112_v1/actual_mode_B.gif), [C](../output/configuration_visualizations_20261112_v1/actual_mode_C.gif), [D](../output/configuration_visualizations_20261112_v1/actual_mode_D.gif). These replays are labeled rejected candidates.

## 5. Reverse validation and forward/backward separation

Three distinct successful balance-bank sources first pass fresh five-second continuation checks. From each, motor commands move toward the 85-degree atlas shape over 0.15, 0.30 or 0.60 s, with bounded native target slew and pitch feedback. All integrations run **forward in physical time**. The arrival is a saved actual integration state with original anchors, accumulated safety/slip history, previous motor target, CPG phase and filtered loads. No robot-state interpolation or negative-time integration is used.

Nine predecessor probes produce two original-metric five-second recoveries at **88.252–88.257 degrees**, with body height about 0.5342 m, COM x around −0.00412 m and body y angular velocity about −0.215 rad/s. Arrivals near 86.3 degrees do not recover in the tested probes.

**These are not low-slip handoff certificates.** The successful reverse cycles accumulate approximately **122 and 130 mm** of loaded slip, compared with about **52 mm** in their matched successful bank continuations. Their physical entry prefixes use roughly 10.3 mm, but recovery adds substantial extra contact motion. They demonstrate local original-policy catchability, not a valid low-slip bridge. No new permissive recovery criterion is introduced to count them as learning references.

Across the final **rejected but physically simulated forward terminal samples** and the two original-metric reverse catches:

| Variable | Forward terminals | Reverse catches |
|---|---:|---:|
| Pitch | 30.93–39.58 degrees | 88.252–88.257 degrees |
| COM x from front support | −0.1673 to −0.1056 m | about −0.00412 m |
| Body height | 0.4095–0.4279 m | about 0.5342 m |
| Body y angular velocity | −0.769 to −0.195 rad/s | about −0.215 rad/s |
| COM x velocity | −0.683 to −0.431 m/s | about −0.0190 m/s |
| COM z velocity | −0.036 to +0.123 m/s | about −0.258 m/s |
| FL vertical contact force | 64.3–76.3 N | about 79.4 N |
| FR vertical contact force | 55.5–64.3 N | about 82.6 N |

Closest twelve-joint Euclidean separation in this comparison is about 3.03 rad. Contact-mode histories and all twelve angles are retained. Angular-rate overlap does not remove the pitch/COM/height/body-shape separation. Marginal empirical gaps are approximately 48.67 degrees, 0.1015 m COM x and 0.1063 m body height; they need not occur in the same state pair and are **not global reachability bounds**. There is no validated overlap between the strictly admissible forward and backward sets.

[Forward/backward comparison](../output/configuration_planning_report_20261112_v2/forward_backward_gap.png).
The authoritative reverse state features are recomputed from saved full integration snapshots in [the audit](../output/configuration_audit_20261112_v1/reverse_state_features.json); this supersedes first-pass height metadata fields without changing any physical state or recovery outcome.

## 6. Interpretation and next physically meaningful change

What is established:

- The full Go2 geometry, joint ranges and static force/torque capacities admit configurations throughout the requested pitch interval.
- A densely sampled geometric/static path is available as a structured initialization.
- The existing kick enters around 30 degrees with COM behind support and a substantially different leg configuration/momentum history from those equilibria.
- Joint launch planning is more useful than attaching a continuation to the existing late kick state, but the tested native-command plans still produce excessive loaded slip and insufficient arrival pitch/recoverability.

What is **not** established:

- No executable contact-aware 30–45-degree transition or recoverable terminal arrival was validated.
- No successful complete standing-to-handstand sequence exists here.
- No proof of global dynamic or geometric infeasibility follows from local solver failures, failed shooting schedules or failed balance counterfactuals.

The most meaningful next offline change is an explicit **four-foot fore-aft load-transfer setup before rear release**, with body height/COM and rear posture planned together, followed by controlled front-load redistribution if required. This would change the entry conditions rather than ask the old terminal kick configuration to supply more rotation. Mapping 0–30-degree four-foot support would test that setup; it has not been validated here. A maximum-margin contact-force allocation and a hard-defect/contact-implicit NLP with better mode-transition sensitivities are also justified by the measured limitations. They should be evaluated before any learning. An arbitrary static pose or geometry animation must not become a training demonstration.

No morphology schedule, BC, PPO or extended PPO is activated. `learning_gate.json` remains closed.

## Tests and reproducibility

**39 tests pass**, including the existing observation/PD/CPG and morphology invariants plus new constrained geometry/static force tests, analytic Jacobian checks, native integration equivalence, contact-quadrature measurement invariance, pre-impact velocity retention, geometric first-touch detection and development/held-out seed separation.

The source checkpoint, model physics and frozen balance policy are preserved. A fresh compatibility audit passes in the final replay. Independent standing-reset perturbations are generated by real motor integration, not arbitrary velocities or contact-force edits. Every candidate is replayed continuously; optimizer state resets never appear in validation. No large random action search was launched. SciPy 1.16.3 was added to the repository Python environment for constrained IK and deterministic least squares.

The structured solver is local, uses a finite deterministic budget and penalty constraints, and is not a globally converged feasibility solver. Contact quadrature is 100 Hz while final validation is 500 Hz. Older exploratory contact records used the loaded threshold for reacquisition; the final frozen-command cohort separately records geometric contact and retains pre/post-touch speed. Force allocations approaching a friction boundary are not claimed robust. Intermediate source revisions/results are preserved; frozen plan commands are the exact physical replay target, while reoptimization can choose a different local solution.

### Portable Molab workflow

Use [the notebook](../notebooks/molab_configuration_planning.ipynb) with `output/configuration_planning_molab_20261112_v1.zip` in Python 3.11. It verifies all exported SHA-256 entries, installs recorded dependencies and uses fresh output paths. Default replay is offline; new optimization is optional. There is no training cell.

Exact module commands from the repository root are:

```bash
python -m pytest -q experiments/recoverability/test_core.py experiments/recoverability/test_transition.py experiments/recoverability/bidirectional/test_bidirectional.py experiments/recoverability/contact_transition/test_contact_transition.py experiments/recoverability/configuration_planning/test_planning.py

python -m experiments.recoverability.configuration_planning.atlas --output output/molab_configuration_atlas --max-nfev 80
python -m experiments.recoverability.configuration_planning.connectivity --atlas output/molab_configuration_atlas --output output/molab_configuration_connectivity

# Replay the archived motor plan on a NEW independent 16+64 cohort.
python -m experiments.recoverability.configuration_planning.replay --source output/configuration_shooting_contact_refined_20261112_v1 --output output/molab_configuration_replay --seed-base 202635120

# Optional fresh deterministic optimization; no neural policy updates.
python -m experiments.recoverability.configuration_planning.plan --atlas output/molab_configuration_atlas --output output/molab_configuration_shooting --max-nfev 12
python -m experiments.recoverability.configuration_planning.reverse --atlas output/molab_configuration_atlas --output output/molab_configuration_reverse
```

Locally replace `python` with `env -u PYTHONPATH .venv/bin/python` to avoid the shell's incompatible Python 3.12 package path. Offscreen MuJoCo render commands use `MUJOCO_GL=egl`; actual EGL rendering was checked locally. Complete report/audit/render/export commands are in the portable notebook. Every output directory and ZIP refuses overwrite. No BC/PPO pilot is prepared from the rejected trajectories.
