# Go2 constrained trajectory optimization: mathematics, implementation, and staff-review handoff

Prepared 20 November 2026. Research branch: `experiment/trajectory_optimization`, based on `a38b193` (`experiment/go2-handstand-robustness`). This is a new **crouched-rest to handstand feasibility experiment**, not a retraining or replacement of the released CapturePolicy. No physical Go2 connection, SDK command, PPO training, model weakening, or modification of `main` was performed.

**Current conclusion: no dynamically feasible handstand has been certified. Physical deployment is disabled.** Solver termination, valid inverse kinematics, a drawn sequence of poses, and passing unit tests are not handstand success certificates. The measured section below distinguishes them.

## 1. Interpretation of the request: zero task reward and zero task cost

The implemented nonlinear program is

$$\begin{aligned}
\text{find }z\quad\text{or equivalently }\min_z\;&0\\
\text{subject to }&h(z)=0,\\
&g(z)\ge0,\\
&z_{min}\le z\le z_{max}.
\end{aligned}$$

There is no RL reward, orientation bonus, energy reward, effort minimization, reference-action loss, time minimization, or terminal soft penalty. The terminal handstand is a boundary constraint. Collision clearance, contact behavior and actuator restrictions are constraints. This answers the instruction to have no rewards/penalties without disguising an unsuccessful attempt as a high-reward trajectory.

A numerical constrained solver still needs a mechanism to move toward feasibility. SciPy `trust-constr` uses internal constraint-merit and barrier terms; these are numerical optimization machinery, not a user-defined robot reward. Row scaling changes numerical conditioning, not the accepted physical constraint definitions. Inverse-kinematics warm starts use least-squares equation solving, and underdetermined linear systems choose a particular minimum-norm solution. Neither adds a task objective to the trajectory NLP. These distinctions should be retained in any presentation of this experiment.

## 2. State, coordinates, decision variables, and units

The unmodified Go2 has 19 generalized position coordinates and 18 velocity coordinates:

$$q=(p_W,Q_{WB},q_j),\quad p_W\in\mathbb R^3,\quad Q_{WB}\in S^3,\quad q_j\in\mathbb R^{12},$$

$$v=(v_W,\omega_B,\dot q_j)\in\mathbb R^{18}.$$

The quaternion is WXYZ; it maps the body frame to the world. MuJoCo's free-joint velocity convention is used without inventing an Euler-angle rate mapping. Joint order is FL, FR, RL, RR, each hip/thigh/calf.

At each node $k=0,\dots,N$, the decision row contains

$$z_k=(y_k,v_k,a_k,\tau_k,\lambda_{FL,k},\lambda_{FR,k},\lambda_{RL,k},\lambda_{RR,k})$$

with dimensions 18+18+18+12+12=78. $y$ is a local configuration chart relative to an identity-orientation anchor; `mj_integratePos` reconstructs a unit-quaternion pose. `mj_differentiatePos` compares poses in the 18-dimensional tangent convention. This avoids treating the four quaternion entries as four independent rotational coordinates or using pitch/yaw Euler angles near a singular orientation.

Positions/clearances are metres, joint angles radians, angular speeds rad/s, accelerations m/s² or rad/s² as appropriate, GRFs newtons in world XYZ, and joint torques N m. Root angular velocities are body-frame coordinates consistent with the model's Jacobians. The norm of every reconstructed quaternion is maintained by the model's manifold integration operation.

The duration is fixed for a run, with $h=T/N$. Initial experiments use $T=2.4$ s and several coarse meshes. The numerical domain additionally bounds base position, speed, orientation chart and acceleration to avoid unbounded exploratory iterates. These numerical domains are not measured hardware limits.

## 3. Full manipulator equation, including passive terms

At every node, the NLP imposes all 18 equations:

$$M(q_k)a_k+b(q_k,v_k)-f_{passive}(q_k,v_k)
=S^T\tau_k+\sum_iJ_i(q_k)^T\lambda_{i,k}.$$

$S$ selects the 12 joint velocities. The first six equations are the unactuated floating-base force/moment equations; they cannot be satisfied by fictitious root torques. $M$ includes the original rotor armature. Bias and passive damping come from the unchanged model. In code the left-side offset is `qfrc_bias - qfrc_passive`.

The solver obtains mass matrices, bias, passive terms and Jacobians from MuJoCo forward calculations. It does **not** read MuJoCo contact-solver forces as its planned forces. The GRFs are optimization variables. No external body wrench is introduced. Independent simulation replay later uses native contacts and never injects the optimized GRFs.

Generalized residuals mix force coordinates (first three, N) with moment/joint-torque coordinates (remaining 15, N m). Reported raw infinity norms therefore cannot be read as an all-force error in newtons. The implementation scales the generalized equation using the numerical body-weight magnitude; this corresponds to a characteristic force $F_0=mg$ and characteristic torque $F_0\times1$ m. Physical components and other constraint groups must be interpreted with their own units.

## 4. Explicit Euler defects on the configuration manifold

The requested Euler discretization is implemented as

$$v_{k+1}=v_k+h a_k,$$

$$q_{k+1}=\operatorname{Integrate}(q_k,v_k,h).$$

The rotational update uses quaternion exponential integration rather than adding angular velocity to quaternion entries. It is still a first-order step in the velocity held over the interval. The position residual is evaluated by `mj_differentiatePos` between the predicted pose and the next node.

This choice has consequences. A curved no-slip contact manifold is not generally preserved exactly by a finite Euler step, even if instantaneous contact velocity and acceleration are zero. Fine meshes and independent replay are essential. A coarse pose path that passes inverse kinematics is not necessarily compatible with the Euler velocity/dynamics equations. Node feasibility alone also does not exclude a between-node collision.

The independent replay runs the existing native simulator at 2 ms, with 20 ms position commands. It does not replace the simulator with the Euler NLP or reset it to each optimized node. This tests whether the approximate transcription and command path produce a real continuous motion.

## 5. Initial crouched pose and velocity

The initial body-root height is explicitly chosen as 0.27 m. Body orientation is ordinary standing orientation; initial base and joint velocities are exactly zero:

$$q_0=q_{crouch},\qquad v_0=0.$$

The four foot support locations are taken from the hash-locked original reference's ordinary standing geometry, then inverse kinematics solves all 12 joint coordinates at the crouch height while preserving those foot locations on the ground. This defines a reproducible boundary state; it is not a claim that an actual robot has automatically been moved there.

IK must meet the explicit residual tolerance; a failed solve raises an error rather than fabricating contacts or changing link lengths. The new body height is a research assumption requiring staff review and a separate supervised preparation procedure. It does not authorize commanding a physical robot into that posture.

With explicit Euler and $v_0=0$, the first position update gives $q_1=q_0$. The solver eliminates those fixed coordinates instead of imposing a redundant duplicate equality. The full post-solve audit still verifies the original boundary and Euler equations.

The first actuator target is not automatically the geometric joint pose: a static load may require PD preload. The relationship between planned joint torque and position target is described below. No unseen hardware initialization or reset is assumed.

## 6. Terminal body and leg posture

The terminal shape is specified from the released successful reference's final posture, not from its action sequence. The body-root height is 0.57 m. Its reference orientation and rear-leg posture are retained, and a small front-leg/body-XY IK projection puts the front feet at the specified anchors and puts the COM projection on the front support line. All final position and joint coordinates become fixed boundary values:

$$q_N=q_{handstand},\qquad v_N=0,\qquad a_N=0.$$

The zero final acceleration together with the manipulator equation requires a static force/torque balance in the ideal point-foot model. It does not certify asymptotic balance or robustness. The independent replay additionally requires five continuous seconds under the existing strict handstand gate and additional terminal leg/body matching.

For two point contacts, static gravitational balance constrains the COM projection to the line segment between the front feet. The terminal IK solves the perpendicular line residual; unilateral, friction and full unactuated equilibrium constraints determine whether admissible support forces actually exist. A line equation alone is not sufficient.

Only the reference's final pose and initial foot geometry define boundaries. The released 320-step action trajectory is not played by this optimizer, and no old checkpoint is retrained. A boundary derived from a previously successful posture is a disclosed model prior, not a learned result of this experiment.

## 7. Contact modes, ground gaps, and zero swing force

The first experiment prescribes the contact schedule:

- Initially four feet in stance.
- At 30% of the planning interval count, rear feet switch to swing.
- Both front feet remain in stance through the terminal state.

Let $c_{i,k}\in\{0,1\}$ be that prescribed mode and $\phi_i(q)$ the foot gap. The constraints are

$$\phi_i(q_k)\ge0\quad\text{for all feet},$$

$$c_{i,k}=1\implies\phi_i(q_k)=0,$$

$$c_{i,k}=0\implies\lambda_{i,k}=0.$$

Together with unilateral normal force, this is a fixed-mode alternative to optimizing complementarity. It prevents an airborne foot from generating a fictitious force. A touching foot may have zero force at a mode transition. Rear forces are eliminated exactly from the optimization variables during swing; the independent audit still checks them.

**Limit:** this version does not discover a new contact schedule, include flight/relanding phases, or model impact impulses. Liftoff is modeled without a velocity jump. If a useful transition requires front-foot release and recontact, an impact-aware hybrid extension is needed. Failure of this schedule is not proof that Go2 cannot perform a handstand.

A contact-implicit extension would require nonpenetration, unilateral force and complementarity $0\le\phi_i\perp\lambda_{n,i}\ge0$, plus appropriate impact and friction treatment. It is discussed here but is not silently claimed to be implemented.

## 8. Stance acceleration, stance velocity, and drift

For the modeled foot point,

$$\dot p_i=J_i(q)v,$$

$$\ddot p_i=J_i(q)a+\dot J_i(q,v)v.$$

Each stance node imposes

$$J_i a+\dot J_i v=0.$$

The directional Jacobian derivative is computed by central differences along the same quaternion-aware integration operation. Initial velocity is zero. A 2 mm/s componentwise stance-velocity bound guards discretization drift, and the horizontal modeled-foot drift from the initial anchor must remain below 4 cm. These tolerances are not foot-pinning simulator constraints: the native simulator is never welded or kinematically reset.

Zero acceleration alone would allow an already-sliding foot to continue at constant speed. That is why initial velocity and velocity/drift guards are also present. At the terminal node, $v_N=a_N=0$ already implies zero foot acceleration; duplicate equalities are removed from the solver to reduce rank deficiency but retained in the complete audit.

### Foot-model qualification

The NLP uses the native foot geometry's center Jacobian and a support-height proxy `center_z - 0.022`. It therefore treats the foot as an ideal point foot with the original spherical radius used for the gap. The finite-sphere rotational/material-point contact kinematics and tangential-force lever arm are not fully represented by this reduced point model. It is not a rigid-sphere rolling contact solver or a six-axis contact-wrench model.

This approximation is consequential. Native MuJoCo replay uses the actual unchanged spherical feet and compliant contacts, so a point-model feasible trajectory could fail replay. That failure must block export; it must not be hidden by injecting the planned forces into the simulator. Hardware equivalence would need additional contact-model qualification.

## 9. Friction cone, unilateral force, and load bounds

For each world-frame foot force $\lambda_i=(f_x,f_y,f_z)$,

$$f_z\ge0,\qquad\sqrt{f_x^2+f_y^2}\le\mu f_z.$$

This is the circular Coulomb cone, not a two-dimensional single-axis friction check. The default research coefficient is $\mu=0.6$, below the nominal simulator ground coefficient. The model refuses a requested planning friction greater than the nominal floor coefficient.

A provisional per-foot normal cap is $f_z\le3mg$. This prevents unlimited forces in a purely feasible solve; it is not a measured leg load rating or a hardware authorization. Actual front-leg load limits, surface friction and impact loads require course-staff input. The NLP's normal cap and the stricter terminal force gate are different checks.

Swing forces are exactly zero rather than merely penalized. At stance the static-friction cone is compatible with near-zero point velocity. A full sliding-contact maximum-dissipation law is not implemented in this first fixed-stick-mode model.

## 10. Ground, knees, body geometry, self clearance, and proximity

Checking a knee joint center alone would miss a calf cylinder hitting the plane. The implementation checks the minimum world height of every non-foot collision primitive, including all calf subdivisions, hip/thigh shapes, head and body:

$$\min_{x\in\mathcal G_j(q_k)}x_z\ge\delta_{body},\qquad\delta_{body}=0.003\text{ m}.$$

The helper uses the actual primitive type, rotation and dimensions. It does not remove, shrink, or disable any collider. Feet separately obey nonpenetration.

Native-mask-compatible, nonadjacent self-geometry pairs must also satisfy a positive signed distance margin. Same-weld and direct-parent geometry pairs are excluded consistently with the model's ordinary filtering; checking all adjacent construction primitives as free collision pairs would incorrectly reject connected links.

Every front/rear foot pair has a 3D Euclidean separation constraint:

$$\|p_{front}-p_{rear}\|_2\ge0.10\text{ m}.$$

Each foot center must also remain at least 0.08 m from its own hip origin. The full self-clearance checks supplement these simple anatomical proximity guards. The distances are research margins, not staff-approved hardware safety distances. They use three-dimensional distance: a rear foot safely elevated above a front foot is not automatically rejected because their horizontal projections overlap.

## 11. Joint limits and a velocity-dependent torque envelope

Joint position and speed limits come from the original Go2 URDF. No speed or torque authority is increased. The research torque envelope is

$$|\tau_i|\le\tau_{stall,i}\max\left(0,1-\frac{|\dot q_i|}{\omega_{0,i}}\right),$$

using original effort/speed values as conservative endpoints:

- Hip/thigh: 23.7 N m and 30.1 rad/s.
- Calf: 45.43 N m and 15.7 rad/s.

This symmetric linear reduction is an **uncalibrated research assumption**. URDF maximum speed is not proven to be an identified zero-torque speed. Real motors can have asymmetric motoring/braking, regenerative, current, voltage, temperature, and controller-specific limits. The envelope must not be described as an identified Go2 motor curve. A Boolean metadata field is not measurement evidence.

The original simulator has box torque clipping and overspeed protection, rather than this complete curve. Independent replay reports any actual torque exceeding the stricter research envelope. Such a mismatch blocks export; the validator does not silently change motor gains or inject the planned torque to make the candidate appear to track.

## 12. Position-command realizability and slew

A torque solution is not directly compatible with Cornell's usual position-PD command path. The NLP therefore reconstructs an absolute target at each node:

$$q^T_{j,k}=q_{j,k}+\frac{\tau_{j,k}+K_d\dot q_{j,k}}{K_p},\qquad K_p=60,\ K_d=3.$$

This inversion matches the unsaturated nodal PD relation. Separate constraints enforce target position bounds, normalized action bounds

$$\left|\frac{q^T_{j,k}-q^0_j}{2}\right|\le1,$$

and target slew

$$|q^T_{j,k+1}-q^T_{j,k}|\le0.7s_jh.$$

The independent replay interpolates the planned target table to a 50 Hz grid, then applies the same per-command slew envelope before native PD. Export uses the **same command-sequence function**. This prevents replaying one target path and exporting a different one, or hiding a coarse-node jump that exceeds the 20 ms actuator interface.

The native simulator's PD torque is recomputed every 2 ms and retains its original torque/overspeed protection. Planned forces and torques are never sent to the base or injected as ground support. Nodal PD inversion alone is not a proof of between-node tracking.

## 13. Solver, initialization, redundant equalities, and numerical tolerances

The implementation uses SciPy `trust-constr` with constant-zero objective and zero objective gradient/Hessian, sparse finite-difference constraint dependencies and a constraint-Hessian approximation. SciPy 1.16.3 was already present in the working environment; it is an additional optimizer dependency not listed in the inherited project lockfile. A small experiment-specific requirements file pins it explicitly for fresh reproduction.

Two initialization methods are available:

1. `linear`: interpolate configuration, compute finite-difference velocities/accelerations and a particular inverse-dynamics force/torque guess. It is deliberately audited rather than labeled feasible.
2. `ik`: solve foot-position constraints along a body-pose path, project stance velocity/acceleration to the instantaneous contact tangent, and initialize forces/torques. This often improves geometry/contact residuals but does not automatically satisfy Euler dynamics.

A projection bug was found in development: replacing joint velocities with a minimum-norm stance solution erased nullspace/swing motion. The corrected code projects a **correction** instead, preserving unconstrained rear-joint motion. A regression test covers this. It still does not turn a kinematic seed into a dynamic trajectory.

Exactly fixed initial/terminal pose/velocity variables and swing force entries are eliminated. The first position step is also fixed by the initial zero velocity. Contact equalities already implied by those fixed boundaries are omitted from the numerical solve to avoid duplicating Jacobian rows. The independent audit checks the full original set again, including eliminated rows. No physical condition is removed from final acceptance.

Default acceptance thresholds are maximum scaled equality violation <=1e-5, scaled inequality violation <=1e-6 and bound violation <=1e-9. Examples: a ground-gap equality scaled by 0.1 m requires approximately 1 micrometre residual; acceleration scaled by 9.81 m/s² requires approximately 9.81e-5 m/s² residual. These tight numerical tolerances are not a statement about real sensor precision.

`--max-seconds` is a solver-callback budget, not a guaranteed hard OS deadline: one Jacobian or factorization can run beyond the budget before the next callback. Reported iteration counts and residuals are retained even when the budget terminates a run. `NOT_FEASIBLE` means the **returned candidate fails the audit**, not that a global infeasibility theorem has been proved.

## 14. Independent native replay and export gate

A fresh MuJoCo data instance starts at the crouched boundary with zero velocity. It receives only the bounded 50 Hz position targets. Physics runs continuously at 500 Hz, with no intermediate state resets, external support, contact locking, gain adjustment or collider change.

Validation checks:

- Complete independent re-audit of the saved decision vector.
- Exact agreement between saved duplicated state/target arrays and reconstructed decisions.
- Hash identity of the candidate archive.
- Original strict handstand predicate for five continuous seconds after the planned terminal time.
- Additional terminal joint and body-XY matching.
- Tracking-error bounds, joint/speed legality, native collision exclusion and stricter speed/torque-envelope mismatch.

Validation uses simulator truth only as a scorer; the deployable reference actor does not read simulator state. It is a new crouched-rest experiment, not the released controller's 67/100 standing-reset test.

Export recomputes replay rather than trusting a `native_replay_pass` flag in a saved JSON file. It refuses to export when mesh feasibility or replay fails, and also refuses while the speed/torque model remains uncalibrated. Even a future numerically accepted archive remains `hardware_ready: false`; staff approval, calibration and a supervised physical procedure remain separate.

## 15. Cornell-compatible staff-review surface

Required group-style file: `go2_deploy/custom_implementation/go2trajectory.py` (provisional filename).

It implements all four exact Cornell components:

- `CustomDeployConfig`
- `get_custom_buffers_dict()`
- `update_custom_buffers(main_controller, t)`
- `get_custom_obs_pieces()`

It is self-contained for reference inference and depends only on NumPy/Torch plus standard Python at runtime. The offline optimizer/validator additionally uses the existing model code, MuJoCo and SciPy. No SDK is imported by the custom file. The instructors' core runner is not modified.

### Observation and action contract

This new position-reference controller uses **43** channels, not the previous CapturePolicy's 47. No dummy COM estimate is supplied:

| Slice | Value |
|---|---|
| 0:3 | Calibrated body projected gravity |
| 3:6 | Body gyro, rad/s |
| 6:18 | Canonical joint angle minus exact nominal |
| 18:30 | Joint velocity / `[30.1,30.1,15.7]` repeated four times |
| 30:42 | Previous acknowledged applied target offset / 2 |
| 42:43 | Completed command steps ×0.02 /8 |

The reference actor selects the 12 normalized position actions from that phase. It is an **open-loop body reference with low-level PD**, not the prior capture/balance feedback controller. It must not be substituted for the successful controller under the same checkpoint identity.

All observation getters read one synchronized software snapshot and already return normalized pieces. Observation divisors are one; Cornell's separate `dof_pos_target=2` applies action scale once. The custom previous-target piece has a separate name to avoid accidental double scaling.

Activation defaults to disabled. Software validation requires a hash-checked, appropriately formatted exported archive. Fresh finite canonical telemetry is required at each control tick. Phase advances only after the prior bounded target is explicitly acknowledged. Stale/invalid data, missing/wrong acknowledgement, incorrect cadence, cancellation, and a physical-publisher attribute disable the interface. Tests use a clearly identified toy archive only to exercise the interface; it is not represented as an optimized handstand.

### Remaining runner capabilities

The unmodified Cornell runner does not satisfy this pipeline through the four hooks alone. Staff would need to review and isolate:

1. Loading the new reference actor and exact nominal posture/gains.
2. Enforcing actual 50 Hz operation with EMA disabled.
3. Supplying synchronized calibrated telemetry and acknowledging actual bounded target application.
4. Verifying the actual motor speed/torque and overload protections.
5. Startup, watchdog, cancellation, loss-of-process behavior, harness, contact loads and recovery.

The original Unitree permutation `[3,4,5,0,1,2,9,10,11,6,7,8]` remains in Cornell's utility code; the new actor stays in canonical order. A custom class declaring 50 Hz does not automatically override the stock runner's runtime defaults.

Do not instantiate the hardware `MainController` for a software dry run. It initializes SDK clients/recovery independently of the custom hook. The software guard prevents this candidate's physical policy activation; it is not a physical emergency-stop implementation.

## 16. Reproduction commands

Use Python 3.11 and MuJoCo 3.11.0, from this branch. Restore the original reference from the existing handstand release; expected SHA256 is `21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`. Keep scratch outputs outside the staff submission. No protected checkpoint is overwritten.

```bash
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/go2-mpl
# After installing the repository dependencies in the chosen environment:
uv pip install --python .venv/bin/python -r experiments/trajectory_optimization/requirements.txt

# Audit a warm start. This is explicitly NOT an optimized success.
python scripts/optimize_go2_trajectory.py \
  --seed ik --intervals 12 --audit-only --output output/trajectory_seed_NEW

# Bounded feasibility solve; no task reward or objective term is added.
python scripts/optimize_go2_trajectory.py \
  --seed ik --intervals 24 --iterations 12 --max-seconds 120 \
  --output output/trajectory_candidate_NEW

# Independent native replay. This is diagnostic even for a failed candidate.
python scripts/validate_go2_trajectory.py \
  --candidate output/trajectory_candidate_NEW

# Export is deliberately blocked unless every numerical/replay/calibration gate passes.
python scripts/validate_go2_trajectory.py \
  --candidate output/trajectory_candidate_NEW --export output/trajectory_staff_NEW

python -m pytest tests/unit_tests/test_trajectory_optimization.py \
  tests/unit_tests/test_trajectory_deployment.py -q
```

A fresh clone must obtain the hash-locked NPZ before the reference-dependent tests can run. Report missing-asset skips explicitly; the measured local validation below used the actual reference. The full old handstand regression suite can also be run with `GO2_HANDSTAND_TRAJECTORY` and `GO2_HANDSTAND_SHA256` set as in the existing reproduction guide.

## 17. Evidence, measured outcome, and next action

This section is completed from the actual local test/solver/replay reports. No invalid trajectory is promoted because it reaches the terminal boundary in a table of optimized poses. The scratch directories retain failed and superseded solver attempts, including the singular-Jacobian attempt and warm-start projection correction.

Validation on Python 3.11 / MuJoCo 3.11.0: **52 tests passed, zero failures, zero skips** across the two new suites and the existing replay, robustness, takeoff and capture suites. Ruff checks passed. Tests include a feasible static crouch fixture, dynamics/contact constraints, zero objective, joint mapping, loader compatibility, phase acknowledgements, stale/invalid telemetry, archive tampering and rejected export. A static crouch fixture is not a successful handstand.

| Attempt | Mesh intervals | Measured outcome |
| --- | ---: | --- |
| Original linear seed / full equations | 6 | Singular-Jacobian warning; 12 iterations, 143.54 s; no feasible solution |
| Reduced-equation linear solve | 6 | 20 iterations, 41.25 s; scaled equality error 0.829, inequality error 163.304 |
| Initial IK solve | 12 | 30 iterations, 118.92 s; failed; subsequently fixed the seed projection that erased rear-leg nullspace motion |
| Corrected IK solve | 12 | 37 iterations, 152.22 s; scaled equality error 3.430, inequality error 2.137; failed |
| Final finer-mesh IK solve | 24 | 12 iterations, 99.93 s; iteration budget reached; failed |

For the final 24-interval candidate, maximum scaled equality violation is **4.294179686**, inequality violation **0.565710752**, and variable-bound violation **0.026746913**. Representative raw violations are 19.29491 in generalized dynamics (mixed N/Nm rows), 0.0143916 m/s² stance acceleration, 0.00113142 m/s excess stance velocity, 0.003 m self-clearance deficit and 0.0232958 rad commanded-joint-limit excess. Euler velocity defect is 4.29418 in mixed generalized velocity units; Euler position defect is 0.198639 in mixed tangent-position units. These residuals exceed acceptance tolerances by large margins.

Independent native replay of that candidate, using the shared interpolated 50 Hz target sequence, detects **RL_calf–ground collision at 1.452 s**, with **zero seconds of continuous terminal hold**. Maximum recorded joint/root-position/root-orientation tracking errors are 0.44550 rad / 0.42849 m / 1.01595 rad. Tracking references use the preceding state knot, so these are knotwise diagnostics rather than interpolated desired-state errors. Prefix joint legality also fails. The provisional torque-speed envelope excess is zero in this failed replay; this does not validate a real motor curve. Export raises a rejection and creates no staff asset directory.

Final candidate SHA256: `477921e5bfe924c5b3e448fa08b079773e8bc6073d90ef4e25fa8f972e15aba0`. Scratch reports and test XML are excluded from Git under `output/trajectory_*` in the experiment worktree. The successful CapturePolicy/checkpoint is unchanged; this new optimizer has **not** reproduced its 67/100 result. No feasible trajectory, deployable asset, or physical readiness is claimed.

The measured candidate failures do not establish global task infeasibility. The smallest useful continuation is to improve the dynamically consistent warm start, refine the mesh with an explicit contact-model study, and consider impact-aware contact phases if continuous front stance is too restrictive. A sparse automatic-differentiation or analytic-Jacobian backend could reduce cost; it should preserve the same zero task cost and independent acceptance criteria. Do not relax collision, stance, torque or success constraints merely to obtain a passing solver message.

## 18. Sources and mathematical interpretation

- [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html): generalized dynamics and compliant contact modeling; native simulator contacts differ from ideal hard point constraints.
- [SciPy trust-constr documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html): constrained solver, internal merit/barrier machinery and termination semantics.
- [Tedrake, Planning and Control through Contact](https://underactuated.mit.edu/contact.html): hybrid contact modes and impact-aware trajectory formulation background.
- Repository implementation: `experiments/trajectory_optimization/model.py`, `transcription.py`, `validation.py`; CLI files `scripts/optimize_go2_trajectory.py` and `scripts/validate_go2_trajectory.py`; staff interface `go2_deploy/custom_implementation/go2trajectory.py`.
- Original scorer/model reference: `gym/envs/go2/handstand_replay.py` at the unchanged research baseline.

These sources motivate the mathematical setup; actual outcomes are measured locally and should not be inferred from the existence of the equations. **Nothing in this report authorizes a physical handstand attempt.**
