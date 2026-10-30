# Backward reachability for Go2 front-foot handstand recovery

The target follows the successful `go2balance` policy: a front-foot-supported handstand, held for two seconds with strict balance score >= 0.5. This does **not** target four-foot standing. `go2reachability` expands rear-leg joint angles and velocities outward from that goal. It samples six independent rear joint positions inside URDF travel limits and six-dimensional velocity directions; front-leg positions and body pitch start at the supported goal geometry. Thus this is a rear-leg recovery slice, not arbitrary full-body pose recovery.

## Train

From the repository root, use the successful balance policy to initialize the networks, with fresh optimizers and a fresh curriculum:

```bash
uv run python -m scripts.train_backwards_reachability \
  --warm_start logs/go2balance/Oct28_22-26-30_/model_1500.pt \
  --device cpu --num_envs 32 --headless --disable_wandb
```

Or train from scratch with `uv run python -m scripts.train --task go2reachability --device cpu --num_envs 32 --headless --disable_wandb`. Resume with the normal `--resume --load_run ...` flags. This task currently requires MuJoCo CPU; it deliberately fails for GPU/Warp or VSim rather than approximating contact validation. Start with modest environment counts because rejection sampling uses scratch forward simulations.

The curriculum increases position-travel and velocity-envelope fractions through 0, .05, .15, .35, .65, 1. Promotion requires >=70% successful completed primary episodes over 128 episodes, twice consecutively. Twenty percent easier-level replay prevents forgetting. Exhausted rejection sampling falls back to goal replay and cannot promote the curriculum. Curriculum progress is checkpointed by the existing runner.

## Torque and speed derivation

Torque alone does not yield a finite maximum angular speed. From rest, with a fixed local inertia metric and bounded joint travel, motor work gives a **conditional sampling envelope**. For sampled rear direction d and joint vector q:

1. Form the stacked front-foot translational Jacobian J. Hold rear velocity to d and solve for base and front-joint velocities u so J u = 0.
2. Compute effective inertia I = uᵀ M(q) u from MuJoCo's full floating-base mass matrix.
3. Let L = min_j(remaining travel_j / |d_j|) over nonzero directions. Available work W = sum_j(tau_max,j |d_j| L), ending at the first rear joint limit.
4. With frozen inertia and zero initial speed, ½ I omega² <= W; sample omega <= sqrt(2W/I), capped by every moving joint's URDF velocity limit / |u_j|.

The rear hip/thigh torque limit is **23.7 Nm**, calf **45.43 Nm**. URDF speed limits are **30.1 rad/s** for hip/thigh and **15.7 rad/s** for calf. These are model limits, not a validated hardware torque-speed curve. `torque_derivation.json` reports local inertias and signed envelopes at the goal pose; `torque_velocity_envelope.png` plots them.

Gravity, changing inertia, front motor work, and nonzero initial energy prevent this local work formula from being a global physical speed bound. The actual coupled dynamics and admissibility are checked forward. No hardware safety claim follows from these numbers.

## Maintaining support

The initial velocity satisfies both front feet's zero-velocity constraints. Every proposal must then pass 40 ms of scratch forward simulation under a pose-holding PD controller: each front foot must retain >=1 N contact force and speed <=0.15 m/s. The probe never changes the accepted initial state. During recovery, the same constraints are checked each physics substep after a 30 ms reset-settling grace period; violations latch a failure and terminate the episode. This rejects invalid states and penalizes contact loss; it does not force the robot onto the ground or guarantee that an untrained policy preserves contact. The probe can reject states that another controller could recover.

## Evaluate and plot

```bash
uv run python -m scripts.backwards_reachability \
  --checkpoint logs/go2reachability/YOUR_RUN/model_1500.pt \
  --samples 256 --seconds 8 --seed 42
```

Without `--checkpoint`, evaluation uses fixed goal-posture PD and labels the plots as a baseline. Each level is sampled independently, without curriculum replay or automatic episode resets. Outputs:

- `phase_space.png`: colored success/failure projections for all six rear joint angle/velocity pairs. These projections marginalize over other sampled coordinates; overlapping colors do not imply a deterministic 2D boundary.
- `recovery_metrics.png`: admissible-proposal fraction (including rejected attempts), recovery probability conditional on acceptance with Wilson 95% intervals, and contact-failure fraction. No accepted samples is recorded as null, not zero recovery probability.
- `samples.csv` / `samples.npz`: all initial joint angles/velocities, level, acceptance, proposal count, directional velocity envelope, best hold, contact failure and recovery outcome.
- `metrics.json`: controller/checkpoint identity and SHA256, random seed, horizon, sample counts and per-level metrics.

Success requires acceptance, a two-second strict hold, and no termination anywhere in the evaluation horizon. These statistics characterize the sampled distribution. They do not measure uniform state-space volume or certify an exact backward reachable set, and the set need not be convex or conical. More samples, repeated seeds, longer horizons and independent full-state slices are required before drawing a reliable boundary. Rejected proposal coordinates are not part of the plotted recovery set; gray marks show their fallback goal replay states.

The initial `pd_baseline/` artifacts contain 16 samples per level over a 3 s horizon. Root artifacts evaluate the existing balance checkpoint; they assess that policy before reachability training. Run the command above after training to replace them with the expanded policy's measured recovery region.

An empirical target set can be defined as R(T, p) = {x: P_pi(two-second strict hold before T, with support retained throughout T | x) >= p}. Here x contains all joint positions and velocities, body state and fixed contact mode. A deterministic simulation yields binary membership per tested x; the per-level Wilson intervals estimate probabilities over the reset distribution, not confidence that every point in a projected region is recoverable. The plots deliberately avoid fitting a cone or convex hull across untested holes.
