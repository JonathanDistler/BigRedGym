# Go2 standing-start kick-up

This task learns only the rear-leg kick, with no handstand balance reward,
checkpoint import, or policy handoff. It retains Go2Handstand inheritance and
uses its completed-episode curriculum accounting and checkpoint persistence.
All reset pitches are zero; yaw is randomized. The front feet are anchored at
their reset world positions after the backend refreshes forward kinematics.

From the repository root in Molab Online (Python 3.11):

```bash
uv sync --frozen
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2kickup --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 1500 --save_interval 25 --headless --disable_wandb
```

For the existing Molab notebook, select `BRANCH = "features/kickup"`,
`TASK = "go2kickup"`, `DEVICE = "cpu"`, `NUM_ENVS = 16`,
`EXPERIMENT_NAME = "go2kickup"`, `RESUME = False`, and `USE_WANDB = False`.
Use `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in its subprocess environment.
The notebook's default balance task and 4096 GPU environments are separate presets.

Do not add `--resume`, `--warm_start`, or `--original_cfg` for a fresh run.
If the notebook host injects a Python path from another Python installation,
prefix the command with `env -u PYTHONPATH`.
Weights & Biases is disabled by the kick-up runner configuration, including when
the CLI flag is omitted; no account is required.
The standard script writes checkpoints and effective configuration under
`logs/go2kickup/` and robot diagnostics under `output/`.

`algorithm.rollout_size = 8192` yields 512 policy steps per environment at 16
environments. `algorithm.batch_size = 1024` controls optimizer minibatches,
independently of the rollout size. Checkpoints are saved every 25 iterations
and at the end of training. OMP/MKL thread limits avoid CPU oversubscription.

| Level | Both rear feet clearance (cm) | Reference amplitude | Desired pitch (degrees) | Front drift (cm) | Front speed (m/s) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.5 | 0.12 | 2 | 7 | 0.30 |
| 1 | 1.2 | 0.20 | 4 | 6 | 0.25 |
| 2 | 2.5 | 0.32 | 7 | 5 | 0.22 |
| 3 | 4.5 | 0.46 | 11 | 4 | 0.18 |
| 4 | 9 | 0.62 | 16 | 3.5 | 0.15 |
| 5 | 16 | 0.80 | 23 | 3 | 0.12 |
| 6 | 24 | 1.00 | 32 | 2.5 | 0.10 |

Each episode lasts up to four seconds. The reference settles for 0.15 seconds,
crouches the rear joints for 0.55 seconds and extends them over 0.30 seconds.
Smoothstep interpolation preserves reference position and endpoint velocity.
The inherited PD controller adds bounded, slew-limited policy residuals to
this absolute reference. Front references remain standing; their small policy
residuals allow stabilizing corrections. Reference timing is independent of the
training script's randomized timeout counters.

Success requires both rear feet to exceed the level target for 0.06 continuous
seconds during the 0.70–1.65 s kick window, with both front feet loaded, slow
and within their anchor tolerance. Both rear contacts must release. Rear takeoff
and extension onset differences must be within the level's tolerance (120, 100,
80, 70, 60, 50, 40 ms); rear height mismatch must be at most 4 cm. Extension
onset means actual knee extension speed exceeds 0.2 rad/s during push-off.
Takeoff requires prior loading, less than 1.5% body weight of rear-foot load,
at least 2 mm world clearance, and positive world vertical foot velocity.
A fall, excessive accumulated drift, or more than 0.08 continuous seconds of
front support loss/sliding disqualifies that episode. Front movement and contact
also shape dense rewards before success. Clearance uses world foot-centre z
minus the URDF sphere radius above the backend's z=0 plane, never knee angles
or the front feet's height. Non-plane terrain is explicitly unsupported.

Promotion requires at least 40% success in two consecutive batches of at least
64 completed episodes at the current level. These are on-policy episode outcome
batches, not a separate deterministic evaluation process. Manual resets and
older-level episodes in flight do not count. Final-level success still measures
kick clearance and front support; it never requires vertical balance.
Torso contact and collapsed base-height termination remain enabled.

The objective uses these phases: settle (0–0.15 s), crouch (0.15–0.70 s),
push-off (0.70–1.00 s), and lift/kick (0.70–1.65 s). Lift shaping overlaps
push-off so early successful releases receive credit. The rear reference has
one shared clock; it does not constrain the two policy residuals to be equal.
The legacy `KickupCPG` also explicitly shares its rear oscillator phase.

Reward weights live in `Go2KickupRunnerCfg.critic.reward.weights`; thresholds
and normalization parameters live in `Go2KickupCfg.reward_settings`.
All rates are multiplied by the control time step by the existing PPO runner.

| Term | Weight | Definition |
| --- | ---: | --- |
| `kick_clearance` | 5.0 | Minimum world rear-foot clearance / stage target, capped at one; gated by front support and drift/speed quality during lift/kick. |
| `kick_upward_velocity` | 1.0 | Positive minimum world rear-foot vertical velocity / 0.6 m/s, capped at one during lift/kick. Folding on the floor earns zero. |
| `kick_synchronization` | 0.6 | Smooth Gaussian height and vertical-velocity agreement (4 cm and 0.4 m/s scales), multiplied by minimum clearance progress. |
| `kick_extension_timing` | 0.15 | Negative squared rear knee extension-rate mismatch (2 rad/s scale) plus onset timing mismatch, capped at four during push-off. Joint angles may differ. |
| `kick_takeoff_timing` | 0.4 | Negative squared takeoff delay / stage tolerance, capped at four. A missing second takeoff accumulates delay. |
| `kick_release` | 0.6 | Supported minimum-clearance reward when both rear feet are unloaded and takeoffs are synchronized. |
| `kick_push_impulse` | 0.3 | Bounded, balanced, useful rear impulse rate during push-off, described below. |
| `kick_force_spikes` | 0.3 | Negative physics-step mean squared excess foot load above 1.5 body weights, covering front/rear feet and all phases. |
| `kick_rear_slip` | 1.0 | Negative squared rear horizontal foot speed while loaded. |
| `kick_front_support` | 1.0 | Minimum front load / 0.06 body weights, capped at one; both feet must support. |
| `kick_front_displacement` | 10.0 | Negative summed squared front world anchor displacement. |
| `handstand_slip` | 4.0 | Existing negative front horizontal slip speed squared while loaded. |
| `kick_motion` | 0.15 | Rear reference tracking during crouch/push-off; front joints remain free to stabilize. |
| `kick_pitch` | 0.5 | Existing supported progressive nose-down pitch shaping. |

Existing torque, action-rate, joint-limit, impact/contact, collision, and angular
instability penalties remain active. Positive lift rewards require valid front
support, speed and drift, so jumping with all four feet cannot earn them.

MuJoCo's `rnePostConstraint` world-frame external foot forces are already exposed
by the CPU backend. Every physics substep integrates positive vertical rear-foot
force times physics dt to log per-foot push impulse in N s. Force credit is
restricted to the extension window, valid front support, and before each foot's
first takeoff; rear force in flight and subsequent landings earns zero.
For reward, subtract 0.25 body weights per foot (static support), cap loads at
1.5 body weights, and multiply by positive actual knee extension speed / 1 rad/s
(capped at one). Thus stationary support earns no push reward. Each foot's
credited impulse saturates at `mass * gravity * 0.18 s` per episode. Divide the
new credited impulse by `mass * gravity * control_dt * 2 * 0.65`, cap at one,
and multiply by `2 * min(left, right) / (left + right)`. The runner's dt factor
then yields a dimensionless impulse return; timestep, mass and gravity changes
do not create an arbitrary force-reward scale. Force has a deliberately smaller
weight than clearance, support, velocity and synchronization.

Local rollout diagnostics include separate rear-left/right clearance and
push impulse, minimum rear clearance, extension/takeoff timing differences,
both-takeoff fraction, front drift/slip, episode success and completed-batch
curriculum success. Timing differences are zero before either event and accrue
while the second event is missing; read them with `both_takeoff_fraction`.

Validation:

```bash
uv run python -m pytest tests/unit_tests/test_go2_kickup.py tests/unit_tests/test_kickup_cpg.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run scripts/train.py \
  --task go2kickup --backend mujoco --device cpu --num_envs 16 \
  --max_iterations 1 --headless --disable_wandb --experiment_name go2kickup_smoke
```

A smoke test establishes initialization and PPO execution, not trained kick
reliability. Inspect clearance, front drift/slip, and curriculum success over a
full training run before evaluating a controller. The separate legacy
`KickupCPG` helper remains available, but this task uses its existing inherited
PD reference hook rather than the helper's handoff interface.

Implementation validation (2026-11-09): 69 tests passed across kick-up, CPG,
handstand/curriculum, CPU MuJoCo physics/reset, contact diagnostics and task
registration. Ruff and `git diff --check` passed. A fresh CPU PPO iteration
completed with 16 environments and 8,192 samples, saving `model_0.pt`,
`model_1.pt` and the requested `kickup/*` vitals under
`logs/go2kickup_sync_final_smoke/Nov09_15-11-46_/`. W&B was disabled by config
without the CLI flag. Value loss was 0.0153 and surrogate loss was 0.00502;
no curriculum successes were recorded in this untrained smoke run. Molab's
hosted runtime was not directly exercised.
