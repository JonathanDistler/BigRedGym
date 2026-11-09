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

Do not add `--resume`, `--warm_start`, or `--original_cfg` for a fresh run.
If the notebook host injects a Python path from another Python installation,
prefix the command with `env -u PYTHONPATH`.
Weights & Biases is disabled with the flag above; no account is required.
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
| 4 | 7 | 0.62 | 16 | 3.5 | 0.15 |
| 5 | 11 | 0.80 | 23 | 3 | 0.12 |
| 6 | 16 | 1.00 | 32 | 2.5 | 0.10 |

Each episode lasts up to four seconds. The reference settles for 0.15 seconds,
crouches the rear joints for 0.55 seconds and extends them over 0.30 seconds.
Smoothstep interpolation preserves reference position and endpoint velocity.
The inherited PD controller adds bounded, slew-limited policy residuals to
this absolute reference. Front references remain standing; their small policy
residuals allow stabilizing corrections. Reference timing is independent of the
training script's randomized timeout counters.

Success requires both rear feet to exceed the level target for 0.06 continuous
seconds with both front feet loaded, slow and within their anchor tolerance.
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
