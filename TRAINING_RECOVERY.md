# Train longer balance holds, then recovery

Balance and recovery now charge a discrete -20 reward for termination, instead
of the previous timestep-scaled -2 rate (-0.02 at the default 100 Hz). Normal timeouts
carry no fall penalty. Balance also terminates after both front feet remain
below 1 N for 0.10 seconds, following a 0.15-second reset grace period. Existing
body-contact and low-height termination still apply. Recovery retains its
stricter individual-foot contact/slip conditions. These settings are starting
values to validate, not evidence that a policy will stop falling.

Use a new warm-start run to apply these reward changes; loading an original
saved configuration can retain the old rewards. Increasing training duration
alone does not apply the new objective. First validate nominal balance, then
expand the initial-condition range gradually. Compare nominal and perturbed
survival separately at fixed checkpoints. A broader reset distribution generally
needs more samples than a single initial pose; use measured survival and
curriculum progress to decide the budget rather than matching iteration counts.

Run from the repository root. Commands below use MuJoCo CPU and your existing
CPG policy at `logs/go2balance/Oct31_23-29-27_/model_1500.pt`.
They start new experiments and preserve the original run.

## 1. Train the longer balance hold

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 uv run --frozen -m scripts.train \
  --task go2balance \
  --warm_start logs/go2balance/Oct31_23-29-27_/model_1500.pt \
  --warm_start_task go2balance \
  --cpg --contact_modes stance stance swing swing \
  --device cpu --num_envs 32 --max_iterations 1500 --save_interval 100 \
  --experiment_name go2balance_longhold --headless --disable_wandb
```

The new task requires a five-second continuous strict hold, uses 45-second
episodes, and promotes at 90% success in two consecutive batches of at least
128 completed primary episodes. A later fall invalidates an earlier hold.
At challenge levels, the scheduled impulse must occur before the successful
recovery hold. Easier replay is retained. This is an initial training budget;
1500 iterations does not guarantee a reliable controller.

Warm-starting loads actor/critic weights with fresh optimizers and curriculum
progress. `--warm_start_task` also imports the source policy's observation,
action, network and motor settings while retaining the new task's rewards and
curriculum. The explicit CPG flags are needed for this older run because its
source snapshot does not record the original CLI overrides. If you used
additional nondefault CPG parameters, restore those too.

On an existing working CUDA setup, balance training can instead use
`--device cuda:0 --num_envs 4096`; GPU dependencies must already be installed.
Recovery training below remains CPU-only. CPU iterations can be slow because
the default rollout contains 32768 transitions, independent of environment count.
`OMP_NUM_THREADS=1` avoids excessive CPU thread overhead. Clearing `PYTHONPATH`
avoids the unrelated Python 3.12/Conda NumPy path present in this shell.

Watch `[handstand curriculum]` messages and the run's `vitals.jsonl`. Compare
strict success, contact support and falls, rather than selecting by reward alone.
New runs save `effective_config.json` alongside their source snapshots; keep
both that file and `files/` when moving a run.

## 2. Test long-duration balance

Replace `YOUR_BALANCE_RUN` with the timestamp folder under
`logs/go2balance_longhold/`. Select a checkpoint explicitly for comparisons:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 uv run --frozen -m scripts.evaluate \
  --task go2balance --experiment_name go2balance_longhold \
  --load_run YOUR_BALANCE_RUN --checkpoint 1500 --original_cfg \
  --device cpu --num_envs 32 --survival --seconds 60 --seed 42 \
  --output_dir output/longhold_seed42
```

Repeat with seeds 43 and 44 and separate output folders. The saved effective
configuration enables CPG automatically for new runs. Survival evaluation
disables balance pushes by default; add `--eval_pushes` for a separate challenge
test. Pushes still obey the saved curriculum level and settled-hold gates.

`survival.json` reports each robot's first termination, observed duration,
best hold, and front-contact loss. A survivor has a null failure time: it has
only been demonstrated to survive the observation horizon. Contact loss itself
is a diagnostic, not a fall. `failure_env_N.json` holds the preceding second,
including joint targets/torques, foot positions/velocities/forces and body state.
The report includes checkpoint identity, seed and joint/foot ordering.

Trials never reset. Once a robot falls, its trial statistics and trace stop;
the vectorized physics batch may continue simulating it until other trials end.
The separate raw `robot_metrics.csv` can therefore contain post-fall samples.
Use `survival.json` for fall counts, not the mean termination flag in `summary.json`.

Before expanding recovery, aim for consistent 60-second survival across seeds.
Any remaining failure trace should guide the next change; these settings do
not guarantee that falls disappear.

## 3. Train rear-leg recovery

Choose the strongest balance checkpoint based on those tests:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 uv run --frozen -m scripts.train_backwards_reachability \
  --warm_start logs/go2balance_longhold/YOUR_BALANCE_RUN/model_1500.pt \
  --device cpu --num_envs 32 --max_iterations 1500 --save_interval 100 \
  --experiment_name go2reachability_expanded --headless --disable_wandb
```

New-run effective settings transfer CPG automatically. The recovery curriculum
starts fresh, retains a two-second hold and 70% promotion threshold, and uses:

| Level | Position fraction | Velocity fraction |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0.025 | 0 |
| 2 | 0.05 | 0.01 |
| 3 | 0.10 | 0.025 |
| 4 | 0.15 | 0.05 |
| 5 | 0.25 | 0.10 |

These fractions scale rear-joint travel and the conditional torque-work speed
envelope. This deliberately starts with a modest region; reaching the last
level is not evidence of full-range recovery. Retest nominal balance after
recovery training to detect forgetting.

## 4. Measure recovery

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 uv run --frozen -m scripts.backwards_reachability \
  --checkpoint logs/go2reachability_expanded/YOUR_RECOVERY_RUN/model_1500.pt \
  --samples 256 --seconds 8 --seed 42 \
  --output output_backwards_reachability/recovery_seed42
```

For a before-training comparison, evaluate the selected new balance checkpoint
with the same command, sample count, seed and horizon, changing the output folder.
Repeat seeds 43 and 44. Check `metrics.json`, `recovery_metrics.png` and
`phase_space.png`. Compare accepted-proposal fractions as well as recovery
conditional on acceptance; rejection is not successful recovery.

For older checkpoints, pass `--config_task go2balance` (or `go2reachability`)
and the original CPG flags if they were only supplied via CLI. Saved settings
are loaded before environment/network construction. `--current_cfg` explicitly
opts out of saved configuration loading when necessary.

If the final level performs reliably, extend both arrays in
`gym/envs/go2/go2reachability_config.py`, with matching entries in the reset and
push arrays. Warm-start a new experiment to apply the new schedule; ordinary
resume should preserve the existing schedule and curriculum state.

This task measures rear-leg recovery with front support and body geometry near
the goal. Broader body tilt and angular-velocity recovery need additional reset
slices. Front-support violations still terminate recovery episodes.

## Resume an interrupted run

For either task, use its experiment and timestamp, without `--warm_start`:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 uv run --frozen -m scripts.train \
  --task go2balance --experiment_name go2balance_longhold \
  --resume --load_run YOUR_BALANCE_RUN --original_cfg \
  --device cpu --num_envs 32 --max_iterations 1500 --headless --disable_wandb
```

This restores optimizers and curriculum state, and retains the saved effective
settings. For recovery, substitute task `go2reachability`, experiment
`go2reachability_expanded`, and its timestamp folder.
