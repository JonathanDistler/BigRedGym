# MuJoCo takeoff diagnosis, 2026-11-09

Subsequent [reference optimization](go2kickup_reference.md) establishes a viable
opt-in trajectory. The measurements below describe the original reference.

Do not start the previously proposed 1,500-iteration runs on the strength of
these results. The reference alone does not establish a viable front-supported
rear takeoff. No PPO training, reward changes, residual curriculum changes, or
checkpoint writes were performed in this diagnosis.

## Checkpoint and method

The newest local trained kick-up checkpoint is
`logs/go2kickup_height_finetune_20261110_000043/Nov10_00-00-53_/model_475.pt`,
SHA256 `e05db5999c9e35cedf8f91142b791d625f8e734e04dc2569389ce3510863f2cb`.
Its saved curriculum level is **3**, with zero successes in the current batch.
This is a different run from the older iteration-1225 sync/GRF checkpoint.
The local kick-up controller and config source files match this run's source
snapshot byte for byte; `--saved-config` additionally restores effective CLI
configuration from `effective_config.json`.

`scripts/diagnose_kickup.py` compares three independent CPU MuJoCo simulations:

1. Reference alone: position/velocity residuals and feedforward torque are zero.
2. Zero residual: zero actions through the actual policy action setter, including
   its clipping, scaling and slew limiter.
3. Learned policy: deterministic actor inference through the same action setter.

Each case uses the same constructor seed (7), reset seed (71), configuration,
and four standing initial states. Initial root/joint state hashes are checked
for equality. Domain randomization is retained with matching seeded draws.
There is no observation noise during inference. Each trajectory lasts two
seconds, covering the complete first kick, without automatic resets. First
termination/timeout is tracked and post-termination samples are excluded from
summary measurements. There were **no terminations/timeouts** in these trials.
CSV samples are taken at 500 Hz, with the state immediately before applying
the recorded torque. Actor inference is at the saved 100 Hz frequency.

The active reference is `Go2Kickup._kick_reference`, called by
`_standing_posture` and the torque controller; the standalone legacy
`KickupCPG` is not the task's controller. Phase boundaries are settling
0–0.15 s, crouching 0.15–0.70 s, extension 0.70–0.90 s, then static hold.
CSV files contain all 12 reference, desired, actual and residual joint positions,
both requested and joint-limit-clipped position targets, all 12 applied motor torques, all four normal forces and world foot positions,
ground-relative clearances, front anchor drift, phase, clock and done flags.
The controller clips position targets to joint limits; the requested columns retain any pre-clip values.

## Quantitative comparison at the saved stage (3)

Clearance is each episode's maximum of the minimum of both rear-foot
clearances. Drift is the mean across episodes of the maximum front-foot drift.
Knee excursions compare the samples nearest 0.70 and 0.90 seconds, averaged
over four episodes. Torque is peak absolute knee torque in the extension phase.

| Case | Rear clearance range | Mean maximum front drift | Desired rear knee extension, L/R | Actual rear knee extension, L/R | Peak rear knee torque, L/R |
| --- | --- | --- | --- | --- | --- |
| Reference alone | 2.55–3.12 mm | 18.77 cm | 1.233 / 1.233 rad | 1.150 / 1.156 rad | 28.85 / 26.95 Nm |
| CPG + zero residual | 2.55–3.12 mm | 18.77 cm | 1.233 / 1.233 rad | 1.150 / 1.156 rad | 28.85 / 26.95 Nm |
| Learned residual | 4.33–7.30 mm | 2.34 cm | 0.513 / 0.433 rad | 0.881 / 0.790 rad | 23.27 / 19.23 Nm |

Reference and zero-residual CSV files are byte-identical. Thus the action
setter does not inadvertently remove a zero-action reference.
Mean peak rear normal forces during extension are 141.4/133.6 N for reference
and 150.8/97.9 N for learned residual. The latter is notably asymmetric.
The calf torque limit is 45.43 Nm; observed peaks do not support calf torque
clipping as the immediate explanation.

The policy cancels **58%/65% of the nominal rear-knee target excursion**. Its
positive knee residual during crouching makes the crouch shallower, then moves
negative during extension. Measured joint extension falls by about 23%/32%.
Nevertheless, the policy improves front anchoring and produces a small rear
release that the reference lacks. Per-episode *total* time with both rear forces
below 0.015 body weights, both clearances above 2 mm, and both front forces
above the saved minimum load is 122, 118, 182 and 120 ms. These durations can
include separate intervals; they are not continuous success holds. They do not
apply the drift, front-speed, upward event, or stage target gates. **None of the
four episodes reaches stage 3's 30 mm clearance target.**

Stage-zero amplitude (0.70) also fails for reference/zero residual: 2.03–2.36 mm
clearance, 14.89 cm mean maximum front drift, no simultaneous rear flight.
The learned policy reaches 1.62–4.03 mm with 1.89 cm drift.
At full amplitude (1.00), reference/zero residual still yields no simultaneous
rear flight: 2.91–4.23 mm clearance, 23.70 cm mean maximum front drift.
Learned residual reaches 9.36–16.02 mm with 2.17 cm drift, still far below the
final-stage 160 mm target. These four seeded trials at each level are diagnostic
comparisons, not an estimate of reliability over training seeds.

![Stage 3 joint, torque, force, clearance and drift traces](../output/kickup_diagnosis_stage3/comparison.png)

## Playback fixes and reset findings

Previously `play.py` used current local configs unless `--original_cfg` was
specified. Its default standing override disabled the curriculum reset, so
kick-up's per-episode level remained zero even after restoring saved level 3.
That changes both reference amplitude (0.85 to 0.70) and policy goal inputs.
All kick-up curriculum reset poses in this run are already standing starts.

Playback also extended every episode to at least 50 seconds and never called
`runner.reset_envs()`. The one-shot kick therefore happened only at startup,
followed by a static extended target for the rest of playback. An observer
opening the viewer later could miss the entire 0.20-second extension.
Early episode resets are not the cause in the measured trajectories.

Playback now defaults to saved configs for `go2kickup` (`--current_cfg` explicitly
selects local configs), retains the saved standing curriculum stage, keeps the
saved episode duration, and repeats episodes on timeout/termination. Playback
clears curriculum eligibility before repeated resets so viewing cannot promote
the saved stage. Other tasks retain their existing reset behavior.
A CPU MuJoCo playback check confirmed saved level 3, four-second episodes,
extension in both episodes, and timeout resets at simulated 4 and 8 seconds;
no premature termination occurred. The reset clock is about 0.01 s ahead in the
first episode because checkpoint restoration performs an initial physics step.
The repeat/frozen-curriculum regression and existing kick-up tests pass (22).

## Decision

A bounded residual curriculum is conditional on a reference that already makes
a useful takeoff. That condition is **not met**: protecting this reference would
also protect its excessive front drift and missing coordinated rear release.
Therefore residual bounds and front success thresholds were left unchanged.
Contact requirements and all checkpoint files were preserved. A later curriculum
should first establish a reference that produces simultaneous rear clearance
with both front contacts retained; only then test restricted residuals against
that measured baseline before relaxing and progressively tightening front drift.
No additional long PPO run is recommended yet.

Reproduce the primary comparison:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/diagnose_kickup.py \
  --checkpoint logs/go2kickup_height_finetune_20261110_000043/Nov10_00-00-53_/model_475.pt \
  --saved-config --level 3 --episodes 4 --seconds 2 \
  --output-dir output/kickup_diagnosis_stage3_repeat
```

Use `--level 0` or `--level 6` to reproduce the amplitude checks. Raw results
are under `output/kickup_diagnosis_saved/` (stage 0),
`output/kickup_diagnosis_stage3/` and `output/kickup_diagnosis_full/`.
The checked-in evaluator uses the current controller with saved configuration;
its provenance explicitly reports whether controller source matches the snapshot.
A false match flag requires separate saved-controller reproduction before
claiming behavior from an older run.

View the corrected policy:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/play.py --task go2kickup --device cpu --num_envs 1 \
  --experiment_name go2kickup_height_finetune_20261110_000043 \
  --load_run Nov10_00-00-53_ --checkpoint 475 --seed 7 --no-keyboard
```
