# Go2 contact control and robot metrics

Training and playback automatically stream metrics for environment 0 to a new
`output/<UTC timestamp>/` folder. Use `--output_dir PATH` to select the folder
and `--metrics_num_envs N` to record more parallel robots. The folder contains
`robot_metrics.csv`, `metadata.json`, and `summary.json`. Playback also saves
its height plot there. Summary statistics are written on normal completion or
Python exception/keyboard interruption; the CSV is flushed every 100 steps.

Existing checkpoint playback (no policy architecture change):

```bash
uv run scripts/play.py --task go2balance --original_cfg --num_envs 1 --output_dir output/balance_eval
```

Train with the optional force-coupled CPG and permanently unloaded rear legs:

```bash
uv run scripts/train.py --task go2balance --cpg --contact_modes stance stance swing swing --headless --output_dir output/cpg_balance
```

The four modes are ordered **FL FR RL RR**:

- `phase`: desired stance when sin(phi) > 0, swing otherwise.
- `stance`: always request support.
- `swing`: always request no contact, regardless of oscillator phase.
- `ignore`: express no contact preference; omit this leg from the CPG reward.

The CPG uses independent clocks with configured initial phase offsets, filtered
upward forces normalized by randomized body weight, and
`phi_dot = 2*pi*frequency_hz - sigma*load*(cos(phi)+zeta)`.
Settings live in `Go2Cfg.contact_cpg`. It adds eight sin/cos phase observations,
four desired-contact observations (1 support, 0 no contact, -1 ignored), and a
bounded contact reward with weight 1. The policy still controls joint targets.
Use `--original_cfg` to restore saved CPG settings during playback. Enabling
CPG inputs changes network dimensions, so old checkpoints cannot directly
resume or warm-start a CPG policy.

These contact modes supplement existing task rewards. Handstand/balance tasks
still require both front feet to support the robot. To practice kick-up, use
`go2handstand` and phase-driven rear contacts rather than permanently forbidding
the rear-leg contact needed to push. Modes do not physically remove limbs.

CSV metrics include per-foot world force xyz and magnitude (N), upward load
share, contact (>1 N upward load), supporting-leg count, and front/rear/overall
left-right symmetry. Symmetry is `1 - abs(L-R)/(L+R)` and is blank when the pair
has <=1 N upward load, so an airborne pair is not scored as perfectly balanced.
Base position, xyzw quaternion, roll/pitch/yaw, nose-down handstand angle error,
body-frame linear/angular velocity, angular speed, joint angles/velocities,
termination flags, episode identifiers, and CPG phase/contact targets are also
recorded. Euler angles become ambiguous near vertical; use the quaternion and
handstand angle error there. Angles are radians and velocities use SI units.

Forces are backend net foot contact forces, sampled at the final physics
substep of each control step. On the configured flat plane these represent
foot GRF; they are not time-integrated impulses or peak impact forces across
substeps. Non-foot collisions are not included in foot load totals. Summary
means pool the selected robots and sampled steps; peak values are absolute
sampled maxima. Logging additional robots increases disk and GPU transfer cost.
