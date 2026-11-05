# CPG kickup prototype

`go2kickup` starts from standing and runs one synchronized oscillator per leg
(FL, FR, RL, RR). The first half-cycle crouches; the second holds the front
legs crouched and extends the hind legs. The primitive supplies absolute joint
angles through the existing torque-limited PD controller. It changes neither
the contact CPG nor the balance actor's observation/action dimensions.

For full-kick playback with an existing balance checkpoint, set
`kickup_curriculum.enabled = False` in the config, then run:

```sh
uv run scripts/play.py --task go2kickup --experiment_name go2balance --load_run RUN --checkpoint ITERATION --num_envs 1
```

See `scripts/play.py --help` for backend and checkpoint options. Match the
checkpoint's observation configuration, including any optional contact CPG.

Tune duration and joint poses in `gym/envs/go2/go2kickup_config.py` in simulation.
These initial values have not demonstrated successful kickup. This is a joint
trajectory CPG, without force entrainment or adaptive momentum control yet.

Override `Go2Kickup.kickup_in_region()` with the existing ROA membership test;
it must return a Boolean tensor with one entry per environment. The default
pitch/speed/front-support gate is only a heuristic. Handoff latches until reset
and restores full policy control immediately. A completed cycle without region
entry terminates as failure rather than repeating the kick. Existing fall
checks remain active after the configured initial grace period.

The task retains the standing joint reference and balance policy interfaces.
Use playback to tune this primitive against a frozen recovery policy before
training anything new. The runner now adds a bounded pitch-progress reward to the inherited balance
rewards. With the kickup curriculum enabled (default), the actor adds corrections
of at most 0.15 radians to the CPG targets during the kick. After handoff it has
full control. Disabling the curriculum restores the original fixed full kick.

## Training curriculum

This is environment-side training logic; the PPO loop needs no special stage
scheduler. Tune `Go2KickupCfg.kickup_curriculum`:

- Pitch targets: 15, 30, 45, 60, 75 degrees.
- CPG amplitudes: 35%, 50%, 70%, 85%, 100% of the configured joint excursion.
- Promotion: at least 70% successes in two consecutive batches of 128 completed
  episodes at the current level. Manual resets and older-level episodes do not
  count. Existing episodes keep their assigned amplitude when the level changes.
- Early stages finish successfully on reaching their pitch target with front
  support. The final stage requires actual handoff and a 0.5-second balance hold;
  the final pitch target shapes reward but cannot certify success.
- Curriculum level and promotion counters are included in training checkpoints.

The policy learns bounded residual joint commands, not oscillator parameters.
Amplitudes and thresholds are initial guesses and need empirical tuning. Train
with `go2kickup` using the existing training entry point. A single actor currently
handles both kick corrections and recovery; freezing a separate recovery actor
would require an additional policy-routing training setup. Actor observation
sizes remain compatible, but no explicit curriculum goal observation is added.

At implementation time `features/kickup` and local `cpg_front` shared commit
`efe13b2`. Changes are new modules/tests/docs plus additive task registrations;
existing contact CPG and balance/reachability files are unchanged. This avoids
conflicts with that branch tip; future changes can still require resolution.
