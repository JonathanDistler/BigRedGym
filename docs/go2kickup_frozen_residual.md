# Frozen balance plus bounded recovery residual

This experiment is separate from the regressing actor on `ppo_kickup`. No
original balance or kick checkpoint, CPG implementation/reference, posture
transition controller, atlas, recovery dataset, or previous evaluation was
changed. No 200-update run was launched. The earlier interrupted experimental
files are retained; the completed diagnostic uses a new output directory.

## Why the direct PPO fine-tune forgets

The first actor update already reduces near-vertical retention from 14/16 to
12/16. This is an immediate measurable loss, rather than the complete collapse
seen by iteration 25. A fresh seed-42 reproduction starts from the original,
with fresh Adam optimizers; it never resumes the failed 50-update actor.
The actor-frozen, critic-only control retains 14/16 after both updates, with
bitwise unchanged actor tensors, zero action change, and zero KL. Identical
first-rollout reward, value, return, and advantage statistics in both arms
confirm matching training inputs before actor optimization.

| Fresh-copy iteration | 0 | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| Near-vertical holds, identical 4000–4015 fixtures | 14/16 | 12/16 | 12/16 | 12/16 | 12/16 |
| Validation + held-out recoveries | 30/123 | 28/123 | 27/123 | 24/123 | 20/123 |
| Mean longest hold (s) | 1.8886 | 1.7468 | 1.7468 | 1.6310 | 1.5277 |

Iteration zero was explicitly replayed in
`output/balance_first_updates_zero_v1/`. The uninterrupted ten-update audit is
`output/balance_first_updates_v2/`; the interrupted v1 audit stops at seven.
Separate fixed 123-state replays of early checkpoints are under
`output/balance_early_replay_v1/`.
All their initial physical measurements and dataset checksums match the frozen
baseline exactly. These trajectories expose continuing forgetting even while
the small near-vertical count stays at 12/16 through update 10.

First-update evidence (`output/balance_first_updates_v2/audit.json` and
`output/balance_critic_control_v1/audit.json`):

| Measurement | Actor update | Critic-only |
|---|---:|---:|
| Initial actor LR | 0.0001 | 0.0001 |
| Actor Adam steps before update | 0 | 0 |
| Actor Adam steps after update | 24 | 0 |
| Ending actor LR | 0.00001 | 0.0001 |
| Analytic KL on rollout observations | 0.43094 | 0 |
| Approximate KL | 0.41247 | 0 |
| PPO clipping fraction, epsilon 0.1 | 88.99% | 0% |
| Mean-action RMS change, action units | 0.04235 | 0 |
| Differential policy entropy | -6.0660 | -6.0676 |
| Critic loss | 27.9708 | 27.9708 |
| Near-vertical holds | 12/16 | 14/16 |

The adaptive LR rule reduces subsequent steps after measuring KL; it does not
reject or roll back an oversized update. PPO clipping is also not a hard
policy-distance bound. The first update moves the narrow front-joint action
distributions considerably. Ending LR alone conceals that movement. Actor
entropy barely changes: exploration collapse is not the observed mechanism.
The saved original has Adam history, but `--warm_start` intentionally loads only
network states. Historical iteration 25/50 optimizers have 600/1200 actor steps;
their LR is 2.25e-5. The interrupted historical continuation restored optimizer
state but restarted worlds and RNG, so it was not uninterrupted training.

First-rollout critic predictions average 9.4244 (SD 1.5326), while GAE returns
average 14.1948 (SD 3.3598). Raw advantages average 4.7704 (SD 2.4719), range
-2.8685 to 8.5296. Normalization produces mean approximately zero and SD one;
there is no observed divide-by-zero or nonfinite-advantage failure. That does
not validate credit assignment: the inherited critic predicts the old reward
and has an inherited normalizer, while recovery adds COM and hold objectives.
Critic losses remain substantial (54.37 on the second actor-arm update).

`output/balance_stage_zero_rewards_v1.json` records the complete first rollout's
individual weighted reward components without updating either network. It
avoids the legacy logger's NaNs before any completed episode. The first rollout
contains no terminations or timeouts. In later stage-zero logs, sustained hold,
handstand balance, and recovery hold dominate positive episode returns, while
termination and COM offset dominate penalties; action and torque penalties are
much smaller. Dense reward and good local stability do not enforce five-second
retention, and no teacher/action-preservation objective protects the old actor.
For the first 2.56-second rollout, mean integrated rewards per environment are
+60.066 sustained balance, +19.655 handstand balance, +12.863 recovery hold,
-0.428 COM offset, -0.084 COM speed, -0.00443 action rate and -0.000332 torque.
All components and their ranges are finite. Zero termination penalty in that
first rollout agrees with its recorded zero episode ends.

Two independent legacy rollout defects are relevant to later targets:

* `terminated` is initialized in storage but not refreshed in rollout updates.
  Current GAE relies on `dones`, so stale `terminated` does not by itself explain
  the measured first update.
* Timeout bootstrapping uses the pre-step value, rather than the terminal
  observation's value. Next observations are captured after resets. Neither
  issue caused the first-update loss, because that rollout had no episode ends.

The new trainer captures terminal observations before reset, bootstraps timeouts
from those observations, masks true terminations, and cuts GAE recursion at
every reset. Unit tests distinguish terminal, timeout, and next-episode reward.
The legacy trainer is preserved for reproduction; these fixes apply only to
the new experiment. Its inherited rollout observation-noise augmentation is
also excluded from the new diagnostic; this changes several variables together
and is not a claim of a noise-only causal ablation.

On identical observations from all 285 saved states, direct fine-tuning changes
final joint targets by RMS 0.01449 rad after update 1, 0.01999 rad at historical
25, and 0.02538 rad at 50; maxima are 0.06395, 0.11188 and 0.14309 rad. These
are initial-state comparisons, distinct from rollout-weighted action changes.
The comparison, including update 10, is saved in
`output/balance_action_comparison_v2.json`.
Historical motor saturation increases only from 3.81% to 3.97%, while retention
collapses. Widespread actuator saturation is therefore not the principal
observed explanation; small target changes and their closed-loop consequences
matter. Original targets can already exceed soft joint bounds, which are not
the same thing as clipped motor torques.

The historical curriculum never reaches frontier states: its final 359 reset
samples are stage-zero demonstrated recoveries, with zero promotion passes.
Distribution breadth cannot explain losing those demonstrated states. The new
run starts with the existing stage-2 mixture: 70% demonstrated recoverables and
30% nearby frontier, plus 20% replay from earlier stages. Sources are sampled
uniformly within strata, then snapshots within source. Two 64-episode primary
batches must each reach 80% success for promotion; replay cannot promote.

## Controller and parity

`gym/envs/go2/frozen_balance_residual.py` implements:

```
target = existing_reference + 0.25 * frozen_balance_actor(obs) + correction
correction = 0.04 rad * gate(obs) * tanh(recovery_actor(obs))
```

The separately optimized network is 59–64–64–12 with tanh hidden activations.
Its final weights and bias are exactly zero initially. The copied teacher
stays in eval mode with gradients disabled, including when the residual enters
train mode. Every completed update checks all teacher tensors and the original
checkpoint hash. Checkpoint evaluation rejects any modified teacher weights.
Only recovery parameters are passed to its optimizer. Fixed Gaussian latent
exploration (SD 0.15) is applied to the bounded correction, never to the teacher.

The gate uses measured projected gravity, angular velocity, and joint velocity
in the existing scaled actor observations. It is exactly zero within 3 degrees
of vertical when angular speed is at most 0.8 rad/s and joint speed at most
6 rad/s. Smoothstep increases authority over another 12 degrees, 1.2 rad/s,
or 10 rad/s respectively, up to 0.04 rad. It does not use COM, simulator
contacts, privileged base velocity, or replay labels during execution.

The existing Go2 path adds its reference once and applies its native PD/torque
limits. The residual adapter rate-limits only additional displacement with the
native joint-speed limits and prevents corrections from worsening baseline
soft-target violations. Existing transition target/rate limits still apply
downstream during full-system evaluation. The original standalone balance path
has no persistent target-slew limiter; imposing a new one on the baseline would
break required parity. No CPG generator or phase override is added.

Parity completed before the fresh diagnostic:

| Check | Result |
|---|---:|
| Original near-vertical five-second holds | 14/16 |
| Validation | 16/64 |
| Held-out | 14/59 |
| Complete replay outcome equality against original | Exact |
| Final commanded joint-target error | 0 rad |
| Lockstep saved states | 285 |
| Checked control steps per saved state | 20 |
| Phase, filtered load, target, controller history, root-state errors | All zero |

Evidence: `output/frozen_residual_zero_v2/{summary,parity}.json` and
`output/frozen_residual_commands_v1.json`. The lockstep test includes all three
dataset splits. The complete 6.5-second replay comparison additionally checks
all historical validation/held-out outcomes, rather than just aggregate counts.

## Short diagnostic

`scripts/train_frozen_residual.py` permits at most 25 updates; this run requests
10. It uses 16 CPU MuJoCo worlds, 256 control steps per rollout, the original
59-actor/63-critic observation mappings, fixed actor LR 1e-5,
two full-batch actor steps, clip 0.05, KL target 0.002, gradient clipping 0.25,
and a new 63–64–64–1 value network without inherited normalization. Four critic
steps use LR 1e-3. Rewards retain existing recovery components and add a small
squared-correction penalty; latent means are regularized even behind a closed
gate. Rewards/GAE are computed at every environment control step. Evaluations
preserve Python, NumPy and Torch RNG state.

Every update is saved separately; fixed evaluations run at 1, 2, 5, 10 and, if
explicitly requested within the cap, 15, 20 and 25. Retention below 13/16 stops
the run without claiming an unreplicated frontier gain offsets the loss.
`diagnostic.json` records update metrics before evaluation so another
interruption does not discard completed-update evidence.
The launcher now also accumulates completed episodes across rollout boundaries,
logging their duration, termination type, longest hold and integrated reward
components before reset. The recorded v2 update series has rollout-integrated
components; a separate paired full-eight-second audit provides completed-episode
reward records in `output/frozen_residual_episode_rewards_v1.json` without
additional optimization.
That paired stage-2 sampler audit achieves 10/16 full-horizon recoveries for
both the frozen teacher and update 10. Initial native-state fingerprints and
per-trial durations match; failed trials end early on termination. It uses
deterministic actions with no exploration and clears initial episode counters
so every trial has an eight-second budget. Completed-episode mean rewards are
approximately +196.16 sustained balance, +74.93 recovery hold, +51.99 handstand
balance, -2.72 COM offset and -7.5 termination for both controllers. This is a
small regression/control audit, not new generalization evidence.

The completed run and decision are recorded in
`output/frozen_residual_diagnostic_v2/` and the final comparison report. The
earlier `v1` contains one saved update and an interrupted evaluation; it was
not resumed. It is excluded from the uninterrupted v2 series.

The ten-update run reaches the requested cap without triggering the retention
stop. Its final comparison is `output/frozen_residual_comparison_v2.json`,
including the early legacy replays and protected controller-code hashes.

| Metric | Original / zero head | Update 1 | Update 2 | Update 5 | Update 10 |
|---|---:|---:|---:|---:|---:|
| Near-vertical five-second holds | 14/16 | 14/16 | 14/16 | 14/16 | 14/16 |
| Validation + held-out recovery | 30/123 | 30/123 | 30/123 | 30/123 | 30/123 |
| Frontier five-second recovery | 0/34 | 0/34 | 0/34 | 0/34 | 0/34 |
| Mean longest strict hold (s) | 1.888618 | 1.888618 | 1.888699 | 1.888455 | 1.888618 |
| Added correction RMS (rad) | 0 | 0.00000401 | 0.00000658 | 0.00001142 | 0.00001603 |
| Added correction activation | 0% | 44.71% | 44.95% | 45.21% | 45.42% |

There are zero paired five-second wins or losses at every interval: all thirty
established recoveries remain successful. Correction telemetry averages control
calls over nominal and replay batches and includes padding in the final replay
batch; it is not an independent-state-weighted activation estimate. Historical
`residual_rms_rad` in the replay traces includes the baseline actor's large
position residual; use `additional_correction_rms_rad` for the new correction.

At update 10, collision-censored COM speed is 0.87527 m/s (original 0.87560),
front support retention is 51.551% (51.477%), and motor saturation is 3.8132%
(3.8145%). Mean endpoint COM planar displacement is 0.28689 m (0.28666 m);
that endpoint measure includes post-collision motion and is not a recovery
improvement claim. Holds and success rates show no useful change.

The paired full standing-to-handstand regression retains 1/32 complete
transitions: validation 1/16 and held-out 0/16, exactly matching the frozen
controller. All per-seed longest hold differences are zero. Outputs are
`output/frozen_residual_standing_v1/`, with identical generated prefixes for
both controllers and the existing target/rate limits enabled. Mean maximum
target step remains approximately 0.0602 rad. No standing improvement is claimed.

Across ten updates, measured latent KL is 6.13e-7 to 5.16e-6, clipping fraction
is zero, and entropy remains -5.73818 with fixed exploration SD. Value loss
is 48.55 initially and 115.73 at update 10; a new critic has not demonstrated
adequate calibration in this short run. Training's sampled added correction
RMS is about 0.0015 rad, while deterministic corrections are far smaller.
The final sampler records 45 stage-2 recoverable, 29 stage-2 frontier, 10
stage-1 and 6 stage-0 resets. Thus this run actually exercises frontier states.
It remains at stage 2 with 59 primary completed episodes and 19 successes,
below the first 64-episode promotion batch. Setup staggers initial episode
counters, so the first cycle includes shortened episodes; these counters are
not a claim that all sampled recoverables failed full eight-second trials.

## Limiting factors and decision

Do not extend the run based on preserved balance alone. The residual has not
established repeatable frontier or generalization gains, and no checkpoint is
promoted over the frozen controller.

`output/frozen_residual_authority_v1.json` measures actual saved inputs at
iteration 10:

| Frontier split | States | Gate initially closed | Mean available gated bound | Learned correction RMS |
|---|---:|---:|---:|---:|
| Training | 67 | 52.24% | 0.00845 rad | 0.00001067 rad |
| Validation | 16 | 37.50% | 0.00781 rad | 0.00000922 rad |
| Held-out | 18 | 50.00% | 0.00541 rad | 0.00000829 rad |

These are initial-state measurements, not whole-episode activation fractions.
The gate is initially closed in 61/64 training recoverables, 15/16 validation
recoverables and 14/14 held-out recoverables. This protects established balance,
but it also hides corrections in many failed frontier states until they move
outside the confidence region. The learned latent means remain approximately
0.0006 RMS on frontier inputs, so the short conservative update recipe barely
uses available authority even when the gate opens. These measurements do not
establish that the 0.04-rad envelope is physically insufficient.

The 59 deployable observations contain no direct base linear velocity, COM
position, height or measured contact force. CPG feedback carries some contact
history, but a feedforward gravity/angular-rate gate cannot distinguish all
translationally unstable near-vertical states. The dataset includes physically
replayed frontiers, and they are now sampled, so distribution coverage alone
does not explain the lack of learning. Reward credit assignment, a poorly
calibrated fresh critic, limited short-run mean movement and an overprotective
gate remain measured or plausible limitations; this run does not isolate their
individual causal effects.

Before considering another recovery training run, investigate a deployable
velocity estimate or short observation history, value calibration with the
actor frozen, and gated-authority probes on recorded frontier trajectories.
Evaluate any changed gate against the exact command/phase parity and retention
suite first. Use new independent source trajectories for a final generalization
test. Failed replay states are not physical-impossibility certificates.

Validation: 70 targeted unit/regression tests passed, including existing CPG,
balance, handstand, posture-transition and recovery tests. Added tests cover
zero-head parity/frozen gradients, bounded gating, adapter target/rate limits,
per-environment memory resets, terminal/timeout bootstrapping, and rejection of
modified teacher weights. Lockstep MuJoCo checks pass on all 285 states.
Both the seven protected historical assets and all eight v4 bank/label/provenance
files match their earlier hashes. The CPG implementation, balance environment
and configuration, and posture-transition controller also match the earlier
manifest; see `output/frozen_residual_protected_code_v1.json` and the final
comparison report. The original balance checkpoint SHA256 remains
`6a56bebf31c26f025274ac18b995b68485fa3fb87e07c011e73872f7b36d755f`.

## Reproduction and Molab

All commands run from a `ppo_kickup` checkout with the transferred assets and
Python 3.11. No Molab connection or execution is claimed. Cloning alone omits
untracked implementation and gitignored datasets/checkpoints; transfer the
listed bundle/manifest assets at their recorded relative paths first. Use new
output names for every invocation; do not overwrite prior results.

The portable artifact is `output/frozen_residual_bundle_v1.tar.gz`, with its
adjacent `.sha256`. Place both files under `output/` in a dedicated `ppo_kickup`
checkout. Its manifest includes both saved-source snapshots and the kick
checkpoint's effective runtime configuration; omitting that JSON can silently
change checkpoint behavior on another machine.

```bash
git branch --show-current
sha256sum -c output/frozen_residual_bundle_v1.tar.gz.sha256
tar -xzf output/frozen_residual_bundle_v1.tar.gz
uv sync --frozen

# After transferring the residual bundle, verify its expected checksums.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/verify_recovery_assets.py \
  --manifest datasets/frozen_residual_assets_manifest_v1.json \
  --environment molab --verify-interface \
  --output output/residual_molab_preflight_new.json

env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/unit_tests/test_frozen_balance_residual.py \
  tests/unit_tests/test_kickup_recovery.py tests/unit_tests/test_recovery_audit.py \
  tests/unit_tests/test_recovery_frontier.py tests/unit_tests/test_go2_balance.py \
  tests/unit_tests/test_handstand_curriculum.py tests/unit_tests/test_contact_cpg_metrics.py \
  tests/unit_tests/test_posture_transition.py tests/unit_tests/test_kickup_cpg.py \
  tests/unit_tests/test_go2_handstand.py -q

# Required baseline gates, with new outputs on this runtime.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_frozen_residual.py --output-dir output/residual_molab_zero_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/verify_frozen_residual_commands.py \
  --output output/residual_molab_commands_new.json

# Bounded diagnostic only; do not increase this based on retention alone.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/train_frozen_residual.py --iterations 10 \
  --dataset datasets/recovery_frontier_v4/train.npz \
  --parity output/residual_molab_zero_new/parity.json \
  --commands-parity output/residual_molab_commands_new.json \
  --output-dir output/residual_molab_diagnostic_new

# Evaluate only an actually completed checkpoint (or the early-stop checkpoint).
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_frozen_residual.py \
  --checkpoint output/residual_molab_diagnostic_new/model_10.pt \
  --output-dir output/residual_molab_final_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_residual_standing.py \
  --checkpoint output/residual_molab_diagnostic_new/model_10.pt \
  --output-dir output/residual_molab_standing_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/diagnose_residual_authority.py \
  --checkpoint output/residual_molab_diagnostic_new/model_10.pt \
  --output output/residual_molab_authority_new.json
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_residual_training_loop.py \
  --checkpoint output/residual_molab_diagnostic_new/model_10.pt \
  --output output/residual_molab_episode_rewards_new.json

# Independent reproduction arms; neither resumes the failed 50-update actor.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_balance_collapse.py --iterations 10 \
  --output-dir output/residual_molab_legacy_audit_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_balance_collapse.py --critic-only --iterations 2 \
  --output-dir output/residual_molab_critic_control_new
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_balance_rewards.py --output output/residual_molab_rewards_new.json

# Repeat separately for --iteration 0, 1, 2, 5 and 10, with new output names.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_balance_update.py \
  --audit output/residual_molab_legacy_audit_new/audit.json --iteration 1 \
  --output-dir output/residual_molab_legacy_iteration_1_new
```

The deterministic baseline and evaluation cohorts are historical regression
tests. Their snapshots are correlated and their held-out labels have already
been inspected. They do not establish independent generalization. Useful
recovery gains on additional source trajectories, stable retention, and
physical plausibility are prerequisites for recommending more training.
