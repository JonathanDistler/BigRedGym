# BC takeoff retention diagnostic, 2026-11-11

This continues `ppo_kickup_e2e`; it does not repeat or modify the validated reward,
CPG controllers, reference trajectories, physical models or earlier results.
No extended run is authorized or automatically launched.

## Readiness

Code inspected: `takeoff.py`, `environment.py`, `ppo.py`, both PPO runners,
`protocol_takeoff.json`, the takeoff learning report, and BC/exploration artifacts.

- Physically valid BC-initialized stochastic takeoff exists at both 90% and 100%.
  It is not a robust deterministic 90% initialization. Fresh resets 31000–31007
  give 0/8 mean-policy and 3/8 stochastic takeoffs at 90%; at 100%, 6/8 and 1/8.
- Both arms load the exact same BC actor. Initial saved actor, critic and latent
  standard-deviation tensors were compared: all equal. Both use 12 direct joint
  targets, native effort/velocity envelopes, 60/3 gains, action scale 2, target
  filter 1 and the same slew bound. No stronger curriculum-only initialization.
- Preparation reward is capped at .5 per episode. Ongoing lean/standing receives
  no positive income. Certified flight requires loaded extension, positive work,
  synchronized release, front support, clearance and valid physical state.
- Saved checkpoints contain the entire actor/critic network, Adam state,
  curriculum state, per-world morphology and simulator/tracker state, reset counts,
  RNG states, protocol and accumulated measurements. Stage changes preserve them.
- Saved checkpoints can be independently evaluated at full-scale geometry; this
  was exercised with the update-four B checkpoint and seeds 41000–41001.

## Bounded experiment

`output/takeoff_retention_20261111_v1` uses training seeds 42 and 43, up to twenty
updates, 1,024 transitions per update and eight environments. Both conditions use
all prepared PPO settings (learning rate 1e-5, latent std .05, target KL .01).
Actor, critic and exploration initialization and interaction budgets are matched
within each seed. Evaluations do not consume the training RNG stream.

Every two updates, the normal checkpoint evaluation is saved, plus independent
mean-policy and stochastic probes at training and original geometry. Probes use
the same reset family/seeds 31000–31007 and action-noise streams, with goal
termination disabled for the full eight-second horizon. These reset seeds are
independent of training. Periodic probes are development data; they are not a
fresh final test set.

Stop both arms of a seed pair if post-update KL exceeds .03, average clipping
exceeds .30, or full-scale deterministic takeoff drops more than 25 percentage
points below its own initial baseline for three consecutive evaluation points.
The latter can stop before ten updates, as requested. Numerical loss/gradient
checks remain active. Stage progression uses the unchanged certified takeoff and
useful rotation gates. No forced promotion.

The post-run `retention_gate.json` additionally requires at least ten updates,
no early stop, and retention across the last three evaluations of both full-scale
mean behavior and training-morphology stochastic behavior (minimum 25%, at most
12.5 percentage points below initialization). This is a conservative engineering
gate, not a statistical superiority test. Two training seeds and eight reset
seeds cannot establish a curriculum advantage.

## Commands

From the repository root; always choose a new output directory:

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_takeoff_retention.py \
  --output output/takeoff_retention_NEW \
  --bc-actor output/takeoff_bc_investigation_v1/bc_actor.pt \
  --validation output/takeoff_learning_certified/validation.json \
  --updates 20 --training-seeds 42 43

env -u PYTHONPATH MPLCONFIGDIR=/tmp/takeoff-mpl .venv/bin/python \
  scripts/report_takeoff_retention.py --run output/takeoff_retention_20261111_v1

env -u PYTHONPATH .venv/bin/python scripts/assess_takeoff_retention.py \
  --run output/takeoff_retention_20261111_v1

env -u PYTHONPATH .venv/bin/python scripts/evaluate_takeoff_checkpoint.py \
  --checkpoint output/takeoff_retention_20261111_v1/B_seed42/checkpoint_000004.pt \
  --output output/takeoff_heldout_NEW
```

`figures/per_seed.csv`, `summary.csv` and `training.csv` accompany measured
PNG/PDF retention and stability figures. Shading denotes one sample standard
deviation across training seeds, not across pooled resets or a confidence
interval. Pitch is in radians, clearance in metres, work in joules, duration in
seconds. `rear_clearance`/`max_pitch` are whole-episode peaks, potentially after
the useful kick; `genuine_flight_pitch` and `genuine_airborne_seconds` are the
certified supported-flight measures. `certified_work_min` is the smaller rear-leg
positive loaded work at certification and is zero if no takeoff was certified.
Later nonfoot collision does not invalidate an earlier valid kick certificate;
it does invalidate interpreting the episode as successful sustained balance.

## Molab cells and source availability

Open `notebooks/molab_takeoff_retention.py` in Molab and run its cells manually.
Upload `output/takeoff_molab_bundle.zip` beside the notebook. The bundle is needed
because the local morphology code and BC checkpoint are untracked and cannot be
obtained by cloning the current remote branch alone. The notebook clones the
correct branch, pins the bundle's base commit and overlays a per-file SHA256
verified snapshot. It installs only CPU MuJoCo/PyTorch analysis dependencies;
W&B is disabled and neither PPO runner imports W&B.

Cells launch the bounded pair, show progress, generate plots and a retention
gate, evaluate saved checkpoints on fresh seeds 40000–40031, and offer a complete
backup download. Source installation is reproducible from the bundle; cloud
execution and dependency downloads have not been tested in Molab here.
The extended launch switch defaults to false and also checks the retention gate.
All paths are unique and every saved checkpoint is included in the backup.

The prepared substantive comparison retains four paired training seeds 42–45,
1,048,576 transitions per arm/seed, performance-driven .90 → .95 → 1.00, two
32-episode passing windows at each stage, no bypass, and a 262,144-transition
full-scale reservation. Both arms receive the same necessary budget extension
up to 1,318,912. The same network and optimizer persist between stages. Evaluate
both current and original geometry, and reserve fresh full-horizon final tests.
This remains a takeoff-first comparison, not a complete handstand trainer.

## Interpretation and next budget

The fresh 90% mean-policy result is 0/8, consistent with the previous 0/4.
Morphology-specific demonstration adaptation should precede the next substantive
BC-initialized curriculum comparison. More supervised epochs on only the old
full-scale dataset do not address the closed-loop morphology gap. Collect or
adapt physically validated 90/95/100% prefixes, fit one shared actor on that
mixture, and load exactly that actor in both arms. Log added offline updates,
simulator interactions, dataset hashes and closed-loop validation separately.
The present run does not modify the BC actor or use a stronger B initialization.

Budget the next investigation at 1,000–3,000 additional supervised updates on the
mixed-morphology data, with early stopping on independent closed-loop takeoff
rather than MSE alone. Then repeat 10–20 PPO updates per arm for at least two
paired training seeds (20,480–40,960 transitions per seed pair), using separate
32-reset validation families at 90% and 100%. Consider a common conservative
actor update or offline BC retention term if forgetting repeats; neither is
implemented or claimed validated here. Keep the same initialization and any
retention mechanism in A and B.

Only after that retention gate passes should the prepared million-transition,
four-seed comparison run. Preserve its full-scale reservation and stage gates.
Do not spend the extended budget on the current initialization merely because
optimizer statistics look stable. The implementation uses CPU MuJoCo loops;
a Molab GPU does not automatically accelerate this runner.

## Actual result

Both seed pairs stopped at update eight, before the requested minimum ten,
because B lost full-scale deterministic takeoff at updates four, six and eight.
A stopped with its partner to preserve matched budgets. Each arm/seed received
8,192 transitions; total diagnostic training was 32,768 transitions. No extended
training ran. The retention gate is **failed**.

| Full-horizon certified takeoff | Initialization | Update 8, seed 42 | Update 8, seed 43 |
| --- | ---: | ---: | ---: |
| A, 100%, deterministic | 6/8 | 8/8 | 0/8 |
| A, 100%, stochastic | 1/8 | 2/8 | 1/8 |
| B, 90%, deterministic | 0/8 | 0/8 | 0/8 |
| B, 90%, stochastic | 3/8 | 2/8 | 2/8 |
| B, 100% transfer, deterministic | 6/8 | 0/8 | 0/8 |
| B, 100% transfer, stochastic | 1/8 | 4/8 | 3/8 |

Across two training seeds, final full-scale deterministic takeoff is 50% ± 70.7
percentage points for A and 0% ± 0 for B (sample standard deviations). Full-scale
stochastic takeoff is 18.75% ± 8.84 points for A and 43.75% ± 8.84 for B. These
small diagnostic samples do not establish a morphology advantage. B's stochastic
transfer improved, so it is inaccurate to say all takeoff behavior was erased;
it failed the prescribed deterministic retention test and never acquired a
reliable shortened-leg mean policy. A improved strongly in one seed but also
failed to retain the mean behavior in the other.

A seed 42's mean certified flight pitch increased from .431 to .721 rad and
mean certified duration from .0393 to .0853 s; the corresponding final A seed 43
values were zero. All final B deterministic certified-flight metrics were zero.
Whole-episode pitch/clearance peaks can be large even without a certified push;
use `final_per_seed.csv` and the certified-flight plots to distinguish them.
The measured tables include work, front support, pitch velocity and collisions.

Post-update KL ranges were .00869–.01469 for A and .00789–.01649 for B.
Average minibatch clipping fractions were 3.05–9.33% for A and 2.69–9.25% for B.
No numerical or excessive-divergence stop occurred. Moderate KL did not assure
retention. Neither B seed promoted to 95% or 100%; five-second handstand success
was zero. The larger comparison is scientifically **not ready**.

Final checkpoints (both actors and critics, optimizers and curriculum included):

- `output/takeoff_retention_20261111_v1/A_seed42/checkpoint_000008.pt`
- `output/takeoff_retention_20261111_v1/B_seed42/checkpoint_000008.pt`
- `output/takeoff_retention_20261111_v1/A_seed43/checkpoint_000008.pt`
- `output/takeoff_retention_20261111_v1/B_seed43/checkpoint_000008.pt`

Initialization and updates two/four/six are preserved alongside them. The source
bundle was extracted into a separate temporary directory, every manifest hash
was checked, reward validation passed, and all physical assets rebuilt without
the original working tree. The 14 targeted takeoff/PPO tests passed; new scripts
passed Ruff and compilation. Notebook Python compilation passed. The local
Marimo checker did not complete and was interrupted; Molab installation/runtime
execution is not claimed validated. The notebook's extended launch remains
false and gated against the measured failed retention result.
