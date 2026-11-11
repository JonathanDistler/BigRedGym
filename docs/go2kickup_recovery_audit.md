# Recovery dataset and checkpoint evaluation audit

Updated frontier dataset, gated curriculum, interrupted 50-update diagnostic and
current commands: [recovery frontier report](go2kickup_recovery_frontier.md).
The measurements below describe the earlier small-bank experiment.

Work remains on `ppo_kickup`. No original controller, checkpoint, reference JSON,
or reference configuration was changed. No additional training was launched.
The requested remote connection was not made: **Molab file availability remains
unverified**, as requested by the user's “don't connect” instruction. Local
availability and compatibility are verified; a target-runtime preflight is below.

## Trajectory diversity and curriculum coverage

Reproducible audit: `scripts/audit_recovery_dataset.py`.
Detailed report: `output/recovery_dataset_audit_v4.json`.

| Cohort | Generated trajectories | Retained trajectories | Snapshots | Successful source trajectories retained |
|---|---:|---:|---:|---:|
| Training, 10000–10015 | 16 | 13 | 98 | 0 |
| Validation, 6000–6015 | 16 | 16 | 80 | 1 |
| Held-out, 7000–7015 | 16 | 14 | 73 | 0 |

Training seeds 10010, 10011 and 10012 have no retained snapshot because the
capture filter excludes early failed/low/colliding states. Held-out seeds 7003
and 7010 similarly contribute no states. Missing snapshots do not remove these
seeds from full-system evaluation: all 16 trials remain in every cohort.

All banks use one frozen kick checkpoint and the same aggressive-prefix,
adaptive reference-first transition family and parameters. Mass and ground
friction vary; training total mass spans 15.489–16.638 kg and ground friction
0.621–0.948. They are not diverse controller/reference families. Training has
three near trajectories and ten failed retained trajectories, with no successful
five-second transition. “Near” labels require only 0.1 s hold, not a near-five-
second outcome. Validation's twenty recovered snapshots all come from seed 6003.

| Training stage | Source | Snapshots | Distinct trajectories | Front support: both / one / neither |
|---|---|---:|---:|---|
| 0 | Procedural nominal reset | Not stored | 32 reset checks | Evaluated after settling in baseline test |
| 1 | Procedural small tilt/angular disturbance | Not stored | 32 reset checks | Not a contact-diversity dataset |
| 2 | Procedural tilt plus base velocity | Not stored | 32 reset checks | Not a contact-diversity dataset |
| 3 | Near-vertical, low-speed trajectory states | 29 | 5 | 27 / 2 / 0 |
| 4 | Source trajectories with at least 0.1 s hold | 46 | 3 | 27 / 17 / 2 |
| 5 | All retained feasible states | 98 | 13 | 40 / 43 / 15 |

Front support here means at least 6% of body weight per front foot, matching the
full-system evaluator. Thirty-two procedural resets per stage were actually
executed, not inferred from configuration. Stage 0 COM speed is zero; stage 1
observed pitch is 1.5367–1.6037 rad and COM/support offset up to 2.15 cm; stage 2
pitch is 1.4324–1.7046 rad, COM speed up to 0.3343 m/s and COM/support offset up
to 8.04 cm. The largest sampled horizontal base-speed component is 0.2407 m/s,
within the 0.25 m/s bound.

The stage sets overlap: 23 samples are in both stages 3 and 4; stages 3 and 4
are each contained in stage 5, but stage 3 is not contained in stage 4. Thus
stage number is not a certified increasing difficulty ordering. Stage 3 has
little unilateral-support coverage and no unsupported states. Stage 4 draws
only from seeds 10002, 10008 and 10014, with seed 10014 supplying 19/46 = 41.3%
of uniformly sampled snapshots. No sampler or curriculum was changed by this
audit.

Within-trajectory snapshot spacing is 50 ms, with less than one second retained
per trajectory. Adjacent states are strongly dependent; 98 snapshots are not
98 independent recovery trials. The audit reports per-seed contributions,
concentration, feature quantiles, phase coverage, temporal spans, stage overlap
and support strata. Concentration statistics are not independence/ESS claims.
No exact native integration-state duplicates or seed overlap were found between
cohorts. These historically inspected cohorts remain regression cohorts, not
new untouched generalization tests. Broader training trajectories and successful
training examples are still needed before claiming adequate recovery coverage.

## Observation mapping, normalization and CPG compatibility

Slices use zero-based, half-open indexing.

| Field | Actor slice | Critic slice |
|---|---|---|
| Base height | — | 0:1 |
| Base linear velocity | — | 1:4 |
| Base angular velocity | 0:3 | 4:7 |
| Projected gravity | 3:6 | 7:10 |
| Commands | 6:10 | 10:14 |
| Joint position observations | 10:22 | 14:26 |
| Joint velocities | 22:34 | 26:38 |
| Previous joint-position residual | 34:46 | 38:50 |
| Handstand stage observation | 46:47 | 50:51 |
| CPG phase features | 47:55 | 51:59 |
| CPG desired contacts | 55:59 | 59:63 |

Totals are exactly **59 actor, 63 critic, 12 actions**. The stage observation is
already divided by two by its property. CPG features interleave sin/cos for
FL, FR, RL and RR; desired contacts are `[1,1,0,0]`.

Environment scaling divides by the saved scales before network inference:
angular velocity by 0.3, base linear velocity by 3, base height by 0.3, commands
by `[3,1,3,0.4]`, joint positions by `[1.0472,2.53075,0.94247]` per leg, joint
velocities by `[2,2,4]` per leg, and previous residual by 0.25. Gravity, stage
and CPG features have unit external scale. No extra actor normalization should
be introduced.

The original checkpoint actor has **no RMS normalizer**. The critic has saved
63-element float64 `obs_rms.running_mean` and `running_var`, plus scalar count
2,360,982,401. All are loaded strictly and checked bit-for-bit against the
checkpoint, including shape, finite values and nonnegative variances. In eval
mode they remain unchanged; training uses the existing normalizer update path.
The normalizer applies `(scaled_obs-mean)/sqrt(var+1e-5)`. Some critic variances
are very small (minimum approximately 4.26e-10); finite normalization alone does
not prove distribution matching. The largest absolute normalized training-bank
critic input is 12.44, so there are substantial out-of-distribution features.

Every one of the **251 stored states** was restored into both the new recovery
environment and the original saved-source balance environment. Actor/critic
observations and deterministic actor actions matched exactly (maximum difference
zero). Both networks loaded strictly. All critic values were finite and saved
normalization statistics stayed unchanged. This directly verifies ordering and
scaling parity, rather than only checking tensor lengths.

All stored CPG phases/filter memories were restored and recomputed sin/cos
features matched exactly. All three bank CPG configurations match the controller:
1 Hz, sigma 2, zeta 0, force filter 0.02 s, FL/FR stance and RL/RR swing. Nominal
resets restore phases `[0,pi,pi,0]` and zero filtered loads; trajectory resets
retain their captured phase/load instead. This preserves the existing CPG
observation/reference/residual architecture. It does not introduce a different
CPG joint-trajectory generator.

## Checkpoint-by-checkpoint paired evaluation

`scripts/evaluate_kickup_recovery.py --run-dir ...` now discovers every completed
`model_<iteration>.pt` numerically, including `model_0.pt`; `.pt.tmp` files are
excluded. `--fine-tuned` remains supported and `--checkpoints` accepts an explicit
list. Run after training to evaluate all saved iterations (normally 0, 25, 50,
75, 100, 125, 150, 175, 200). It does not launch or resume training.

For each cohort, generate the standing/kick prefix **once**, evaluate the frozen
balance controller once, then evaluate every recovery checkpoint against those
same native states, model properties, solver warmstarts and initial targets.
Validation is 6000–6015; held-out is 7000–7015, always the full batches. Actual
input fingerprints are stored and checked for mutation after each candidate.
Actor/critic network and normalization digests are checked before and after
evaluation. The original checkpoint/reference input files are also hashed and
checked unchanged. Model-0 parity tests the comparison itself.

Outputs include per-checkpoint/cohort traces, success rates and paired outcomes,
longest and total stable duration, support retention, collisions, saturation,
target continuity and handoff dynamics. `summary.json` includes paired wins,
losses and mean hold differences; `provenance.json` records checkpoint/input
hashes and fixed seed lists. No automatic best-checkpoint selection uses held-out
results. Evaluating many checkpoints on historical held-out seeds does not make
those seeds an untouched final test.

Actual series test: `output/recovery_checkpoint_series_v2/` evaluated the two
existing smoke-run checkpoints. Original controller: validation 1/16, held-out
0/16. Model 0: 1/16 and 0/16, with the same per-seed outcomes as the original.
Model 1: 0/16 and 0/16. `summary_metrics.json` adds aggregated telemetry from
these completed traces using the current summary function. No improvement or
checkpoint promotion is claimed.

```bash
# Reproduce the complete local audit.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_recovery_dataset.py --output output/recovery_audit_new.json

# Evaluate all saved checkpoints of the explicitly chosen training run.
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/evaluate_kickup_recovery.py \
  --run-dir logs/REPLACE_EXPERIMENT/REPLACE_RUN \
  --dataset datasets/go2kickup_recovery_train_v2/train.npz \
  --output-dir output/recovery_checkpoint_series_new
```

## Molab availability preflight

Local preflight `output/recovery_asset_preflight_v1.json` verified every listed
asset on host `JonnyDistler`. It is not evidence of Molab availability. Nothing
was connected, transferred or executed remotely.

`datasets/recovery_assets_manifest_v1.json` records relative paths, SHA256 and
sizes for the training NPZ and labels/provenance, original balance checkpoint
and adjacent saved source, CPU implementation and robot resources, plus the
kick checkpoint, atlas and reference assets needed for paired evaluation. Keep
the manifest from this audited checkout; regenerating it on a different machine
would replace the expected evidence. Transfer assets with these relative paths.
Uncommitted/untracked files and gitignored checkpoints are not brought to Molab
by cloning the branch alone. The existing generic Molab notebook still defaults
to `main` and a GPU task; it is not this recovery CPU recipe.

Run **inside the Molab checkout**, without switching branches:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  scripts/verify_recovery_assets.py --environment molab --verify-interface \
  --output output/recovery_molab_assets_new.json
```

It fails on missing/corrupt files or a branch other than `ppo_kickup`, verifies
both network loads and the training-bank interface when assets match, and
records hostname, Python path/version, runtime label, manifest checksum and
individual asset checks. The `molab` label is user supplied; it is evidence of
remote availability only when the command actually runs there. It never starts
training and never changes a branch, controller or checkpoint.

In the existing Molab notebook, the same command can be invoked with its scrubbed
`ENV` and `REPO` variables:

```python
subprocess.run(
    [str(REPO / ".venv/bin/python"), "scripts/verify_recovery_assets.py",
     "--environment", "molab", "--verify-interface",
     "--output", "output/recovery_molab_assets_new.json"],
    cwd=REPO, env=ENV, check=True,
)
```

Until that report exists from the target runtime, Molab availability remains
unconfirmed. No remote action is required from or performed by this agent.

Added files: `scripts/audit_recovery_dataset.py`,
`scripts/verify_recovery_assets.py`, `tests/unit_tests/test_recovery_audit.py`,
`datasets/recovery_assets_manifest_v1.json` and this document. Updated only the
recovery evaluation script to support series evaluation. Validation: 54 targeted
unit/regression tests passed, including mapping, normalization/phase restoration,
stage coverage, numeric checkpoint discovery, seed alignment and corrupt/missing
asset detection. Ruff and whitespace checks pass.
