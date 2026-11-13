# Dynamic bridge Molab experiment

Base: `8162cf4`, branch `experiment/unified_go2_handstand`. This is an opt-in extension, not a new actor. Original experts, corrected initializer, old ZIP, previous checkpoints and the separate balance branch remain unchanged. No hardware transmission. Live Molab installation/training has **not** been performed: this session has no Molab connection.

## What changes

Use `scripts/dynamic_bridge_go2.py` and `notebooks/molab_dynamic_bridge_go2.py`. The launcher loads **the existing corrected student** at `output/retained_go2_distillation_v3/student.pt`, SHA256 `0d08913c5f306f4090f62f74001a62a795feee533958e3561adcfee816c984ab`. No distillation is rerun. Native model, 93/123 observations, normalization buffers, 12 position-target actions, 50Hz control, nominal kp60/kd3, all real limits/colliders, teacher demonstrations and reward terms remain unchanged.

Bridge sampling now gives smooth priority to actual native snapshots around the pre-stall/partial-rotation region. This is **a reset distribution**, not a pitch-triggered controller switch. Per-state recovery-failure EMA, trials, selected lane indices and reset RNG are saved/restored. At least20% of bridge sampling remains uniform; standing/balance mixture floors remain intact. No arbitrary joint/pitch interpolation, external forces or mid-episode resets are introduced. The shared neural residual remains free to learn all12 joint targets. This is a testable sampling hypothesis; improved motion has not yet been demonstrated.

Competence-conditioned imitation remains offline replay: validated ADC preparation/kick observations and demonstrated successful balance frames. Replay is class-balanced; front-leg kick anchoring is reduced. **No teacher targets are imposed on unproven bridge rollout states.** Policy/critic/Adam/exploration, native worlds, normalization, curriculum, counters and RNG are saved every completed update (more frequent than the requested2–5). All generations are exclusive; best kick/balance/bridge/combined and initial verified models remain separate. Exact resume rejects changed runtime/source/preparation/retention contracts. Old retained PPO generations are preserved, but the new sampler starts a new run from the corrected student; it does not pretend to exact-resume a different sampler.

## Targeted diagnosis, without repeating rollouts

`output/dynamic_bridge_diagnosis_v2/` reads the saved native CSV from the corrected student's matched standing replay. The first stalled forward rotation is at0.824s,36.11°, before the episode's later global peak. At that instant:

- pitch velocity−0.0302rad/s; whole-body pitch angular momentum0.0692kgm²/s;
- COM/support displacement−0.1923m and world x COM velocity−0.5580m/s;
- front ground moment about COM−8.97Nm, rear ground moment0, front vertical force159.04N;
- front slip speed0.0351m/s, calf clearance17.88mm, head clearance63.12mm;
- zero instantaneous saturated-joint fraction, target tracking RMSE0.2274rad;
- FL thigh/knee positions0.343/−2.185rad, velocities−4.213/+3.066rad/s;
- RL thigh/knee positions0.757/−1.321rad, velocities+2.894/−5.798rad/s (right leg values and all torques in JSON/CSV).

The CSV's certified-flight flag plus15mm rear-clearance criterion first coincides at0.682s; this differs from the earlier0.664s bilateral-clearance diagnostic. Neither is a future policy input. Existing exact-clock shooting reached48.25° versus47.74° baseline by reducing opposing front impulse, without connecting to80°. These measurements motivate earlier support learning; they do not establish that stronger actuators or more angular momentum alone solve it. No extra peak-pitch reward was added.

## Evaluation and honest capture accounting

Periodic PPO evaluation now covers the original4 standing and4 balance resets **plus all5 held-out actual bridge states**, instead of only2 bridge samples. Original endpoint retention criteria stay separate and unchanged. Expanded evaluation (`scripts/evaluate_dynamic_bridge.py`) adds8 new standing seeds and up to8 physically checked native60ms balance-perturbation descendants:12 standing, up to12 balance and5 bridge episodes. New balance states come from two legacy families, are correlated, and are not independent robustness evidence. They are evaluation-only, not teacher or training data.

Reports separate legacy/new/family cohorts and record certified takeoff, mean/max collision-free pitch, crossings45/60/75/85, front support/recontacts, genuine push-offs, illegal contacts/terminations, five-second holds, and strict standing completion requiring **both certified takeoff and legal five-second hold**. Crossing flags denote reaching the angle and may already be true at an intermediate reset. Empirical capture-neighborhood entry records incoming pitch velocity/COM motion; it is explicitly a diagnostic, **not certified recoverability**. Only real legal five-second hold demonstrates successful recovery. Full integration states generate best-standing and representative-failure GIFs. No pose reconstruction from CSV or trajectory splicing.

A local `--smoke` evaluator runs only two control steps in each of three cohorts and marks `complete_evaluation=false`. Its rows are software checks, never success-rate evidence. The previous4/4 takeoff,44.86° mean and3/4 balance are the last substantive validated endpoint results. No new learning improvement/full maneuver is claimed here.

## Copy-paste Marimo workflow

Upload `output/dynamic_bridge_go2_molab.zip` and open `notebooks/molab_dynamic_bridge_go2.py` (also inside ZIP). Its cells provide extraction/manifest validation, isolatedPython3.11 using uv from a Python3.13 notebook, pinned installation, deterministic inference, matched standing/balance validation, optional expanded baseline, measured one-process benchmark, pilot/resume/evaluation/rendering and complete backup download. The venv executable is **not resolved** before `uv pip install`. W&B is disabled; lock inheritance prevents overlapping jobs after a notebook disconnect. Installation must actually pass on Molab before training. Local clean-extraction checks reuse installed binaries and do not prove live dependency installation.

The notebook first runs the existing installation smoke, then evaluates ADC and the unchanged corrected student; failed endpoint validation blocks PPO. Benchmark measures n1/2/4 serial environments, one process, with the new sampler. It uses conservative memory limits and throughput to select n; no16-worker setup. The old WSL n2 result is not used as a Molab measurement. Monitor actual memory during full rollouts: short benchmark RSS is not a guarantee for a long job.

After setup exports ROOT, run, PREP, POLICY, CERT, OUT, N and H (H=128):

```python
# Optional expanded initialization baseline; correlated cohorts are reported separately.
run('scripts/evaluate_dynamic_bridge.py','--preparation',PREP,'--initial',POLICY,
    '--output','output/molab_expanded_initial','--render')

# A: bounded five-update pilot, 5*N*128 training interactions.
run('scripts/dynamic_bridge_go2.py','train','--preparation',PREP,'--certificate',CERT,
    '--initial',POLICY,'--output',OUT,'--n',N,'--horizon',H,'--updates',5,
    '--eval-every',5,'--interaction-budget',5*N*H,'--max-seconds',1800)

# After the pilot button has saved output/molab_training_config.json:
# B: explicit20–50 cumulative updates, numerical/endpoint checks before extension.
run('scripts/continue_dynamic_bridge.py','--preparation',PREP,'--certificate',CERT,
    '--output',OUT,'--target',25)
# Later, after reviewing evaluations:
run('scripts/continue_dynamic_bridge.py','--preparation',PREP,'--certificate',CERT,
    '--output',OUT,'--target',50)

# Major-checkpoint expanded evaluation: use an exclusive output folder each time.
latest=json.loads((ROOT/OUT/'latest.json').read_text())['checkpoint']
run('scripts/evaluate_dynamic_bridge.py','--preparation',PREP,'--initial',OUT+'/'+latest,
    '--output','output/molab_expanded_update_000050','--render')

# C: optional100,200 or300 only after actual recovery/contact evidence.
run('scripts/continue_dynamic_bridge.py','--preparation',PREP,'--certificate',CERT,
    '--output',OUT,'--target',100,'--baseline','output/molab_expanded_initial',
    '--evaluation','output/molab_expanded_update_000050')
```

Use buttons for the complete setup/configuration path; the equivalent pilot command alone does not create the notebook's configuration JSON. Pilot may stop early at a saved boundary when its30-minute limit expires. The resume cell loads latest.json and recorded n/horizon/target/budget. It revokes an explicit user pause but never clears a regression STOP automatically. Subsequent extension can target20,25,50,100,200 or300; it never runs an unbounded loop. Larger-than50 extensions require complete matched expanded baseline/current reports for the latest checkpoint, recovery/entry progress, no worsening illegal-contact fraction, and no substantial support-quality loss. Peak pitch alone is insufficient. CPU Torch may differ from local+cu130: load portable initializer, start a fresh Molab optimizer, then exact-resume only within that runtime.

At n2/H128,5/25/50/100/300 updates mean1,280/6,400/12,800/25,600/76,800 training interactions; evaluations are additional, separately reported experience. These are pilot budgets, not promises of enough training to solve the maneuver. Retention checks occur every5 updates; original/full generations are never overwritten. Longer progression requires human review; absence of45° in early updates alone is not a stop rule.

Pause at a checkpoint boundary, wait for the worker to exit, then use the backup cell. It exports all source/assets/data/models/optimizer/RNG/skill bests/latest/logs/hashes without venv/cache. Large backups are downloaded via the file browser to avoid notebook RAM pressure.
