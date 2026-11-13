# Retained-skill Go2 fusion — measured local result

Starting commit `d314ef1`, branch `experiment/unified_go2_handstand`. All earlier research branches, checkpoints and the previous ZIP remain preserved. This experiment is opt-in through `scripts/retained_go2.py`; it does not rewrite the original ADC, balance or d314ef1 actor classes. No hardware commands or live Molab jobs were run. The user explicitly requested local continuation because no Molab tool is connected.

## Takeoff regression: diagnosis and smallest correction

The old initializer did **not** represent the ADC policy: it added global balance affine feedback in standing and multiplied the ADC latent output by a gravity factor before interpreting it as a physical target. This is a function/action-domain initialization error, not a nominal-gain change. Before fitting, kick-frame action MSE was0.34745 loaded/0.32332 airborne. After the old fit it was still0.002879/0.004836. Both teacher input conventions remained unchanged; neither had a running observation normalizer. The old supervised minibatches were already balanced64/64, so simply adding more balance data was not the remedy. Earlier plain-MLP forgetting was a separate observed problem; no broad gradient-interference ablation is claimed.

In matched seed96000 replay, original ADC first rear-foot clearance occurs at0.664s; forward pitch velocity first stalls at0.824s,36.11°. The old student's clearance at0.122s was **not certified takeoff** and its episode ended0.22s. Positive rear-leg work was30.51J for ADC over0–0.8s versus31.13J for the old student over its available0–0.22s prefix: extra work alone did not create useful rotation. Rear ground angular impulse was10.284 versus2.254Nms over those respective prefixes; opposing front moment reached−47.72Nm at0.124s in the failed student. Windows are intentionally labeled unequal; no missing trajectory is extrapolated.

`output/retained_go2_analysis_v1/` contains same-observation phase/per-joint losses, fixed-normalization records and a synchronized motion/torque plot. Raw joint target, angle, velocity, torque, foot loading, moment and impulse trajectories are preserved in the native physics CSVs. For the matched ADC/corrected standing replay, **every compared target/torque/velocity/force/momentum channel has maximum difference0**. Against the original73-input actor, padded93-input initialization is tested within2e−6.

The new actor copies the ADC base/critic weights, zero-pads the additional20 observation columns and initializes a shared correction's output layer to zero. Its balance specialization is disabled initially: deterministic ADC means are exactly reproduced against the expanded adapter. Then a learned competence router is fitted to reliable kick/capture demonstrations, while both specialist coefficients stay protected. One actor forward pass returns all12 targets:

`z = z_ADC + g(observation)*(z_balance − z_ADC) + correction(observation)`; action=`tanh(z)`.

Exact0/1 branches avoid cancellation at competent endpoints. The gate is learned from geometry and COM-height data, with empirical support bounds; **there is no fixed pitch or time controller switch and no averaged teacher labels**. Intermediate logits are an internal neural parameterization, not evidence that either teacher is competent there. The shared correction receives all93 inputs and can change all12 targets anywhere, including before rear liftoff and outside the balance domain. Frozen neural coefficients do not freeze physical joints. During PPO, router/correction, critic and exploration remain trainable under competence-limited teacher replay.

A targeted probe found that a contact-centroid jump during legitimate unloading could spuriously reject the balance specialization. Router features therefore exclude discontinuous support centroids, contact forces and velocities; velocity/force information remains available to the balance feedback and shared correction. This improved the routing design, but the already-failing fourth balance case still fails earlier than the standalone expert. The gate is not a viability oracle.

## Endpoint results and limits

Nominal physics:50Hz commands,500Hz native simulation, kp60/kd3, original12-joint order, offsets, action_scale2, torque/speed/joint/target-rate limits and real colliders. No physics limits were relaxed. Actor inputs93/critic123 retain the original73/103 prefixes. ADC and balance heads use unchanged physical feature scales. Only the router has fixed, checkpointed center/scale/support buffers. Initial exploration std is deliberately reduced to0.002479 for PPO; **only deterministic ADC means are claimed preserved**, not the old stochastic exploration distribution. `retained_spec.json` and the checkpoint state dict record the current architecture/statistics; inherited compatibility.json describes ancestral architecture but unchanged expert/control conventions.

Matched standing seeds96000–96003; four correlated held-out balance configurations are distinct from teacher demonstration starts:

| Controller | Certified takeoff | Mean standing peak | Five-second balance |
|---|---:|---:|---:|
| ADC expanded adapter |4/4|44.859°|not its competent skill|
| Corrected shared actor |4/4|44.859°|3/4|
| Shared actor after32 PPO interactions |4/4|45.335°|3/4|

Corrected valid holds:6.176,6.176,0.040,6.180s. Post-smoke:6.176,6.176,0.020,6.180s. The standalone balance expert's failed perturbation held3.692s before self-collision; the shared actor fails that same case earlier by non-foot collision. **Five-second success membership is preserved, not all partial-recovery quality.** No new failed starts occur among the known successful balance starts; all standing attempts still eventually fail as the ADC baseline does. These are tiny, correlated suites and one training seed, not robust success evidence.

Standing five-second handstand success remains **0/4**. No standing episode contains a reset or artificial impulse. The four-way exact-state GIF compares ADC, the previous regressed student, the corrected student, and the post-smoke student on matched standing seed96000. Ended episodes are explicitly held on their final frame; no trajectory is spliced into a handstand. Separate near-vertical GIFs visibly start near vertical.

## Physical bridge: actual states and bounded shooting

Final bank:10 training and5 held-out bridge snapshots. These come from **real ADC rollouts** on separate train95120/95121 and held-out96120/96121 seeds, plus native controlled continuations. Sampling occurs on actual25/35/45° crossings; no body pitch or joint interpolation creates these states. Complete integration state, actuator/target history and tracker state are retained. Shooting restores the original policy clock and full state, applies bounded±0.08rad front thigh/knee and rear-thigh target perturbations for120ms, then continues native dynamics to300ms. Every accepted intermediate is checked for legal contact, joint limits, speed and torque constraints. The executed command sequences are offline feasibility probes, **not a scripted deployed kick/balance sequence**.

From a real45.54° state, unmodified continuation reached47.74°; the tested front-thigh−0.08/knee+0.08 variation reached48.25°. Its front angular impulse was−2.308Nms versus−2.877Nms baseline over300ms. Total angular momentum still decreased; this was reduced counter-braking, not new free-air angular momentum. The tested rear-posture change did not improve the connection. All four exact-clock shooting prefixes were physically valid, but **none reached the capture region**. Full traces/state lists and provenance are shipped.

At the first ADC stall, mechanical energy54.95J contrasts with88.00J at the measured80°, +0.6rad/s capture reference: about33.05J reference-state difference. At a real45.54° sample energy is50.47J and L_COM0.799kgm²/s; the80° reference has L_COM0.329kgm²/s. Thus a larger raw angular-momentum reward is not a demonstrated solution: front contacts dissipate/reverse useful rotation, and COM elevation/support geometry differ substantially. These are articulated-state differences, **not exact required-energy/momentum lower bounds**; actuators can add work through contact.

Direct balance-expert takeovers from the tested real35°/45° ADC states failed by self-collision. No connected45→80° trajectory is claimed. The successful80° point remains one measured training-family capture configuration, not an80–90° viability interval. The new bank removes the old arbitrarily pitched bridge proposals; previously validated balance-capture resets remain legitimate endpoint examples.

On two fixed actual held-out bridge starts (correlated samples from one held-out ADC episode), initial shared-actor peaks were47.80°/44.69°; post-smoke peaks48.05°/46.20°. Both had0/2 five-second success and zero valid hold. Small peak changes after32 interactions are not reliable learning evidence; failure classes and contact fractions are retained in JSON. The full maneuver remains unsolved.

## Teacher retention, adaptive curriculum and checkpointing

There are1,870 **unique** competent teacher frames:320 kick and1,550 balance. Distillation uses exactly64/64 class minibatches. PPO replay has1,550 rows per class through explicit repeat indices (3,100 replay rows): equal expected class sampling, **not extra experience or fabricated states**. The original files remain unchanged. Duplicate provenance is recorded in replay_balance.json. The endpoint certificate is explicitly transferred only across byte-identical physical reset banks/configs/centers for this replay-only change; it does not fabricate a new evaluation or bridge result. Live Molab must validate both endpoints afresh.

Adaptive sampling reserves at least25% standing and25% balance, allocating remaining experience toward measured failures, principally the bridge. Completed-episode competence EMAs, trial counts, reset RNG and curriculum stage are checkpointed. Promotion requires both endpoint competencies, then real bridge hold≥1s, then five-second bridge recovery. Genuine rear recontact/reload/push-off and foot recontacts are monitored; internal leg swing is never counted as newly generated total angular momentum. No repeated-kick bonus is added.

All original rewards/real contact termination remain; shared residual learning is the new mechanism. Competent replay regularizes known skills, with reduced front-leg ADC anchoring so new preparation/support motion remains learnable. No teacher label is used in a bridge region where neither expert demonstrated competence.

Every completed update creates an exclusive atomic full checkpoint with actor/critic, Adam, fixed normalization buffers, native worlds, NumPy/Torch/reset RNG, adaptive sampler/curriculum, counters and best-selection state. Source/model/runtime/preparation and retention-contract checks reject incompatible resume. Best kick/balance/combined pointers select **admissible unified checkpoints**; best bridge is recorded separately. Every candidate full checkpoint is retained, including rejected ones; original standalone experts remain separately preserved. The corresponding full checkpoint generation retains optimizer state for continuation. Restoring a checkpoint reconciles best-selection metadata, archiving any interrupted newer selection first.

A loss of4/4 takeoff, a mean peak more than1° below matched ADC, or fewer than3/4 balance successes writes STOP. The current boundary is saved before the next update; known-good initial and prior generations remain intact. This is a deliberately strict skill-retention gate, not an assertion that the failed fourth reset is solved. Numerical KL backtracking preserves optimizer state and consumes no additional environment samples.

The final release smoke uses n2/horizon8: update1 then exact resume to update2,32 training interactions total. KL0.01235/0.00454. Earlier superseded debugging smokes remain preserved and excluded from the selected model's training count. Six focused tests passed, including original73-input mean tolerance, joint order/control scales, fixed normalization, demonstrated skill evidence, all native reset validity, equal-class replay/unique counts, adaptive sampling and exact next PPO update including Adam/router/RNG. A changed retention contract is rejected.

## Molab: copy-paste jobs, after installation and fresh retention

Open `notebooks/molab_retained_go2.py` with `retained_go2_molab.zip` beside it. Setup works from a Python3.13 Marimo kernel and creates isolatedPython3.11 via uv. `uv pip install` receives the **unresolved** `.venv311/bin/python` path. W&B is disabled. Actual installation/inference smoke precedes benchmark/training. Endpoint validation replays both ADC and the student on that host; failure blocks PPO. A one-worker benchmark tests n1/2/4, records memory/throughput and recommends a measured conservative count. No16-worker configuration is used.

Buttons prevent automatic jobs. A file lock is inherited by the worker so a kernel disconnect cannot permit overlapping heavy jobs. Source/data/model hashes are verified. Source/reference normalization states remain fixed; initial CPU weights are portable but local+cu130 full checkpoints must not be exact-resumed under a different CPU Torch build. Start a fresh Molab run from the initializer/new distillation, then resume within that same runtime.

After the notebook defines `run`, PREP, POLICY, CERT, OUT, N and H:

```python
# StageB:20 bounded updates; H=128, N selected from1/2/4 after measured benchmark.
run('scripts/retained_go2.py','train','--preparation',PREP,'--certificate',CERT,
    '--initial',POLICY,'--output',OUT,'--n',N,'--horizon',H,
    '--updates',20,'--eval-every',5,'--interaction-budget',20*N*H,'--max-seconds',1800)
# Exact continuation: use latest.json and identical n/h/evaluation semantics.
latest=json.loads((ROOT/OUT/'latest.json').read_text())['checkpoint']
run('scripts/retained_go2.py','train','--preparation',PREP,'--certificate',CERT,
    '--output',OUT,'--resume',OUT+'/'+latest,'--n',N,'--horizon',H,
    '--updates',20,'--eval-every',5,'--interaction-budget',20*N*H,'--max-seconds',1800)
```

StageC50–100 cumulative updates is a manual, guarded decision after actual bridge improvement and endpoint retention; n4/h128 target100 means51,200 PPO interactions. StageD300 means153,600, only after further evidence/resources. Counts exclude extra separately logged evaluation rollouts. Do not stop merely because early updates fail to cross45°. Wall-time bounds are soft at update boundaries. Regression STOP is not automatically cleared. Use explicit separate forks/known-good full generations after review rather than bypassing source/retention checks.

Pause via STOP, wait for the worker to exit, then use the backup cell. The backup ZIP includes source/data, complete native/optimizer/RNG checkpoints, all skill bests, latest pointers, logs and hashes, excluding installed environments/caches. Large backups use the file browser rather than allocating the ZIP in notebook RAM.

Local verification uses a clean extracted tree with original editable-repository hooks disabled: imports, focused tests, full checkpoint restoration and actual short MuJoCo inference/GIF. Pinned installed binaries are reused. **Live Molab dependency installation and launch were not performed** because no connection is available, as confirmed by the user. The first notebook stages must pass there before any longer job.

The prior ADC/balance research branches remain suitable for their separate PRs. This opt-in change is a shared actor/real bridge data/retention-safe learning addition, not a claim of a complete handstand controller.
