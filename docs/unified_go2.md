# Unified Go2 handstand — opt-in fusion pilot

Branch `experiment/unified_go2_handstand`, based on validated `f74f34e`. The prior research branch remains unchanged. All original ADC, B3/update50, update52 and balance files are preserved. `output/unified_go2_preparation_v1/compatibility.json` records discovered paths, SHA256s, architectures, joint order, offsets/scales, effort/speed/position limits and control parameters.

## Implemented policy and conventions

One trainable actor selects all12 normalized joint-position actions in one forward pass. Actor inputs93, privileged critic inputs123; first73/103 inputs retain the existing ADC ordering. No running observation normalization exists: original physical scales remain unchanged. Both experts use50Hz control,500Hz MuJoCo, kp60/kd3 and action_scale2; original torque/speed/target-rate/joint/contact constraints are retained. Scalar/vector kd values are explicitly checked equal. The original8s observation-clock divisor remains8 even in12s episodes. ADC and saved balance checkpoint target heights are identical0.5600034658m; no energy-feature renormalization is mixed.

The final student contains a trainable observable-feedback affine layer plus a learned93→256→256→128→12 ELU residual. Residual value is multiplied by `(1−projected_gravity_x)^2`, making its value and first derivative vanish at upright. This is a local feedback-preserving network parameterization, **not an average of expert actions or a60°/75° switch**. No teacher module is called at inference. Unknown intermediate states use that same student's output. Their behavior is unvalidated and must be learned through real interaction.

ADC weights initialize the residual and critic, with20 extra input columns initialized to zero. Balance coefficients initialize the affine layer in the existing physical channels. Affine parameters are frozen during initial residual distillation, then trainable in PPO. Distillation minimizes pre-clamp physical-position targets to avoid zero gradients on initially saturated targets, plus local velocity-feedback derivative error at competent balance samples. Final physical actions are bounded; the actor returns corresponding atanh latents for the existing PPO tanh likelihood convention. Do not apply an additional transform to expert balance actions when building labels. Critic and PPO Adam start from compatible weights/fresh state; every subsequent checkpoint preserves optimizer and RNG state.

## Capture dataset and competence

Finite serial native probes span60,70,75,80,85,88,89,90°, incoming velocities0–1.2rad/s and negative velocities, COM offsets ±4mm, front/rear posture perturbations. Initial feasibility is checked using original geometry, contact penetration, joint limits and velocities. Failed proposals are recorded, never silently repaired or labeled successful. Valid initial states can still fail immediately.

The original balance actor **did recover from one80°, +0.6rad/s training proposal**, obtaining approximately6.0s of continuous valid hold. It also succeeded at selected89–90° proposals. Several88°,85°,75°,70° and60° proposals failed; this is a sparse, nonmonotonic set, not a certified80–90° interval. All details and physical reset snapshots are in capture_region.json and bank.pt. The successful80° point is promising but is a training state, not held-out generalization.

There are5 successful balance-training resets and11 valid bridge-training starts (including real saved post-takeoff/partial states), plus distinct held-out states. Teacher labels are collected only from successful balance-training trajectories and safe kick-up portions of certified ADC standing trajectories:1,870 frames total. No unsuccessful bridge teacher is imitated. Balance held-out measurements from the earlier four-state study are reused rather than relabeled for training. These near-vertical states are correlated; no broad robustness claim is made. A normalized neighborhood of measured successful states supplies a small bounded shaping credit and diagnostics; **neighborhood entry is not proof of recoverability**.

## Local results and failures

Local evaluation uses nominal identical physics, one held-out standing seed96000, one held-out bridge state and two correlated held-out balance starts per controller, with12s horizons. This tiny evaluation is not comparable to a four-seed mean as a learning claim.

| Controller | Standing certified takeoff | Standing peak | Near-vertical holds | Bridge five-second success |
|---|---|---:|---|---|
| Original ADC | yes |49.65°|0,0s|no|
| Original balance expert |no|27.08°|11.976,11.976s|no|
| Plain MLP student (preserved failed pilot) |yes|53.00°|0.10,0.08s|no|
| Feedback-residual student after2,000 BC steps |no|10.64°|11.976,11.976s|no|
| Same student after two numerical PPO smoke updates |no|7.51°|11.976,11.976s|no|

The longer reported balance duration follows the12s evaluation horizon; the previous6.176s result used6.2s episodes. The revised student retained balance on these two evaluated starts but **lost standing takeoff**. The plain MLP showed the opposite failure. There is no validated unified standing-to-handstand maneuver. A bridge trial reached about96° without any valid hold; that is not success. The two PPO updates are only16 interactions total, not a learning study. No claim of learned bridge improvement follows from them.

The first structured BC attempt used a saturated action loss with poor gradients; it is preserved as distillation_v2. The final pre-clamp-loss fit is distillation_v3, exported unchanged as `output/unified_go2_initial_v1/student.pt`. Preparation metadata was reconciled to the final architecture; export provenance records the previous preparation hash. Physical states, labels and expert checkpoints were unchanged. Plain MLP/v1 smoke sources and outputs are preserved separately; their checkpoint architecture is not silently resumed as the new actor.

Actual exact-state MuJoCo GIFs, full integration-state NPZs and synchronized numerical/contact/torque traces exist under evaluation_initial_v1, evaluation_initial_v2 and evaluation_smoke_v2. No approximate robot geometry or scripted resets are used inside a recorded attempt.

## Mixed-start PPO and continuation

`Environment` retains native multi-impulse safety/success validation, contact-aware progress, near-target braking, pose/support and low-motion hold rewards, illegal-contact termination and bounded clearance/effort penalties. Added terms are a once-per-standing-episode certified takeoff bonus, bounded capture-neighborhood progress, a small opposing-front-moment penalty only during clearly insufficient upswing, and a penalty for losing an established1s hold. Components are separately logged in replay traces. Repeated kicks receive no standalone bonus. Physical rear-foot recontact/reload and renewed push-off are monitored; free-air leg motion is not counted as new total momentum. No within-episode state resets, freezes or external impulses exist.

Serial mixtures (standing/bridge/balance) are initially10/15/75%, then20/30/50%,35/40/25%,60/30/10%. Promotion requires held-out actual balance success, then bridge recovery≥1s, then bridge five-second recovery with takeoff preservation. A decline below the best measured balance success returns the mixture to stage0. Distillation regularization uses only competent offline data, with reduced front-joint anchoring in ADC frames and full balance anchoring. No teacher supplies actions in an unvalidated gap.

Every completed PPO update gets an exclusive atomic checkpoint with actor, critic, Adam, source/runtime contract, preparation hash, full native simulation state, lane/reset RNG, NumPy/Torch RNG, curriculum, observation buffers and interaction counts. Resumption rejects changed sources/runtime/n/horizon and requires the latest boundary in the same output. Immutable checkpoints are retained; latest.json and evaluation-selected best.json point separately to generations. Selection logs balance success, standing success/takeoff, bridge hold and collision-free pitch, not reward alone. Best exists after the first scheduled evaluation. Keep evaluation frequency and other command semantics unchanged on resume. No normalization state is omitted: there is no running normalizer.

```bash
# Local-equivalent tiny smoke; a fresh output is mandatory.
python scripts/unified_go2.py train --output output/SMOKE --initial output/unified_go2_initial_v1/student.pt --n 2 --horizon 4 --updates 1 --eval-every 0 --interaction-budget 16
python scripts/unified_go2.py train --output output/SMOKE --resume output/SMOKE/checkpoint_000001.pt --n 2 --horizon 4 --updates 2 --eval-every 0 --interaction-budget 16
# Real evaluation/rendering, with distinct standing/bridge/balance records.
python scripts/unified_go2.py evaluate --output output/EVAL --initial output/SMOKE/checkpoint_000002.pt --controllers adc balance student --render
```

The final local smoke restored exactly and completed update2. Post-update KL was0.003685 and0.000407. Three focused tests passed, including compatible ADC weight mapping, exact next physical step after restore, all reset validity, and success-dependent curriculum gates. No long PPO or parallel heavyweight jobs were run.

## Molab package and staged execution

Use `output/unified_go2_molab.zip` and `notebooks/molab_unified_go2.py`. The Marimo notebook can use Python3.13; setup creates isolatedPython3.11 through uv and passes the **unresolved virtualenv executable path** to uv pip. Requirements are pinned, CPU Torch is selected and W&B is disabled. Setup verifies the complete archive manifest. Buttons prevent automatic training; a nonblocking file lock prevents concurrent heavy jobs. All MuJoCo environments are serial, with zero subprocess workers. Benchmark n2/4/8 on the actual host, review throughput/RSS, then hold rollout semantics fixed. No16-worker configuration is reused.

Stages: bounded initial distillation (notebook10,000 gradients), explicit BC takeoff/balance evaluation, tiny numerical smoke,20-update mixed-start diagnostic, then guarded continuation toward100 and optionally300 cumulative updates. With n4/horizon32 these targets are2,560/12,800/38,400 PPO interactions; larger meaningful budgets require a new explicit decision. The bundled structured initializer has a takeoff regression: **inspect the BC evaluation and choose whether more distillation or stage0 repair is justified before spending a longer budget**. Notebook stage1 explicitly uses the new Molab distillation rather than silently substituting the bundled candidate. Stage2 checks measured balance preservation. Curriculum promotion, rather than a timed expert switch, controls bridge and standing exposure.

Resume uses the latest pointer and identical n/horizon/evaluation settings. Wall-time limits are soft at complete-update boundaries. STOP halts between updates; do not interrupt checkpoint writes. The backup cell locks against active jobs and ZIPs all source, data and checkpoints, excluding only installed environments/caches. Download uses Marimo's UI.

Local exact-resume checkpoints use Torch2.13.0+cu130; Molab CPU Torch has a different build. Start a **new** Molab run from the portable student weights (or new distillation), then exact-resume within that unchanged Molab runtime. Never bypass runtime guards to claim bitwise cross-build continuation.

The ZIP verifier extracts to a clean temporary tree, verifies all SHA256s, disables editable/original-repo imports, executes focused tests, restores checkpoint2 and generates an actual short MuJoCo GIF from the extracted package. It reuses installed pinned binaries; it does not prove a live Molab network install. No hardware commands or physical deployment are added.

Future PR: keep the original balance research branch/PR independent; review this opt-in branch as one actor/data/runner/deployment addition. Before claiming fusion, recover certified takeoff without losing held-out balance, expand the capture map beyond its sparse80–90° points, then demonstrate a nominal standing-start transition and five continuous legal seconds. Longer learning belongs on Molab.
