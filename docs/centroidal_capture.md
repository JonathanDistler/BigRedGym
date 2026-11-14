# Go2 dynamic front-foot capture — 2026-11-13

**Level 2 achieved: continuous standing → certified takeoff → rotation → real front-foot contact and braking above 80°. Level 3 (one second of validated stability) and Level 4 (five-second handstand) remain unachieved.** The strongest trajectory reaches 91.14°, remains legal through 1.82 s, and continues under the unchanged balance actor until head-ground collision at 2.14 s. Its longest loose stability interval is 0.726 s; the original strict handstand gate only reaches 0.058 s. Sustained support is not equivalent to stable handstand.

Branch `experiment/centroidal_capture_handstand` was created from `15fab3f` after inspecting branch/history/tracked and untracked files. Existing artifacts and expert checkpoints were preserved. Git metadata required sandbox escalation to create the branch. The Molab update-5 checkpoint remains unavailable; all comparisons use the original local retained student, SHA256 `0d08913c5f306f4090f62f74001a62a795feee533958e3561adcfee816c984ab`. No update-5 comparison, hardware readiness, neural-policy improvement, or complete handstand is claimed.

## Why the airborne seeds fail

Both original trajectories were reconstructed using their actual saved spline parameters, standing state and actor. At 500 Hz, both the unchanged unified student and the standalone validated balance actor were tried after the 1.14 s seed endpoint. All four nominal-physics trials terminate by head-ground collision around 1.18 s. The original 83° sequence's first illegal contact is 1.170 s with balance feedback, 1.172 s with the unified actor. Lower-head contact occurs before calf, thigh or rear-leg ground contact. This is not an unavoidable rear-leg collision.

| At seed endpoint 1.14 s | 61.83° seed | 83.39° seed |
|---|---:|---:|
| front foot gaps FL / FR |0.166 / 0.157 m|0.264 / 0.268 m|
| COM velocity x / z |−1.110 / −1.621 m/s|−0.600 / −1.859 m/s|
| COM speed |1.965 m/s|1.954 m/s|
| mechanical energy |97.003 J|98.741 J|
| pitch-axis centroidal momentum |0.61267|1.26516 kg m²/s|
| lowest non-foot body clearance |0.0682 m|0.0598 m|
| front thigh angles FL / FR |1.377 / 1.357 rad|1.583 / 1.571 rad|
| front knee angles FL / FR |−2.300 / −2.307 rad|−2.163 / −2.165 rad|

The 61° seed is **not substantially lower in total mechanical energy or COM speed**; only its rotational momentum is substantially lower. Both command a front-leg configuration that leaves the feet above the head's approaching ground envelope. The 83° front feet rise from ~0.081 m gap at 0.9 s to ~0.295 m at 1.1 s while the body is already descending. The late front-thigh command goes toward +1.82 rad rather than the demonstrated balance target near −0.97 rad. Knee retraction and hip reconfiguration together prevent useful early support. Available joint torque is not exhausted: at 1.14 s front thigh torque margins exceed 21 N m and knee margins exceed 44 N m. Geometric positioning and time to act, rather than simply more torque, are the primary observed barriers.

`reachability.json` contains fixed-body, joint-limited 41×41 thigh/knee scans for both front legs at 0.8, 0.9, 1.0, 1.1 and 1.14 s. Scans reject sampled front calf/thigh penetration. At 0.9 s, neither front foot of the 83° seed can reach the floor in the sampled collision-free configurations (minimum gaps ~2–3 cm): vertical launch trajectory matters too. At 1.0 s, ground-reaching sampled configurations exist, with optimistic speed-only travel times ~0.040 s. At 1.14 s they require ~0.046 s, versus ~0.030 s until the observed head impact. These are optimistic kinematic diagnostics, **not dynamically feasible trajectories, exhaustive reachability certificates, or a proof of an exact irrecoverability time**. Body motion, acceleration and friction also matter. The latest useful intervention boundary is not rigorously certified; experiments establish that beginning at 0.62 s works substantially better than appending actions at 1.14 s. The final 40 ms are demonstrably inadequate for either existing actor.

High-resolution `*_dt1.pt` files preserve full matrices (Euler roll/yaw become singular near vertical), body angular velocities, COM position/velocity, foot positions/velocities/gaps, joint angles/targets/velocities/torques/margins, front/rear loads, individual link momentum and inertia, clearance groups and illegal contact identities. The dedicated original audit adds named head/trunk clearance at 500 Hz. Rear link contributions are available in each diagnostic row's independent `link_audit`.

## Contact impulse discrepancy: cause identified and reduced

Previously, contact forces were sampled after `mj_forward` at the **new** state. Those are recomputed forces for that state, not necessarily the forces that advanced the preceding step. The torque command is also replaced before the next step. Interpolating that stream across a contact/command transition misaligns impulse with velocity integration.

The new audit wraps the existing `mj_step` solely to record contact forces **after the solver but before the following `mj_forward`**, using its still-corresponding contact positions, frames and COM. No solver settings, commands or states are changed. Its prefix integration state matches the saved original 83° state exactly (maximum difference 0). It includes all external contacts, free contact torques and `xfrc_applied`; self-contact is internal, gravity has zero moment about whole-robot COM, and generalized applied forces on the floating base are zero. Original actuator joint torques remain internal. `frame.T` transforms contact-frame wrenches; geom2 gets the positive wrench. Existing independent contact-sign/link-Jacobian tests remain passing.

| Pitch-momentum residual through 1.14 s | Nominal 2 ms | Diagnostic 1 ms |
|---|---:|---:|
| 83° seed, old post-forward forces |−0.0092474|−0.0079708|
| 83° seed, actual solver-step forces |+0.00076222|+0.00040325|
| 61° seed, actual solver-step forces |−0.00276870|−0.00137059|

Units kg m²/s. The 83° residual falls by about 12× after fixing sampling. Remaining residual approximately halves with timestep, consistent with first-order discretization effects. The model uses Euler with implicit damping treatment, Newton solver, 100 iteration cap and tolerance 1e−8; the saved generalized finite-difference velocity acceleration differs from `qacc`, as expected with implicit damping integration. Armature/damping are preserved and logged. This is quantified numerical closure, not exact conservation to machine precision, and no tight equality on a tiny residual is used in the objective. The 1 ms rollouts are diagnostics only; every claimed motion result uses the original 2 ms model. See [MuJoCo pipeline documentation](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) and [integrator documentation](https://mujoco.readthedocs.io/en/stable/computation/).

## Bounded capture search and measured outcomes

All trials use one matched standing initial state, seed 96000. A genuine forward-simulated prefix supplies the initial state at 0.62 s. Planning corrections begin before loss of front contact; there is no reset within any evaluated motion. Reference commands come from the actual 61° or 83° seed and continue with the last seed command until the specified offline balance takeover. Five linear interpolation knots at [0, .16, .34, .56, .8] s control front-left thigh/knee, front-right thigh/knee and symmetric rear thigh/knee corrections. Position actions are clipped to [−1,1] and passed through the original target-rate/joint/torque/speed limits; normalized residual bounds up to ±1.5 allow cancelling a seed command near its opposing limit. No force, support pin, gain/friction/collision modification or pose injection is used.

The objective includes front gap, near-ground impact velocity, slip speed, COM/foot alignment, body velocity, full centroidal momentum norm, legal contact loading and sustained capture. Later versions add full orientation, demonstrated joint posture, empirical capture distance and COM-height error. Signed moment is **not** rewarded as universally positive: measured contact braking is permitted and desired. Torque/speed/joint/collision violations end a legal prefix; clearance and authority are logged, with collision avoidance enforced through actual geometry rather than a smooth approximate collision model. A full dynamic reachability constraint is not implemented.

| Condition | Front contact above 60° | Loose stable capture | Original strict hold | Subsequent result |
|---|---:|---:|---:|---|
| Original 61° + balance |0|0|0|head impact ~1.18 s|
| Original 83° + balance |0|0|0|head impact ~1.18 s|
| 61° capture CEM v2 |0.34 s|0.08 s pitch-based proxy|0|actuator-state limit at 1.52 s|
| 83° capture CEM v2 |0.50 s|0.16 s pitch-based proxy|0|self-collision at 1.56 s|
| 83° capture CEM v3, 1.82 s endpoint |0.80 s|0.44 s with full orientation|0.058 s|legal, still running|
| v3 continued unchanged balance |support until ~2.13 s|**0.726 s**, 500 Hz|**0.058 s**|head impact at 2.14 s|
| additional height/posture v4 pilot |0.36 s|0.06 s|0|self-collision at 1.46 s|

Loose stable capture requires legal cumulative prefix, both front feet loaded, full orientation error <0.2 rad, |pitch rate|<1 rad/s, COM speed<0.6 m/s and rear clearance>0.08 m. It does **not** replace the stricter established gate (height, <0.25 m/s COM speed, <0.5 rad/s body angular speed, low slip, support geometry, force and collision checks). Early v2 used pitch instead of full orientation for its proxy; these numbers are not directly comparable. Some v2 arrivals have substantial twist about the near-vertical body axis; Euler pitch alone hid this. Near-symmetric front corrections plus full orientation/capture-posture penalties improved the arrival in v3, but several changes were combined, so this is not an isolated causal ablation.

There were **320 CEM candidate evaluations**: 48 initial 61° pilot, 80 refined 61°, 80 refined 83°, 80 orientation-aware 83°, and 32 final height/posture attempts. Fourteen additional matched balance takeover-time tests (1.16–1.40 s, both v2 seeds) failed. Every best trial was replayed from standing with a freshly compiled model; final integration-state errors were 0. Generation checkpoints retain parameters, distribution, history, RNG and protocol. No concurrent MuJoCo pools, PPO or distillation were run. Observed WSL RAM remained roughly 0.8–1.3 GiB used with 248 KiB swap, well below available memory.

## Strongest trajectory and remaining failure

The retained best is `output/centroidal_capture_83_v3/best.pt`, with the full failed continuation in `output/centroidal_capture_83_continuation_v1/replay.pt`. This is a planning diagnostic followed by an unchanged neural actor, **not** a deployed single-policy improvement. Balance outputs are already bounded position actions and are not tanh-squashed again; observation ordering is unchanged. Seven focused tests cover conventions, contact signs, momentum and objective/correction behavior.

The first both-front load after 0.9 s occurs at **1.054 s, 66.93°**. The left foot approaches with velocity approximately (−1.33, −0.41, −0.415) m/s, the right approximately (−0.187, −0.031, +0.001) m/s. Thus normal impact speed is moderate but substantial initial tangential motion remains; this is not an ideal low-slip landing. Actual contact braking from that event through 1.32 s supplies pitch-axis impulse **−0.145205 kg m²/s**, versus measured ΔH **−0.145192**, confirming useful external angular-momentum reduction. Both front feet support the body as it progresses above 80°. Rear feet remain clear during the loose stable interval. The strongest candidate's 1.82 s endpoint has pitch87.724°, pitch rate−0.0413 rad/s, COM velocity(−0.0337,−0.00047,−0.1964)m/s and both feet loaded.

The balance takeover at 1.32 s enters the empirical capture neighborhood (distance~0.749), but that distance is not a viability certificate. COM height is~0.507 m versus demonstrated target0.560 m, and front knees are about−1.52 rad versus demonstrated−1.18 rad. Under the unchanged balance actor, COM height falls to~0.459 m by1.8 s, then continues downward despite small pitch error. Head clearance decays to zero. This is predominantly a **height/posture capture failure after successful pitch braking**, not continued unchecked angular rotation. The extra height/posture pilot did not improve it and is preserved as a failed intervention.

**Next recommended action:** use the v3 contact transition as the reference, retain near-symmetry, and optimize a separate 1.05–1.65 s support-phase spline with stronger terminal vertical velocity, base/COM height and front-knee posture constraints. Optimize takeover time jointly with that support-phase spline. Score actual strict hold over at least a 2–3 s terminal rollout, not just the best instantaneous arrival. Keep the early launch/contact portion fixed initially to avoid losing the achieved landing. The current last two knots are poorly placed for independent post-contact height regulation; add support-relative knots or event-triggered phase onset rather than increasing launch impulse. Broaden starts only after one genuine five-second recovery. Do not distill these failed continuations.

## Files, replay and portable continuation

- `output/centroidal_capture_diagnosis_v1/`: 500 Hz original seed analyses, timestep refinement and kinematic contact options.
- `output/centroidal_capture_61_v2/`, `...83_v2/`, `...83_v3/`, `...83_v4/`: bounded searches, complete environment starts/finals, actual standing commands/states, parameters, progress and GIFs. v2/v3 source snapshots preserve objective provenance.
- `output/centroidal_capture_83_continuation_v1/`: full cold standing replay with failed balance continuation and GIF.
- `output/centroidal_capture_best_audit_v2/`: 500 Hz contact/braking audit, complete qpos/qvel and exact replay check.
- `output/centroidal_capture_molab_v2.zip`: separate self-contained extension of the earlier portable bundle, experts, preparation, seeds, results and scripts. The first archive failed clean extraction because inherited packaging had omitted preparation manifests; v2 explicitly includes the complete preparation. CRC/member SHA256 checks and a clean extracted cold replay are performed locally; live Molab execution is not claimed.

```bash
# Existing Python3.11 workflow; inherited Python3.12 PYTHONPATH must be removed.
env -u PYTHONPATH .venv/bin/python scripts/replay_centroidal_capture.py --trajectory output/centroidal_capture_83_v3/best.pt --output output/capture_REPLAY --continue-balance --render
env -u PYTHONPATH .venv/bin/python scripts/audit_saved_centroidal_capture.py --trajectory output/centroidal_capture_83_continuation_v1/replay.pt --output output/capture_AUDIT
env -u PYTHONPATH .venv/bin/python scripts/search_centroidal_capture.py --output output/capture_NEW --seed-trajectory 83 --iterations 1 --population 6 --steps 140 --warm-start output/centroidal_capture_83_v3/best.pt --handoff 1.32 --render
env -u PYTHONPATH .venv/bin/python -m pytest tests/unit_tests/test_centroidal_capture.py tests/unit_tests/test_centroidal_bridge.py tests/unit_tests/test_balance_feedback.py -q
```

On Molab extract into a fresh directory, use the existing bundle's Python3.11 dependency workflow, and run the six-candidate pilot first with a new output. Current source uses objective version4; do not resume version2/3 with it. To exactly resume v3, restore its two archived source files into the **fresh extraction**, then repeat its original command: `--output output/centroidal_capture_83_v3 --seed-trajectory 83 --iterations 5 --population 20 --steps 60 --warm-start output/centroidal_capture_83_v2/best.pt --handoff 1.32`. The existing four-generation checkpoint continues from generation4. Preserve all protocol settings and original actors. Current-version runs resume by repeating identical flags/output with larger `--iterations`; mismatched protocol is rejected. To back up, copy the complete output directory including progress.pt, inputs.pt and source snapshot; use the package script to create a new, never-overwritten archive. No long job or remote installation is launched automatically.
