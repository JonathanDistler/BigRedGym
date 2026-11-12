# Go2 handstand curriculum V2

This opt-in experiment lives in `experiments/handstand_v2/` on
`experiment/handstand_curriculum_v2`. The original `curriculum.py`, morphology
learning code, checkpoints, and imported experiment results are preserved.
No demonstration from the failed 51-degree airborne trajectory is used.

## Diagnosis

Ten identical fresh reset seeds (70400–70409) and paired independent Torch noise
seeds are used in both deterministic and stochastic modes for the original
update-36 policy and imported updates 4, 20, and 44. Actor/critic/action dimensions
remain **44/74/12**, with strict state-dict loading, native gains, torque clipping,
overspeed protection, joint ranges, contacts, and MuJoCo integration.

The original curriculum explicitly sets `cfg.rotation_enabled = False` in
`load_source()`. The parent TakeoffEnvironment also replaces the generic grounded
pitch reward. Thus the continuing learning signal is primarily the bounded
one-use takeoff credits and the new strict-gated potential. The latter computes
`phi = progress if valid else 0.0`: no useful progress potential survives when
any strict condition fails. In particular, cumulative loaded foot-center travel
exceeds 2 cm near takeoff, and the historical maximum support gap later exceeds
20 ms. Neither condition can recover during the episode. Disabling the native
rotation objective is intentional replacement, but the replacement is ineffective.
The native `airborne_rotation` term is still nonzero between roughly 25° and
34°: for seed 70400 its episode credit is 0.725, with 0.223 from
`supported_flight`. Thus the diagnosis is **not absence of every rotation
reward**; it is loss of the added Stage 1 signal, weak post-takeoff incentives
relative to the 3-point launch bonus, and physical slip/COM-rotation limitations.
Potential shaping alone would also cancel on returning to zero/termination;
it should not be mistaken for persistent positive progress income.

For the original deterministic seed 70400:

- True contact-point slip reaches 2 cm at **0.762 s**.
- Body pitch peaks at **34.52 degrees, 0.880 s**, while both front feet are loaded.
- Bilateral front support is first lost at **0.932 s**.
- First non-foot collision is **FL_calf at 1.082 s**.
- No speed or joint-limit failure precedes the peak.

All 80 measured original/imported-policy rollouts have their first collision
**after** peak pitch. The collision rate therefore does not explain takeoff
failure: takeoff succeeds in all of them. Material slip exceeds the certification
limit before peak pitch in all 80. Loaded peak and certified peak are distinct.

| Policy | Deterministic loaded peak | Stochastic loaded peak |
|---|---:|---:|
| Original update 36 | 34.30° | 33.71° |
| Imported update 4 | 34.43° | 34.06° |
| Imported update 20 | 30.48° | 31.11° |
| Imported update 44 | 32.03° | 30.91° |

For the original, deterministic certified pitch averages 29.18°, but stochastic
certified pitch averages only 2.69°: noisy actions often exhaust the slip budget
before a qualifying takeoff. This is a physical robustness limitation that a raw
loaded peak hides. Zero means no certified post-takeoff state, not zero body pitch.

## Measurements and rewards

`Measurement` reports independent raw body peak, loaded peak, strict certified
peak, raw dynamic crossing snapshots (with loaded/safety labels), strict supported
crossing snapshots, recovered support, and caught duration. Controlled crossing
certificates require body pitch rate between -0.5 and 2 rad/s (the upper bound
matches the shaping profile cap); faster raw crossings remain diagnostic only. A later fall cannot
erase a previously certified crossing. Historical maximum gap persists; current
gap resets on reloading. Flight is never counted as supported motion. Safety
failure remains permanent for that episode. Reset clears all episode certificates.

Foot-center drift is geometric displacement from reset. True loaded slip is
integrated tangential **material velocity at actual ground contact points**, using
MuJoCo point Jacobians, per-foot loading thresholds and the contact normal. Rolling
and airborne center displacement do not automatically count as slip. Flight
center travel is independently recorded. Slip certification keeps the 2 cm limit;
drift keeps 4 cm; touchdown speed keeps 1 m/s. No collision or motor limit is relaxed.

V2 retains certified takeoff rewards and adds smooth supported progress income
from 25° onward, desired angular-velocity tracking with braking near the target,
rear clearance, loading, slip-speed penalties and contact-moment penalties. These
are integrated at the physics rate. Positive shaping requires takeoff, bilateral
loading, and safety, but does not require already satisfying the final success
gate. Cumulative slip can disqualify certification while still providing graded
feedback and a slip penalty. The distinction is explicit: continuous reward is
not a safety or success certificate. No positive V2 income is paid during flight.
Native motor penalties remain; a strict speed/joint/torque safety violation stops
V2 at the next control boundary in addition to native collision termination.

The fixed, sequential stages are 0 (certified takeoff), 1A (35°), 1B (40°), 1C
(45°), 2 (45° crossing plus .15 s catch), 3 (60° plus .15 s), 4 (85° plus .30 s),
and 5 (90° plus 5 s). Reliable takeoff is preflight-tested before starting stage
1A. Two passing evaluation windows promote a stage. Targets are fixed within a
training block; no unobserved randomized command is added to the 44-D actor.
Native takeoff shaping remains active throughout to resist forgetting.

## Model guidance and rotational authority

The existing rotating variable-length COM pendulum diagnostic is reused only in
its eligible intervals: front support sufficiently stationary, rear feet unloaded,
and no invalid contact. All actions still run through full MuJoCo dynamics.
Desired COM rate is energy-deficit based, bounded at 2 rad/s, with sign toward the
upright COM angle; predicted and measured acceleration are plotted separately.
It is guidance, not a guarantee or replacement for contact dynamics.

For the original first reset at peak pitch, COM pendulum angular velocity is
already -1.00 rad/s (away from upright), and angular momentum about the front
support is -2.16 kg m²/s. Rear contact moment is zero; front contact moment about
COM is -8.37 Nm. The approximate energy deficit is 9.64 J. Front calf joints move
from about -2.29 to -2.18 rad between .80 and .88 s while rear calves retract from
about -1.10 to -1.35 rad. This shape change accompanies body pitch but fails to
maintain useful COM rotation. It does not demonstrate that the target is infeasible.

Median eligible pendulum-acceleration RMSE across all 80 rollouts is about
7.96 rad/s². Moving support and articulated shape dynamics materially limit this
approximation, even within its eligibility filter. Do not optimize solely against
its predictions. `authority_probe.py` performs a separate bounded front-calf and
rear-calf action perturbation assay to investigate yielding/retraction. It never
exports demonstration targets or labels an assisted action sequence as a learned
policy. Results are in its separate output directory.

## Validation and execution

Use the existing Python 3.11 environment with its injected Python 3.12 PYTHONPATH
removed. The portable ZIP includes pinned minimal dependencies, all necessary
repository sources/resources and copies of all four input checkpoints. Its
manifest identifies the base Git commit and SHA-256 of every bundled file.
Each run records its full config, input checksum and exact source ZIP/checksum;
this also preserves the originally untracked `curriculum.py`.

```bash
# Existing local environment
export MPLCONFIGDIR=/tmp/handstand_matplotlib
export XDG_CACHE_HOME=/tmp/handstand_cache
export MUJOCO_GL=egl
unset PYTHONPATH
.venv/bin/python -m pytest -q experiments/handstand_v2/test_measurement.py tests/unit_tests/test_rotation_learning.py tests/unit_tests/test_takeoff_learning.py
.venv/bin/python -m experiments.handstand_v2.run diagnose --output output/my_v2_diagnostics
.venv/bin/python -m experiments.handstand_v2.report output/my_v2_diagnostics
.venv/bin/python -m experiments.handstand_v2.run smoke --output output/my_v2_smoke
.venv/bin/python -m experiments.handstand_v2.run pilot --gate output/my_v2_smoke/gate.json --output output/my_v2_pilot
.venv/bin/python -m experiments.handstand_v2.authority_probe
.venv/bin/python -m experiments.handstand_v2.package --output output/my_v2_molab.zip
```

Every output directory must be new. Existing artifacts are never overwritten.
The smoke run is exactly two 1024-interaction PPO updates; the pilot is bounded at
ten updates with an automatic early stop for takeoff regression or excessive KL.
Evaluation runs inside an isolated Torch RNG context and cannot reset the training
random stream; a full-replay regression test verifies this. Full policy evaluations
run every two updates; finite-loss and KL checks run on every update.
The inherited tiny policy std (~.01) and frequent PPO KL early stopping are logged;
no architecture or exploration parameter is silently changed.

Open `notebooks/molab_handstand_v2.ipynb` with Python 3.11 and upload the portable
ZIP. The notebook verifies checksums, installs pinned dependencies, runs tests and
diagnostics, then smoke and gated pilot. It offers a downloadable complete ZIP.
It does not execute extended training. A fresh local setup can similarly extract
the ZIP and install `handstand_v2_requirements.txt` in a new Python 3.11 environment.

For a trusted exported checkpoint and a fresh GIF output:

```bash
.venv/bin/python -m experiments.handstand_v2.playback --checkpoint output/my_v2_pilot/policy_0010.pt --output output/my_v2_pilot/playback.gif
# Interactive MuJoCo viewer (with a display)
MUJOCO_GL=glfw .venv/bin/python -m experiments.handstand_v2.run playback --checkpoint output/my_v2_pilot/policy_0010.pt --output output/my_v2_viewer
```

The original remains selectable as playback default. Rank candidates by certified
pitch and retained takeoff in **both** modes, not raw airborne peak. Checkpoint
exports contain network weights, stage, hyperparameters and update index; source
config is retained in the neighboring manifest and original checkpoint archive.

Extended mode additionally requires `--authorize-extended`, a passing pilot gate,
and `--replication-gate` from a different training seed with >1° certified mean gain
in both evaluation modes and >=90% takeoff. This flag is only for a separately
user-authorized run. No such authorization is inferred from this experiment.

## Current result

The two-update smoke preserved 100% takeoff and finite learning, but failed to improve both modes of
certified pitch: deterministic 29.18° to 29.69°, stochastic 2.69° to 0°. It passed
only the bounded pilot gate, not an improvement gate. Final pilot results and the
go/no-go decision are recorded in `output/handstand_v2_final_report.md`.

The bounded three-seed action assay found front-calf flexion increased mean loaded
peak from 34.37° to 37.11°, but certified pitch only changed from 29.40° to 29.59°,
with no recovery. Combined front flexion/rear retraction gave 36.93° loaded and
29.39° certified. Rear retraction alone gave 34.30° loaded and 29.26° certified.
These results identify some body-rotation authority without demonstrating safe
COM capture or a feasible 40–45° supported crossing.
