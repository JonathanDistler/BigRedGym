# Go2 actuator dynamics and front-support capture experiment

Branch: `experiment/adc_front_capture_handstand`. Parent: `1b4ad58`.
The initial tree had only untracked `datasets/` and `output/`; nothing was reset,
deleted or overwritten. New outputs use exclusive directories and checkpoints.

## Reference audit

The [paper and appendix](https://arxiv.org/html/2609.09492) describe annealing
stiffness with a completed-episode-length EMA and recomputing critical damping.
Their cart-pole argument assumes bounded command margins and does not establish
monotonic viability for saturated, contacting Go2 dynamics. Their Spot training
uses 4096 environments, 50 Hz policy control, a 20 s horizon, and a different
network/action scale. Those settings were not transferred to Go2.

The [released source](https://github.com/SaxionMechatronics/spot_rl_train), audited
at `6bd031d8f89aa99ad7a4b2050a792ddb40345f6d`, is not identical to the textual
algorithm: `mdp/curriculums.py` applies inverse stiffness/action-scale coupling,
but does not update damping. Its handstand config also adds slip, hopping, joint
velocity and acceleration penalties beyond the appendix's compact reward table.
The Go2 implementation keeps action scaling fixed and implements damping explicitly.

[Project demonstration](https://saxionmechatronics.github.io/adc/) and released
`media/handstand.gif` frames were inspected. The GIF shows foreleg reconfiguration
through rotation; image evidence does not determine foot velocity, loads or the
correct Go2 movement. The [YouTube URL](https://youtu.be/S5hcQCiXyU0) was accessible
only as a page shell in the browser tool; no claim of full video inspection.

Capture geometry, mixed intermediate resets, collision-clearance shaping,
per-joint BC freedom, recovery-based progress and Go2 probes are **our extensions**,
not methods attributed to the authors.

## Physical audit and mechanics

Both `go2_config.py` and the preserved initialization use K_nominal=60 N m/rad,
B_nominal=3 N m s/rad. These are repository control parameters, not newly
identified hardware measurements. The saved actor uses action_scale=2 rad,
50 Hz decisions, 500 Hz physics, no target low-pass filtering and a target slew
bound of 0.7 times the original URDF joint speed limit. Commands are clipped to
joint position limits. Applied PD torque is clipped to original URDF effort;
positive work is disabled at overspeed. This is a limiter, not a full measured
motor torque-speed curve. No extra motor or sensor delay is modeled or identified.

The new schedule never mutates shared models. Each lane owns gains and state.
Effective inertia uses 1/diag(M^-1) with MuJoCo's mass matrix including armature.
For gains above nominal, active damping is 2 sqrt(K I_eff) minus passive damping;
within the last 1% above nominal it blends to the original B=3. Thus the final
endpoint restores **both** original gains. This is a local free-coordinate
approximation, not critical damping of the coupled contact system. Damping is
recomputed at update boundaries, with actual values checkpointed per lane.

Positive task pitch rotates the initial forward trunk axis downward: initial +X
heading, +Y rotation axis. Front-foot body-frame forward (+trunk X), initial-heading
world forward, planted-leg extension, unloaded stepping and loaded sliding are
logged separately through exact transforms, joint motion, world foot velocities,
loads and unload/recontact events. A planted foot can move substantially in body
coordinates without moving in world coordinates. Neither sliding nor stepping is
hard-coded into the action interface. The actor controls all 12 position targets.

`mechanics.py` logs front hip/thigh/knee/calf/foot positions, joint angles/rates,
feet in world/body/COM frames, actual loaded-foot support region, COM velocity,
LIP validity, COM capture estimate, ballistic forecast, bounded candidate placements,
per-leg native Jacobian IK, and conservative single-leg vertical-load torque ratios.
The reachability test is at the current floating-base pose, includes illegal-contact
rejection, and does not certify future dynamics or a six-axis single-support wrench.
An empty support set is undefined, never interpreted as perfect support.

The LIP capture approximation is flagged valid only with bilateral loading,
small vertical velocity and small pitch rate/angular momentum. Outside that regime,
the forecast is diagnostic. No precomputed foot trajectory or backward displacement
target is used for C/D. Targets are bounded candidates, not deployed IK control.

## Learning implementation and limits

A is the old anticipatory B objective at nominal dynamics. B adds only ADC. C
replaces the old displacement/leaning reward with bounded contact/COM geometry,
collision-free rotation/clearance progress, target orientation/height, low motion
near upright, effort, illegal-impact penalty and a five-second success bonus.
D combines C with ADC. C/D additionally use current-state/contact phases and a
performance-gated intermediate-reset curriculum; this is a bundled front-support
intervention, not an isolated reward ablation. No future peak timing is observed.
Both old and new observations have 73 actor / 103 critic inputs, shared weights and
fixed feature scaling; there is no running normalization to reset. C/D change phase
semantics while preserving input dimensions. All conditions have equal interaction
budgets and the same nominal held-out standing/intermediate evaluation starts.

All start with the same legacy mixture (35% standing), so A/B remain faithful to
the previous mixed-start baseline. C/D may increase standing proportion as held-out
recovery improves. Original reset states are revalidated for geometry, limits,
applied torque, and a 100 ms native actor rollout. These are legitimate transient
states, not certified invariant balance states. Near-vertical samples still fail
quickly under the initial policy. No collider is removed/resized, no state is frozen,
and intermediate-start success is never counted as standing success.

The capture objective rewards improving actual support leverage and clearance,
including planted-leg adjustments; displacement itself earns no reward. However,
this is a learning hypothesis: no dynamic capture planner or demonstrated Go2
capture basin has yet been established. IK candidates/load checks remain diagnostic.
A successful five-second hold must satisfy the inherited physics-rate orientation,
COM energy/height, front loading, rear clearance/unloading, low linear/angular
motion, low slip, collision/limit-safe prefix and impact gates continuously.

`--phase-bc` is optional and applied consistently across all matched conditions.
It preserves rear-joint preparation/push-off labels strongly while reducing front
anchoring before liftoff and reducing all anchors in swing/balance. No uniform
smoothing is introduced. Existing balance labels are archived targets,
not validated five-second demonstrations, and are excluded from this experiment's
imitation dataset. Successful balancing supervision is not claimed.

`run_adc_front_capture.py` uses exact absolute update targets and an explicit
cumulative interaction cap. It saves actor, critic, Adam, RNGs, integration state,
per-lane gains, histories, curriculum and observations. Source/runtime/input hashes
must match on resume. Intermediate step counters start at zero, while physical
MuJoCo time and real event history are retained. The EMA uses elapsed post-reset
survival. A `STOP` file in the output directory requests a checkpoint-safe stop. Policy import from old combined checkpoints is an explicit
new experiment using one selected lane. `run_front_catch.py` also accepts longer
bounded targets without changing horizon, epochs or update definition.

## Validation and deployment

See `notebooks/molab_adc_front_capture.md` for Python 3.13-compatible notebook cells
that launch an isolated Python 3.11 child environment, benchmark real 16/32-env
throughput, run 30 updates, continue to 100–300, render and download full backups.
W&B is disabled. Stage 3 is not launched automatically.

The local results, checkpoint-restoration evidence, probe numbers, rendered
artifacts and remaining research gaps are recorded in the generated report under
`output/adc_front_capture_report_v1`. Local CPU evidence must not be represented
as a benchmark or a training run on Molab.

## Evaluation correction and interpretation

The first long diagnostic used one baseline-phase evaluation lane for every actor.
C/D train with a new event-phase encoding, so those aggregate evaluations were
invalidated. `repair_adc_evaluation.py` performs an explicit evaluator-only source
migration after confirming no reset-curriculum stage had advanced. It preserves
training tensors, Adam, RNGs, gains, worlds and interaction counts, archives the
old evaluations, and re-evaluates checkpoints 0/10/20/25 using each actor's own
phase semantics at the same original nominal dynamics. The new source contract
then governs exact subsequent resumes. This migration is disclosed rather than
presented as an unchanged-source resume. Earlier artifacts remain untouched.

The metric `first_useful_front_time` is a heuristic: joint motion coinciding with
an increase in a joint support/clearance/rotation score. It can fire during early
preparation and **does not establish causal usefulness**. Interpret it alongside
world/body foot trajectories, measured liftoff/recontact, pitch timing, clearance,
and actual recovery. Render CSVs additionally record the native active-contact
support interval without the 22 mm foot-radius margin used by the shaped distance.

The packaged local checkpoint was produced with PyTorch `2.13.0+cu130` running
on CPU. Molab setup installs the CPU build in its isolated environment. Fresh
training / weight import works across those builds; the exact-resume guard
intentionally refuses to call a different runtime an exact continuation. Molab
checkpoints can resume in the same Molab runtime. The clean-extraction validation
reuses installed binary dependencies with editable imports disabled; it does not
claim a fresh network dependency installation or a live Molab session.

The first corrected final evaluation exposed a CSV schema mismatch between the
old and new reward sets. Evaluation logs now have one schema/file per condition.
Checkpoint 25 was preserved and updates 26–30 replayed after an explicit logging-only
migration. The report distinguishes 61,440 committed training interactions from
10,240 discarded/replayed interactions; all four conditions incurred the same
repeat. This does not increase any final actor's committed learning experience.

## Remaining research work

This implementation does not establish a full articulated capture basin or a
six-axis contact-wrench feasibility certificate. IK and single-leg vertical-load
checks are approximations; actual acceptance is measured in native MuJoCo.
Staggered steps, coordinated reach and loaded sliding remain available to one
actor but have not been isolated in separate controlled training ablations.
No successful near-vertical recovery or validated five-second balance demonstration
has been produced. No new hardware actuator measurements were available. Molab
training, its CPU benchmark, and fresh CPU-wheel installation remain to be run;
the delivered local experiment must not be represented as remote execution.
