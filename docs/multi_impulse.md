# State-aware multi-impulse experiment

Parent `d1bcf45`, isolated branch `experiment/phase_multi_impulse_handstand`.
Previous results, checkpoints and source stay intact. The diagnosis was written
before changing a reward: `output/multi_impulse_diagnosis_v1/DIAGNOSIS.md`.
B's best available and final actor coincide at update 30; two byte-identical
preserved combined checkpoints have separate names and SHA256 provenance.

## Diagnosis

B already makes two certified physical push-offs in each held-out standing
trajectory. In seed 96000, peak pitch is 49.248 degrees at 6.792 seconds. Integrated
front/rear moments to peak are -78.298/+78.307 N m s: most rear impulse is opposed
by the front contacts. Energy at peak is 55.46 J against a target potential of
89.42 J. Whole-robot body angular momentum at peak is about 0.25 kg m²/s. Mean
saturation is only 0.16% in that episode, with about 0.162 rad tracking RMSE.
Every original B standing trial ends in a front-calf impact. These measurements
do not identify independent causal percentages for energy loss, geometry, gains
or intentional braking. Native angular momentum excludes implicit rotor momentum;
the measured momentum-balance residual is retained rather than hidden.

## One learned actor

The old 73/103 actor/critic MLPs and their Adam moments are copied by parameter
name. A zero-initialized 93-to-64-to-12 residual adds twenty current-state features;
the critic gets a corresponding residual. There is one deployable actor and one
12-position-target convention. B0 freezes the new residual and ignores new context.
The residual gates smoothly with measured tilt; the original base policy remains
free to adjust early front-leg preparation in corrective training. No additional
deployed controller chooses kicks or supplies trajectories.

The context comprises smooth PRELOAD, KICK, ROTATE, CATCH, REACCELERATE, RETRY and
HOLD weights, angular velocity/momentum, COM/support error and velocity, actual
foot loads, rear loading duration, and current validated hold duration. Candidate
phases must persist for 60 ms, then blend smoothly. Predictions use current pitch
and angular velocity, never future peaks or success. The old time input is kept
on its original 8-second normalization even when episodes are extended to 12s.

New conditions reduce exploration near vertical using measured tilt. B0 retains
the original distribution. This is part of the bundled intervention and is not an
isolated reward-only comparison. Target authority, original joint bounds, motor
speed limits and clipped PD torque are unchanged. There is no uniform action
smoothing, joint freezing, artificial impulse, or within-episode reset.

## Physical corrections and rewards

A rear push is counted only after bilateral loading, positive actuator work,
measured knee extension, synchronized rear release and real rear-foot clearance.
A new certificate is armed after a real bilateral rear recontact/loading dwell;
its front drift reference is local to that attempt. Historical standing takeoff
still uses the original certification. Free-air swinging cannot increment either
push-off counter. Ground-supported front moments, front recontacts and approximate
internal redistribution are recorded separately. Internal redistribution requires
small measured external impulse and a compatible total-momentum change; it is
never labelled new external angular momentum.

Corrective learning pays bounded new collision-free rotation progress, target
pose/support geometry, useful braking near upright, sustained low-motion balance,
and legal clearance/effort. There is no reward per kick, per recontact, or per foot
displacement. New record progress can occur after a retry; repeatedly returning
to the same angle receives no new progress credit. Five-second success requires
continuous native 500 Hz orientation, base/COM height, support, low-slip/low-motion,
rear-airborne and collision/limit checks. Success from an intermediate reset is
never labelled standing success.

## Separate balancing and curriculum

`balance.py` constructs a native equilibrium and a finite-difference LQR solely
for offline validation/supervision. Its discounted Riccati design does not alter
simulated dynamics. Some geometrically attractive poses were rejected for tiny
self-contact or incompatible actuator authority. Accepted near-vertical states
include nonzero angular velocity and must survive a full five-second hold under
the native validator. This establishes teacher feasibility, not learned success.
The admitted set is narrow: two training states and one held-out state from the
same perturbation family. Lower partial states require native short-rollout
validation and are not declared five-second recoverable.

A separate imitation fit and a failed DAgger attempt are preserved. Low imitation
loss did not transfer the validator's five-second performance; the selected
learned fit held about 0.32 seconds. B1/B3 begin on the admitted balancing states
with the original base actor frozen, then unlock mixed standing/partial learning
only after recovery gates pass. Available tiers are near-vertical and one 75–90
partial sample per split; 60/45 tiers remain empty and block expansion. No invalid
atlas samples or fabricated successful trajectories are admitted.

## Controlled study and checkpoints

B0 continues the old B objective; B1 adds balancing supervision/reset training;
B2 adds corrective state/context/objective; B3 enables both. All main runs have
matched nominal evaluation seeds, architecture, environment counts, horizon,
PPO update definition and main transition budgets. The parent is B update 30.
The local target is 100 further main-study updates, 16 environments and horizon
32: 51,200 PPO transitions per condition. Balance initialization uses 620 physical
teacher frames and 1,000 supervised steps. B0/B2 receive an equal-count 620-step
control-policy PPO warmup. Thus data exposure is 51,820 per actor, 207,280 total;
supervised optimization counts and policy-versus-teacher data sources differ by
intervention and are explicitly disclosed. Teacher data collection is shared;
executed physics transitions and per-actor data exposure are distinct quantities.

Each condition has independent Adam tensors. The first pilot was excluded after
an aliasing bug was found and regression-tested. PPO updates that exceed measured
KL are rolled back, including Adam and RNG, and retried at a smaller learning
rate without new physics data. Source/runtime/data guards protect exact resume.
A complete training boundary is saved before evaluation, so an evaluator failure
cannot discard completed policy updates. Immutable checkpoints and nondominated
outcome-vector milestone aliases protect earlier improvements.

ADC remains available, defaulting to the original nominal endpoint. Elevated
initial gains require a fresh B-specific probe. Every held-out evaluation uses
K=60, B=3 and the original geometry. The reported local study does not claim a
new hardware actuator identification, live Molab execution, or a three-seed result.

See the generated report for actual outcomes and remaining failures. Native
videos contain full integration state, floating-base/joint state, synchronized
CSV and 500 Hz physical traces. A shared fixed world camera is chosen from
offline extents; those future extents never enter policy observations.
