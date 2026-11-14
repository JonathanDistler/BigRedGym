# Centroidal bridge experiment, 2026-11-13

Result: real nominal-physics, continuous-standing replay reaches **83.388° without collision or cumulative joint/speed legality violation**, with certified takeoff retained. This is **rotation progress only**. The unchanged actor collides 40 ms later at 88.40°. No five-second handstand, capture, student improvement, or hardware readiness is demonstrated.

## Provenance and comparison limits

Source branch `experiment/unified_go2_handstand`, starting commit `15fab3f`. Existing untracked artifacts were preserved. The requested Molab update-5 checkpoint was not found locally. Instead use the retained unified student `output/retained_go2_distillation_v3/student.pt`, SHA256 `0d08913c5f306f4090f62f74001a62a795feee533958e3561adcfee816c984ab`. Its seed-96000 baseline peaks at 36.12°, **not the reported update-5 48–50° plateau**. Therefore this experiment cannot claim a matched update-5 improvement or definitively explain that checkpoint's stall. Exact matching applies to the local actor comparison below. One initial state, no held-out performance claim.

| Quantity | Matched baseline | Planned prefix |
|---|---:|---:|
| collision-free pitch, physics samples |36.1166°|83.3880°|
| terminal pitch rate |0.5235 rad/s|2.9870 rad/s|
| safe seconds above 50 / 55 / 60°|0 / 0 / 0|0.236 / 0.190 / 0.150|
| safe seconds above 70 / 80°|0 / 0|0.076 / 0.018|
| positive front impulse, standing to 1.14 s|0|0.8973 N m s|
| negative front impulse, same interval|−15.1348|−9.9582 N m s|
| terminal capture distance (entry threshold 1)|3.1947|2.6625|
| certified takeoff at 1.14 s|yes|yes|
| independent suffix replay max state error|0|0|
| continuous standing replay max state error|0|0|
| five-second success after actor continuation|no|no|

The optimized prefix is legal only over its saved horizon. Front support is absent at its terminal state. COM-to-support quantities use the legacy zero fallback when support is absent and must **not** be interpreted as valid support geometry then.

## Independent mechanics

`experiments/centroidal_bridge/physics.py` selects every positive-mass body rooted at the floating robot root, verifies that root has world as parent, and computes `A_G` from body-COM translational and angular Jacobians. Link contribution is `I_world omega + (r_link-r_COM) cross m v_link`. It sums full inertia tensors with parallel-axis terms. All vectors are world-frame; positive pitch axis follows initial heading, approximately world +y. Trunk pitch rate is a separate kinematic quantity, never substituted for momentum. `mj_subtreeVel` supplies an independent check: maximum link-sum/subtree angular momentum difference is 6.22e−15 kg m²/s. Full link contributions, matrices, inertia tensors, contact-point world positions/velocities, and free contact torques are saved.

Contact wrench is obtained using `mj_contactForce`; contact-frame basis vectors are rows, so world conversion uses `frame.T`. The wrench acts positively on geom2 and negatively on geom1. External contacts are classified by robot membership, with all nonfoot contacts included. Internal self contacts cancel from total external wrench. Both contact torque and force lever arm are included. The focused plane-contact test confirms an upward robot normal force and the angular impulse sign.

At the available actor's pitch peak (0.822 s), front moment is independently **−9.2338 N m**, rear moment zero, COM angular momentum only **+0.08719 kg m²/s**, and pitch rate +0.01216 rad/s. COM is 0.1911 m behind the support in world x. This is external braking with nearly depleted useful centroidal momentum, not torque saturation or a fixed-pivot explanation. The support angular momentum is **−1.16758 kg m²/s**: its difference from COM momentum is the COM translational term, not contradictory sign conventions. The tensor-projected COM inertia is 0.55367 kg m². Front-contact braking and whole-body momentum exhaustion explain this local reproduction, with the missing update-5 qualification above.

The improvement begins at **0.50 s**, while rear feet remain loaded. First rear unloading is 0.584 s; takeoff/clearance condition appears at 0.656 s. Peak positive whole-body momentum is 2.12485 kg m²/s at 0.572 s. Front moment becomes positive at 0.696 s. Last front load greater than 1 N is at 0.774 s. This sequence changes real contact loading, reduces negative front impulse, adds positive front impulse, then flies. It does not keep a fixed support pin. At 0.9 s H is 1.26551; at 1.14 s it is 1.26516, while inertia decreases from 0.59465 to 0.46748 kg m² and pitch rate rises. Internal leg motion changes trunk motion without creating aerial total angular momentum. Mechanical energy at those times is 99.12 and 98.74 J. The candidate solves insufficient early rotation by changing ground interaction, but creates a new landing/capture problem.

### Accounting discrepancy: not fully closed

For the full standing replay through 1.14 s, measured pitch ΔH is 0.13185 baseline / 1.26187 candidate. Left-endpoint physics-step contact integration gives 0.14821 / 1.27112 kg m²/s. Residual is −0.01636 / −0.00925; maximum 3D accumulated residual norm is 0.20628 / 0.12461. The candidate terminal pitch residual is 0.73% of its momentum change. Baseline near-cancellation makes a relative error misleading. World-frame sign, every external contact group, pure contact torque, and native-vs-Jacobian momentum were checked. Right-endpoint and trapezoidal integration did not eliminate the baseline transient (maximum norms 0.921 / 0.502 in the longer first-run replay). Forward integration agrees best with MuJoCo's step ordering. Contact-switch discretization and solver/integration effects remain plausible, **not established** explanations. No claim of exact wrench-integral closure is made. A timestep-refinement audit and generalized momentum balance including passive/armature terms remain required before using tiny moment differences as evidence. Existing support momentum includes `H_COM + (COM-support) cross M v_COM`; a moving-support derivative additionally requires `−v_support cross M v_COM`, so its moment cannot be integrated as if the support were fixed.

## Search and artifacts

Serial CEM: five temporal knots at [0,.12,.26,.44,.64] s relative to 0.50 s onset, four symmetric front/rear thigh/calf residual channels, actor evaluated at every candidate's actual observation. Residuals clipped to ±0.8 normalized action, commands to ±1, real joint targets/filter, torque/speed gating, collisions, friction, and native forward dynamics retained. No artificial forces, pins, resets during trajectory, or injected pitch. The optimizer teacher uses simulator state only offline; actor interface and checkpoint structure are unchanged.

First run: 6 × 24 candidates, 0.84 s horizon; exposed an objective failure that rewarded rollover to ~150°. Second run: 5 × 24 candidates, 0.64 s horizon, pitch credit capped at 80°, penalties for overshoot, legality loss, terminal speed and capture distance. Search uses pitch progression and capture/speed surrogates; it does **not yet explicitly optimize energy or a signed early-impulse schedule**. Those quantities are audited afterward. It also has symmetric front timing, not independent left/right knots. These are limitations, not completed ablations. Initial six-candidate smoke plus main searches total 270 candidate evaluations, excluding replay/baseline calls. No PPO or concurrent simulation worker pools. Observed RAM about 1.6 GiB used, swap 248 KiB; output progress saved each generation. Resume is at generation boundaries, not midway through a generation.

`output/centroidal_bridge_capture_v2/` contains baseline/best GIFs, 10-panel mechanics plots, JSON results, complete initial/final environment snapshots, integration-state arrays, all actions, per-step audits, per-generation CEM state/RNG, and incumbent parameters including the separately useful legal **61.83°** candidate (`incumbent_2_8.pt`, terminal rate 1.883 rad/s). The latest candidate reaches 83.39° but has no front support. Separate `*_continuation.pt` and `continuation.json` retain the failed capture test. `output/centroidal_bridge_search_v1/` preserves the rejected rollover objective and results.

No imitation/PPO was launched: no verified recoverable bridge target exists. No three-way A/B/C or matched update-5 ablation is claimed. Next feasible intervention: replay the saved 61.83° incumbent with a landing-focused suffix; add a front-foot placement/contact-return objective and optimize braking **after** positive impulse, aiming to recontact before the body descends. Enforce a supported terminal state and reject high-speed airborne endpoints even if pitch improves. Then test existing balance recovery and nearby initial states before distillation.

## Reproduction and Molab

Use the existing Python 3.11 environment. Remove inherited incompatible PYTHONPATH; no packages added.

```bash
env -u PYTHONPATH .venv/bin/python scripts/search_centroidal_bridge.py --output output/centroidal_new_run --iterations 5 --population 24 --steps 32 --render
env -u PYTHONPATH MPLCONFIGDIR=/tmp/mpl-centroidal .venv/bin/python scripts/report_centroidal_bridge.py --run output/centroidal_new_run
env -u PYTHONPATH .venv/bin/python scripts/validate_centroidal_capture.py --run output/centroidal_new_run
env -u PYTHONPATH .venv/bin/python -m pytest tests/unit_tests/test_centroidal_bridge.py tests/unit_tests/test_dynamic_bridge.py -q
```

To replay 61.83° without optimization, use a fresh output with `--iterations 0 --steps 32 --parameters output/centroidal_bridge_capture_v2/incumbent_2_8.pt`. Resume by repeating the original command/output with a larger `--iterations`; keep checkpoint, onset, seed, steps and population unchanged. Do not reuse an output for another protocol.

The separate `output/centroidal_bridge_molab_v1.zip` extends the existing portable Molab source/preparation bundle and includes the result artifacts; ZIP CRC and per-member SHA256 are verified. On Molab, extract into a fresh directory and use the same Python 3.11 package setup from the existing bundle. Replace `--checkpoint` with the actual retained-structure update-5 checkpoint and use a fresh `--output`. Start `--iterations 1 --population 6 --steps 32`, replay and audit, then extend to a bounded search. Exact update-5 checkpoint compatibility and remote runtime execution are **not verified locally**. Original expert checkpoints and old Molab results are untouched.

## Methodological references

[MIT centroidal dynamics](https://underactuated.mit.edu/humanoids.html): exact articulated momentum relation, without assuming all contact forces are achievable. [Chignoli and Kim](https://dspace.mit.edu/entities/publication/c63ab250-24d4-4391-b7d2-99484c688877): momentum-based aerial planning motivates separating launch from airborne redistribution; its Mini Cheetah method is not copied. [Carius et al.](https://jan.carius.io/assets/optimization_with_hard_contacts/trajectory_optimization_with_implicit_hard_contacts.pdf): contact switches should emerge from actual constrained dynamics; native simulation is used instead of implementing their bilevel iLQR. [ADC paper](https://arxiv.org/html/2609.09492): actuator-dependent viability motivates preserving nominal Go2 dynamics, not importing Spot gains. [MuJoCo functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html) and [computation](https://mujoco.readthedocs.io/en/stable/computation/): subtree velocities, Jacobians, contact-frame forces and simulator constraints. ETH item endpoint returned 403; the author-hosted Carius manuscript was read instead.
