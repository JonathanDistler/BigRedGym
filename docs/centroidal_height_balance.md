# Post-contact support correction — 2026-11-14

**Continuous standing-start simulation achieves 5.014 seconds under the unchanged strict handstand gate.** Cold model replay confirms certified takeoff, no illegal contact and zero final integration-state error. This is an offline planned transition followed by the unchanged neural balance actor, not a newly trained unified actor or hardware result.

The original capture commands are frozen through1.32s. One symmetric front hip/knee residual is applied with100ms rise and150ms fade, then removed. The winning normalized hip/knee amplitudes are approximately−0.0204364/+0.1025612 (joint target changes−0.04087/+0.20512rad), duration0.64517s, and balance delay0.05048s after1.32s (effective50Hz takeover1.38s). Residual ends before1.98s; subsequent control is the unchanged balance actor without blending or double squashing. Actual exact parameters and all commands are in the saved trajectory. All actuator/contact/model constraints are original.

Only **3 candidates** were evaluated, stopping immediately at success, within the40-candidate cap. Two focused tests pass, including comparison of logged gate conditions with the original simulator gate.

| Matched standing-start quantity | Original capture | Support correction |
|---|---:|---:|
| Maximum legal pitch |91.1368°|92.0555°|
| Actual front-foot capture |yes|yes|
| Minimum body height after1.32s, full rollout |0.3415m|0.5229m|
| COM vertical velocity at1.82s |−0.19643m/s|+0.01553m/s|
| Minimum post-contact head clearance |−0.000981m|0.18218m|
| Minimum post-contact calf clearance |0.02037m|0.03011m|
| Continuous original strict hold |0.058s|**5.014s**|
| Five-second success |no|**yes**|

The first brief strict interval loses support/slip conditions at1.342s in both trials. The baseline's longest strict interval ends1.448s and **first fails COM height at1.450s** (0.499921m, target0.560003±0.06m), before body-height failure1.526s. Front knee targets continue flexing, tracking knees near−1.52rad toward−2.43rad, without saturation. Initial knee torque/target tracking lag and contact settling coexist, but subsequent controller-commanded flexion drives sustained compression. The short extension pulse keeps the body inside the local balance controller's viable height/posture region. The original strict gate, including rear-foot airborne, support, slip, velocity, clearance and force checks, is unchanged.

Read-only success copy: `output/centroidal_height_verified_20261114/`; hashes: `reproducibility/centroidal_height_balance.json`. Full500Hz gate/torque/target/contact logs and integration states: `output/centroidal_height_40_v1/{baseline,best}.pt`. Original reference hashes were all reverified unchanged. Back up the success manifest's files together with the original reference manifest's files using the existing [backup procedure](centroidal_capture_preservation.md); no binaries are committed to ordinary Git.

Reproduce with existing Python3.11:

```bash
env -u PYTHONPATH .venv/bin/python scripts/replay_centroidal_capture.py --trajectory output/centroidal_height_verified_20261114/trajectory.pt --output output/height_REPLAY --render
env -u PYTHONPATH .venv/bin/python scripts/stabilize_centroidal_capture.py --output output/height_NEW
env -u PYTHONPATH .venv/bin/python -m pytest tests/unit_tests/test_centroidal_height.py -q
```

Next: test a small neighboring-start set and prepare supervised demonstrations for the existing unified actor. No PPO, new architecture, Molab rebuild or additional search was launched.
