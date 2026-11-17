# Software validation evidence

These results are software-only and do not authorize a physical attempt.
Baseline source: a38b193; Cornell interface: 8ca9bf98373a234de662f9fb8242a228b7efde21.

- `original100.json`: exact original CLI run, checkpoint and source fingerprints,
  all 101 rows (100 held-out plus nominal), conditions and metrics.
- `adapter100.json`: new inference/COM/target adapter, all matched trial metrics,
  per-channel errors and inference timing.
- `matched_trials.json`: paired outcomes and complete metrics including failed trials.
- `nominal_sensors.npz`: recorded simulated IMU stream (`samples`: time, wxyz,
  body specific force xyz), and 50 Hz synchronized snapshots (`controls`: time,
  wxyz, gyro xyz, q12, dq12, previous target12, step, original observation47).
  All values are synthetic calibrated body-origin telemetry, not hardware logs.
- `synthetic_drift.json`: controlled 8-second root observer drift experiment.
- `recorded_sensor_replay.json`: open-loop perturbation replay on the recorded
  maneuver. Targets/history stay fixed to separate sensor-input sensitivity from
  code parity; these numbers are not closed-loop success probabilities.
- `dry_run.json`: complete Session/hook replay timing and applied-target parity.
  Last ten physics samples occur after the final control snapshot and require
  no new command; all samples up to each snapshot are consumed once.
- `report.md`: measured summary, limits and readiness status.

Reproduce with commands in `docs/go2_handstand_deployment.md`. Reports are
committed for review; regenerated large scratch outputs remain ignored.

`pytest.xml` records the final focused suite: 55 tests passed, zero failures or
skips. TorchScript deprecation warnings are expected from the pinned Torch
version; this format is retained for Cornell's actual bundle-loader contract.
