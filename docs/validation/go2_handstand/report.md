# Measured handstand deployment validation

**Software candidate tested; NOT READY FOR HARDWARE.** No robot communication,
SDK commands or training were performed. Research baseline source is a38b193;
Cornell interface is 8ca9bf98373a234de662f9fb8242a228b7efde21.

| Evaluation | Original | Adapter |
|---|---:|---:|
| Held-out complete strict successes | 67/100 | 67/100 |
| Separate nominal success | PASS | PASS |
| Changed matched-trial success outcomes | — | 0 |

All 101 trials use the same seed/starting perturbation rules, full standing start,
8-second budget and original continuous five-second strict gate. Failed trials
remain in the denominator. Per-trial reports contain strict hold, rear clearance,
illegal collisions, orientation RMS, front-foot path and failure reasons. The
selected metrics above match exactly on all paired rows: 0 differences.

| Maximum absolute parity error | Value |
|---|---:|
| All 47 observations | 2.3283064365386963e-10 |
| Root observer state (height/velocity components) | 2.8332891588433995e-13 |
| Capture correction | 0 |
| Balance correction | 0 |
| Normalized action | 0 |
| Final bounded joint target | 0 |

Worst observation channel: 46 (zero based, COM estimate). The difference is
roundoff from timestamp differences/quaternion conversion versus research fixed
2 ms integration; it is far below the 1e-7 observation tolerance and does not
change any action. All nine requested boundary steps have dedicated tests.

The focused suite passed **55/55**, with zero failures/skips. Ruff and Git whitespace
checks passed. Eleven deprecation warnings concern the TorchScript format required
by Cornell's loader. The exact Cornell loader successfully loads the complete
scripted controller; its unchanged 100 Hz/EMA-on behavior is also a demonstrated
failing compatibility condition. Four hooks alone do not implement the sensor or
bounded actuator path.

Inference time (all matched trials, including cold startup): median
0.240348 ms, p99 0.631752 ms,
maximum 35.869320 ms. Cold startup exceeds 20 ms; the review loader
warms up before activation. Complete warmed Session/hook replay: 320
commands, median 0.955357 ms,
p99 2.032322 ms,
maximum 2.346758 ms, 0 cycles above 20 ms.
All 3190 samples preceding snapshots were consumed once;
10 trailing samples occur after the final
snapshot/cancellation boundary. These WSL2 host timings are not real-time guarantees.

Synthetic 8-second integration with 0.03 m/s² vertical accelerometer bias reaches
0.96024 m root-height drift. Recorded-maneuver replay with declared 10 ms delayed
IMU signals reaches approximately 0.634 m COM-height and 0.249 m/s COM-velocity
difference; combined perturbations reach approximately 0.756 m and 0.757 m/s.
Noisy streams change normalized actions by up to approximately .05. Those are
intentionally different observations, not code-parity failures. The complete
conditions and channelwise differences are in the adjacent JSON files; no noisy
closed-loop success probability or measured hardware noise model is claimed.

Hardware blockers remain: measured frame/bias/lever-arm calibration, lossless
500 Hz telemetry and synchronized clocks, 50 Hz/EMA-off runtime enforcement,
bounded target acknowledgement, verified effort/overspeed protections, deployment
computer timing, contact-load/harness assessment, and staff-approved activation,
cancellation and recovery. Cornell's supported PD command path does not reproduce
the research 500 Hz torque clipping/overspeed law. No safety limit was bypassed.

The smallest next action is staff review of sensor ingestion and supported motor
protections, followed by an approved software runner integration. A physical
attempt is not authorized by these results.

See [deployment instructions](../../go2_handstand_deployment.md) and
[exact changed files](changed_files.txt).
