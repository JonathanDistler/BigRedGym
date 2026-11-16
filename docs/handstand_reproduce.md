# Reproduce the released Go2 handstand controller

Source branch: [`experiment/go2-handstand-robustness`](https://github.com/JonathanDistler/BigRedGym/tree/experiment/go2-handstand-robustness).
Artifacts: [handstand-capture-67of100-v1](https://github.com/JonathanDistler/BigRedGym/releases/tag/handstand-capture-67of100-v1).
The branch includes the deployment audit commit `c8006aad6a85b939856d1c61493824878afab9cf`
in [go2_deployment_gaps.md](go2_deployment_gaps.md).

The evaluated update-2 controller achieved **67/100 held-out simulation successes**
on seeds 301001–301100, plus a separate nominal success. Each success requires
five continuous seconds of the unchanged strict handstand gate. This is the
planned-prior capture feedback controller described in [handstand_capture.md](handstand_capture.md);
the earlier failed full-maneuver student in the robustness runbook is a different controller.
Hardware readiness remains unvalidated; consult the deployment audit.

## Release contents and identity

| Asset | Contents | SHA256 |
| --- | --- | --- |
| `handstand_capture_bundle.tar.gz` | Source, configs, tests, lockfile, notebook, reference trajectory, balance/takeoff teachers, demonstrations, BC and evaluated PPO checkpoints; per-file `bundle_manifest.json` | `2cf3a3afe6db87827b60b73c741617f64d054eb1c39a46b5858eb6ab678a77a4` |
| `handstand_capture_results.tar.gz` | Original 100-trial results, comparisons, diagnostics, training records, resume evidence, report and checkpoint copies | `69e4d2c391292c9b80a5e2232b71d8a9cffd9f19dcef0bd51f4624bcffff709e` |
| `capture_ppo_update2_67of100.pt` | Standalone evaluated checkpoint, identical to bundle `artifacts/capture_ppo_latest.pt` | `90117fc2ff80033663578e12ba664e4b1ff4c15c4c04f19619409a800458ba6c` |
| `go2_deployment_gaps.md` | Audit from `c8006aa` | `ed086dd68f6ee516e6a3762f49b8788b8ceb35740395b0fb83d3ecf8bdd960b1` |

`SHA256SUMS` accompanies these assets. The unchanged bundle records source commit
`a545e5b02a81a0877dc7cb99ecfff26962d714f5`; all its source files match this
branch at publication. The later audit and this download guide are in Git and
the audit is also a release asset. The reference trajectory SHA256 is
`21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`.
Torch checkpoints use pickle: load the trusted release after checking hashes.

## Download and run

Use a fresh directory to avoid overwriting local work. GitHub CLI authentication
is required if repository access is restricted. Alternatively download the named
assets from the release page. Python 3.11, `uv`, Git and GitHub CLI are required
for the commands below; CPU simulation requires no GPU or robot connection.

```bash
git clone --branch experiment/go2-handstand-robustness --single-branch \
  https://github.com/JonathanDistler/BigRedGym.git BigRedGym-handstand
cd BigRedGym-handstand
mkdir release-download
gh release download handstand-capture-67of100-v1 \
  --repo JonathanDistler/BigRedGym --dir release-download
(cd release-download && sha256sum --check SHA256SUMS)

# Extract in a separate directory; retain the Git checkout and audit above.
mkdir reproduce
tar -xzf release-download/handstand_capture_bundle.tar.gz -C reproduce
tar -xzf release-download/handstand_capture_results.tar.gz -C reproduce
cd reproduce
unset PYTHONPATH VIRTUAL_ENV UV_PROJECT_ENVIRONMENT UV_PYTHON
uv sync --frozen --no-dev
export HANDSTAND_PYTHON="$PWD/.venv/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/go2-mpl
export GO2_HANDSTAND_TRAJECTORY="$PWD/artifacts/handstand_reference/trajectory.npz"
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92

# Immutable nominal replay: exactly 5.014 s strict hold and zero state error.
"$HANDSTAND_PYTHON" scripts/replay_go2_handstand.py \
  --trajectory "$GO2_HANDSTAND_TRAJECTORY" \
  --trajectory-sha256 "$GO2_HANDSTAND_SHA256"

# Full held-out evaluation; includes nominal separately from the 100 trials.
"$HANDSTAND_PYTHON" scripts/handstand_capture.py full \
  --checkpoint artifacts/capture_ppo_latest.pt --controllers student \
  --evaluation-seed 301000 --trials 100 --output output/reproduced_100

# Inspect heldout_successes=67 and heldout_trials=100 under result.student.
"$HANDSTAND_PYTHON" -c 'import json; r=json.load(open("output/reproduced_100/full.json"))["result"]["student"]; print(r["heldout_successes"], r["heldout_trials"])'

# Focused tests; uv run supplies pytest without changing the project lockfile.
uv run --frozen --no-dev --with pytest python -m pytest \
  tests/unit_tests/test_handstand_replay.py \
  tests/unit_tests/test_handstand_robustness.py \
  tests/unit_tests/test_handstand_takeoff.py \
  tests/unit_tests/test_handstand_capture.py -q
```

Original evaluation evidence is `output/capture_final100/full.json`; its provenance
records the evaluated checkpoint/reference hashes. The bundle resolves teacher,
balance and reference paths internally, so the original author's neighboring
checkout is unnecessary. See [handstand_capture.md](handstand_capture.md) for BC,
training continuation, exact-resume verification, metrics and known limitations,
and [handstand_takeoff_molab.md](handstand_takeoff_molab.md) for notebook use.
Reproduction is supported for the pinned CPU simulator; Molab throughput and
physical deployment have not been validated.
