# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo==0.23.16"]
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Interactive CPU MuJoCo takeoff pilot. Every expensive step requires a click.

Upload this notebook to Molab, then upload the takeoff bundle and paste its
SHA256 sidecar value. No repository cloning, hardware access or automatic long
training. The project subprocess environment is separately pinned to Python 3.11.
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import tarfile
    import marimo as mo

    return Path, hashlib, io, json, mo, os, shutil, subprocess, sys, tarfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Go2 standing takeoff — bounded CPU pilot

    Run sections in order. The learned actor is a **bounded feedback residual
    around an explicit nominal demonstration prior**, trained for 0.96 seconds.
    It is not a controller for the complete handstand. A phase-prior-only
    ablation is included; takeoff retention alone does not prove PPO improvement.

    All simulation uses the original physics and strict gate. No physical robot
    interface is imported. Runs are synchronous and limited to 50 cumulative
    updates; save checkpoints before restarting or disconnecting compute.
    Molab throughput is measured on your runtime, never inferred from local data.
    """)
    return


@app.cell
def _(Path, mo):
    bundle_upload = mo.ui.file(
        filetypes=[".gz"], multiple=False, label="Upload takeoff bundle"
    )
    bundle_sha = mo.ui.text(label="Bundle SHA256 (paste .sha256 sidecar)")
    workspace = mo.ui.text(
        value=str(Path.home() / "go2_takeoff_work"), label="Work directory"
    )
    extract_button = mo.ui.run_button(label="1. Verify and extract")
    mo.vstack([bundle_upload, bundle_sha, workspace, extract_button])
    return bundle_sha, bundle_upload, extract_button, workspace


@app.cell
def _(
    Path,
    bundle_sha,
    bundle_upload,
    extract_button,
    hashlib,
    io,
    json,
    mo,
    tarfile,
    workspace,
):
    mo.stop(
        not extract_button.value,
        mo.md("Upload the bundle and its hash, then click verify."),
    )
    if not bundle_upload.value:
        raise ValueError("Upload a .tar.gz bundle first")
    _payload = bundle_upload.value[0].contents
    _expected = bundle_sha.value.strip().split()[0] if bundle_sha.value.strip() else ""
    if len(_expected) != 64 or hashlib.sha256(_payload).hexdigest() != _expected:
        raise ValueError("Bundle SHA256 mismatch; use the external sidecar")
    source_root = (
        Path(workspace.value).expanduser().resolve() / f"source-{_expected[:12]}"
    )
    source_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(_payload), mode="r:gz") as _archive:
        # Explicit validation also works when the notebook kernel is Python 3.13.
        _members = _archive.getmembers()
        if sum(_m.size for _m in _members) > 250_000_000:
            raise ValueError("Unexpectedly large source bundle")
        for _member in _members:
            _dest = (source_root / _member.name).resolve()
            if not _dest.is_relative_to(source_root) or not _member.isfile():
                raise ValueError("Unsafe archive member")
        _manifest_bytes = _archive.extractfile("bundle_manifest.json").read()
        _manifest_path = source_root / "bundle_manifest.json"
        if _manifest_path.exists():
            if _manifest_path.read_bytes() != _manifest_bytes:
                raise ValueError("Existing extraction has a different manifest")
        else:
            _archive.extractall(source_root, members=_members, filter="data")
    _manifest = json.loads((source_root / "bundle_manifest.json").read_text())
    for _name, _digest in _manifest["files"].items():
        _path = (source_root / _name).resolve()
        if (
            not _path.is_relative_to(source_root)
            or hashlib.sha256(_path.read_bytes()).hexdigest() != _digest
        ):
            raise ValueError(f"Manifest mismatch: {_name}")
    _reference = source_root / "artifacts/handstand_reference/trajectory.npz"
    if (
        hashlib.sha256(_reference.read_bytes()).hexdigest()
        != "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
    ):
        raise ValueError("Immutable reference mismatch")
    print(f"Verified {len(_manifest['files'])} files in {source_root}")
    return (source_root,)


@app.cell
def _(os, source_root, subprocess):
    # Notebook venv / UV_PYTHON / ROS PYTHONPATH must not select project packages.
    project_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
        }
        and not k.startswith("UV_")
    }
    project_env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
        UV_PYTHON="3.11",
        UV_PROJECT_ENVIRONMENT=str(source_root / ".venv"),
    )
    project_python = source_root / ".venv/bin/python"

    def project_run(arguments, timeout=900, interpreter=None):
        command = [str(interpreter or project_python), *map(str, arguments)]
        print("cwd:", source_root, "command:", command, flush=True)
        completed = subprocess.run(
            command,
            cwd=source_root,
            env=project_env,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(completed.stdout)
        completed.check_returncode()
        return completed.stdout

    return project_env, project_python, project_run


@app.cell
def _(mo):
    install_button = mo.ui.run_button(label="2. Install pinned Python 3.11 environment")
    regression_button = mo.ui.run_button(label="3. Exact nominal replay")
    diagnosis_button = mo.ui.run_button(label="4. Diagnose original student")
    clone_button = mo.ui.run_button(label="Optional: rebuild corrective BC warm start")
    mo.vstack([install_button, regression_button, diagnosis_button, clone_button])
    return clone_button, diagnosis_button, install_button, regression_button


@app.cell
def _(
    install_button, json, mo, project_env, project_run, shutil, source_root, subprocess
):
    mo.stop(not install_button.value)
    _uv = shutil.which("uv")
    if not _uv:
        raise RuntimeError(
            "Install uv in the runtime terminal: python -m pip install uv"
        )
    subprocess.run(
        [_uv, "python", "install", "3.11"],
        cwd=source_root,
        env=project_env,
        check=True,
        timeout=600,
    )
    subprocess.run(
        [_uv, "sync", "--frozen", "--no-dev", "--python", "3.11"],
        cwd=source_root,
        env=project_env,
        check=True,
        timeout=900,
    )
    _version = project_run(
        [
            "-c",
            "import sys, json, mujoco, torch, numpy; "
            "assert sys.version_info[:2]==(3,11); assert mujoco.__version__=='3.11.0'; "
            "print(json.dumps(dict(python=sys.version, mujoco=mujoco.__version__, "
            "torch=torch.__version__, numpy=numpy.__version__)))",
        ]
    )
    (source_root / ".environment_verified.json").write_text(_version)
    return


@app.cell
def _(json, mo, project_run, regression_button, source_root):
    mo.stop(not regression_button.value)
    _stdout = project_run(
        [
            "scripts/replay_go2_handstand.py",
            "--trajectory",
            "artifacts/handstand_reference/trajectory.npz",
            "--trajectory-sha256",
            "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92",
        ]
    )
    _result = json.loads(_stdout)
    assert _result["max_state_error"] == 0 and _result["strict_hold_seconds"] >= 5
    (source_root / ".reference_regression.json").write_text(_stdout)
    return


@app.cell
def _(diagnosis_button, mo, project_run):
    mo.stop(not diagnosis_button.value)
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "diagnose",
            "--checkpoint",
            "artifacts/baseline_student.pt",
            "--output",
            "output/molab_diagnosis",
        ]
    )
    return


@app.cell
def _(clone_button, mo, project_run):
    mo.stop(not clone_button.value)
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "bc",
            "--teacher",
            "artifacts/takeoff_teacher.npz",
            "--demonstrations",
            "16",
            "--epochs",
            "200",
            "--trials",
            "4",
            "--output",
            "output/molab_bc",
        ]
    )
    return


@app.cell
def _(mo):
    warm_checkpoint = mo.ui.text(
        value="artifacts/student.pt", label="Warm checkpoint (or output/molab_bc/bc.pt)"
    )
    smoke_button = mo.ui.run_button(
        label="5. Simulator smoke: 2 updates, restart/resume to 3, exact comparison"
    )
    benchmark_button = mo.ui.run_button(
        label="6. Benchmark serial envs 1/4/16 and rollout lengths 64/128"
    )
    pilot_button = mo.ui.run_button(label="7. Run bounded resumed pilot")
    cumulative_updates = mo.ui.number(
        start=4, stop=50, value=10, label="Cumulative update target (maximum 50)"
    )
    resume_path = mo.ui.text(
        value="output/molab_smoke/latest.pt", label="Resume checkpoint"
    )
    mo.vstack(
        [
            warm_checkpoint,
            smoke_button,
            benchmark_button,
            cumulative_updates,
            resume_path,
            pilot_button,
        ]
    )
    return (
        benchmark_button,
        cumulative_updates,
        pilot_button,
        resume_path,
        smoke_button,
        warm_checkpoint,
    )


@app.cell
def _(mo, project_run, smoke_button, source_root, warm_checkpoint):
    mo.stop(not smoke_button.value)
    if not (source_root / ".reference_regression.json").exists():
        raise RuntimeError("Run the nominal reference regression first")
    _common = ["--envs", "1", "--rollout-steps", "64", "--trials", "2"]
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "train",
            "--checkpoint",
            warm_checkpoint.value,
            "--updates",
            "2",
            *_common,
            "--output",
            "output/molab_smoke",
        ]
    )
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "train",
            "--checkpoint",
            "output/molab_smoke/latest.pt",
            "--resume",
            "--updates",
            "3",
            *_common,
            "--output",
            "output/molab_smoke",
        ]
    )
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "train",
            "--checkpoint",
            warm_checkpoint.value,
            "--updates",
            "3",
            *_common,
            "--output",
            "output/molab_continuous",
        ]
    )
    project_run(
        [
            "scripts/verify_handstand_checkpoint.py",
            "--left",
            "output/molab_smoke/latest.pt",
            "--right",
            "output/molab_continuous/latest.pt",
            "--output",
            "output/molab_smoke/resume_verification.json",
        ]
    )
    return


@app.cell
def _(benchmark_button, mo, project_run, source_root):
    mo.stop(not benchmark_button.value)
    if not (source_root / "output/molab_smoke/resume_verification.json").exists():
        raise RuntimeError("Complete the simulator resume smoke first")
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "benchmark",
            "--checkpoint",
            "output/molab_smoke/latest.pt",
            "--env-counts",
            "1",
            "4",
            "16",
            "--lengths",
            "64",
            "128",
            "--output",
            "output/molab_benchmark",
        ]
    )
    return


@app.cell
def _(
    cumulative_updates, json, mo, pilot_button, project_run, resume_path, source_root
):
    mo.stop(not pilot_button.value)
    if not (source_root / "output/molab_smoke/resume_verification.json").exists():
        raise RuntimeError("Resume verification must pass before pilot")
    _benchmark = source_root / "output/molab_benchmark/benchmark.json"
    if not _benchmark.exists():
        raise RuntimeError("Measure runtime throughput/memory first")
    print("Runtime benchmark:", json.loads(_benchmark.read_text())["result"])
    # Keep the exact-resume smoke configuration. A different count/length needs
    # its own fresh smoke test; benchmark counts are not parallel-worker counts.
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "train",
            "--checkpoint",
            resume_path.value,
            "--resume",
            "--updates",
            str(int(cumulative_updates.value)),
            "--envs",
            "1",
            "--rollout-steps",
            "64",
            "--trials",
            "2",
            "--output",
            "output/molab_pilot",
        ]
    )
    return


@app.cell
def _(mo):
    checkpoint_upload = mo.ui.file(
        filetypes=[".pt"],
        multiple=False,
        label="Upload your trusted complete checkpoint after restart",
    )
    checkpoint_sha = mo.ui.text(
        label="Expected checkpoint SHA256 from download manifest"
    )
    restore_button = mo.ui.run_button(label="Restore uploaded checkpoint")
    evaluate_path = mo.ui.text(
        value="output/molab_pilot/best.pt", label="Checkpoint to evaluate"
    )
    full_evaluation = mo.ui.checkbox(
        value=False,
        label="Also test untrained continuation through full 6.4 s (no balance replay fallback)",
    )
    evaluate_button = mo.ui.run_button(
        label="8. Evaluate 8 held-out standing starts + nominal"
    )
    download_button = mo.ui.run_button(
        label="9. Package latest/best checkpoints and measured results"
    )
    mo.vstack(
        [
            checkpoint_upload,
            checkpoint_sha,
            restore_button,
            evaluate_path,
            full_evaluation,
            evaluate_button,
            download_button,
        ]
    )
    return (
        checkpoint_sha,
        checkpoint_upload,
        download_button,
        evaluate_button,
        evaluate_path,
        full_evaluation,
        restore_button,
    )


@app.cell
def _(checkpoint_sha, checkpoint_upload, hashlib, mo, restore_button, source_root):
    mo.stop(not restore_button.value)
    if not checkpoint_upload.value:
        raise ValueError("Upload only your own trusted checkpoint; Torch uses pickle")
    _bytes = checkpoint_upload.value[0].contents
    if (
        len(_bytes) > 64_000_000
        or hashlib.sha256(_bytes).hexdigest() != checkpoint_sha.value.strip()
    ):
        raise ValueError("Checkpoint size/hash check failed")
    _restored = (
        source_root / "artifacts" / f"restored-{checkpoint_sha.value.strip()[:12]}.pt"
    )
    _restored.write_bytes(_bytes)
    print(
        "Paste this relative path into Resume checkpoint:",
        _restored.relative_to(source_root),
    )
    print(
        "Re-run install/regression/smoke/benchmark on a fresh runtime, then resume to a higher cumulative target."
    )
    return


@app.cell
def _(
    evaluate_button, evaluate_path, full_evaluation, json, mo, project_run, source_root
):
    mo.stop(not evaluate_button.value)
    _extra = ["--full-maneuver"] if full_evaluation.value else []
    project_run(
        [
            "scripts/handstand_takeoff.py",
            "evaluate",
            "--checkpoint",
            evaluate_path.value,
            "--trials",
            "8",
            *_extra,
            "--output",
            "output/molab_evaluation",
        ]
    )
    _rows = json.loads(
        (source_root / "output/molab_evaluation/evaluate.json").read_text()
    )["result"]["rows"]
    mo.ui.table(
        [
            {
                k: r[k]
                for k in [
                    "seed",
                    "takeoff_success",
                    "max_simultaneous_rear_clearance_m",
                    "max_pitch_deg",
                    "rear_contact_loss",
                    "front_capture",
                    "illegal_contacts",
                    "strict_hold_seconds",
                    "complete_success",
                    "torque_ratio_max",
                ]
            }
            for r in _rows
        ]
    )
    return


@app.cell
def _(download_button, hashlib, io, json, mo, source_root, tarfile):
    mo.stop(not download_button.value)
    _paths = []
    for _folder in [
        "output/molab_pilot",
        "output/molab_smoke",
        "output/molab_evaluation",
        "output/molab_benchmark",
    ]:
        _directory = source_root / _folder
        _paths.extend(
            p
            for p in _directory.glob("*")
            if p.suffix == ".json" or p.name in ["latest.pt", "best.pt"]
        )
    if not _paths:
        raise ValueError("No completed checkpoints/results available")
    _manifest = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in _paths
    }
    _target = source_root / "artifacts/takeoff_training_download.tar.gz"
    with tarfile.open(_target, "w:gz") as _archive:
        for _path in _paths:
            _archive.add(_path, arcname=str(_path.relative_to(source_root)))
        _contents = json.dumps(_manifest, indent=2).encode()
        _entry = tarfile.TarInfo("checkpoint_hashes.json")
        _entry.size = len(_contents)
        _archive.addfile(_entry, io.BytesIO(_contents))
    mo.vstack(
        [
            mo.download(
                data=lambda: _target.read_bytes(),
                filename=_target.name,
                label="Download complete checkpoints + results",
            ),
            mo.md(
                f"Archive SHA256: `{hashlib.sha256(_target.read_bytes()).hexdigest()}`"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
