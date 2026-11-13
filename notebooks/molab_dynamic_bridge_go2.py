"""Marimo Python3.13 kernel; isolatedPython3.11 workers, manual staged jobs."""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import os, sys, json, hashlib, zipfile, shutil, subprocess, fcntl
    from pathlib import Path
    import marimo as mo

    ROOT = Path("dynamic_bridge_bundle").absolute()
    if not ROOT.exists():
        ROOT.mkdir()
        with zipfile.ZipFile("dynamic_bridge_go2_molab.zip") as _zip:
            for _m in _zip.infolist():
                if not (ROOT / _m.filename).resolve().is_relative_to(ROOT):
                    raise ValueError("Unsafe archive")
            _zip.extractall(ROOT)
    for _name, _hash in json.loads(
        (ROOT / "retained_manifest.json").read_text()
    ).items():
        assert hashlib.sha256((ROOT / _name).read_bytes()).hexdigest() == _hash, _name
    PY = ROOT / ".venv311/bin/python"  # DO NOT resolve this virtualenv executable.
    _uv = shutil.which("uv") or str(ROOT / "bootstrap_uv/bin/uv")
    if not Path(_uv).exists() and not shutil.which(_uv):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(ROOT / "bootstrap_uv"),
                "uv",
            ]
        )
    with (ROOT / ".job.lock").open("a") as _install_lock:
        fcntl.flock(_install_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not PY.exists():
            subprocess.check_call(
                [_uv, "venv", "--python", "3.11", str(ROOT / ".venv311")]
            )
        subprocess.check_call(
            [
                _uv,
                "pip",
                "install",
                "--python",
                str(PY),
                "torch==2.13.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        subprocess.check_call(
            [
                _uv,
                "pip",
                "install",
                "--python",
                str(PY),
                "-r",
                str(ROOT / "deployment/retained_go2/requirements.txt"),
            ]
        )
    ENV = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        MUJOCO_GL="egl",
    )
    ENV.pop("PYTHONPATH", None)

    def run(*args):
        with (ROOT / ".job.lock").open("a") as _lock:
            fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Keep the lock in the child even if the notebook kernel disconnects.
            subprocess.check_call(
                [str(PY), *map(str, args)],
                cwd=ROOT,
                env=ENV,
                pass_fds=(_lock.fileno(),),
            )

    PREP = "output/retained_go2_preparation_v3"
    POLICY = "output/retained_go2_distillation_v3/student.pt"
    CERT = "output/molab_retention/retention.json"
    OUT = "output/molab_retained_ppo"
    H = 128
    return ROOT, PY, ENV, run, PREP, POLICY, CERT, OUT, H, mo, json, zipfile, fcntl


@app.cell
def _(mo):
    install_go = mo.ui.run_button(label="1 Installation + inference smoke")
    endpoint_go = mo.ui.run_button(label="2 Verify existing student: BOTH skills")
    bench_go = mo.ui.run_button(label="3 Benchmark one worker")
    pilot_go = mo.ui.run_button(label="4 Five-update bridge pilot")
    resume_go = mo.ui.run_button(label="5 Resume latest")
    extend_go = mo.ui.run_button(label="6 Continue to25 cumulative updates")
    render_go = mo.ui.run_button(label="7 Evaluate + render")
    stop_go = mo.ui.run_button(label="Pause after update boundary")
    backup_go = mo.ui.run_button(label="8 Download complete backup")
    mo.vstack(
        [
            mo.md(
                "Manual sequence. One job at a time. No automatic long training. Inspect all sample counts and regression reports."
            ),
            install_go,
            endpoint_go,
            bench_go,
            pilot_go,
            resume_go,
            extend_go,
            render_go,
            stop_go,
            backup_go,
        ]
    )
    return (
        install_go,
        endpoint_go,
        bench_go,
        pilot_go,
        resume_go,
        extend_go,
        render_go,
        stop_go,
        backup_go,
    )


@app.cell
def _(run, install_go, mo):
    mo.stop(not install_go.value)
    run(
        "scripts/smoke_retained_install.py",
        "--output",
        "output/molab_install_smoke",
        "--require-venv",
    )
    INSTALL_OK = True
    return (INSTALL_OK,)


@app.cell
def _(run, ROOT, PREP, POLICY, CERT, json, INSTALL_OK, endpoint_go, mo):
    mo.stop(not endpoint_go.value)
    assert INSTALL_OK
    # Re-evaluate ADC AND student on this runtime; do not reuse local CPU counts.
    run(
        "scripts/validate_retained_go2.py",
        "--preparation",
        PREP,
        "--initial",
        POLICY,
        "--output",
        "output/molab_retention",
        "--render",
    )
    _certificate = json.loads((ROOT / CERT).read_text())
    mo.stop(
        not _certificate["passed"],
        mo.md(
            "Endpoint retention FAILED. Do not run PPO. Inspect retention.json and videos."
        ),
    )
    ENDPOINT_OK = True
    return (ENDPOINT_OK,)


@app.cell
def _(mo):
    expanded_baseline_go = mo.ui.run_button(
        label="Optional expanded baseline:12 standing, up to12 balance,5 bridge"
    )
    expanded_baseline_go
    return (expanded_baseline_go,)


@app.cell
def _(run, PREP, POLICY, ENDPOINT_OK, expanded_baseline_go, mo):
    mo.stop(not expanded_baseline_go.value)
    assert ENDPOINT_OK
    run(
        "scripts/evaluate_dynamic_bridge.py",
        "--preparation",
        PREP,
        "--initial",
        POLICY,
        "--output",
        "output/molab_expanded_initial",
        "--render",
    )
    return


@app.cell
def _(run, ROOT, POLICY, json, ENDPOINT_OK, bench_go, mo):
    mo.stop(not bench_go.value)
    assert ENDPOINT_OK
    run(
        "scripts/benchmark_dynamic_bridge.py",
        "--initial",
        POLICY,
        "--output",
        "output/molab_benchmark",
    )
    _benchmark = json.loads(
        (ROOT / "output/molab_benchmark/benchmark.json").read_text()
    )
    N = _benchmark["recommended_n"]
    mo.md(
        f"Selected n={N}, ONE worker process, zero subprocess workers. Review measured RSS/throughput before StageB."
    )
    return (N,)


@app.cell
def _(run, ROOT, PREP, POLICY, CERT, OUT, H, N, ENDPOINT_OK, json, pilot_go, mo):
    mo.stop(not pilot_go.value)
    assert ENDPOINT_OK
    _config = dict(
        n=N,
        horizon=H,
        eval_every=5,
        seed=42,
        target_updates=5,
        interaction_budget=5 * N * H,
    )
    (ROOT / "output/molab_training_config.json").write_text(
        json.dumps(_config, indent=2)
    )
    run(
        "scripts/dynamic_bridge_go2.py",
        "train",
        "--preparation",
        PREP,
        "--certificate",
        CERT,
        "--initial",
        POLICY,
        "--output",
        OUT,
        "--n",
        N,
        "--horizon",
        H,
        "--updates",
        "5",
        "--eval-every",
        "5",
        "--interaction-budget",
        5 * N * H,
        "--max-seconds",
        "1800",
    )
    return


@app.cell
def _(run, ROOT, PREP, CERT, OUT, json, resume_go, mo):
    mo.stop(not resume_go.value)
    _config = json.loads((ROOT / "output/molab_training_config.json").read_text())
    _latest = json.loads((ROOT / OUT / "latest.json").read_text())["checkpoint"]
    _stop = ROOT / OUT / "STOP"
    if _stop.exists():
        mo.stop(
            "Endpoint regression" in _stop.read_text(),
            mo.md(
                "Regression stop is active. Keep known-good checkpoints; inspect retention reports before deciding on a separate fork."
            ),
        )
        _stop.unlink()  # Explicit resume revokes a user pause, not a regression stop.
    run(
        "scripts/dynamic_bridge_go2.py",
        "train",
        "--preparation",
        PREP,
        "--certificate",
        CERT,
        "--output",
        OUT,
        "--resume",
        OUT + "/" + _latest,
        "--n",
        _config["n"],
        "--horizon",
        _config["horizon"],
        "--updates",
        _config["target_updates"],
        "--eval-every",
        _config["eval_every"],
        "--interaction-budget",
        _config["interaction_budget"],
        "--max-seconds",
        "1800",
    )
    return


@app.cell
def _(run, ROOT, PREP, CERT, OUT, json, extend_go, mo):
    mo.stop(not extend_go.value)
    mo.stop(
        (ROOT / OUT / "STOP").exists(),
        mo.md("Resolve the stop reason before continuation."),
    )
    _retentions = sorted((ROOT / OUT).glob("retention_*.json"))
    mo.stop(not _retentions, mo.md("No completed retention evaluation yet."))
    mo.stop(
        not json.loads(_retentions[-1].read_text())["passed"],
        mo.md("Both endpoint skills must remain valid."),
    )
    run(
        "scripts/continue_dynamic_bridge.py",
        "--preparation",
        PREP,
        "--certificate",
        CERT,
        "--output",
        OUT,
        "--target",
        "25",
    )
    return


@app.cell
def _(run, ROOT, PREP, OUT, json, render_go, mo):
    mo.stop(not render_go.value)
    _latest = json.loads((ROOT / OUT / "latest.json").read_text())["checkpoint"]
    _iteration = int(_latest.split("_")[-1].split(".")[0])
    run(
        "scripts/evaluate_dynamic_bridge.py",
        "--preparation",
        PREP,
        "--initial",
        OUT + "/" + _latest,
        "--output",
        f"output/molab_expanded_update_{_iteration:06}",
        "--render",
    )
    return


@app.cell
def _(ROOT, OUT, stop_go, mo):
    mo.stop(not stop_go.value)
    (ROOT / OUT / "STOP").write_text(
        "User pause: finish and checkpoint the current update.\n"
    )
    return


@app.cell
def _(ROOT, fcntl, zipfile, backup_go, mo):
    mo.stop(not backup_go.value)
    import time

    with (ROOT / ".job.lock").open("a") as _lock:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _backup = ROOT.parent / f"retained_go2_backup_{time.time_ns()}.zip"
        with zipfile.ZipFile(_backup, "x", zipfile.ZIP_DEFLATED) as _zip:
            for _file in ROOT.rglob("*"):
                if _file.is_file() and not any(
                    x in _file.parts
                    for x in [".venv311", "bootstrap_uv", "__pycache__"]
                ):
                    _zip.write(_file, str(_file.relative_to(ROOT)))
    mo.stop(
        _backup.stat().st_size > 200 * 1024**2,
        mo.md(
            f"Complete backup ready at `{_backup}`. Use the notebook file browser to download it without loading a large ZIP into kernel RAM."
        ),
    )
    mo.download(
        _backup.read_bytes(),
        filename=_backup.name,
        label="Download complete source, data, checkpoints, RNG and best/latest backup",
    )
    return


if __name__ == "__main__":
    app.run()
