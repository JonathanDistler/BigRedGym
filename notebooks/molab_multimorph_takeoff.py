# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Run cells manually; upload multimorph_takeoff_molab_bundle.zip beside this file."""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import sys
    import json
    import hashlib
    import zipfile
    from pathlib import Path
    from datetime import datetime, timezone
    import marimo as mo

    work = Path.cwd() / (
        "takeoff_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    work.mkdir()
    repo = work / "BigRedGym"
    env = dict(
        os.environ,
        WANDB_MODE="disabled",
        MPLCONFIGDIR="/tmp/takeoff-mpl",
        XDG_CACHE_HOME="/tmp/takeoff-cache",
    )
    env.pop("PYTHONPATH", None)

    def command(args, cwd=None):
        subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)

    return Path, command, env, hashlib, json, mo, repo, subprocess, sys, work, zipfile


@app.cell
def _(Path, command, hashlib, json, repo, zipfile):
    # Upload the locally supplied bundle first. It contains the uncommitted work.
    bundle = Path("multimorph_takeoff_molab_bundle.zip").resolve()
    if not bundle.exists():
        raise FileNotFoundError(
            "Upload multimorph_takeoff_molab_bundle.zip beside the notebook"
        )
    command(
        [
            "git",
            "clone",
            "--branch",
            "ppo_kickup_e2e",
            "--single-branch",
            "https://github.com/JonathanDistler/BigRedGym.git",
            repo,
        ]
    )
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("molab_manifest.json"))
        command(["git", "checkout", "--detach", manifest["base_commit"]], cwd=repo)
        for name, digest in manifest["files"].items():
            target = (repo / name).resolve()
            if not target.is_relative_to(repo.resolve()):
                raise ValueError("Invalid bundle path")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("Hash mismatch: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(
        "Verified",
        len(manifest["files"]),
        "files; original branch ppo_kickup_e2e, pinned",
        manifest["base_commit"],
    )
    return


@app.cell
def _(command, repo, sys, work):
    command([sys.executable, "-m", "venv", work / "venv"])
    python = work / "venv/bin/python"
    command(
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.13.0",
            "tensordict==0.13.0",
            "wandb==0.28.1",
            "mujoco==3.11.0",
            "numpy==2.4.6",
            "pandas==3.0.5",
            "matplotlib==3.11.1",
        ]
    )
    command(
        [
            python,
            "-c",
            "from scripts.run_takeoff_ppo import check_validation\n"
            "from pathlib import Path\n"
            "import torch\n"
            "check_validation(Path('output/takeoff_learning_certified/validation.json'))\n"
            "s=torch.load('output/multimorph_bc_20261111_v2/bc_actor.pt', "
            "weights_only=False,map_location='cpu')\n"
            "assert s['actor_dim']==44\n"
            "assert Path('resources/robots/go2/urdf/go2.urdf').exists()\n"
            "print('BC, reward hash and URDF verified')",
        ],
        cwd=repo,
    )
    return (python,)


@app.cell
def _(env, python, repo, subprocess, work):
    # Ten matched updates, two seeds, common gated BC actor and regularizer.
    import time as _time

    diagnostic = repo / "output" / ("retention_" + str(_time.time_ns()))
    diagnostic_log = work / (diagnostic.name + ".log")
    diagnostic_handle = diagnostic_log.open("x")
    diagnostic_process = subprocess.Popen(
        [
            str(python),
            "scripts/run_takeoff_retention.py",
            "--output",
            str(diagnostic),
            "--bc-actor",
            "output/multimorph_bc_20261111_v2/bc_actor.pt",
            "--validation",
            "output/takeoff_learning_certified/validation.json",
            "--protocol",
            "experiments/morphology/learning/protocol_takeoff_bc_anchor.json",
            "--bc-gate",
            "output/multimorph_bc_20261111_v2/bc_gate.json",
            "--updates",
            "10",
        ],
        cwd=repo,
        env=env,
        stdout=diagnostic_handle,
        stderr=subprocess.STDOUT,
    )
    print("PID", diagnostic_process.pid, "log", diagnostic_log)
    return diagnostic, diagnostic_log, diagnostic_process


@app.cell
def _(diagnostic_log, diagnostic_process):
    print("Process status:", diagnostic_process.poll())
    print(diagnostic_log.read_text()[-12000:])
    return


@app.cell
def _(command, diagnostic, diagnostic_process, python, repo):
    if diagnostic_process.poll() != 0:
        raise RuntimeError("Diagnostic must finish successfully before plotting")
    command(
        [python, "scripts/report_takeoff_retention.py", "--run", diagnostic], cwd=repo
    )
    command(
        [python, "scripts/assess_takeoff_retention.py", "--run", diagnostic], cwd=repo
    )
    return


@app.cell
def _(command, diagnostic, json, python, repo, work):
    import csv

    # Select the substantive run here only after it has been launched.
    ANALYZE_EXTENDED = False  # This notebook launches only the short diagnostic.
    selected_run = (
        (repo / "output" / ("matched_bc_" + work.name))
        if ANALYZE_EXTENDED
        else diagnostic
    )
    for _folder in sorted(f for f in selected_run.glob("*_seed*") if f.is_dir()):
        _training = _folder / "training.csv"
        if _training.exists():
            with _training.open() as _f:
                _rows = list(csv.DictReader(_f))
            print(_folder.name, _rows[-1] if _rows else "no updates")
        _events = _folder / "curriculum_events.json"
        if _events.exists():
            print("Promotions:", json.loads(_events.read_text()))
    if (selected_run / "completion.json").exists():
        command(
            [python, "scripts/report_takeoff_retention.py", "--run", selected_run],
            cwd=repo,
        )
    return (selected_run,)


@app.cell
def _(command, python, repo, selected_run, work):
    # Replay each latest checkpoint, full eight-second horizon, fresh held-out seeds.
    for folder in sorted(f for f in selected_run.glob("*_seed*") if f.is_dir()):
        latest = sorted(folder.glob("checkpoint_*.pt"))[-1]
        command(
            [
                python,
                "scripts/evaluate_takeoff_checkpoint.py",
                "--checkpoint",
                latest,
                "--output",
                work / ("heldout_" + selected_run.name + "_" + folder.name),
            ],
            cwd=repo,
        )
    return


@app.cell
def _(diagnostic_process, mo, repo, work, zipfile):
    if diagnostic_process.poll() is None:
        raise RuntimeError("Wait for the diagnostic to finish before final backup")
    # Includes ALL checkpoints, logs, protocol, code, BC source and results.
    import time as _time

    backup = work.parent / (work.name + "_complete_" + str(_time.time_ns()) + ".zip")
    with zipfile.ZipFile(backup, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for file in work.rglob("*"):
            if (
                file.is_file()
                and "venv" not in file.relative_to(work).parts
                and ".git" not in file.relative_to(work).parts
            ):
                z.write(file, file.relative_to(work))
    mo.download(
        data=lambda: backup.read_bytes(),
        filename=backup.name,
        label="Download complete backup",
    )
    return


if __name__ == "__main__":
    app.run()
