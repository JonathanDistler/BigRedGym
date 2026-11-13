"""Open as a marimo notebook under Python3.13; training uses isolated Python3.11."""

import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os, sys, json, zipfile, hashlib, shutil, subprocess
    from pathlib import Path
    import marimo as mo

    # Upload unified_go2_molab.zip beside this notebook before executing setup.
    BUNDLE = Path("unified_go2_molab.zip")
    ROOT = Path("unified_go2_bundle").absolute()
    if not ROOT.exists():
        ROOT.mkdir(exist_ok=False)
        with zipfile.ZipFile(BUNDLE) as _archive:
            for _m in _archive.infolist():
                if not (ROOT / _m.filename).resolve().is_relative_to(ROOT):
                    raise ValueError("Unsafe archive path")
            _archive.extractall(ROOT)
    for _name, _sha in json.loads((ROOT / "unified_manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / _name).read_bytes()).hexdigest() == _sha, _name
    TRAIN_PY = ROOT / ".venv311/bin/python"
    if not TRAIN_PY.exists():
        _uv = shutil.which("uv")
        if not _uv:
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
            _uv = str(ROOT / "bootstrap_uv/bin/uv")
        subprocess.check_call([_uv, "venv", "--python", "3.11", str(ROOT / ".venv311")])
        # Never resolve this executable symlink: uv must target the virtual env.
        subprocess.check_call(
            [
                _uv,
                "pip",
                "install",
                "--python",
                str(TRAIN_PY),
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
                str(TRAIN_PY),
                "-r",
                str(ROOT / "deployment/unified_go2/requirements.txt"),
            ]
        )
    ENV = dict(
        os.environ,
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MUJOCO_GL="egl",
    )
    ENV.pop("PYTHONPATH", None)

    def run(*args):
        import fcntl

        with (ROOT / ".heavy_job.lock").open("a") as _lock:
            try:
                fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    "Another training/evaluation job is active; wait before starting this cell"
                )
            subprocess.check_call([str(TRAIN_PY), *map(str, args)], cwd=ROOT, env=ENV)

    return ROOT, TRAIN_PY, ENV, run, mo, Path, json, zipfile


@app.cell
def _(mo):
    bench_go = mo.ui.run_button(label="1 Benchmark")
    distill_go = mo.ui.run_button(label="2 Optional distillation")
    bc_eval_go = mo.ui.run_button(label="Evaluate BC retention")
    smoke_go = mo.ui.run_button(label="3 Smoke and resume")
    train_go = mo.ui.run_button(label="4 Bounded20 updates")
    resume_go = mo.ui.run_button(label="5 Resume")
    extend_go = mo.ui.run_button(label="6 Guarded continuation")
    evaluate_go = mo.ui.run_button(label="7 Evaluate/render")
    backup_go = mo.ui.run_button(label="8 Backup")
    mo.vstack(
        [
            mo.md(
                "Run ONE job at a time. These buttons prevent automatic training on notebook open."
            ),
            bench_go,
            distill_go,
            bc_eval_go,
            smoke_go,
            train_go,
            resume_go,
            extend_go,
            evaluate_go,
            backup_go,
        ]
    )
    return (
        bench_go,
        distill_go,
        bc_eval_go,
        smoke_go,
        train_go,
        resume_go,
        extend_go,
        evaluate_go,
        backup_go,
    )


@app.cell
def _(run, bench_go, mo):

    mo.stop(not bench_go.value)
    # Conservative serial benchmark on THIS Molab host. No worker spawning.
    run("scripts/unified_go2.py", "benchmark", "--output", "output/molab_benchmark")
    return


@app.cell
def _(run, distill_go, mo):

    mo.stop(not distill_go.value)
    # Optional new distillation; existing verified local student is already bundled.
    # Use a new directory. No teacher labels are invented in the bridge gap.
    run(
        "scripts/unified_go2.py",
        "distill",
        "--output",
        "output/molab_distillation",
        "--steps",
        "10000",
    )
    return


@app.cell
def _(run, bc_eval_go, mo):
    mo.stop(not bc_eval_go.value)
    run(
        "scripts/unified_go2.py",
        "evaluate",
        "--output",
        "output/molab_BC_evaluation",
        "--initial",
        "output/molab_distillation/student.pt",
        "--controllers",
        "adc",
        "balance",
        "student",
        "--quick",
        "--render",
    )
    return


@app.cell
def _(run, smoke_go, mo):

    mo.stop(not smoke_go.value)
    # Tiny numerical smoke, followed by exact update-boundary continuation.
    run(
        "scripts/unified_go2.py",
        "train",
        "--output",
        "output/molab_smoke",
        "--initial",
        "output/unified_go2_initial_v1/student.pt",
        "--n",
        "2",
        "--horizon",
        "4",
        "--updates",
        "1",
        "--eval-every",
        "0",
        "--interaction-budget",
        "16",
        "--max-seconds",
        "120",
    )
    run(
        "scripts/unified_go2.py",
        "train",
        "--output",
        "output/molab_smoke",
        "--resume",
        "output/molab_smoke/checkpoint_000001.pt",
        "--n",
        "2",
        "--horizon",
        "4",
        "--updates",
        "2",
        "--eval-every",
        "0",
        "--interaction-budget",
        "16",
        "--max-seconds",
        "120",
    )
    return


@app.cell
def _(run, train_go, mo):

    mo.stop(not train_go.value)
    # Stage1: fixed conservative n4 serial;20*4*32=2560 PPO interactions.
    # Choose n2/4/8 only after benchmark/memory review, then keep n/horizon fixed.
    run(
        "scripts/unified_go2.py",
        "train",
        "--output",
        "output/molab_stage1",
        "--initial",
        "output/molab_distillation/student.pt",
        "--n",
        "4",
        "--horizon",
        "32",
        "--updates",
        "20",
        "--eval-every",
        "10",
        "--interaction-budget",
        "2560",
        "--max-seconds",
        "1800",
    )
    return


@app.cell
def _(ROOT, run, json, resume_go, mo):

    mo.stop(not resume_go.value)
    # Resume the latest saved boundary after notebook timeout/interruption.
    _latest = json.loads((ROOT / "output/molab_stage1/latest.json").read_text())[
        "checkpoint"
    ]
    run(
        "scripts/unified_go2.py",
        "train",
        "--output",
        "output/molab_stage1",
        "--resume",
        "output/molab_stage1/" + _latest,
        "--n",
        "4",
        "--horizon",
        "32",
        "--updates",
        "20",
        "--eval-every",
        "10",
        "--interaction-budget",
        "2560",
        "--max-seconds",
        "1800",
    )
    return


@app.cell
def _(ROOT, run, json, extend_go, mo):

    mo.stop(not extend_go.value)
    # Stage2: explicit bounded continuation only after preserved balance is measured.
    _best = json.loads((ROOT / "output/molab_stage1/best.json").read_text())
    if _best["score"][0] < 0.75:
        raise RuntimeError(
            "Balance not preserved: inspect evaluation and improve distillation/balance learning before bridge expansion"
        )
    _latest = json.loads((ROOT / "output/molab_stage1/latest.json").read_text())[
        "checkpoint"
    ]
    run(
        "scripts/unified_go2.py",
        "train",
        "--output",
        "output/molab_stage1",
        "--resume",
        "output/molab_stage1/" + _latest,
        "--n",
        "4",
        "--horizon",
        "32",
        "--updates",
        "100",
        "--eval-every",
        "10",
        "--interaction-budget",
        "12800",
        "--max-seconds",
        "1800",
    )
    # Curriculum promotion requires bridge recovery; standing fine-tuning is stage3,
    # not an iteration-triggered switch. Further target300 costs38400 interactions.
    return


@app.cell
def _(ROOT, run, json, evaluate_go, mo):

    mo.stop(not evaluate_go.value)
    _best = json.loads((ROOT / "output/molab_stage1/best.json").read_text())[
        "checkpoint"
    ]
    run(
        "scripts/unified_go2.py",
        "evaluate",
        "--output",
        "output/molab_best_evaluation",
        "--initial",
        "output/molab_stage1/" + _best,
        "--controllers",
        "adc",
        "balance",
        "student",
        "--render",
    )
    return


@app.cell
def _(ROOT, zipfile, mo, backup_go):

    mo.stop(not backup_go.value)
    # Stop training at an update boundary first: create output/molab_stage1/STOP.
    import fcntl

    _lock = (ROOT / ".heavy_job.lock").open("a")
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _backup = ROOT.parent / "unified_go2_complete_backup.zip"
    with zipfile.ZipFile(_backup, "x", zipfile.ZIP_DEFLATED) as _z:
        for _p in ROOT.rglob("*"):
            if _p.is_file() and not any(
                x in _p.parts for x in [".venv311", "bootstrap_uv", "__pycache__"]
            ):
                _z.write(_p, str(_p.relative_to(ROOT)))
    mo.download(
        _backup.read_bytes(),
        filename=_backup.name,
        label="Download complete source/data/checkpoint backup",
    )
    return


if __name__ == "__main__":
    app.run()
