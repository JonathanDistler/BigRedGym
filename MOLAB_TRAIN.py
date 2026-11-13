def _bigredgym_train():
    # Second Molab cell. Setup must have completed; all ML runs in .venv311.
    from pathlib import Path
    import json
    import os
    import subprocess
    import tempfile

    PROJECT = Path(
        os.environ.get("BIGREDGYM_PROJECT", "/marimo/bigredgym_molab")
    ).resolve()
    PYTHON311 = PROJECT / ".venv311/bin/python"
    NUM_ENVS, WORKERS, UPDATES, HORIZON = 16, 4, 10, 32
    SEEDS, MAX_RUNTIME_SECONDS = [42, 43], 1200
    NEW_RUN, BENCHMARK_FIRST = False, False
    # HORIZON is the reference horizon at 16 worlds. Counts 8/16/32/64 use
    # horizons 64/32/16/8, preserving 512 training interactions per PPO update.
    config = json.loads((PROJECT / "deployment/molab/experiment.json").read_text())
    config.update(
        num_envs=NUM_ENVS,
        workers=WORKERS,
        updates=UPDATES,
        horizon=HORIZON,
        reference_horizon=HORIZON,
        seeds=SEEDS,
        max_seconds=MAX_RUNTIME_SECONDS,
    )
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    environment.update(
        WANDB_DISABLED="true",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if not PYTHON311.exists():
        raise RuntimeError("Run MOLAB_START.py first.")
    if BENCHMARK_FIRST:
        subprocess.run(
            [
                str(PYTHON311),
                str(PROJECT / "deployment/molab/benchmark.py"),
                "--workers",
                str(WORKERS),
            ],
            cwd=PROJECT,
            env=environment,
            check=True,
        )
    with tempfile.NamedTemporaryFile(
        "w", dir=PROJECT / ".molab", suffix=".json", delete=False
    ) as file:
        json.dump(config, file)
        configuration = file.name
    command = [
        str(PYTHON311),
        str(PROJECT / "deployment/molab/workflow.py"),
        "train",
        "--config",
        configuration,
        "--periodic-backups",
        "--export-directory",
        "/marimo",
    ]
    if NEW_RUN:
        command.append("--new-run")
    print(
        "Launching matched A/B/C, seeds", SEEDS, "; existing runs resume automatically."
    )
    subprocess.run(command, cwd=PROJECT, env=environment, check=True)


_bigredgym_train()
