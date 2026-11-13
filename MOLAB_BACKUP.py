def _bigredgym_backup():
    # Export cell: all new checkpoints, metrics, figures, source and reproduction data.
    from pathlib import Path
    import os
    import subprocess

    PROJECT = Path(
        os.environ.get("BIGREDGYM_PROJECT", "/marimo/bigredgym_molab")
    ).resolve()
    PYTHON311 = PROJECT / ".venv311/bin/python"
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    # Generate reports, then export an immutable full backup under /marimo.
    subprocess.run(
        [str(PYTHON311), str(PROJECT / "deployment/molab/workflow.py"), "report"],
        cwd=PROJECT,
        env=environment,
        check=False,
    )
    subprocess.run(
        [
            str(PYTHON311),
            str(PROJECT / "deployment/molab/workflow.py"),
            "backup",
            "--export-directory",
            "/marimo",
        ],
        cwd=PROJECT,
        env=environment,
        check=True,
    )
    backups = sorted(
        Path("/marimo").glob("bigredgym_backup_*.zip"), key=lambda p: p.stat().st_mtime
    )
    print("Download:", backups[-1])
    try:
        from IPython.display import FileLink, display

        display(FileLink(str(backups[-1])))
    except ImportError:
        pass  # The printed /marimo path is also available in Molab's file browser.


_bigredgym_backup()
