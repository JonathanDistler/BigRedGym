"""Run with the notebook Python (3.11--3.13+); install a separate pinned runtime."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, PYTHON, atomic_json, child_env, run, stamp, verify_manifest


def packages_match(python, lock):
    """Use the isolated interpreter's metadata; never resolve/download on a hit."""
    code = (
        "from importlib.metadata import version; from pathlib import Path; import sys; "
        "lines=Path(sys.argv[1]).read_text().splitlines(); "
        "pairs=[line.split('==',1) for line in lines "
        "if '==' in line and not line.startswith(('#',' ','--'))]; "
        "assert pairs and all(version(name)==pinned for name,pinned in pairs)"
    )
    result = subprocess.run(
        [str(python), "-c", code, str(lock)],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--skip-tests", action="store_true", help="Imports/assets are still validated"
    )
    args = p.parse_args()
    verify_manifest()
    uv = shutil.which("uv")
    if uv is None and Path("/usr/local/bin/uv").is_file():
        uv = "/usr/local/bin/uv"
    if uv is None:
        raise RuntimeError(
            "uv was not found. Install uv with: "
            + sys.executable
            + " -m pip install uv ; then rerun MOLAB_START.py (or put uv on PATH)."
        )
    state = ROOT / ".molab"
    state.mkdir(exist_ok=True)
    # Installation and environment state belong to this project, not the notebook.
    import fcntl

    with (state / "setup.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another setup is running for this project.") from error
        run([uv, "python", "install", "--no-bin", "3.11.14"])
        if not PYTHON.exists():
            run(
                [
                    uv,
                    "venv",
                    "--no-project",
                    "--managed-python",
                    "--python",
                    "3.11.14",
                    ROOT / ".venv311",
                ]
            )
        run(
            [
                PYTHON,
                "-c",
                (
                    "import sys; assert sys.version_info[:2]==(3,11), sys.version; "
                    "print(sys.executable, sys.version)"
                ),
            ]
        )
        already_installed = packages_match(
            PYTHON, ROOT / "deployment/molab/requirements.lock"
        )
        if already_installed:
            print("Pinned environment already installed; skipping package downloads.")
        else:
            run(
                [
                    uv,
                    "pip",
                    "sync",
                    "--python",
                    PYTHON,
                    "--default-index",
                    "https://pypi.org/simple",
                    "--index",
                    "https://download.pytorch.org/whl/cpu",
                    "--index-strategy",
                    "unsafe-best-match",
                    ROOT / "deployment/molab/requirements.lock",
                ]
            )
        run([uv, "pip", "check", "--python", PYTHON])
        run([PYTHON, ROOT / "deployment/molab/validate.py"])
        if not args.skip_tests:
            run(
                [
                    PYTHON,
                    "-m",
                    "pytest",
                    "-c",
                    "deployment/molab/pytest.ini",
                    "deployment/molab/tests",
                    "-q",
                    "--junitxml",
                    state / ("tests_" + stamp() + ".xml"),
                ]
            )
        atomic_json(
            state / "setup_success.json",
            dict(
                python=str(PYTHON),
                uv=uv,
                timestamp=stamp(),
                tests=not args.skip_tests,
                packages_preexisting=already_installed,
            ),
        )
    print("\nSETUP SUCCESS: validated Python 3.11 subprocess environment at", PYTHON)
    print("No PPO training was started. Run MOLAB_TRAIN.py when ready.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nSETUP FAILED:", error, file=sys.stderr, flush=True)
        raise
