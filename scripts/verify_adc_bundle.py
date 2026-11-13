"""Extract into a clean tree and test with repo .pth/editable imports disabled."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(tempfile.mkdtemp(prefix="adc_clean_"))
    with zipfile.ZipFile(args.bundle) as archive:
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe path")
        archive.extractall(root)
    hashes = json.loads((root / "adc_bundle_manifest.json").read_text())
    for name, digest in hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(name)
    # Installed binary dependencies are reused. -I -S prevents user PYTHONPATH,
    # user site, and editable .pth hooks from importing the original repository.
    prefix = (
        f"import sys,os,runpy; sys.path.insert(0,{str(root)!r}); "
        f"sys.path.append({sysconfig.get_paths()['purelib']!r}); "
        f"os.chdir({str(root)!r}); "
    )
    env = dict(
        os.environ,
        MUJOCO_GL="egl",
        WANDB_MODE="disabled",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    env.pop("PYTHONPATH", None)
    commands = [
        "import gym; "
        "from experiments.adc_front_capture.core import CaptureEnvironment; "
        f"assert gym.GYM_ROOT_DIR == {str(root)!r}; "
        "print('isolated import',gym.GYM_ROOT_DIR)",
        "sys.argv=['pytest','tests/unit_tests/test_adc_front_capture.py',"
        "'tests/unit_tests/test_front_catch.py',"
        "'tests/unit_tests/test_front_catch_anticipatory.py','-q']; "
        "runpy.run_module('pytest',run_name='__main__')",
        f"sys.argv=['render',{args.checkpoint!r},'--modes','A_seed42',"
        "'--output','output/extracted_replay']; "
        "runpy.run_path('scripts/render_adc_front_capture.py',run_name='__main__')",
        "sys.argv=['resume','--output',"
        f"{str(Path(args.checkpoint).parent)!r},'--resume',{args.checkpoint!r},"
        "'--workers','0','--interaction-budget','800000']; "
        "runpy.run_path('scripts/run_adc_front_capture.py',run_name='__main__')",
    ]
    logs = []
    for i, command in enumerate(commands):
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", prefix + command],
            env=env,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (args.output / f"check_{i}.log").write_text(result.stdout)
        print(result.stdout, flush=True)
        logs.append(dict(check=i, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f"Extracted package check {i} failed; see log")
    (args.output / "verification.json").write_text(
        json.dumps(
            dict(
                extracted=str(root),
                checks=logs,
                files=len(hashes),
                bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                dependencies=(
                    "Existing installed binaries; editable repository paths disabled"
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
