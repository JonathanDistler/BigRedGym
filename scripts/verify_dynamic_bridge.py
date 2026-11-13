"""Clean extracted-tree verification, excluding original editable repository hooks."""

import argparse, hashlib, json, os, subprocess, sys, sysconfig, tempfile, zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    root = Path(tempfile.mkdtemp(prefix="retained_clean_"))
    with zipfile.ZipFile(a.bundle) as z:
        for m in z.infolist():
            if not (root / m.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe archive path")
        z.extractall(root)
    manifest = json.loads((root / "retained_manifest.json").read_text())
    for f, h in manifest.items():
        if hashlib.sha256((root / f).read_bytes()).hexdigest() != h:
            raise ValueError(f)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_MODE="disabled",
        MUJOCO_GL="egl",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    env.pop("PYTHONPATH", None)
    prefix = f"import sys,os,runpy;sys.path.insert(0,{str(root)!r});sys.path.append({sysconfig.get_paths()['purelib']!r});os.chdir({str(root)!r});"
    checks = [
        "import gym;from experiments.retained_go2.actor import Student;assert gym.GYM_ROOT_DIR==os.getcwd();print('clean imports',gym.GYM_ROOT_DIR)",
        "sys.argv=['pytest','tests/unit_tests/test_retained_go2.py','tests/unit_tests/test_dynamic_bridge.py','-q'];runpy.run_module('pytest',run_name='__main__')",
        "sys.argv=['resume','train','--output','output/dynamic_bridge_pipeline_smoke_v2','--resume','output/dynamic_bridge_pipeline_smoke_v2/checkpoint_000002.pt','--preparation','output/retained_go2_preparation_v3','--certificate','output/retained_go2_validation_v4/retention.json','--n','1','--horizon','2','--updates','2','--eval-every','0','--interaction-budget','4'];runpy.run_path('scripts/dynamic_bridge_go2.py',run_name='__main__')",
        "sys.argv=['install','--output','output/extracted_inference'];runpy.run_path('scripts/smoke_retained_install.py',run_name='__main__')",
    ]
    results = []
    for i, code in enumerate(checks):
        r = subprocess.run(
            [sys.executable, "-I", "-S", "-c", prefix + code],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (a.output / f"check_{i}.log").write_text(r.stdout)
        print("check", i, r.returncode, flush=True)
        results.append(dict(check=i, returncode=r.returncode))
        if r.returncode:
            raise RuntimeError("Check failed: " + r.stdout[-3000:])
    (a.output / "verification.json").write_text(
        json.dumps(
            dict(
                extracted=str(root),
                files=len(manifest),
                checks=results,
                dependency_scope="Reused installed pinned binaries; no original repository or editable hooks; live Molab install not executed",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
