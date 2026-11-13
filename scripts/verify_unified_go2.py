"""Clean extracted-tree verification, excluding original editable repository hooks."""

import argparse, hashlib, json, os, subprocess, sys, sysconfig, tempfile, zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    root = Path(tempfile.mkdtemp(prefix="unified_clean_"))
    with zipfile.ZipFile(a.bundle) as z:
        for m in z.infolist():
            if not (root / m.filename).resolve().is_relative_to(root):
                raise ValueError("Unsafe archive path")
        z.extractall(root)
    manifest = json.loads((root / "unified_manifest.json").read_text())
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
        "import gym;from experiments.unified_go2.core import Student;assert gym.GYM_ROOT_DIR==os.getcwd();print('clean imports',gym.GYM_ROOT_DIR)",
        "sys.argv=['pytest','tests/unit_tests/test_unified_go2.py','-q'];runpy.run_module('pytest',run_name='__main__')",
        "sys.argv=['resume','train','--output','output/unified_go2_smoke_v2','--resume','output/unified_go2_smoke_v2/checkpoint_000002.pt','--n','2','--horizon','4','--updates','2','--eval-every','0','--interaction-budget','16'];runpy.run_path('scripts/unified_go2.py',run_name='__main__')",
        "import torch;from scripts.unified_go2 import setup,actions,render_episode;from experiments.unified_go2.core import *;out=Path('output/extracted_replay');out.mkdir();cfg,data,bank,assets=setup(Path('output/unified_go2_preparation_v1'),out);e=Environment(assets,cfg);o,_=e.restore_start(bank['evaluation']['balance'][0],'near_vertical');net=Student();net.load_state_dict(load('output/unified_go2_smoke_v2/checkpoint_000002.pt')['network']);states=[];exec(\"for i in range(10):\\n x=np.empty(__import__('mujoco').mj_stateSize(e.m,__import__('mujoco').mjtState.mjSTATE_INTEGRATION));__import__('mujoco').mj_getState(e.m,e.d,x,__import__('mujoco').mjtState.mjSTATE_INTEGRATION);states.append(x);(o,_),_,done,info=e.step(actions(net,o[None])[0]);\\n if done: break\");render_episode(e,np.array(states),out/'replay.gif','extracted student');print('actual extracted MuJoCo replay verified')",
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
