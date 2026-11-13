"""Dependency/virtualenv/nominal-inference smoke, before Molab training."""

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
    WANDB_DISABLED="true",
    MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"),
)
import numpy as np, torch, mujoco
from scripts.unified_go2 import setup, actions, render_episode
from experiments.retained_go2.actor import Student
from experiments.unified_go2.core import (
    Environment,
    load,
    save_json,
    digest,
    adc_network,
)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-venv", action="store_true")
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Training subprocess must use Python3.11")
    if a.require_venv and sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Base interpreter selected; pass the UNRESOLVED virtualenv python path to uv pip"
        )
    prep = Path("output/retained_go2_preparation_v2")
    cfg, data, bank, assets = setup(prep, a.output)
    e = Environment(assets, cfg)
    net = Student()
    model = Path("output/retained_go2_distillation_v3/student.pt")
    net.load_state_dict(load(model)["network"])
    adc = adc_network(load(prep / "adc.pt"))
    obs, _ = e.reset(1.0, 96000)
    states = []
    with torch.no_grad():
        error = float(np.max(abs(actions(net, obs[None]) - actions(adc, obs[None]))))
    if error > 2e-6:
        raise RuntimeError("Standing initialization differs from ADC")
    for _ in range(10):
        state = np.empty(mujoco.mj_stateSize(e.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(e.m, e.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        states.append(state)
        (obs, _), _, done, info = e.step(actions(net, obs[None])[0])
        if done:
            break
    render_episode(
        e, np.array(states), a.output / "inference.gif", "retained install smoke"
    )
    result = dict(
        python=sys.version,
        executable=sys.executable,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
        torch=str(torch.__version__),
        numpy=np.__version__,
        mujoco=mujoco.__version__,
        first_action_ADC_error=error,
        frames=len(states),
        model_sha=digest(model),
        scope="Installation/inference only; no takeoff or handstand success claim",
    )
    save_json(a.output / "installation.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
