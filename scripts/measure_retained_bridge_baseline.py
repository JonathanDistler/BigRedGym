"""Matched initial-student bridge reference for the actual-state curriculum."""

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled")
import torch
from scripts.unified_go2 import setup, actions
from experiments.unified_go2.core import Environment, load, save_json
from experiments.retained_go2.actor import Student


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(
        Path("output/retained_go2_preparation_v3"), a.output
    )
    cfg.seconds = 6.2
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    net = Student()
    net.load_state_dict(
        load("output/retained_go2_distillation_v3/student.pt")["network"]
    )
    rows = []
    for index, state in enumerate(bank["evaluation"]["bridge"][:2]):
        obs, _ = e.restore_start(state, "partial")
        while True:
            (obs, _), _, done, info = e.step(actions(net, obs[None])[0])
            if done:
                break
        rows.append(
            dict(
                index=index,
                distribution="bridge",
                success=bool(e.ever_hold_success),
                handstand_seconds=e.multi_best_hold,
                peak_collision_free_pitch=info["peak_collision_free_pitch"],
                termination=info["termination"],
                front_support_fraction=info["front_support_fraction"],
            )
        )
    save_json(a.output / "evaluation.json", rows)
    print(rows, flush=True)


if __name__ == "__main__":
    main()
