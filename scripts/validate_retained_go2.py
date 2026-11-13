"""Matched four-start skill retention; endpoint proof required before PPO."""

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
    MUJOCO_GL="egl",
)
import torch
from scripts.unified_go2 import setup
from experiments.unified_go2.core import adc_network, load, save_json, digest
from experiments.retained_go2.actor import Student
from experiments.retained_go2.evaluation import evaluate, retention


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--preparation", type=Path, default=Path("output/unified_go2_preparation_v1")
    )
    p.add_argument("--render", action="store_true")
    p.add_argument("--baseline", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, data, bank, assets = setup(a.preparation, a.output)
    reference = adc_network(load(a.preparation / "adc.pt"))
    student = Student()
    student.load_state_dict(load(a.initial)["network"])
    if a.baseline:
        import json

        rows = json.loads(a.baseline.read_text())["adc"]
    else:
        rows = evaluate(
            reference,
            cfg,
            bank,
            assets,
            data["centers"],
            "adc",
            a.output / "videos" if a.render else None,
        )
    corrected = evaluate(
        student,
        cfg,
        bank,
        assets,
        data["centers"],
        "student",
        a.output / "videos" if a.render else None,
    )
    result = retention(corrected, rows)
    result.update(
        baseline_file=str(a.baseline or (a.output / "evaluation.json")),
        initial=str(a.initial),
        initial_sha=digest(a.initial),
        preparation_sha=digest(a.preparation / "manifest.json"),
    )
    save_json(a.output / "evaluation.json", dict(adc=rows, student=corrected))
    save_json(a.output / "retention.json", result)
    print("RETENTION", result, flush=True)


if __name__ == "__main__":
    main()
