"""Replay one fresh legacy audit checkpoint on the fixed 123-state cohorts."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.evaluate_recovery_frontier import summary
from scripts.recovery_nominal_fixture import evaluate_nominal
from scripts.optimize_posture_transition import digest


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--iteration", type=int, choices=(0, 1, 2, 5, 10), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checkpoint = (
        Path(json.loads(a.audit.read_text())["log_dir"]) / f"model_{a.iteration}.pt"
    )
    p = Policy(str(checkpoint), "go2kickup_recovery", 16, True)
    try:
        result = dict(
            checkpoint=str(checkpoint),
            sha256=digest(checkpoint),
            retention=evaluate_nominal(p, a.output_dir),
        )
        for split in ("validation", "heldout"):
            path = Path(f"datasets/recovery_frontier_v4/{split}.npz")
            with np.load(path) as f:
                arrays = {k: f[k] for k in f.files}
            outcomes = []
            for start in range(0, len(arrays["seed"]), 16):
                samples = [
                    sample_from_arrays(arrays, i)
                    for i in range(start, min(start + 16, len(arrays["seed"])))
                ]
                rows, _ = replay_batch(p, samples)
                outcomes.extend(rows)
            result[split] = dict(
                all_states=summary(outcomes, range(len(outcomes))),
                strata={
                    str(k): summary(outcomes, np.flatnonzero(arrays["stratum"] == k))
                    for k in range(3)
                },
                outcomes=outcomes,
                dataset_sha256=digest(path),
            )
            dump(a.output_dir / "partial.json", result)
            print(
                a.iteration, split, result[split]["all_states"]["successes"], flush=True
            )
        dump(a.output_dir / "summary.json", result)
    finally:
        p.env._backend.close()


if __name__ == "__main__":
    main()
