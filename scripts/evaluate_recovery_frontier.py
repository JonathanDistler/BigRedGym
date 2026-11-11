"""Evaluate unchanged reset cohorts and nominal retention at iterations 0/25/50."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.recovery_nominal_fixture import evaluate_nominal
from scripts.evaluate_kickup_recovery import network_digest
from scripts.optimize_posture_transition import digest
from scripts.reproduce_transition_hold import BALANCE


def summary(outcomes, indices):
    rows = [outcomes[i] for i in indices]
    if not rows:
        return dict(count=0)
    result = dict(count=len(rows), successes=sum(r["success"] for r in rows))
    for key in (
        "hold_s",
        "total_stable_duration_s",
        "front_support_retention",
        "motor_saturation_fraction",
        "residual_rms_rad",
        "alive_front_support_retention",
        "mean_alive_com_speed",
        "mean_alive_abs_pitch_rate",
    ):
        result["mean_" + key] = float(np.mean([r[key] for r in rows]))
    for endpoint in ("initial", "final"):
        for key in ("com_speed", "pitch_rate", "com_support_x"):
            result[f"mean_abs_{endpoint}_{key}"] = float(
                np.mean([abs(r[endpoint][key]) for r in rows])
            )
    return result


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--wait-for-checkpoints", action="store_true")
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import os

    os.environ["GO2_RECOVERY_DATASET"] = str((a.dataset_dir / "train.npz").resolve())
    paths = [("original", Path(BALANCE), "go2balance")] + [
        (f"iteration_{i}", a.run_dir / f"model_{i}.pt", "go2kickup_recovery")
        for i in (0, 25, 50)
    ]
    protected = digest(BALANCE)
    results = {}
    for name, path, task in paths:
        deadline = time.monotonic() + 1800
        if a.wait_for_checkpoints and not path.is_file():
            print("Waiting for completed checkpoint", path, flush=True)
        while (
            a.wait_for_checkpoints
            and not path.is_file()
            and time.monotonic() < deadline
        ):
            time.sleep(5)
        policy = Policy(str(path), task, 16, True)
        try:
            saved = torch.load(path, weights_only=True, map_location="cpu")
            policy.runner.alg.critic.load_state_dict(
                saved["critic_state_dict"], strict=True
            )
            policy.runner.switch_to_eval()
            before = network_digest(policy)
            output = a.output_dir / name
            output.mkdir()
            # Original sixteen near-vertical probe starts, same RNG and geometry.
            retention = evaluate_nominal(policy, output)
            model_result = dict(
                checkpoint=str(path), sha256=digest(path), retention=retention
            )
            for split in ("validation", "heldout"):
                with np.load(a.dataset_dir / f"{split}.npz", allow_pickle=False) as d:
                    data = {k: d[k] for k in d.files}
                rows = json.loads((a.dataset_dir / f"{split}_labels.json").read_text())
                outcomes = []
                for start in range(0, len(rows), 16):
                    batch = [
                        sample_from_arrays(data, i)
                        for i in range(start, min(start + 16, len(rows)))
                    ]
                    classified, _ = replay_batch(policy, batch)
                    outcomes.extend(classified)
                    print(name, split, len(outcomes), "/", len(rows), flush=True)
                stage = {
                    str(i): summary(outcomes, np.flatnonzero(data[f"stage_{i}"]))
                    for i in range(6)
                }
                stratum = {
                    str(k): summary(outcomes, np.flatnonzero(data["stratum"] == k))
                    for k in range(3)
                }
                model_result[split] = dict(
                    all_states=summary(outcomes, range(len(rows))),
                    stages=stage,
                    strata=stratum,
                    outcomes=outcomes,
                    source_ids=data["source_id"].tolist(),
                    dataset_sha256=digest(a.dataset_dir / f"{split}.npz"),
                )
                dump(output / f"{split}.json", model_result[split])
                results[name] = model_result
                dump(a.output_dir / "summary.json", results)
            if network_digest(policy) != before:
                raise RuntimeError("Evaluation mutated weights or normalization")
        finally:
            policy.env._backend.close()
    if digest(BALANCE) != protected:
        raise RuntimeError("Original checkpoint changed")
    dump(a.output_dir / "summary.json", results)


if __name__ == "__main__":
    main()
