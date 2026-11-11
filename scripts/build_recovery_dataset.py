"""Replay real standing transitions; save native recovery states separately."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_state import capture
from gym.envs.go2.posture_transition import PostureTransition
from scripts.optimize_posture_transition import bank, Evaluator, digest
from scripts.evaluate_kickup_handoff import Policy, Experiment, measurements, dump
from scripts.reproduce_transition_hold import BALANCE, KICK


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--balance-checkpoint", default=BALANCE)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    samples, labels = [], []
    with torch.inference_mode():
        balance = Policy(a.balance_checkpoint, "go2balance", 16, True)
        kick = Policy(KICK, "go2frontyield", 16)
        seeds = list(range(a.seed, a.seed + 16))
        data = bank(kick, Experiment(), seeds, "aggressive")
        parameters = PostureTransition(
            **json.loads(
                Path(
                    "gym/envs/go2/references/posture_transition_adaptive.json"
                ).read_text()
            )
        )
        atlas = json.loads(Path("output/handoff_cpg_comparison/atlas.json").read_text())

        def collect(e, time, handoff, failed):
            # Samples are taken only once policy blending has finished: the
            # temporary transition torque controller has no remaining authority.
            if time < handoff or time > handoff + 1.0:
                return
            if round(time / e.dt) % 5:
                return
            rows = measurements(e)
            for i, row in enumerate(rows):
                if failed[i] or row["body_contact"] or row["base_z"] < 0.3:
                    continue
                samples.append(capture(e, i))
                labels.append(dict(seed=seeds[i], time=time, **row))

        result = Evaluator(balance, atlas, True, True).evaluate(
            parameters,
            data,
            list(range(16)),
            a.output_dir / "trajectory.csv",
            state_callback=collect,
        )
        outcomes = {r["seed"]: r for r in result["trials"]}
        for row in labels:
            r = outcomes[row["seed"]]
            row["outcome"] = (
                "recovered"
                if r["success"]
                else "near"
                if r["hold_s"] >= 0.1
                else "failed"
            )
            row["hold_s"] = r["hold_s"]
        if not samples:
            raise RuntimeError("No feasible trajectory states collected")
        arrays = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
        # Contact variations and progressively broader approach states. These
        # flags are eligibility sets, not claims that every sample recovers.
        arrays.update(version=np.array(1), seed=np.array([r["seed"] for r in labels]))
        arrays["stage_3"] = np.array(
            [
                abs(r["pitch_rad"] - np.pi / 2) < 0.3 and r["com_speed"] < 0.5
                for r in labels
            ]
        )
        arrays["stage_4"] = np.array([r["outcome"] != "failed" for r in labels])
        arrays["stage_5"] = np.ones(len(labels), dtype=bool)
        np.savez_compressed(a.output_dir / "train.npz", **arrays)
        dump(a.output_dir / "labels.json", labels)
        dump(a.output_dir / "evaluation.json", result)
        dump(
            a.output_dir / "provenance.json",
            dict(
                balance=balance.meta,
                kick=kick.meta,
                seeds=seeds,
                parameters=parameters.to_dict(),
                samples=len(samples),
                dataset_sha256=digest(a.output_dir / "train.npz"),
            ),
        )
        balance.env._backend.close()
        kick.env._backend.close()
    print(
        f"Saved {len(samples)} states; {result['successes']}/16 five-second holds",
        flush=True,
    )


if __name__ == "__main__":
    main()
