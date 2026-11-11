"""Add a whole new training family without replacing the first expanded bank."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.build_recovery_frontier import collect_pose_rollouts, save_bank, PROTECTED
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.optimize_posture_transition import digest
from scripts.reproduce_transition_hold import BALANCE
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = {p: digest(p) for p in PROTECTED}
    samples, labels, trajectories = collect_pose_rollouts("train", 16, 21000, 0.3)
    policy = Policy(BALANCE, "go2balance", 16, True)
    try:
        for start in range(0, len(samples), 16):
            outcomes, _ = replay_batch(policy, samples[start : start + 16])
            for i, outcome in enumerate(outcomes):
                labels[start + i]["frozen"] = outcome
            print(
                "extension labelled",
                min(start + 16, len(samples)),
                "/",
                len(samples),
                flush=True,
            )
    finally:
        policy.env._backend.close()
    source_trajectories = json.loads((a.source / "trajectories.json").read_text())
    for split in ("train", "validation", "heldout"):
        with np.load(a.source / f"{split}.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        old_samples = [
            sample_from_arrays(arrays, i) for i in range(len(arrays["seed"]))
        ]
        old_labels = json.loads((a.source / f"{split}_labels.json").read_text())
        if split == "train":
            valid = [
                i
                for i, row in enumerate(labels)
                if row["frozen"]["validity"] == "valid"
            ]
            old_samples += [samples[i] for i in valid]
            old_labels += [labels[i] for i in valid]
        save_bank(a.output_dir / f"{split}.npz", old_samples, old_labels)
        dump(a.output_dir / f"{split}_labels.json", old_labels)
        print(
            split,
            len(old_labels),
            "states",
            len(set(r["source_id"] for r in old_labels)),
            "sources",
            flush=True,
        )
    dump(a.output_dir / "trajectories.json", source_trajectories + trajectories)
    dump(
        a.output_dir / "provenance.json",
        dict(
            parent=str(a.source),
            parent_sha256={
                s: digest(a.source / f"{s}.npz")
                for s in ("train", "validation", "heldout")
            },
            protected_sha256=hashes,
            rollouts=len(source_trajectories) + len(trajectories),
            extension_family="train_heading_0.3; source seeds21000-21015",
            saved_npz_sha256={
                s: digest(a.output_dir / f"{s}.npz")
                for s in ("train", "validation", "heldout")
            },
        ),
    )
    if any(digest(p) != h for p, h in hashes.items()):
        raise RuntimeError("Protected input changed")


if __name__ == "__main__":
    main()
