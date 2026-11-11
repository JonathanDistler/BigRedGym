"""Hold out diagonal/lateral velocity families; keep training bytes unchanged."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import replay_batch, sample_from_arrays
from scripts.build_recovery_frontier import collect_pose_rollouts, save_bank, PROTECTED
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE
from scripts.optimize_posture_transition import digest


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    trajectories = json.loads((a.source / "trajectories.json").read_text())
    retained = [
        r for r in trajectories if r["split"] == "train" or "_kick_" in r["source_id"]
    ]
    for name in ("train.npz", "train_labels.json"):
        shutil.copy2(a.source / name, a.output_dir / name)
    for split, seed, heading, angle in (
        ("validation", 22000, 0.6, np.pi / 4),
        ("heldout", 23000, -1.1, np.pi / 2),
    ):
        samples, labels, extra = collect_pose_rollouts(split, 8, seed, heading, angle)
        retained.extend(extra)
        policy = Policy(BALANCE, "go2balance", 16, True)
        try:
            for start in range(0, len(samples), 16):
                outcomes, _ = replay_batch(policy, samples[start : start + 16])
                for i, outcome in enumerate(outcomes):
                    labels[start + i]["frozen"] = outcome
                print(
                    split,
                    "new family labelled",
                    min(start + 16, len(samples)),
                    "/",
                    len(samples),
                    flush=True,
                )
        finally:
            policy.env._backend.close()
        valid = [i for i, r in enumerate(labels) if r["frozen"]["validity"] == "valid"]
        new_samples = [samples[i] for i in valid]
        new_labels = [labels[i] for i in valid]
        with np.load(a.source / f"{split}.npz", allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        old_labels = json.loads((a.source / f"{split}_labels.json").read_text())
        indices = [i for i, r in enumerate(old_labels) if "_kick_" in r["source_id"]]
        new_samples.extend(sample_from_arrays(data, i) for i in indices)
        new_labels.extend(old_labels[i] for i in indices)
        save_bank(a.output_dir / f"{split}.npz", new_samples, new_labels)
        dump(a.output_dir / f"{split}_labels.json", new_labels)
    if digest(a.output_dir / "train.npz") != digest(a.source / "train.npz"):
        raise RuntimeError("Training bank changed")
    dump(a.output_dir / "trajectories.json", retained)
    dump(
        a.output_dir / "provenance.json",
        dict(
            parent=str(a.source),
            training_unchanged=True,
            protected_sha256={p: digest(p) for p in PROTECTED},
            split_rule=(
                "Axial training velocity, diagonal validation, lateral heldout. "
                "Whole source and timing families remain disjoint."
            ),
            saved_npz_sha256={
                s: digest(a.output_dir / f"{s}.npz")
                for s in ("train", "validation", "heldout")
            },
        ),
    )


if __name__ == "__main__":
    main()
