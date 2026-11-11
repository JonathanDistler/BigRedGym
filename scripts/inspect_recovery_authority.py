"""Inspect native contact/actuator constraints without declaring failures impossible."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gym.envs.go2.recovery_frontier import (
    authority,
    install_samples,
    sample_from_arrays,
)
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    policy = Policy(BALANCE, "go2balance", 16, True)
    result = {}
    try:
        for split in ("train", "validation", "heldout"):
            with np.load(a.dataset_dir / f"{split}.npz", allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            labels = json.loads((a.dataset_dir / f"{split}_labels.json").read_text())
            records = []
            for start in range(0, len(labels), 16):
                batch = [
                    sample_from_arrays(arrays, i)
                    for i in range(start, min(start + 16, len(labels)))
                ]
                install_samples(policy, batch)
                for i in range(len(batch)):
                    row = labels[start + i]
                    value = authority(policy.env, i)
                    value.update(
                        index=start + i,
                        source_id=row["source_id"],
                        kind=row["kind"],
                        frozen_success=row["frozen"]["success"],
                    )
                    contacts = value["front_contacts"]
                    if contacts:
                        d = policy.env._backend._datas[i]
                        rot = d.xmat[1].reshape(3, 3)
                        forward = np.cross(rot[:, 1], np.array([0.0, 0.0, 1.0]))
                        forward /= max(np.linalg.norm(forward), 1e-8)
                        positions = (
                            np.array([c["position"] for c in contacts]) @ forward
                        )
                        com = float(d.subtree_com[1] @ forward)
                        margin = min(com - positions.min(), positions.max() - com)
                        value["front_point_contact_static_margin_m"] = float(margin)
                        value["static_interpretation"] = (
                            "Negative static support margin does not rule out "
                            "dynamic recovery."
                        )
                    else:
                        value["front_point_contact_static_margin_m"] = None
                        value["static_interpretation"] = (
                            "No front contact now; future contact remains possible."
                        )
                    records.append(value)
            result[split] = dict(
                states=records,
                requested_torque_clipped_count=sum(
                    r["requested_torque_limit_ratio"] > 1 for r in records
                ),
                no_front_contact_count=sum(not r["front_contacts"] for r in records),
                initial_joint_limit_violation_count=sum(
                    r["maximum_joint_limit_violation_rad"] > 0.03 for r in records
                ),
                certified_physically_unrecoverable_count=0,
                interpretation=(
                    "No optimal-controller search or viability proof. "
                    "Constraint proxies do not establish impossibility."
                ),
            )
    finally:
        policy.env._backend.close()
    dump(a.output, result)


if __name__ == "__main__":
    main()
