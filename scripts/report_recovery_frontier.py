"""Measured stage coverage and descriptive recovery-boundary pairs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gym.envs.go2.recovery_state import RecoveryBank

MIX = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.7, 0.3, 0.0),
    (0.6, 0.3, 0.1),
    (0.5, 0.35, 0.15),
    (0.4, 0.4, 0.2),
)


def describe(rows):
    if not rows:
        return dict(states=0, trajectories=0)
    result = dict(
        states=len(rows),
        trajectories=len(set(r["source_id"] for r in rows)),
        frozen_successes=sum(r["frozen"]["success"] for r in rows),
        frozen_success_rate=float(np.mean([r["frozen"]["success"] for r in rows])),
        kind_counts=dict(Counter(r["kind"] for r in rows)),
        family_counts=dict(Counter(r["family"] for r in rows)),
    )
    for key in ("com_support_x", "com_speed", "pitch_rad", "pitch_rate", "base_z"):
        a = np.array([r["frozen"]["initial"][key] for r in rows])
        result[key] = dict(
            mean=float(a.mean()),
            p10=float(np.quantile(a, 0.1)),
            median=float(np.median(a)),
            p90=float(np.quantile(a, 0.9)),
        )
    result["front_support"] = dict(
        Counter(
            "both"
            if min(r["frozen"]["initial"]["force_0"], r["frozen"]["initial"]["force_1"])
            >= r["weight_N"] * 0.06
            else "one_or_none"
            for r in rows
        )
    )
    return result


def weighted_rate(data, indices, stage):
    values = []
    for k, weight in enumerate(MIX[stage]):
        ids = indices[data["stratum"][indices] == k]
        if len(ids) and weight:
            sources = np.unique(data["source_id"][ids])
            rate = np.mean(
                [
                    data["frozen_success"][ids[data["source_id"][ids] == s]].mean()
                    for s in sources
                ]
            )
            values.append((weight, float(rate)))
    return sum(w * r for w, r in values) / sum(w for w, _ in values)


def sampling_distribution(data, rows, indices, stage):
    weights = np.zeros(len(indices))
    mixture = np.array(MIX[stage])
    present = np.array([(data["stratum"][indices] == k).any() for k in range(3)])
    mixture = mixture * present
    mixture /= mixture.sum()
    for k in range(3):
        positions = np.flatnonzero(data["stratum"][indices] == k)
        sources = np.unique(data["source_id"][indices[positions]])
        for source in sources:
            local = positions[data["source_id"][indices[positions]] == source]
            weights[local] = mixture[k] / len(sources) / len(local)
    result = {}
    for field in ("com_support_x", "com_speed", "pitch_rad", "pitch_rate", "base_z"):
        values = np.array([rows[i]["frozen"]["initial"][field] for i in indices])
        order = np.argsort(values)
        quantiles = np.interp([0.1, 0.5, 0.9], np.cumsum(weights[order]), values[order])
        result[field] = dict(
            mean=float(weights @ values),
            p10=float(quantiles[0]),
            median=float(quantiles[1]),
            p90=float(quantiles[2]),
        )
    support = {"both": 0.0, "one": 0.0, "none": 0.0}
    for weight, i in zip(weights, indices):
        r = rows[i]
        n = sum(
            r["frozen"]["initial"][f"force_{j}"] >= 0.06 * r["weight_N"]
            for j in range(2)
        )
        support[("none", "one", "both")[n]] += float(weight)
    result["front_support_probability"] = support
    return result


def report(root):
    root = Path(root)
    result = {}
    split_sources, split_families = {}, {}
    for split in ("train", "validation", "heldout"):
        bank = RecoveryBank(root / f"{split}.npz")
        data = bank.data
        rows = json.loads((root / f"{split}_labels.json").read_text())
        for i, row in enumerate(rows):
            row["weight_N"] = float(data["model_body_mass"][i].sum() * 9.81)
        split_sources[split] = {r["source_id"] for r in rows}
        split_families[split] = {r["family"] for r in rows}
        stages = {}
        for stage in range(6):
            ids = bank.indices(stage)
            stages[str(stage)] = dict(
                **describe([rows[i] for i in ids]),
                sampling_mixture=MIX[stage],
                sampling_distribution=sampling_distribution(data, rows, ids, stage),
                source_uniform_expected_frozen_success=weighted_rate(data, ids, stage),
                advancement=(
                    "Two 64-episode primary batches with >=80% five-second "
                    "recovery; terminal falls invalidate early holds."
                ),
            )
        features = np.array(
            [
                [
                    r["frozen"]["initial"]["com_support_x"] / 0.05,
                    r["frozen"]["initial"]["com_speed"] / 0.2,
                    (r["frozen"]["initial"]["pitch_rad"] - np.pi / 2) / 0.15,
                    r["frozen"]["initial"]["pitch_rate"] / 0.5,
                    r["frozen"]["initial"]["base_z"] / 0.05,
                ]
                for r in rows
            ]
        )
        good = np.flatnonzero(data["frozen_success"])
        bad = np.flatnonzero(~data["frozen_success"])
        pairs = []
        for i in bad:
            if not len(good):
                continue
            j = int(good[np.linalg.norm(features[good] - features[i], axis=1).argmin()])
            pairs.append(
                dict(
                    failure_index=int(i),
                    success_index=j,
                    scaled_distance=float(np.linalg.norm(features[i] - features[j])),
                    failure_source=rows[i]["source_id"],
                    success_source=rows[j]["source_id"],
                    failure_initial=rows[i]["frozen"]["initial"],
                    success_initial=rows[j]["frozen"]["initial"],
                    failure_authority=rows[i]["frozen"]["authority"],
                )
            )
        result[split] = dict(
            **describe(rows),
            stages=stages,
            by_stratum={
                str(k): describe(
                    [rows[i] for i in np.flatnonzero(data["stratum"] == k)]
                )
                for k in range(3)
            },
            boundary_pairs=sorted(pairs, key=lambda p: p["scaled_distance"])[:12],
            feasibility_counts=dict(Counter(r["frozen"]["feasibility"] for r in rows)),
        )
    for a, b in (
        ("train", "validation"),
        ("train", "heldout"),
        ("validation", "heldout"),
    ):
        if split_sources[a] & split_sources[b] or split_families[a] & split_families[b]:
            raise ValueError("Source rollout/family leakage")
    result["interpretation"] = (
        "Measured controller frontier, not a physical impossibility certificate. "
        "Stage success is conditional on selection; independent cohorts test transfer."
    )
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    with a.output.open("x") as f:
        json.dump(report(a.dataset_dir), f, indent=2)


if __name__ == "__main__":
    main()
