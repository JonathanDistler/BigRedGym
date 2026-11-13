"""Explicit evaluator-only migration, preserving every training boundary tensor.

Old artifacts stay untouched. Only run_adc_front_capture.py may have changed.
Re-evaluate existing weights using their own phase encoding at nominal gains.
"""

import argparse
import copy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.core import CatchConfig
from experiments.front_catch.vector import Vector
from experiments.front_catch.learning import evaluate
from experiments.front_catch.io import atomic_checkpoint
from experiments.front_catch.data import sha
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from scripts.run_adc_front_capture import sources
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", nargs="+", type=int, default=[0, 10, 20, 25])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--reuse-correct-evaluations",
        action="store_true",
        help="Only for the per-condition CSV schema correction",
    )
    p.add_argument("--discarded-interactions", type=int, default=0)
    a = p.parse_args()
    if a.discarded_interactions < 0:
        p.error("Negative interaction count")
    torch.set_num_threads(1)
    latest = a.directory / f"checkpoint_{max(a.iterations):06}.pt"
    saved = torch.load(latest, weights_only=False, map_location="cpu")
    old = saved["protocol"]["source_sha256"]
    current = sources()
    changed = {
        key for key in set(old) | set(current) if old.get(key) != current.get(key)
    }
    if changed != {"scripts/run_adc_front_capture.py"}:
        raise ValueError("Not an evaluator-only change: " + str(changed))
    if any(run["contact"].stage for run in saved["runs"].values()):
        raise ValueError(
            "Old evaluations changed reset curriculum; new experiment required"
        )
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = copy.deepcopy(saved["protocol"])
    protocol["source_sha256"] = current
    protocol["evaluator_migration"] = dict(
        parent=str(latest),
        parent_sha256=sha(latest),
        changed_files=sorted(changed),
        training_state_unchanged=True,
        reason=(
            "Each policy must receive its own phase encoding "
            "under identical nominal dynamics"
        ),
    )
    if a.reuse_correct_evaluations:
        protocol["evaluator_migration"]["reason"] = (
            "Separate per-condition evaluation CSV schemas; "
            "training and evaluations unchanged"
        )
        protocol["evaluator_migration"]["discarded_replayed_interactions"] = (
            a.discarded_interactions
        )
    bank = torch.load(
        Path(protocol["preparation"]) / "reset_bank.pt",
        weights_only=False,
        map_location="cpu",
    )
    configs = {}
    for mode, run in saved["runs"].items():
        cfg = CatchConfig(**run["config"])
        cfg.kp = protocol["nominal_k"]
        cfg.kd = protocol["nominal_b"]
        configs[mode] = cfg
    corrected = saved["history"]["evaluation"] if a.reuse_correct_evaluations else []
    steps = 0
    with Vector(
        Assets(a.output / "assets"),
        configs,
        bank,
        protocol["num_envs"],
        workers=a.workers,
    ) as vector:
        for iteration in [] if a.reuse_correct_evaluations else a.iterations:
            snapshot = torch.load(
                a.directory / f"checkpoint_{iteration:06}.pt",
                weights_only=False,
                map_location="cpu",
            )
            for mode, run in snapshot["runs"].items():
                net = ActorCritic(73, 103)
                net.load_state_dict(run["network"])
                rows, traces, count = evaluate(net, vector, mode, "B", bank)
                steps += count
                corrected.extend(
                    dict(**r, mode=mode, iteration=iteration, actual_steps=run["steps"])
                    for r in rows
                )
                print("correct evaluation", iteration, mode, flush=True)
    saved["history"]["legacy_phase_mismatch_evaluation"] = saved["history"][
        "evaluation"
    ]
    saved["history"]["evaluation"] = corrected
    saved["history"]["collapse_counts"] = {}
    saved["evaluation_steps"] += steps
    saved["protocol"] = protocol
    # Preserve states without reconstructing, resetting, or remapping weights/Adam/RNG.
    atomic_checkpoint(a.output / latest.name, saved)
    restored = torch.load(
        a.output / latest.name, weights_only=False, map_location="cpu"
    )
    for mode, run in saved["runs"].items():
        for key, value in run["network"].items():
            assert torch.equal(value, restored["runs"][mode]["network"][key])
    dump(a.output / "protocol.json", protocol)
    dump(
        a.output / "migration.json",
        dict(
            **protocol["evaluator_migration"],
            reevaluation_steps=steps,
            original_evaluation_steps=saved["evaluation_steps"] - steps,
            note=(
                "Training interaction count unchanged; "
                "old aggregate evaluations invalidated"
            ),
        ),
    )
    for path in current:
        dest = a.output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)


if __name__ == "__main__":
    main()
