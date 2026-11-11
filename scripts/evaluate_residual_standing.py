"""Paired standing-to-handstand regression for the frozen-balance residual."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from gym.envs.go2.posture_transition import PostureTransition
from scripts.evaluate_frozen_residual import attach, load_controller
from scripts.evaluate_kickup_handoff import Policy, Experiment, dump
from scripts.evaluate_kickup_recovery import (
    ATLAS,
    PARAMETERS,
    COHORTS,
    bank_digest,
    summarize_pair,
)
from scripts.optimize_posture_transition import bank, Evaluator, digest
from scripts.reproduce_transition_hold import BALANCE, KICK, REFERENCE
from gym.utils.helpers import set_seed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    a = parser.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protected = {
        str(p): digest(p) for p in (BALANCE, KICK, REFERENCE, PARAMETERS, ATLAS)
    }
    baseline = Policy(BALANCE, "go2balance", 16, True)
    residual = Policy(BALANCE, "go2balance", 16, True)
    kick = Policy(KICK, "go2frontyield", 16)
    attach(residual, load_controller(residual, a.checkpoint))
    atlas = json.loads(ATLAS.read_text())
    parameters = PostureTransition(**json.loads(PARAMETERS.read_text()))
    results = {}
    try:
        for cohort, seeds in COHORTS.items():
            data = bank(kick, Experiment(), seeds, "aggressive")
            fingerprint = bank_digest(data)
            reports = []
            for name, policy in (("original", baseline), ("residual", residual)):
                set_seed(1729)
                report = Evaluator(policy, atlas, True, True).evaluate(
                    parameters,
                    data,
                    list(range(16)),
                    a.output_dir / f"{cohort}_{name}.csv",
                )
                dump(a.output_dir / f"{cohort}_{name}.json", report)
                reports.append(report)
                assert bank_digest(data) == fingerprint
            results[cohort] = summarize_pair(*reports)
            dump(a.output_dir / "summary.json", results)
        assert all(digest(p) == h for p, h in protected.items())
        dump(
            a.output_dir / "provenance.json",
            dict(
                checkpoint=str(a.checkpoint),
                sha256=digest(a.checkpoint),
                protected_sha256=protected,
                cohorts=COHORTS,
            ),
        )
    finally:
        for p in (baseline, residual, kick):
            p.env._backend.close()


if __name__ == "__main__":
    main()
