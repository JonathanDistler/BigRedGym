"""Paired full-system evaluation of every saved recovery checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.optimize_posture_transition import bank, Evaluator, digest
from scripts.evaluate_kickup_handoff import Policy, Experiment, dump
from scripts.reproduce_transition_hold import BALANCE, KICK, REFERENCE
from gym.envs.go2.posture_transition import PostureTransition
from gym.utils.helpers import class_to_dict, set_seed

PARAMETERS = Path("gym/envs/go2/references/posture_transition_adaptive.json")
ATLAS = Path("output/handoff_cpg_comparison/atlas.json")
COHORTS = {"validation": list(range(6000, 6016)), "heldout": list(range(7000, 7016))}


def checkpoint_iteration(path):
    match = re.fullmatch(r"model_(\d+)\.pt", Path(path).name)
    if match is None:
        raise ValueError(f"Expected model_<iteration>.pt, got {path}")
    return int(match.group(1))


def discover_checkpoints(run_dir):
    paths = sorted(Path(run_dir).glob("model_*.pt"), key=checkpoint_iteration)
    if not paths:
        raise ValueError(f"No completed checkpoints in {run_dir}")
    return paths


def bank_digest(data):
    """Fingerprint actual shared prefix inputs, not just the requested seed IDs."""
    h = hashlib.sha256()
    h.update(json.dumps(data["seeds"]).encode())
    for t, sample in sorted(data["samples"].items()):
        h.update(str(t).encode())
        for q, v in sample["states"]:
            h.update(q.tobytes())
            h.update(v.tobytes())
        for key in ("targets", "velocity"):
            h.update(sample[key].numpy().tobytes())
        h.update(sample["collision"].tobytes())
        for warm in sample["warmstart"]:
            h.update(warm.tobytes())
    for m in data["models"]:
        for key in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "geom_friction",
            "body_invweight0",
            "dof_M0",
            "dof_invweight0",
        ):
            h.update(getattr(m, key).tobytes())
    for key in ("kp", "kd"):
        h.update(data[key].numpy().tobytes())
    return h.hexdigest()


def network_digest(policy):
    h = hashlib.sha256()
    for role in ("actor", "critic"):
        for key, value in getattr(policy.runner.alg, role).state_dict().items():
            h.update((role + key).encode())
            h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_evaluation_policy(checkpoint, task):
    policy = Policy(str(checkpoint), task, 16, True)
    try:
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        # Policy's historical evaluator needs only actor inference. Load the
        # critic here too so its saved normalization is verified and frozen.
        policy.runner.alg.critic.load_state_dict(
            saved["critic_state_dict"], strict=True
        )
        policy.runner.alg.critic.requires_grad_(False)
        policy.runner.switch_to_eval()
        return policy
    except Exception:
        policy.env._backend.close()
        raise


def check_interface(original, candidate):
    for role, dim in (("actor", 59), ("critic", 63)):
        left = getattr(original.runner, f"{role}_cfg")
        right = getattr(candidate.runner, f"{role}_cfg")
        for key in ("obs", "normalize_obs", "frequency"):
            if left.get(key) != right.get(key):
                raise ValueError(f"Incompatible {role}.{key}")
        if candidate.runner.get_obs(right["obs"]).shape != (16, dim):
            raise ValueError(f"Incorrect {role} dimension")
        for name in left["obs"]:
            a = torch.as_tensor(original.env.scales.get(name, 1.0))
            b = torch.as_tensor(candidate.env.scales.get(name, 1.0))
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if candidate.runner.actor_cfg["actions"] != original.runner.actor_cfg["actions"]:
        raise ValueError("Incompatible actuator interface")
    if candidate.runner.get_inference_actions().shape != (16, 12):
        raise ValueError("Incorrect action dimension")
    if class_to_dict(original.env.cfg.contact_cpg) != class_to_dict(
        candidate.env.cfg.contact_cpg
    ):
        raise ValueError("CPG configuration mismatch")
    torch.testing.assert_close(
        original.env._standing_posture(), candidate.env._standing_posture()
    )


def aggregate_report(report):
    def mean_field(field):
        values = [
            r[field] for r in report["trials"] if field in r and r[field] is not None
        ]
        return float(np.mean(values)) if values else None

    return dict(
        success_rate=report["successes"] / len(report["trials"]),
        mean_longest_hold_s=mean_field("hold_s"),
        mean_total_stable_duration_s=mean_field("total_stable_duration_s"),
        mean_front_support_retention=mean_field("support_fraction"),
        nonfoot_collision_rate=mean_field("collision"),
        mean_motor_saturation_fraction=mean_field("torque_saturation_fraction"),
        mean_maximum_target_step_rad=mean_field("max_target_step_rad"),
    )


def summarize_pair(baseline, candidate):
    left = {r["seed"]: r for r in baseline["trials"]}
    right = {r["seed"]: r for r in candidate["trials"]}
    if set(left) != set(right):
        raise ValueError("Paired evaluation seed mismatch")
    pairs = [
        dict(
            seed=s,
            original_success=left[s]["success"],
            candidate_success=right[s]["success"],
            original_hold_s=left[s]["hold_s"],
            candidate_hold_s=right[s]["hold_s"],
            hold_difference_s=right[s]["hold_s"] - left[s]["hold_s"],
        )
        for s in sorted(left)
    ]
    return dict(
        count=len(pairs),
        original_successes=baseline["successes"],
        candidate_successes=candidate["successes"],
        success_rate_difference=(candidate["successes"] - baseline["successes"])
        / len(pairs),
        wins=sum(p["candidate_success"] and not p["original_success"] for p in pairs),
        losses=sum(p["original_success"] and not p["candidate_success"] for p in pairs),
        mean_hold_difference_s=float(np.mean([p["hold_difference_s"] for p in pairs])),
        paired_trials=pairs,
        original_metrics=aggregate_report(baseline),
        candidate_metrics=aggregate_report(candidate),
        note="Historical regression seeds; no heldout checkpoint selection.",
    )


def evaluate_series(checkpoints, output_dir, dataset=None, cohorts=None):
    cohorts = COHORTS if cohorts is None else cohorts
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    # Relocate only runtime replay lookup, never the checkpoint's saved config.
    if dataset is not None:
        import os

        os.environ["GO2_RECOVERY_DATASET"] = str(Path(dataset).resolve())
    torch.set_num_threads(1)
    protected = {
        str(p): digest(p)
        for p in (Path(BALANCE), Path(KICK), Path(REFERENCE), PARAMETERS, ATLAS)
    }
    baseline_policy = kick = None
    summary = {str(i): {} for i in range(len(checkpoints))}
    with torch.inference_mode():
        try:
            baseline_policy = load_evaluation_policy(BALANCE, "go2balance")
            kick = Policy(KICK, "go2frontyield", 16)
            frozen_digest = network_digest(baseline_policy)
            atlas = json.loads(ATLAS.read_text())
            parameters = PostureTransition(**json.loads(PARAMETERS.read_text()))
            provenance = dict(
                cohorts=cohorts,
                protected_sha256=protected,
                checkpoints=[dict(path=str(p), sha256=digest(p)) for p in checkpoints],
                cohort_prefix_sha256={},
                frozen_network_sha256=frozen_digest,
                baseline=baseline_policy.meta,
                kick=kick.meta,
                checkpoint_selection="All supplied checkpoints; no heldout selection.",
            )
            dump(dest / "provenance.json", provenance)
            for cohort, seeds in cohorts.items():
                data = bank(kick, Experiment(), seeds, "aggressive")
                prefix_hash = bank_digest(data)
                provenance["cohort_prefix_sha256"][cohort] = prefix_hash
                set_seed(1729)
                baseline = Evaluator(baseline_policy, atlas, True, True).evaluate(
                    parameters, data, list(range(16)), dest / f"{cohort}_original.csv"
                )
                baseline["prefix_sha256"] = prefix_hash
                dump(dest / f"{cohort}_original.json", baseline)
                if network_digest(baseline_policy) != frozen_digest:
                    raise RuntimeError("Frozen baseline parameters/statistics changed")
                for i, checkpoint in enumerate(checkpoints):
                    policy = None
                    try:
                        set_seed(1729)
                        policy = load_evaluation_policy(
                            checkpoint, "go2kickup_recovery"
                        )
                        check_interface(baseline_policy, policy)
                        before = network_digest(policy)
                        child = dest / f"checkpoint_{i:03d}_{Path(checkpoint).stem}"
                        child.mkdir(exist_ok=True)
                        set_seed(1729)
                        result = Evaluator(policy, atlas, True, True).evaluate(
                            parameters, data, list(range(16)), child / f"{cohort}.csv"
                        )
                        result.update(
                            checkpoint=str(checkpoint),
                            checkpoint_sha256=digest(checkpoint),
                            prefix_sha256=prefix_hash,
                        )
                        dump(child / f"{cohort}.json", result)
                        if network_digest(policy) != before:
                            raise RuntimeError("Evaluation mutated policy/statistics")
                        if bank_digest(data) != prefix_hash:
                            raise RuntimeError("Shared prefix input was mutated")
                        summary[str(i)].update(
                            checkpoint=str(checkpoint),
                            sha256=digest(checkpoint),
                            iteration=checkpoint_iteration(checkpoint),
                        )
                        summary[str(i)][cohort] = summarize_pair(baseline, result)
                        dump(dest / "summary.json", summary)
                        print(
                            f"{cohort} {checkpoint}: {result['successes']}/16 "
                            f"(frozen {baseline['successes']}/16)",
                            flush=True,
                        )
                    finally:
                        if policy is not None:
                            policy.env._backend.close()
            for p, expected in protected.items():
                if digest(p) != expected:
                    raise RuntimeError(f"Protected input modified: {p}")
            if network_digest(baseline_policy) != frozen_digest:
                raise RuntimeError("Frozen baseline changed during series evaluation")
            provenance["completed"] = True
            dump(dest / "provenance.json", provenance)
        finally:
            for policy in (baseline_policy, kick):
                if policy is not None:
                    policy.env._backend.close()
    return summary


def main():
    p = argparse.ArgumentParser(__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--fine-tuned", type=Path)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Evaluate every completed model_*.pt numerically, including model_0.",
    )
    source.add_argument("--checkpoints", type=Path, nargs="+")
    p.add_argument("--dataset", type=Path, help="Replay NPZ location on this machine.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-seed", type=int, default=6000)
    p.add_argument("--heldout-seed", type=int, default=7000)
    a = p.parse_args()
    checkpoints = (
        discover_checkpoints(a.run_dir)
        if a.run_dir
        else a.checkpoints
        if a.checkpoints
        else [a.fine_tuned]
    )
    if len(set(p.resolve() for p in checkpoints)) != len(checkpoints):
        p.error("Duplicate checkpoint paths")
    cohorts = {
        "validation": list(range(a.validation_seed, a.validation_seed + 16)),
        "heldout": list(range(a.heldout_seed, a.heldout_seed + 16)),
    }
    if set(cohorts["validation"]) & set(cohorts["heldout"]):
        p.error("Validation and heldout cohorts overlap")
    evaluate_series(checkpoints, a.output_dir, a.dataset, cohorts)


if __name__ == "__main__":
    main()
