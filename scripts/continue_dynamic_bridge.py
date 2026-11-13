"""Manual bounded continuation; larger budgets require measured recovery evidence."""

import argparse, csv, json, math, subprocess, sys
from pathlib import Path


def permission(out, target, baseline=None, evaluation=None):
    if (out / "STOP").exists():
        raise ValueError(
            "STOP exists; preserve the known-good generation and inspect the reason"
        )
    reports = sorted(out.glob("retention_*.json"))
    if not reports or not json.loads(reports[-1].read_text())["passed"]:
        raise ValueError("A completed, passing endpoint evaluation is required")
    logs = sorted(out.glob("session_*/training.csv"))
    stats = [r for log in logs for r in csv.DictReader(log.open())]
    if not stats or any(
        not math.isfinite(float(r["post_update_kl"]))
        or float(r["post_update_kl"]) > 0.08
        for r in stats[-5:]
    ):
        raise ValueError("Recent numerical-health evidence is missing or unhealthy")
    if target > 50:
        if baseline is None or evaluation is None:
            raise ValueError(
                "Above50 updates requires baseline and current expanded evaluation folders"
            )

        def bridge(folder):
            summary = json.loads((folder / "summary.json").read_text())
            if not summary["complete_evaluation"]:
                raise ValueError("Truncated smoke is not performance evidence")
            rows = json.loads((folder / "episodes.json").read_text())
            return [r for r in rows if r["distribution"] == "bridge"]

        import hashlib

        current = json.loads((evaluation / "summary.json").read_text())
        original = json.loads((baseline / "summary.json").read_text())
        latest = json.loads((out / "latest.json").read_text())["checkpoint"]
        if (
            current["checkpoint_sha"]
            != hashlib.sha256((out / latest).read_bytes()).hexdigest()
        ):
            raise ValueError("Expanded evaluation must describe the latest checkpoint")
        if (
            current["reference_sha"] != original["reference_sha"]
            or current["preparation_sha"] != original["preparation_sha"]
        ):
            raise ValueError(
                "Expanded comparisons must use identical reset/physics conventions"
            )
        before, after = bridge(baseline), bridge(evaluation)
        if len(before) != len(after) or len(after) < 5:
            raise ValueError("Matched full held-out bridge suite required")
        avg = lambda rows, k: sum(r[k] for r in rows) / len(rows)
        # Progress requires actual partial hold or more capture-neighborhood entries,
        # not peak-pitch-only improvement. Neighborhood entry is still diagnostic.
        progress = max(r["valid_hold"] for r in after) >= max(
            0.25, max(r["valid_hold"] for r in before) + 0.1
        )
        progress |= sum(
            r["empirical_neighborhood_entry"] is not None for r in after
        ) > sum(r["empirical_neighborhood_entry"] is not None for r in before)
        if (
            not progress
            or avg(after, "illegal_collision") > avg(before, "illegal_collision")
            or avg(after, "front_support_fraction")
            < avg(before, "front_support_fraction") - 0.05
        ):
            raise ValueError(
                "Recovery/contact evidence does not yet warrant a larger budget"
            )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation", default="output/retained_go2_preparation_v3")
    p.add_argument("--certificate", default="output/molab_retention/retention.json")
    p.add_argument(
        "--target", type=int, choices=[20, 25, 50, 100, 200, 300], required=True
    )
    p.add_argument("--baseline", type=Path)
    p.add_argument("--evaluation", type=Path)
    a = p.parse_args()
    permission(a.output, a.target, a.baseline, a.evaluation)
    cfg = json.loads(Path("output/molab_training_config.json").read_text())
    latest = json.loads((a.output / "latest.json").read_text())
    if a.target <= latest["iteration"]:
        raise ValueError("Target must exceed completed update")
    cfg.update(
        target_updates=a.target, interaction_budget=a.target * cfg["n"] * cfg["horizon"]
    )
    Path("output/molab_training_config.json").write_text(json.dumps(cfg, indent=2))
    subprocess.check_call(
        [
            sys.executable,
            "scripts/dynamic_bridge_go2.py",
            "train",
            "--preparation",
            a.preparation,
            "--certificate",
            a.certificate,
            "--output",
            str(a.output),
            "--resume",
            str(a.output / latest["checkpoint"]),
            "--n",
            str(cfg["n"]),
            "--horizon",
            str(cfg["horizon"]),
            "--updates",
            str(a.target),
            "--eval-every",
            str(cfg["eval_every"]),
            "--interaction-budget",
            str(cfg["interaction_budget"]),
            "--max-seconds",
            "1800",
        ]
    )


if __name__ == "__main__":
    main()
