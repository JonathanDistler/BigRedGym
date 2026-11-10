"""Evaluate pilot 0/50/100/200 atomically saved checkpoints on matched starts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument(
        "--experiment-name",
        help="Resolve the newest saved run once, then pin its path.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    if args.experiment_name:
        candidates = [
            x.parent
            for x in (Path("logs") / args.experiment_name).glob(
                "*/effective_config.json"
            )
        ]
        if not candidates:
            p.error("No saved runs found for this experiment")
        args.run_dir = max(candidates, key=lambda x: x.stat().st_mtime)
    print(f"Evaluation run: {args.run_dir.resolve()}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for iteration in (0, 50, 100, 200):
        checkpoint = args.run_dir / f"model_{iteration}.pt"
        while not checkpoint.is_file():
            if not args.watch:
                p.error(f"Missing checkpoint: {checkpoint}")
            time.sleep(5)
        out = args.output_dir / f"iteration_{iteration}"
        with (args.output_dir / f"iteration_{iteration}.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_frontyield.py")),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(out),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = {}
        for done in (0, 50, 100, 200):
            result = args.output_dir / f"iteration_{done}" / "evaluation.json"
            if not result.is_file():
                continue
            report = json.loads(result.read_text())
            rs = report["results"]
            summary[done] = dict(
                level=report["level"],
                valid_takeoff=sum(r["valid_takeoff"] for r in rs),
                trials=len(rs),
                mean_peak_min_clearance_mm=float(
                    np.mean([r["peak_min_clearance_m"] for r in rs]) * 1000
                ),
                worst_peak_min_clearance_mm=min(r["peak_min_clearance_m"] for r in rs)
                * 1000,
                mean_max_front_drift_cm=float(
                    np.mean([r["max_front_drift_m"] for r in rs]) * 100
                ),
                worst_max_front_drift_cm=max(r["max_front_drift_m"] for r in rs) * 100,
                mean_peak_pitch_rad=float(np.mean([r["peak_pitch_rad"] for r in rs])),
                mean_peak_abs_pitch_velocity=float(
                    np.mean([r["peak_abs_pitch_velocity"] for r in rs])
                ),
                mean_terminal_pitch_rad=float(
                    np.mean([r["terminal_pitch_rad"] for r in rs])
                ),
                mean_terminal_pitch_velocity=float(
                    np.mean([r["terminal_pitch_velocity"] for r in rs])
                ),
                front_contact_fraction=float(
                    np.mean([r["front_contact_fraction"] for r in rs])
                ),
                mean_push_abs_rear_residual_rad=float(
                    np.mean([r["push_mean_abs_residual_rad"] for r in rs])
                ),
                curriculum_success=sum(r["curriculum_success"] for r in rs),
                nonfoot_contact=sum(r["nonfoot_contact"] for r in rs),
                overrotation=sum(r["overrotation"] for r in rs),
                handoff_candidates=sum(r.get("handoff_candidate", False) for r in rs),
            )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(iteration, summary[iteration], flush=True)


if __name__ == "__main__":
    main()
