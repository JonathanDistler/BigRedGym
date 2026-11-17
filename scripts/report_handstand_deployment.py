"""Compare complete matched trial reports without changing success definitions."""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    original = json.loads(args.original.read_text())["result"]["student"]
    adapter = json.loads(args.adapter.read_text())
    left, right = original["rows"], adapter["rows"]
    if len(left) != len(right):
        raise ValueError("Different trial counts")
    rows = []
    for a, b in zip(left, right):
        if a["seed"] != b["seed"]:
            raise ValueError("Different trial seeds")
        rows.append(
            {
                "seed": a["seed"],
                "original_success": a["complete_success"],
                "adapter_success": b["complete_success"],
                "original": a,
                "adapter": b,
            }
        )
    mismatches = [
        r["seed"] for r in rows if r["original_success"] != r["adapter_success"]
    ]
    report = {
        "original_heldout_successes": original["heldout_successes"],
        "adapter_heldout_successes": adapter["heldout_successes"],
        "heldout_trials": len(rows) - 1,
        "original_nominal_success": left[0]["complete_success"],
        "adapter_nominal_success": right[0]["complete_success"],
        "outcome_mismatch_seeds": mismatches,
        "maximum_absolute_errors": adapter["maximum_absolute_errors"],
        "inference_ms": adapter["inference_ms"],
        "rows": rows,
        "criteria": (
            "Unchanged research five-second strict gate, full standing start, "
            "seeds 301001–301100, 8s horizon, joint perturbations "
            "±.003rad/±.03rad/s; nominal reported separately."
        ),
        "hardware_status": (
            "NOT READY: Cornell sensor ingestion, actuator protection "
            "and timing integration blocked."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"}, indent=2
        )
    )
    if mismatches or max(adapter["maximum_absolute_errors"].values()) > 1e-6:
        raise SystemExit("Regression mismatch: review required")


if __name__ == "__main__":
    main()
