"""Regenerate concise comparison tables from measured evaluation JSON files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help="LABEL=path/to/evaluate.json (repeat for each controller)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = {}
    for spec in args.evaluation:
        label, path = spec.split("=", 1)
        evaluations[label] = json.loads(Path(path).read_text())["result"]
    lines = [
        "# Measured handstand comparison",
        "",
        "Nominal values use the first five seconds of the longest strict hold. "
        "Shorter holds are partial failures. "
        "Replay success does not establish autonomy.",
        "",
        "| Controller | Nominal successes/trials | Best strict hold (s) | "
        "Axis mean/RMS/max (deg) | Pitch RMS (deg) | Slip (m) | Front-leg KE (J) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, report in evaluations.items():
        case = report["nominal"]
        row = case["results"][0]
        metrics = row.get("hold_metrics")
        if metrics:
            axis = "/".join(
                f"{metrics[k]:.5f}"
                for k in [
                    "orientation_mean_deg",
                    "orientation_rms_deg",
                    "orientation_max_deg",
                ]
            )
            detail = (
                f"{axis} | {metrics['pitch_rms_deg']:.5f} | "
                f"{metrics['front_slip_m']:.7f} | "
                f"{metrics['front_leg_kinetic_J']:.8f}"
            )
        else:
            detail = "N/A | N/A | N/A | N/A"
        lines.append(
            f"| {label} | {case['successes']}/{case['trials']} | "
            f"{row.get('strict_hold_seconds', 0):.3f} | {detail} |"
        )
    labels = list(evaluations)
    lines += [
        "",
        "| Condition | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for condition in evaluations[labels[0]]:
        cells = []
        for label in labels:
            case = evaluations[label].get(condition)
            cells.append(f"{case['successes']}/{case['trials']}" if case else "N/E")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The test matrix is a small diagnostic pilot. Nominal and fixed-push "
        "repetitions are deterministic duplicates, not independent population samples. "
        "Initial-condition seeds are matched across controllers. See raw JSON for "
        "rejected resets, Wilson intervals, failure reasons, "
        "action changes, and recovery.",
        "",
        "Mass tests require recomputed MuJoCo constants; compare with the "
        "mass_recompute_control condition. That control also fails, so mass alone "
        "cannot explain the difference from the original replay.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
