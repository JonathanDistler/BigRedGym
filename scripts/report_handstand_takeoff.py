"""Generate the measured takeoff report; generated results remain outside Git."""

import argparse
import json
from pathlib import Path


def result(path):
    return json.loads(path.read_text())["result"]


def table_row(cells):
    return "| " + " | ".join(map(str, cells)) + " |"


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ["bc", "pilot", "evaluation", "full-evaluation", "benchmark", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    bc, pilot, evaluation, full, bench = map(
        result,
        [args.bc, args.pilot, args.evaluation, args.full_evaluation, args.benchmark],
    )
    lines = [
        "# Measured takeoff pilot",
        "",
        "Local CPU measurements. Molab throughput: **NOT MEASURED**.",
        "The actor retains an explicit nominal demonstration prior and learns",
        "bounded state feedback. The prior-only ablation also succeeds; these",
        "results do not establish PPO improvement.",
        "",
        table_row(
            [
                "Checkpoint",
                "Nominal",
                "Held-out",
                "Rear clear (m)",
                "Pitch (deg)",
                "Rear contact loss",
                "Capture",
                "Collisions",
                "Strict hold (s)",
                "Critic loss",
                "RSS (MiB)",
            ]
        ),
        table_row(["---"] * 11),
    ]
    entries = []
    if args.baseline:
        entries.append(("Original student", result(args.baseline), None, None))
    entries.append(("BC", bc["student"], None, None))
    entries += [
        (
            f"PPO {r['update']}",
            r["validation"],
            r["diagnostics"]["critic_loss"],
            r["peak_rss_MiB"],
        )
        for r in pilot["history"]
        if r["update"] in [1, 2, 3, 5, 10, 25, 50]
    ]
    for label, ev, critic, rss in entries:
        row = ev["nominal"]
        lines.append(
            table_row(
                [
                    label,
                    int(row["takeoff_success"]),
                    f"{ev['heldout_successes']}/{ev['heldout_trials']}",
                    f"{row['max_simultaneous_rear_clearance_m']:.5f}",
                    f"{row['max_pitch_deg']:.3f}",
                    row["rear_contact_loss"],
                    row["front_capture"],
                    row["illegal_contacts"],
                    f"{row['strict_hold_seconds']:.3f}",
                    f"{critic:.5f}" if critic is not None else "N/A",
                    f"{rss:.1f}" if rss else "N/A",
                ]
            )
        )
    successes, trials = evaluation["heldout_successes"], evaluation["heldout_trials"]
    lines += [
        "",
        f"Expanded PPO evaluation: **{successes}/{trials}** plus nominal.",
        "Held-out reset bounds: joint position ±0.003 rad and velocity",
        "±0.03 rad/s. Independent seeds in a narrow local reset family.",
        "This is not broad robustness evidence.",
        "",
        f"Full continuation: **{full['full_maneuver_successes']}/"
        f"{len(full['rows'])}** strict successes.",
        "After 0.96 s, the last learned takeoff phase is held with live",
        "feedback. No recorded balance output is replayed. Later calf",
        "collisions invalidate safety despite earlier certified takeoff.",
        "",
        table_row(
            [
                "Serial envs",
                "Rollout steps",
                "Transitions/s",
                "RSS (MiB)",
                "CPU % of one core",
            ]
        ),
        table_row(["---"] * 5),
    ]
    for row in bench:
        lines.append(
            table_row(
                [
                    row["envs"],
                    row["rollout_steps"],
                    f"{row['transitions_per_second']:.2f}",
                    f"{row['peak_rss_MiB']:.1f}",
                    f"{row['cpu_percent_one_core']:.1f}",
                ]
            )
        )
    last = pilot["history"][-1]
    throughput = last["end_to_end_training_transitions_per_second"]
    lines += [
        "",
        f"End-to-end training throughput: **{throughput:.2f}** transitions/s.",
        "Includes optimizer and validation overhead; excludes initial setup.",
        "",
        "Final update reward sums:",
        "```json",
        json.dumps(last["reward_components"], indent=2),
        "```",
        "Legacy reward is logged but excluded from the optimized reward.",
        "",
        "Final update diagnostics:",
        "```json",
        json.dumps(last["diagnostics"], indent=2),
        "```",
        "",
        "Torque ratio stays <=1.0. The nominal maneuver reaches the torque",
        "cap; saturation is not a torque-limit violation.",
        "",
        "Decision: stop this bounded pilot. Takeoff is retained, but no",
        "superiority to the nominal prior is demonstrated. Validate a",
        "longer corrective capture prefix before 200–500 updates.",
        "A 5000-update experiment is not justified.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
