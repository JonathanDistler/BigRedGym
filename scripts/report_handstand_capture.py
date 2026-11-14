"""Generate capture root-cause, prefix, full-success and Assignment 6 tables."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def result(path):
    return read(path)["result"]


def row(cells):
    return "| " + " | ".join(map(str, cells)) + " |"


def interval(successes, trials):
    z = 1.96
    p = successes / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / (1 + z * z / trials)
    )
    return [center - half, center + half]


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in [
        "diagnostics",
        "viability",
        "rescue",
        "prefix",
        "comparison",
        "final",
        "ppo",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    traces = {
        name: read(args.diagnostics / (name + "_trace.json"))
        for name in ["reference", "prior", "hybrid", "student"]
    }
    comparison = result(args.comparison)
    prefixes = result(args.prefix)
    final = result(args.final)["student"]
    population = final["rows"][1:]
    good = [r for r in population if r["complete_success"]]
    bad = [r for r in population if not r["complete_success"]]
    low, high = interval(len(good), len(population))
    metrics = [r["hold_metrics"] for r in good]
    aggregation = {}
    if metrics:
        for field in [
            "orientation_mean_deg",
            "orientation_max_deg",
            "front_slip_m",
            "front_leg_kinetic_J",
            "pitch_sign_changes_hz",
        ]:
            values = [m[field] for m in metrics]
            aggregation[field] = dict(
                mean=float(np.mean(values)),
                maximum=float(max(values)),
                minimum=float(min(values)),
            )
        for field in ["orientation_rms_deg", "pitch_rms_deg"]:
            aggregation[field] = dict(
                pooled_rms=float(np.sqrt(np.mean([m[field] ** 2 for m in metrics])))
            )
    lines = [
        "# Capture diagnosis and measured progress",
        "",
        "The takeoff-only actor clamps effective phase at control step 47.",
        "Continuing it beyond 0.96 s omits the optimized rotation, extension",
        "pulse and balance handoff. It was not a trained capture controller.",
        "The collision is a later consequence, not the first deviation.",
        "",
        row(
            [
                "Controller",
                "First >0.005 action deviation (s)",
                "First >5 mrad filtered target deviation (s)",
                "First geometry contact (s)",
                "First solver interval end (s)",
            ]
        ),
        row(["---"] * 5),
    ]
    reference = traces["reference"]["physics"]
    for name, trace in traces.items():
        controls = trace["controls"]
        command = next(
            (r["time_s"] for r in controls if r["action_error_max"] > 0.005), None
        )
        target = next(
            (
                s["time_s"]
                for r, s in zip(reference, trace["physics"])
                if np.max(abs(np.array(r["q_target"]) - s["q_target"])) > 0.005
            ),
            None,
        )
        summary = trace["result"]
        geometric = summary.get("first_geometric_illegal")
        solver = summary.get("first_illegal")
        lines.append(
            row(
                [
                    name,
                    command,
                    target,
                    geometric["time_s"] if geometric else "none",
                    solver["time_s"] if solver else "none",
                ]
            )
        )
    lines += [
        "",
        "Full 0.8–2.3 s diagnostics:",
        f"[Overview]({args.diagnostics.name}/capture_comparison.png), "
        f"[all channels PDF]({args.diagnostics.name}/capture_diagnostics.pdf).",
        "Post-failure tails are native simulation diagnostics and never count",
        "as successful continuation. Raw traces include geom IDs, body pairs,",
        "world points/normals, per-step solver forces and impulses. Geometry",
        "timestamps and solver-interval end timestamps are distinguished.",
        "",
        "## Causal rescue and local viability",
        "",
        row(["Switch to reference (s)", "Prefix safe", "Strict hold (s)"]),
        row(["---"] * 3),
    ]
    for r in result(args.rescue):
        lines.append(
            row(
                [
                    r["switch_time_s"],
                    r["prefix_safe"],
                    f"{r['strict_hold_seconds']:.3f}",
                ]
            )
        )
    probes = result(args.viability)
    valid = [r for r in probes if not r["invalid_reset"]]
    lines += [
        "",
        f"Snapshot probes: {len(valid)}/{len(probes)} physically valid, "
        f"{sum(r['collision_free_through_2_2'] for r in valid)}/{len(valid)} "
        "collision-free through 2.2 s.",
        f"Only {sum(r['local_five_second_strict'] for r in valid)}/{len(valid)} "
        "meet five strict seconds within the reference horizon.",
        "These are explicit 1.28 s snapshot interventions, not full-maneuver",
        "successes. They establish tested sensitivities, not a safe-ball radius.",
        "The nominal capture segment is viable nearby through 2.2 s;",
        "recorded-action long-term balancing is much narrower.",
        "",
        "## Continuous standing-start prefixes",
        "",
        "Each cell is prefix successes / trials (current 0.1 s strict capture",
        "stability successes). Prefix means safe powered takeoff, both front",
        "feet supporting, rear clearance >0.1 m and orientation within 5 deg",
        "of the reference at the endpoint. Neither gate substitutes for five",
        "continuous strict handstand seconds.",
        "",
    ]
    names = list(dict.fromkeys(r["controller"] for r in prefixes))
    times = sorted(set(r["prefix_seconds"] for r in prefixes))
    lines += [row(["Prefix (s)"] + names), row(["---"] * (len(names) + 1))]
    for t in times:
        cells = [f"{t:.2f}"]
        for name in names:
            values = [
                r
                for r in prefixes
                if r["controller"] == name and r["prefix_seconds"] == t
            ]
            cells.append(
                f"{sum(r['prefix_success'] for r in values)}/{len(values)} "
                f"({sum(r['capture_success'] for r in values)})"
            )
        lines.append(row(cells))
    lines += [
        "",
        "Prefix seeds are matched across controllers and reused across",
        "durations; durations are correlated, not additional independent trials.",
        "",
        "## Matched full maneuvers",
        "",
        row(["Controller", "Nominal", "Held-out strict successes", "Held-out trials"]),
        row(["---"] * 4),
    ]
    for name, c in comparison.items():
        lines.append(
            row(
                [
                    name,
                    c["nominal"]["complete_success"],
                    c["heldout_successes"],
                    c["heldout_trials"],
                ]
            )
        )
    lines += [
        "",
        "Hybrid uses privileged simulator COM and is a teacher/baseline.",
        "The new student uses IMU/encoder-derived estimated COM, not simulator",
        "world position/velocity, COM or contact-force observations. It retains",
        "a disclosed nominal prior and the learned balance feedback matrix.",
        "The maximum full-evaluation duration is 8 s, common to controllers;",
        "successful trials stop after the original five-second gate. Reference",
        "commands only exist to 6.4 s; any later fixed-prior continuation holds",
        "its final command and is not an exact-reference benchmark.",
        "",
        "## Sealed larger held-out evaluation",
        "",
        f"**{len(good)}/{len(population)} strict complete successes**, "
        f"Wilson 95% interval **{100 * low:.1f}–{100 * high:.1f}%**.",
        "The nominal trial is separate. All failed episodes remain in the",
        "denominator. This is only the bounded joint reset family: position",
        "±0.003 rad and velocity ±0.03 rad/s, ideal 500 Hz IMU, calibrated",
        "standing height/zero base velocity, nominal physics. It is not broad",
        "perturbation robustness or hardware readiness.",
        "",
        "## Assignment 6 — successful held-out five-second intervals",
        "",
        "Aggregate orientation/pitch RMS values pool equal-duration trials.",
        "Other values are trial means, except overall maximum orientation.",
        "Failed episodes are excluded from hold-only metrics, never from",
        "the success probability. Raw metrics remain available per trial.",
        "",
        row(["Metric", "Value"]),
        row(["---"] * 2),
    ]
    if metrics:
        lines += [
            row(
                ["Orientation mean (deg)", aggregation["orientation_mean_deg"]["mean"]]
            ),
            row(
                [
                    "Orientation pooled RMS (deg)",
                    aggregation["orientation_rms_deg"]["pooled_rms"],
                ]
            ),
            row(
                [
                    "Orientation maximum (deg)",
                    aggregation["orientation_max_deg"]["maximum"],
                ]
            ),
            row(["Pitch pooled RMS (deg)", aggregation["pitch_rms_deg"]["pooled_rms"]]),
            row(
                [
                    "Pitch zero-crossing rate (s^-1)",
                    aggregation["pitch_sign_changes_hz"]["mean"],
                ]
            ),
            row(["Front-foot path mean (m)", aggregation["front_slip_m"]["mean"]]),
            row(
                [
                    "Front-leg kinetic energy mean (J)",
                    aggregation["front_leg_kinetic_J"]["mean"],
                ]
            ),
        ]
    lines += [
        "",
        "Pitch crossings ignore |error| <=0.1 deg. Crossing rate is not",
        "automatically a full oscillation-cycle frequency. Foot path and leg",
        "kinetic energy use original SI-unit 500 Hz scoring.",
        "",
        "## PPO and stop/go",
        "",
    ]
    history = result(args.ppo)["history"]
    for r in history:
        lines += [
            f"Update {r['update']}: current captures "
            f"{r['validation']['heldout_successes']}/"
            f"{r['validation']['heldout_trials']}; final KL "
            f"{r.get('final_approx_kl', 'not logged')}; RSS "
            f"{r['peak_rss_MiB']:.1f} MiB."
        ]
    lines += [
        "",
        "Two capture PPO updates were retained. The first higher-rate smoke",
        "had excessive KL and is preserved separately. Lower learning rate",
        "and KL stopping improve update control, but matched BC/PPO tests do",
        "not establish a PPO benefit. Do not spend 250 or 5000 updates now.",
        "Any >50 capture continuation requires paired objective improvement",
        "and actual Molab throughput/exact-resume proof. Molab remains unmeasured.",
        "",
        "## Failed held-out maneuvers",
        "",
        row(
            [
                "Seed",
                "First failure / end (s)",
                "Reason",
                "Body pair",
                "Max pitch (deg)",
                "Best strict hold (s)",
            ]
        ),
        row(["---"] * 6),
    ]
    failure_rows = []
    for r in bad:
        event = r.get("first_geometric_illegal") or r.get("first_safety_violation")
        failure_time = event["time_s"] if event else r.get("control_steps", 400) * 0.02
        pair = (r.get("first_illegal") or {}).get("bodies", [])
        reason = (r.get("first_safety_violation") or {}).get("reason", r["failure"])
        item = dict(
            seed=r["seed"],
            failure_time_s=failure_time,
            reason=reason,
            body_pair="/".join(pair),
            max_pitch_deg=r["max_pitch_deg"],
            strict_hold_seconds=r["strict_hold_seconds"],
        )
        failure_rows.append(item)
        lines.append(
            row(
                [
                    r["seed"],
                    f"{failure_time:.3f}",
                    reason,
                    "/".join(pair),
                    f"{r['max_pitch_deg']:.3f}",
                    f"{r['strict_hold_seconds']:.3f}",
                ]
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    with args.output.with_suffix(".failures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "seed",
                "failure_time_s",
                "reason",
                "body_pair",
                "max_pitch_deg",
                "strict_hold_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(failure_rows)
    summary = dict(
        heldout_successes=len(good),
        heldout_trials=len(population),
        wilson95=[low, high],
        assignment6=aggregation,
        failure_reasons=dict(Counter(r["reason"] for r in failure_rows)),
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
