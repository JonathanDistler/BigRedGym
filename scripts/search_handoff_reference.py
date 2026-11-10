"""Small reference search for controlled rotation; never trains PPO.

Screening seeds and held-out validation seeds are disjoint. The score saturates
rotation at the target, penalizes speed and falls, and gives no clearance bonus.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gym.envs.go2.kickup_reference import KickReference
from scripts.search_kickup_reference import Experiment


def objective(result, rows, target):
    viable = [
        r
        for r in rows
        if 0.85 <= r["time"] <= 1.5
        and min(r["force_0"], r["force_1"]) > 0.04 * result["body_weight_N"]
        and min(r["clearance_2"], r["clearance_3"]) > 0.02
        and not r["body_contact"]
    ]
    if not viable:
        return -10.0
    # Finite desired rotation, terminal rate, contact, and tolerable front travel.
    cost = min(
        abs(r["pitch_rad"] - target) / 0.25
        + 0.25 * r["pitch_rate"] ** 2
        + 2 * max(0.0, r["front_drift"] - 0.12)
        for r in viable
    )
    return -float(cost) - 3 * result["body_contact"]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parameters", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--target-degrees", type=float, default=90.0)
    args = p.parse_args()
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    ex = Experiment()
    initial = json.loads(Path(args.parameters).read_text())
    rng = np.random.default_rng(621)
    best, best_score = initial.copy(), -np.inf
    history = []
    bounds = dict(
        extend=(0.10, 0.26),
        rear_push_hip=(-0.3, 0.7),
        rear_push_knee=(-1.2, -0.85),
        front_push_hip=(-1.15, 1.3),
        front_push_knee=(-2.4, -0.95),
        front_gain=(0.2, 0.9),
        front_damping=(0.5, 1.5),
        tuck_delay=(0.0, 0.16),
        tuck_duration=(0.12, 0.4),
        rear_tuck_hip=(0.5, 2.1),
        rear_tuck_knee=(-2.6, -0.95),
        hip_delay=(-0.15, 0.15),
        knee_delay=(-0.15, 0.15),
    )
    for i in range(args.candidates):
        values = (initial if i < args.candidates // 2 else best).copy()
        if i:
            for key, (lo, hi) in bounds.items():
                values[key] = float(
                    np.clip(values[key] + rng.normal(0, (hi - lo) * 0.22), lo, hi)
                )
        trials = [
            ex.trial(KickReference(**values), seed=s, record=True) for s in (5100, 5101)
        ]
        score = min(
            objective(r, rows, np.deg2rad(args.target_degrees)) for r, rows in trials
        )
        peak = np.mean([max(x["pitch_rad"] for x in rows) for _, rows in trials])
        history.append(
            dict(
                candidate=i,
                score=score,
                peak_deg=float(np.rad2deg(peak)),
                parameters=values,
            )
        )
        if score > best_score:
            best_score, best = score, values
            print(
                f"candidate {i}: score {score:.3f}, peak {np.rad2deg(peak):.1f}deg",
                flush=True,
            )
        (dest / "search.json").write_text(json.dumps(history, indent=2))
    (dest / "candidate_reference.json").write_text(json.dumps(best, indent=2))
    validation = []
    for seed in range(6000, 6016):
        result, rows = ex.trial(KickReference(**best), seed=seed, record=True)
        validation.append(
            dict(
                **result,
                objective=objective(result, rows, np.deg2rad(args.target_degrees)),
                maximum_pitch_deg=float(np.rad2deg(max(x["pitch_rad"] for x in rows))),
                terminal_pitch_velocity=rows[-1]["pitch_rate"],
            )
        )
    (dest / "validation.json").write_text(json.dumps(validation, indent=2))
    print(
        "held-out peak",
        np.mean([r["maximum_pitch_deg"] for r in validation]),
        flush=True,
    )


if __name__ == "__main__":
    main()
