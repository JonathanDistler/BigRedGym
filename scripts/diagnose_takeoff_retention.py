"""Read saved matched-seed native physics, without rerunning experiments."""

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    roots = {
        "adc": Path("output/unified_go2_evaluation_initial_v1/videos/adc_standing_0"),
        "previous_student": Path(
            "output/unified_go2_evaluation_initial_v2/videos/student_standing_0"
        ),
    }
    result = {}
    series = {}
    for name, root in roots.items():
        rows = list(csv.DictReader(Path(str(root) + "_physics.csv").open()))
        series[name] = rows

        def num(k):
            return np.array([float(r[k]) for r in rows])

        t = num("time")
        dt = np.diff(t, prepend=t[0] - 0.002)
        valid = np.array([r["prefix_safe"] == "True" for r in rows])
        pitch = num("pitch")
        omega = num("pitch_velocity")
        air = num("rear_clearance") > 0.015
        liftoff = (
            np.flatnonzero(
                air
                & (num("load_RL") < 0.02 * num("mass") * 9.81)
                & (num("load_RR") < 0.02 * num("mass") * 9.81)
            )
            if "load_RL" in rows[0]
            else np.flatnonzero(
                air & np.array([r["rear_loaded"] == "False" for r in rows])
            )
        )
        first = int(liftoff[0]) if len(liftoff) else None
        stalls = np.flatnonzero(
            (omega <= 0) & (t > (t[first] if first is not None else 100))
        )
        stall = int(stalls[0]) if len(stalls) else int(np.argmax(pitch * valid))
        end = t <= min(0.8, t[-1])
        keys = [
            "time",
            "pitch",
            "pitch_velocity",
            "L_com",
            "mechanical_energy",
            "front_moment_com",
            "rear_moment_com",
            "rear_positive_power",
            "com_support_x",
            "rear_clearance",
        ]
        result[name] = dict(
            peak_degrees=float(np.max(pitch[valid]) * 180 / np.pi),
            rear_liftoff_time=float(t[first]) if first is not None else None,
            first_stall={k: float(rows[stall][k]) for k in keys},
            first_08s_rear_work=float(
                np.sum(num("rear_positive_power")[end] * dt[end])
            ),
            first_08s_rear_impulse=float(np.sum(num("rear_moment_com")[end] * dt[end])),
            first_08s_front_impulse=float(
                np.sum(num("front_moment_com")[end] * dt[end])
            ),
            terminal_time=float(t[-1]),
        )
    comparison = []
    for t in [0.002, 0.1, 0.3, 0.5, 0.65, 0.7]:
        r = {}
        for name, rows in series.items():
            x = min(rows, key=lambda x: abs(float(x["time"]) - t))
            r[name] = x if t <= float(rows[-1]["time"]) else None
        for leg in ["FL", "FR", "RL", "RR"]:
            for part in ["hip", "thigh", "calf"]:
                joint = leg + "_" + part + "_joint"
                row = dict(requested_time=t, joint=joint)
                for name, x in r.items():
                    row[name + "_sample_time"] = (
                        float(x["time"]) if x is not None else None
                    )
                    for k in ["target", "q", "v", "tau"]:
                        row[name + "_" + k] = (
                            float(x[joint + "_" + k]) if x is not None else None
                        )
                row["target_difference"] = (
                    row["previous_student_target"] - row["adc_target"]
                    if row["previous_student_target"] is not None
                    else None
                )
                comparison.append(row)
    with (a.output / "joint_comparison.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0]))
        w.writeheader()
        w.writerows(comparison)
    (a.output / "diagnosis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
