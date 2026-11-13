"""Read existing native traces once; no training or trajectory re-creation."""

import argparse, csv, json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--trace",
        type=Path,
        default=Path(
            "output/retained_go2_validation_v3/videos/student_standing_0_physics.csv"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = list(csv.DictReader(a.trace.open()))

    def val(r, k):
        return float({"True": 1, "False": 0}.get(r[k], r[k]))

    flight = next(
        r for r in rows if val(r, "rear_clearance") > 0.015 and val(r, "takeoff") > 0
    )
    stall = next(
        r
        for r in rows
        if val(r, "time") > val(flight, "time") and val(r, "pitch_velocity") <= 0
    )
    window = [
        r
        for r in rows
        if val(flight, "time") - 0.08 <= val(r, "time") <= val(stall, "time") + 0.06
    ]
    keys = [
        "pitch",
        "pitch_velocity",
        "com_support_x",
        "com_velocity_x",
        "L_com",
        "front_moment_com",
        "rear_moment_com",
        "front_slip_speed",
        "front_force_z",
        "calf_clearance",
        "head_clearance",
        "saturated_fraction",
        "tracking_rmse",
        "rear_clearance",
        "mechanical_energy",
    ]
    for prefix in ["FL", "FR", "RL", "RR"]:
        keys += [
            f"{prefix}_{joint}_joint_{suffix}"
            for joint in ["thigh", "calf"]
            for suffix in ["q", "v", "tau"]
        ]
    result = dict(
        source=str(a.trace),
        flight_time=val(flight, "time"),
        first_stall_time=val(stall, "time"),
        flight={k: val(flight, k) for k in keys},
        stall={k: val(stall, k) for k in keys},
        critical_interval={
            k: dict(
                min=min(val(r, k) for r in window), max=max(val(r, k) for r in window)
            )
            for k in keys
        },
        interpretation="Local first stall, not global episode maximum. Opposing front contact moment is measured; no causal reward ablation claimed. Reset prioritization uses existing pre-stall native states, without invented teacher actions.",
    )
    (a.output / "diagnosis.json").write_text(json.dumps(result, indent=2))
    with (a.output / "critical_interval.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["time", *keys])
        w.writeheader()
        w.writerows({k: r[k] for k in w.fieldnames} for r in window)
    print(
        json.dumps(
            {k: result[k] for k in ["flight_time", "first_stall_time", "stall"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
