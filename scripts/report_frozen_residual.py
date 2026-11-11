"""Summarize completed residual evaluations and verify protected inputs."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.optimize_posture_transition import digest
from scripts.evaluate_kickup_handoff import dump


def metrics(result):
    outcomes = [r for s in ("validation", "heldout") for r in result[s]["outcomes"]]
    return dict(
        near_vertical=result["retention"]["successes"],
        recoveries=sum(r["success"] for r in outcomes),
        validation=result["validation"]["all_states"]["successes"],
        heldout=result["heldout"]["all_states"]["successes"],
        frontier_successes=sum(
            result[s]["strata"]["1"]["successes"] for s in ("validation", "heldout")
        ),
        mean_hold_s=float(np.mean([r["hold_s"] for r in outcomes])),
        mean_alive_com_speed=float(
            np.mean([r["mean_alive_com_speed"] for r in outcomes])
        ),
        mean_alive_front_support=float(
            np.mean([r["alive_front_support_retention"] for r in outcomes])
        ),
        mean_motor_saturation=float(
            np.mean([r["motor_saturation_fraction"] for r in outcomes])
        ),
        mean_endpoint_com_xy_displacement_m=float(
            np.mean(
                [
                    sum(
                        (r["final"][f"com_world_{a}"] - r["initial"][f"com_world_{a}"])
                        ** 2
                        for a in ("x", "y")
                    )
                    ** 0.5
                    for r in outcomes
                ]
            )
        ),
        mean_final_front_drift_m=float(
            np.mean([r["final"]["front_drift"] for r in outcomes])
        ),
        additional_correction_rms_rad=result.get("additional_correction_rms_rad"),
        additional_correction_activation_frequency=result.get(
            "additional_correction_activation_frequency"
        ),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    baseline = json.loads(
        Path("output/frozen_residual_zero_v2/summary.json").read_text()
    )
    results = dict(original=metrics(baseline))
    for path in sorted(a.run_dir.glob("evaluation_*/summary.json")):
        result = json.loads(path.read_text())
        row = metrics(result)
        left = [r for s in ("validation", "heldout") for r in baseline[s]["outcomes"]]
        right = [r for s in ("validation", "heldout") for r in result[s]["outcomes"]]
        row["paired_wins"] = sum(
            not x["success"] and y["success"] for x, y in zip(left, right)
        )
        row["paired_losses"] = sum(
            x["success"] and not y["success"] for x, y in zip(left, right)
        )
        results[path.parent.name] = row
    expected = json.loads(
        Path("output/recovery_frontier_protected_after_resume.json").read_text()
    )
    protection = {p: digest(p) == entry["sha256"] for p, entry in expected.items()}
    manifest = json.loads(
        Path("datasets/recovery_frontier_assets_manifest_v2.json").read_text()
    )
    data_protection = {
        p: digest(p) == entry["sha256"]
        for p, entry in manifest["files"].items()
        if p.startswith("datasets/")
    }
    protected_code = {
        p: digest(p) == manifest["files"][p]["sha256"]
        for p in (
            "gym/envs/go2/contact_cpg.py",
            "gym/envs/go2/go2balance.py",
            "gym/envs/go2/go2balance_config.py",
            "gym/envs/go2/posture_transition.py",
        )
    }
    assert all(protection.values()) and all(data_protection.values())
    assert all(protected_code.values())
    legacy = {}
    if a.legacy_dir:
        for path in a.legacy_dir.glob("*/summary.json"):
            result = json.loads(path.read_text())
            for split in ("validation", "heldout"):
                assert (
                    result[split]["dataset_sha256"] == baseline[split]["dataset_sha256"]
                )
                left, right = baseline[split]["outcomes"], result[split]["outcomes"]
                assert len(left) == len(right)
                assert all(x["initial"] == y["initial"] for x, y in zip(left, right))
            legacy[path.parent.name] = dict(
                **metrics(result),
                checkpoint=result["checkpoint"],
                sha256=result["sha256"],
                exact_initial_measurement_parity=True,
            )
    latest = max(
        results.keys() - {"original"}, key=lambda name: int(name.split("_")[-1])
    )
    useful = results[latest]["frontier_successes"] > 0
    dump(
        a.output,
        dict(
            comparisons=results,
            protected=protection,
            datasets_unchanged=data_protection,
            protected_code=protected_code,
            early_legacy=legacy,
            recommendation="Do not extend training: no repeatable frontier gain."
            if not useful
            else "Repeat frontier gains on new cohorts before further training.",
            limitations="Snapshots are correlated. Endpoint COM drift includes "
            "post-collision "
            "motion; alive speed/support are collision-censored. Correction telemetry "
            "averages calls across nominal and replay batches, including padding. "
            "Historical held-out cohorts have already been inspected.",
        ),
    )


if __name__ == "__main__":
    main()
