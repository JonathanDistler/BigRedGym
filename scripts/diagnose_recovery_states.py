"""Descriptive handoff comparisons; one positive seed cannot establish causality."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    records = []
    for cohort, path in (("validation", a.validation), ("heldout", a.heldout)):
        for trial in json.loads(path.read_text())["trials"]:
            if trial["handoff"] is not None:
                records.append(dict(cohort=cohort, **trial))
    positive = [r for r in records if r["success"]]
    negative = [r for r in records if not r["success"]]
    fields = [
        k
        for k, v in records[0]["handoff"].items()
        if isinstance(v, (int, float)) and k != "time"
    ]
    differences = []
    for key in fields:
        good = np.array([r["handoff"][key] for r in positive])
        bad = np.array([r["handoff"][key] for r in negative])
        if not len(good) or not len(bad):
            continue
        differences.append(
            dict(
                variable=key,
                recovered_mean=float(good.mean()),
                failed_mean=float(bad.mean()),
                failed_std=float(bad.std()),
                descriptive_separation=float(
                    abs(good.mean() - bad.mean()) / max(bad.std(), 1e-6)
                ),
            )
        )
    report = dict(
        positive_handoffs=len(positive),
        negative_handoffs=len(negative),
        warning=(
            "Descriptive comparison only; one successful seed, correlated "
            "variables and survivor selection preclude causal or "
            "generalization claims."
        ),
        ranked=sorted(differences, key=lambda r: -r["descriptive_separation"]),
        trials=records,
    )
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
