"""Conservative retention gate from measured probes, never from episode reward."""

import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.run / "probes.csv")
    outcomes = json.loads((a.run / "outcome.json").read_text())
    checks = []
    for outcome in outcomes:
        g = d[(d.method == outcome["method"]) & (d.training_seed == outcome["seed"])]
        final = g.iteration.max()
        criteria = []
        for scale, stochastic in [(1.0, False), (None, True)]:
            # A promoted run no longer evaluates the previous training geometry.
            # Follow its actual current geometry; never substitute a missing final
            # old-stage measurement or call one new-stage point three repeats.
            selected = (
                (g.scale == scale)
                if scale is not None
                else (g.scale == g.groupby("iteration").scale.transform("min"))
            )
            h = g[selected & (g.stochastic == stochastic)]
            initial = float(h[h.iteration == 0].takeoff.mean())
            last = (
                h[h.iteration >= max(2, final - 4)].groupby("iteration").takeoff.mean()
            )
            criteria.append(
                dict(
                    scale=float(scale) if scale is not None else "current_training",
                    scales_evaluated=sorted(h.scale.unique().tolist()),
                    stochastic=stochastic,
                    initial=initial,
                    final=float(h[h.iteration == final].takeoff.mean()),
                    last_three_mean=float(last.mean()),
                    passed=bool(
                        len(last) >= 3 and last.mean() >= max(0.25, initial - 0.125)
                    ),
                )
            )
        checks.append(
            dict(
                **outcome,
                criteria=criteria,
                passed=outcome["updates"] >= 10
                and outcome["stop_reason"] == "completed bounded diagnostic"
                and all(c["passed"] for c in criteria),
            )
        )
    result = dict(
        passed=all(c["passed"] for c in checks),
        checks=checks,
        note=(
            "At least ten updates; no early safety stop; last three evaluations "
            "retain full-scale deterministic and training-scale stochastic "
            "takeoff within 12.5 percentage points, with minimum 25% takeoff. "
            "Diagnostic sample sizes do not establish curriculum superiority."
            " Training-scale retention follows the current stage at each checkpoint;"
            " it does not certify three repeated evaluations at a newly entered stage."
        ),
    )
    (a.run / "retention_gate.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
