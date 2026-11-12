"""Aggregate observed trials, merge/deduplicate recovery banks and render figures."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.recoverability.run import dump, wilson


def prepare(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    first = json.loads((a.phase1 / "candidate_results.json").read_text())
    if a.roll:
        first = [r for r in first if not r["case"].startswith("roll_")]
        first += json.loads((a.roll / "candidate_results.json").read_text())
    # Initial acquisition was conflated with recovery in the early measurement.
    # It never affected success; omit the early flag from the authoritative aggregate.
    for r in first:
        r.pop("recovered", None)
    summary = []
    for label in dict.fromkeys(r["controller"] for r in first):
        rs = [r for r in first if r["controller"] == label]
        n = sum(r["accepted"] for r in rs)
        k = sum(r["success"] for r in rs)
        nominal = [r for r in rs if r["case"] == "nominal"]
        summary.append(
            dict(
                controller=label,
                trials=len(rs),
                accepted=n,
                successes=k,
                probability=k / n if n else None,
                wilson95=wilson(k, n),
                nominal_success=sum(r["success"] for r in nominal) / len(nominal),
            )
        )
    dump(out / "balance_trials.json", first)
    dump(out / "balance_summary.json", summary)
    eligible = [r for r in summary if r["nominal_success"] >= 0.8]
    selected = max(eligible, key=lambda x: x["probability"]) if eligible else None
    dump(
        out / "reference_gate.json",
        dict(
            selected=selected,
            criterion="Nominal >=80%; rank by observed stress-trial success, not a statistical superiority claim",
        ),
    )
    cells = []
    region = []
    unique = {}
    arrays = {}
    state_count = 0
    for tag, directory in [("coarse", a.coarse), ("fine", a.fine)]:
        if directory is None:
            continue
        cc = json.loads((directory / "recovery_cells.json").read_text())
        for r in cc:
            cells.append(dict(source=tag, **r))
        rr = json.loads((directory / "validated_region.json").read_text())
        with np.load(directory / "recoverable_states.npz") as bank:
            for index, entry in enumerate(rr):
                values = {
                    k: bank[f"{index}_{k}"]
                    for k in ("integration", "target", "phase", "filtered_load")
                }
                fingerprint = hashlib.sha256(
                    b"".join(values[k].tobytes() for k in values)
                ).hexdigest()
                if fingerprint not in unique:
                    unique[fingerprint] = state_count
                    for k, value in values.items():
                        arrays[f"{state_count}_{k}"] = value
                    state_count += 1
                region.append(
                    dict(
                        **{k: v for k, v in entry.items() if k != "cell"},
                        cell=f"{tag}:{entry['cell']}",
                        unique_state=unique[fingerprint],
                        source_map=str(directory),
                    )
                )
    dump(out / "recovery_cells.json", cells)
    dump(out / "validated_region.json", region)
    np.savez_compressed(out / "unique_recoverable_states.npz", **arrays)
    dump(
        out / "bank_provenance.json",
        dict(
            region_memberships=len(region),
            unique_full_states=state_count,
            independent_seeds=sorted(set(r["seed"] for r in region)),
            caution="Prefix snapshots and repeated zero-offset controls are correlated, not independent recovery trials.",
        ),
    )
    for tag in ("coarse", "fine"):
        rs = [r for r in cells if r["source"] == tag and r["family"] == "pitch_omega"]
        if not rs:
            continue
        x = sorted(set(r["x"] for r in rs))
        y = sorted(set(r["y"] for r in rs))
        z = np.full((len(y), len(x)), np.nan)
        for r in rs:
            z[y.index(r["y"]), x.index(r["x"])] = (
                r["probability"] if r["probability"] is not None else np.nan
            )
        fig, ax = plt.subplots(figsize=(12, 6))
        mesh = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xticks(range(len(x)), [str(v) for v in x])
        ax.set_yticks(range(len(y)), [str(v) for v in y])
        for r in rs:
            ax.text(
                x.index(r["x"]),
                y.index(r["y"]),
                f"{r['successes']}/{r['accepted']}" + ("*" if r["validated"] else ""),
                ha="center",
                va="center",
                color="black" if (r["probability"] or 0) > 0.6 else "white",
                fontsize=8,
            )
        ax.set(
            xlabel="Initial pitch [degrees; categorical measured columns]",
            ylabel="Initial pitch angular velocity [rad/s]",
            title=f"{tag}: five-second recovery | feasible reset family\n* six-seed validation gate; no interpolation guarantee",
        )
        fig.colorbar(mesh, ax=ax, label="Conditional success probability")
        fig.tight_layout()
        fig.savefig(out / f"{tag}_recovery_heatmap.png")
        plt.close(fig)
    coarse_trials = json.loads((a.coarse / "recovery_trials.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, family, field, title in [
        (axes[0], "front_config", "q2", "Front calf configuration [rad]"),
        (
            axes[1],
            "com_shift",
            "com_support_x",
            "COM position relative to front support [m]",
        ),
    ]:
        rs = [r for r in cells if r["source"] == "coarse" and r["family"] == family]
        xx = []
        yy = []
        lo = []
        hi = []
        for r in rs:
            actual = [
                t["initial"][field]
                for t in coarse_trials
                if t["family"] == family and t["x"] == r["x"] and t["accepted"]
            ]
            if not actual or r["probability"] is None:
                continue
            xx.append(float(np.mean(actual)))
            yy.append(r["probability"])
            lo.append(r["probability"] - r["wilson95"][0])
            hi.append(r["wilson95"][1] - r["probability"])
        axis.errorbar(xx, yy, yerr=[lo, hi], fmt="o", capsize=4)
        axis.set(
            xlabel=title, ylabel="Five-second recovery probability", ylim=(-0.05, 1.05)
        )
        axis.grid()
    fig.tight_layout()
    fig.savefig(out / "configuration_com_recovery.png")
    plt.close(fig)
    cases = list(dict.fromkeys(r["case"] for r in first))
    labels = [r["controller"] for r in summary]
    z = np.full((len(labels), len(cases)), np.nan)
    for i, label in enumerate(labels):
        for j, case in enumerate(cases):
            rs = [
                r
                for r in first
                if r["controller"] == label and r["case"] == case and r["accepted"]
            ]
            if rs:
                z[i, j] = np.mean([r["success"] for r in rs])
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(z, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cases)), cases, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(cases)):
            ax.text(
                j,
                i,
                f"{z[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if z[i, j] < 0.5 else "black",
            )
    ax.set_title("Five-second success on physically feasible perturbation trials")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "balance_policy_comparison.png")
    plt.close(fig)
    print(
        "Prepared",
        out,
        "validated cells",
        sum(r["validated"] for r in cells),
        "unique states",
        state_count,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--phase1", type=Path, required=True)
    p.add_argument("--roll", type=Path)
    p.add_argument("--coarse", type=Path, required=True)
    p.add_argument("--fine", type=Path)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
