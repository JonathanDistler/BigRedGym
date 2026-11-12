"""Summarize the verified bounded experiment and export its best observed policy."""

# Markdown tables and links below intentionally occupy a single line.
# ruff: noqa: E501
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.handstand_v2.run import ORIGINAL


def main():
    root = Path("output")
    pilot = root / "handstand_v2_verified_pilot_20261111"
    diagnostics = root / "handstand_v2_verified_diagnostics_20261111"
    out = root / "handstand_v2_final"
    out.mkdir(exist_ok=False)
    comparison = json.loads((pilot / "comparison.json").read_text())
    gate = json.loads((pilot / "gate.json").read_text())
    stats = json.loads((pilot / "training.json").read_text())

    def means(rs, k, mode):
        return float(np.mean([r[k] for r in rs if r["stochastic"] == mode]))

    baseline = [means(comparison["before"], "certified", m) for m in (False, True)]
    curves = []
    for i in range(1, 11):
        files = list(pilot.glob(f"probe{i}_*_physics.csv"))
        if not files:
            continue
        for mode in (False, True):
            rs = []
            for f in files:
                if not f.name.endswith(f"_{int(mode)}_physics.csv"):
                    continue
                rows = list(csv.DictReader(f.open()))
                rs.append(
                    dict(
                        certified=max(
                            [
                                math.degrees(float(r["pitch"]))
                                for r in rows
                                if r["certified"] == "True"
                            ]
                            + [0.0]
                        ),
                        loaded=max(
                            [
                                math.degrees(float(r["pitch"]))
                                for r in rows
                                if r["support_valid"] == "True"
                            ]
                            + [0.0]
                        ),
                        takeoff=any(r["takeoff"] == "True" for r in rows),
                    )
                )
            curves.append(
                dict(
                    iteration=i,
                    stochastic=mode,
                    **{
                        k: float(np.mean([r[k] for r in rs]))
                        for k in ("certified", "loaded", "takeoff")
                    },
                )
            )
    candidates = []
    for iteration in sorted(set(r["iteration"] for r in curves)):
        rs = [r for r in curves if r["iteration"] == iteration]
        if len(rs) == 2 and min(r["takeoff"] for r in rs) >= 0.9:
            gains = [
                next(r["certified"] for r in rs if r["stochastic"] == m)
                - baseline[int(m)]
                for m in (False, True)
            ]
            candidates.append((min(gains), iteration))
    score, best_new = max(candidates)
    selected = pilot / f"policy_{best_new:04d}.pt" if score > 0 else ORIGINAL
    shutil.copyfile(selected, out / "selected_policy.pt")
    shutil.copyfile(pilot / f"policy_{best_new:04d}.pt", out / "best_new_policy.pt")
    (out / "selection.json").write_text(
        json.dumps(
            dict(
                best_observed_new_update=best_new,
                worst_mode_certified_gain=score,
                selected_source=str(selected),
                validation="Evaluation-seed selection only; no independent replication or deployment certification.",
            ),
            indent=2,
        )
    )
    (out / "pilot_curves.json").write_text(json.dumps(curves, indent=2))
    checkpoint = torch.load(
        pilot / "policy_0001.pt", map_location="cpu", weights_only=False
    )
    (out / "effective_training_hyperparameters.json").write_text(
        json.dumps(checkpoint["hp"], indent=2)
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for mode in (False, True):
        rs = [r for r in curves if r["stochastic"] == mode]
        label = "stochastic" if mode else "deterministic"
        for axis, k in zip(axes, ("certified", "loaded", "takeoff")):
            axis.plot(
                [r["iteration"] for r in rs],
                [r[k] for r in rs],
                marker="o",
                label=label,
            )
            axis.axhline(means(comparison["before"], k, mode), ls="--", alpha=0.5)
            axis.set_ylabel(k)
            axis.grid()
            axis.legend()
    axes[-1].set_xlabel("PPO update")
    fig.tight_layout()
    fig.savefig(out / "pilot_progression.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for axis, mode in zip(axes, (False, True)):
        for field in ("certified", "loaded"):
            before = [r[field] for r in comparison["before"] if r["stochastic"] == mode]
            after = [r[field] for r in comparison["after"] if r["stochastic"] == mode]
            axis.plot(range(70400, 70410), before, "o--", label="original " + field)
            axis.plot(range(70400, 70410), after, "o-", label="final " + field)
        axis.set_title("stochastic" if mode else "deterministic")
        axis.set_ylabel("degrees")
        axis.legend()
        axis.grid()
    fig.tight_layout()
    fig.savefig(out / "paired_pitch_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for axis, mode in zip(axes, (False, True)):
        for prefix in ("before", "after"):
            rows = list(
                csv.DictReader(
                    (pilot / f"{prefix}_70400_{int(mode)}_rewards.csv").open()
                )
            )
            for key in (
                "reward_airborne_rotation",
                "reward_v2_progress",
                "reward_v2_slip",
            ):
                axis.plot(
                    [float(r["time"]) for r in rows],
                    np.cumsum([float(r.get(key, 0)) for r in rows]),
                    label=prefix + " " + key,
                )
        axis.set_title("stochastic" if mode else "deterministic")
        axis.legend(fontsize=7)
        axis.grid()
    fig.tight_layout()
    fig.savefig(out / "before_after_rewards.png")
    plt.close(fig)
    table = "| Policy/mode | Takeoff | Loaded peak | Certified peak | Useful progress income |\n|---|---:|---:|---:|---:|\n"
    for group in ("before", "after"):
        for mode in (False, True):
            rs = comparison[group]
            table += f"| {group} / {'stochastic' if mode else 'deterministic'} | {means(rs, 'takeoff', mode):.0%} | {means(rs, 'loaded', mode):.2f}° | {means(rs, 'certified', mode):.2f}° | {means(rs, 'useful_feedback', mode):.3f} |\n"
    events = json.loads((diagnostics / "events.json").read_text())
    text = f"""# Go2 handstand V2 — final diagnosis and bounded pilot

**NO-GO for extended training.** No independent training-seed replication with
supported-pitch gain has been demonstrated. No extended training was launched.

The final implementation is isolated on `experiment/handstand_curriculum_v2`.
Original source files, original checkpoints and imported Molab results are unchanged.
The actor/critic/action interface remains 44/74/12.

## Root cause

Across all 80 matched original/imported checkpoint evaluations, the loaded-slip
budget is exceeded before peak pitch; the first invalid contact is FL_calf after
the peak in {sum(r["collision_after_peak"] for r in events)}/80 cases. No motor or
joint-limit event explains the plateau. On original seed 70400 the ordered events
are: slip limit .762 s, peak .880 s, front-support loss .932 s, calf contact 1.082 s.
Thus collision rate is an eventual outcome, not evidence of a pre-launch collision.

The old replacement progress potential is disabled by strict cumulative slip and
historical gap gates. The native rotation objective is disabled explicitly, but
native bounded takeoff `airborne_rotation` income remains (0.725 on the example
reset), plus 0.223 supported-flight income and a 3-point launch bonus. There is
some 25–34° signal; the problem is not a total absence of rotation reward.

True slip is material velocity at loaded ground contact points, distinct from
rolling center travel and flight displacement. Certification still retains the
2 cm slip budget, 4 cm drift budget, native collision exclusion and motor limits.
V2 supplies graded safe, bilateral-loaded shaping without calling it certification.
Current gaps reset on reloading; historical maximum gaps persist diagnostically.
A supported crossing before a later fall is retained as an event. Raw dynamic
crossings, loaded peaks, strict crossings, recovery and upright catch are separate.

## Verified training result

{table}

Final run: {len(stats)} PPO updates / {len(stats) * 1024:,} interactions, starting
from original update 36 with a fresh optimizer. Two-update smoke preceded it.
Maximum post-update KL: {max(r["post_update_kl"] for r in stats):.5f}; all logged
losses and metrics remained finite. No stage promotion beyond 1A is established.
Worst-mode final certified gain: {gate["supported_gain_degrees"]:.3f}°.
No stable handstand or successful front-support recovery was established.

Best observed new checkpoint among evaluated update windows: update {best_new},
worst-mode certified gain {score:.3f}°. Selected playback source: `{selected}`.
This selection uses the evaluation seeds and is not independent confirmation.
No raw airborne trajectory is labeled successful or used for imitation.

The initial pilot and smoke are preserved in their separate output directories.
A runner RNG issue was found during review: evaluation reset the training stream.
It was fixed with a Torch RNG isolation context, regression-tested, and the
smoke and pilot were repeated. The `verified_*` directories are the authoritative
runs. Exact per-run source snapshots and input hashes accompany each run. Effective
optimizer hyperparameters are exported here from the actual checkpoint; the older
run manifest was written before the learning-rate cap was applied.

## Physical model and feasibility

The variable-length pendulum is used only for eligible intervals; full MuJoCo
contact dynamics remain authoritative. Median eligible acceleration RMSE is about
7.96 rad/s². At the original peak, COM pendulum rotation is already moving away
from upright, support angular momentum is -2.16 kg m²/s, and approximate energy
deficit is 9.64 J. Rear contact moment is zero. Joint traces show front extension
and rear retraction, but these do not sustain useful COM rotation through the peak.

A bounded three-reset action assay increased loaded pitch from 34.37° to 37.11°
with front-calf flexion, while certified pitch changed only 29.40° to 29.59° and
recovery still failed. Combined yielding/retraction gave 36.93° loaded, 29.39°
certified. Rear retraction alone gave 34.30° loaded, 29.26° certified. These are
physical diagnostic probes, not policies, successful demonstrations, or a proof
that a 40–45° supported crossing is infeasible.

## Recommendation

Keep the original policy as the reference and do not increase the PPO budget.
The new dense signal is finite and nonzero, but the bounded pilot does not establish
reproducible safe supported-pitch improvement. A next separately reviewed experiment
should address pre-takeoff material slip and useful COM angular momentum together.
V2's new slip penalty currently acts after certified takeoff; early sliding can
already exhaust the strict episode budget before that penalty becomes active.
Do not hide this limitation by increasing the certification budget or counting
flight as support. The inherited action std is about .01 and PPO frequently stops
early for KL; exploration and optimization remain additional limitations.

## Artifacts and reproduction

- [Implementation and exact setup/training/playback commands](../docs/go2_handstand_curriculum_v2.md)
- [Molab notebook](../notebooks/molab_handstand_v2.ipynb)
- [Offline diagnostics and eight detailed policy/mode figures](handstand_v2_verified_diagnostics_20261111/)
- [Verified two-update smoke](handstand_v2_verified_smoke_20261111/)
- [Verified ten-update pilot and checkpoint exports](handstand_v2_verified_pilot_20261111/)
- [Paired seed comparison](handstand_v2_final/paired_pitch_comparison.png)
- [Pilot progression](handstand_v2_final/pilot_progression.png)
- [Before/after reward traces](handstand_v2_final/before_after_rewards.png)
- [Validation record](handstand_v2_validation.json): 30 targeted tests, 13 standalone
  archive tests, and Ruff passed. Legacy curriculum matches the initial snapshot
  byte-for-byte; the three tracked legacy learning sources have no diff.

Base commit: `cc0d7a8d2db621d0d9f866e9e1274e18daef245b`. New files remain
uncommitted; exact source ZIPs and file hashes capture the experiment beyond this
base commit, including the pre-existing untracked curriculum. The portable Molab
ZIP contains complete sources/resources, four input checkpoint copies, pinned
requirements and a per-file checksum manifest. The complete artifact ZIP also
contains diagnostics, plots, final checkpoint exports and MuJoCo GIF playback.
"""
    (root / "handstand_v2_final_report.md").write_text(text)
    for name in ("selected_policy", "best_new_policy"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.handstand_v2.playback",
                "--checkpoint",
                str(out / f"{name}.pt"),
                "--output",
                str(out / f"{name}.gif"),
            ],
            check=True,
        )
    print(text[:800])


if __name__ == "__main__":
    main()
