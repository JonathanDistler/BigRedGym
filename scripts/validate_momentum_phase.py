"""Bounded physical validation, six teacher resets and seven post-takeoff schedules."""

import argparse
import copy
from pathlib import Path
import shutil
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.com_energy_phase.core import EnergyConfig, EnergyEnvironment
from experiments.com_energy_phase.data import reference, sha
from experiments.morphology.learning.environment import Assets
from experiments.momentum_phase.tools import (
    DEFAULT_INITIAL,
    network,
    configuration,
    rollout,
    balance_demonstrations,
    save_frames,
)
from experiments.momentum_phase.core import MomentumEnvironment
from scripts.run_com_energy_phase import dump
from scripts.morphology_feasibility import save_csv


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    p.add_argument(
        "--reference", type=Path, default=Path("output/recoverability_maps_20261112")
    )
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    test = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit_tests/test_momentum_phase.py",
            "tests/unit_tests/test_com_energy_phase.py",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    (out / "tests.txt").write_text(test.stdout + test.stderr)
    if test.returncode:
        raise RuntimeError("Unit validation failed")
    assets = Assets(out / "assets")
    ref = reference(EnergyEnvironment(assets, EnergyConfig()), a.reference)
    dump(out / "reference.json", ref)
    net, saved = network(a.initial)
    times = saved["config"]["phase_times"]
    env = MomentumEnvironment(assets, configuration(ref, times, "A"))
    rows = []
    all_frames = []
    frame_seeds = []
    observed = []
    events = []
    for seed in range(93000, 93006):
        info, frames = rollout(net, env, seed, capture=True)
        if not info["takeoff"]:
            raise RuntimeError("Initialization failed certified takeoff")
        rows.append(info)
        all_frames.extend(frames)
        frame_seeds.extend([seed] * len(frames))
        observed.extend(
            s for s in env.physics_trace if s["prefix_safe"] and s["support_valid"]
        )
        events.append(
            dict(
                seed=seed,
                takeoff_time=info["takeoff_time"],
                loading_time=next(
                    (s["time"] for s in env.physics_trace if s["phase"] >= 0.2), None
                ),
                peak_rear_extension_time=max(
                    [
                        s
                        for s in env.physics_trace
                        if s["prefix_safe"] and s["time"] <= info["takeoff_time"]
                    ],
                    key=lambda s: (s["joint_8"] + s["joint_11"]) / 2,
                )["time"],
            )
        )
        save_csv(out / f"teacher_{seed}.csv", env.physics_trace)
        print(
            "teacher",
            seed,
            "pitch",
            np.rad2deg(info["peak_certified_pitch"]),
            flush=True,
        )
    save_frames(out / "takeoff_demonstrations.npz", all_frames, frame_seeds)
    save_csv(out / "teacher_summary.csv", rows)
    dump(out / "phase_events.json", events)
    # Candidate schedules have fixed, small amplitudes and only begin after certificate.
    candidates = []
    frames_by_kind = {}
    later_frames = []
    later_seeds = []
    for kind in (
        "none",
        "rear_tuck",
        "rear_extend",
        "front_fold",
        "front_open",
        "front_compliance",
        "combined",
        "phase_filter",
    ):
        for seed in (92000, 92001):
            cfg = configuration(ref, times, "A")
            cfg.phase_filter = kind == "phase_filter"
            w = MomentumEnvironment(assets, cfg)
            info, frames = rollout(net, w, seed, kind, capture=True)
            candidates.append(info)
            frames_by_kind.setdefault(kind, []).extend(
                (frame, seed) for frame in frames if 0.4 <= frame[3] < 0.8
            )
            save_csv(out / f"schedule_{kind}_{seed}.csv", w.trace)
            save_csv(out / f"schedule_physics_{kind}_{seed}.csv", w.physics_trace)
            observed.extend(
                s for s in w.physics_trace if s["prefix_safe"] and s["support_valid"]
            )
            # Save actual prefixes; these are not handstand demonstrations.
            if kind != "none" and info["takeoff"]:
                for frame in frames:
                    if 0.4 <= frame[3] < 0.8:
                        later_frames.append(frame)
                        later_seeds.append(seed)
            print(
                "schedule",
                kind,
                seed,
                "pitch",
                np.rad2deg(info["peak_certified_pitch"]),
                flush=True,
            )
    save_csv(out / "schedule_summary.csv", candidates)
    if later_frames:
        save_frames(out / "post_takeoff_candidates.npz", later_frames, later_seeds)
    balance, meta = balance_demonstrations(assets, ref, times, a.reference)
    save_frames(out / "balance_demonstrations.npz", balance, [-1] * len(balance))
    dump(out / "balance_labels.json", meta)
    envelope = dict(
        com_cap=max(s["L_com"] for s in observed),
        support_cap=max(s["L_support"] for s in observed),
        source=(
            "Maximum positive momentum from safe supported prefixes; "
            "no speculative energy-equivalent expansion"
        ),
        sigma_support=max(0.2, 0.25 * max(s["L_support"] for s in observed)),
    )
    dump(out / "momentum_envelope.json", envelope)

    # Normalize offline features with the final measured envelope used by PPO.
    def normalize_frames(frames):
        result = []
        for actor, critic, action, phase in frames:
            actor = actor.copy()
            critic = critic.copy()
            actor[47] *= 1.3 / envelope["com_cap"]
            actor[48] *= 4.5 / envelope["support_cap"]
            critic[77] *= 1.3 / envelope["com_cap"]
            critic[78] *= 4.5 / envelope["support_cap"]
            result.append((actor, critic, action, phase))
        return result

    # Rewrite only files created in this new validation directory, before training.
    save_frames(
        out / "takeoff_demonstrations.npz", normalize_frames(all_frames), frame_seeds
    )
    save_frames(
        out / "balance_demonstrations.npz",
        normalize_frames(balance),
        [-1] * len(balance),
    )
    scores = {
        kind: np.mean(
            [r["peak_certified_pitch"] for r in candidates if r["schedule"] == kind]
        )
        for kind in frames_by_kind
    }
    best = max(scores, key=scores.get)
    selected = (
        frames_by_kind[best] if scores[best] > scores["none"] + np.deg2rad(1) else []
    )
    if selected:
        save_frames(
            out / "selected_rotation_demonstrations.npz",
            normalize_frames([f for f, seed in selected]),
            [seed for f, seed in selected],
        )
    dump(
        out / "selected_schedule.json",
        dict(
            schedule=best,
            mean_pitch_degrees=float(np.rad2deg(scores[best])),
            baseline_degrees=float(np.rad2deg(scores["none"])),
            supervised_frames=len(selected),
            scope="Actual safe post-takeoff prefix; no handstand claim",
        ),
    )
    fine_assets = copy.deepcopy(assets)
    fine_assets.models[1.0].opt.timestep *= 0.5
    fine = MomentumEnvironment(fine_assets, configuration(ref, times, "A"))
    fine_result, _ = rollout(net, fine, 93000)
    save_csv(out / "fine_timestep_physics.csv", fine.physics_trace)
    dump(
        out / "timestep_check.json",
        dict(
            native_dt=0.002,
            fine_dt=0.001,
            native_residual=rows[0]["max_momentum_residual_com"],
            fine_residual=fine_result["max_momentum_residual_com"],
            fine_pitch_degrees=float(np.rad2deg(fine_result["peak_certified_pitch"])),
        ),
    )
    max_residual = max(r["max_momentum_residual_com"] for r in rows)
    gate = dict(
        passed=max_residual < 0.15 * envelope["com_cap"],
        certified_initialization=all(r["takeoff"] for r in rows),
        max_integrated_com_momentum_residual=max_residual,
        relative_com_residual=max_residual / envelope["com_cap"],
        max_integrated_fixed_reference_residual=max(
            r["max_momentum_residual_fixed"] for r in rows
        ),
        units="kg m^2/s; native-dt trapezoidal ground moment integration",
        initial_checkpoint=str(a.initial),
        initial_sha256=sha(a.initial),
        initial_bc_sha256=saved["manifest"]["bc_sha256"],
        reference_sha256=ref["bank_sha256"],
        target_height=ref["height"],
        target_energy=ref["potential"],
        mass=ref["mass"],
        teacher_resets=6,
        schedule_rollouts=16,
        numerical_convergence_rollouts=1,
        extended_training=False,
        source_sha256={
            str(f): sha(f)
            for root in ("experiments/momentum_phase", "experiments/com_energy_phase")
            for f in Path(root).glob("*.py")
        },
    )
    dump(out / "validation.json", gate)
    source = out / "source"
    source.mkdir()
    for f in [Path(__file__), *Path("experiments/momentum_phase").glob("*.py")]:
        shutil.copyfile(f, source / f.name)
    print("validation", gate, flush=True)
    if not gate["passed"]:
        raise RuntimeError("Momentum balance gate failed; do not train")


if __name__ == "__main__":
    main()
