"""Matched local tracking/viability probes; never claim a certified viability kernel."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.front_catch.data import config, sha, validate_state
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.adc_front_capture.curriculum import effective_inertia
from experiments.morphology.learning.environment import Assets
from scripts.run_adc_front_capture import initial
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--initial", type=Path)
    p.add_argument("--initial-mode")
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.preparation / "manifest.json").read_text())
    init = a.initial or Path(manifest["initial"])
    net, parent = initial(init, a.initial_mode)
    cfg = config(parent, "B")
    cfg.adc_enabled = True
    cfg.comparison_suite = "anticipatory_v1"
    env = CaptureEnvironment(Assets(a.output / "assets"), cfg)
    nominal_k = cfg.kp
    nominal_b = cfg.kd
    bank = torch.load(
        a.preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    records = []
    reset_audit = []
    for split, groups in bank.items():
        for kind, states in groups.items():
            for i, item in enumerate(states):
                env.restore_start(item["state"], kind)
                valid, details = validate_state(env)
                reset_audit.append(dict(split=split, kind=kind, index=i, **details))
    # Same actor, starts, horizon and original motor constraints. Loaded/post-takeoff
    # states sample distinct support configurations; no qpos modifications.
    for k in [nominal_k, 1.125 * nominal_k, 1.25 * nominal_k, 1.5 * nominal_k]:
        for kind in [
            "standing",
            "loaded",
            "pre_liftoff",
            "post_liftoff",
            "partial",
            "near_vertical",
        ]:
            for seed in range(3):
                if kind == "standing":
                    obs, _ = env.reset(1.0, 96000 + seed)
                else:
                    obs, _ = env.restore_start(
                        bank["train"][kind][seed % len(bank["train"][kind])]["state"],
                        kind,
                    )
                gains = env.set_dynamics(k, nominal_k, nominal_b)
                inertia = effective_inertia(env)
                errors = []
                saturations = []
                maxratio = 0.0
                elapsed = 0.0
                clear = []
                takeoff = False
                for step in range(60):
                    with torch.no_grad():
                        action = net.actor(torch.tensor(obs)).tanh().numpy()
                    (obs, _), _, done, info = env.step(action)
                    elapsed += cfg.dt
                    errors.append(
                        float(np.mean((env.target - env.d.qpos[env.qids]) ** 2))
                    )
                    saturations.append(info["saturation"])
                    maxratio = max(maxratio, info["max_torque_ratio"])
                    clear.append(env.trace[-1]["illegal_clearance"])
                    takeoff |= info["certified_standing_takeoff"]
                    if done:
                        break
                records.append(
                    dict(
                        k=k,
                        kind=kind,
                        seed=seed,
                        tracking_mse=np.mean(errors),
                        saturation=np.mean(saturations),
                        max_torque_ratio=maxratio,
                        survival=elapsed,
                        takeoff=takeoff,
                        peak_pitch=info["peak_collision_free_pitch"],
                        min_clearance=min(clear),
                        kd=gains["kd"],
                        effective_inertia=inertia.tolist(),
                        collision=info["head_collision"] or info["calf_collision"],
                    )
                )
                print("probe", k, kind, seed, elapsed, flush=True)
    base = [r for r in records if r["k"] == nominal_k]
    accepted = []
    summaries = []
    for k in sorted(set(r["k"] for r in records)):
        rows = [r for r in records if r["k"] == k]
        tracking = np.mean([r["tracking_mse"] for r in rows])
        survival = np.mean([r["survival"] for r in rows])
        takeoff = np.mean([r["takeoff"] for r in rows if r["kind"] == "standing"])
        useful = tracking < 0.98 * np.mean(
            [r["tracking_mse"] for r in base]
        ) or survival > 1.02 * np.mean([r["survival"] for r in base])
        safe = all(r["max_torque_ratio"] <= 1.000001 for r in rows)
        retained = takeoff >= np.mean(
            [r["takeoff"] for r in base if r["kind"] == "standing"]
        )
        # A local gate, not evidence of learned task improvement.
        if k > nominal_k and useful and safe and retained:
            accepted.append(k)
        summaries.append(
            dict(
                k=k,
                tracking_mse=tracking,
                survival=survival,
                takeoff=takeoff,
                accepted=k in accepted,
            )
        )
    dump(
        a.output / "probe.json",
        dict(
            initial_sha256=sha(init),
            nominal_k=nominal_k,
            nominal_b=nominal_b,
            accepted_stiffness=accepted,
            summaries=summaries,
            records=records,
            reset_audit=reset_audit,
            limitation=(
                "Local fixed-policy probes; no global viability "
                "or hardware identification claim"
            ),
        ),
    )


if __name__ == "__main__":
    main()
