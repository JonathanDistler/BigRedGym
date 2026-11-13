"""Actual ADC-state sampling and bounded native shooting; no pose interpolation."""

import argparse, copy, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
)
import numpy as np, torch
from scripts.unified_go2 import setup, actions
from experiments.unified_go2.core import (
    Environment,
    adc_network,
    load,
    digest,
    capture_distance,
    save_json,
)
from experiments.front_catch.data import validate_state
from experiments.balance_feedback.actor import BalanceActor


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--certificate", type=Path, required=True)
    a = p.parse_args()
    certificate = json.loads(a.certificate.read_text())
    if not certificate["passed"]:
        raise RuntimeError("Both endpoint skills must pass before bridge work")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path("output/unified_go2_preparation_v1")
    cfg, data, old, assets = setup(source, a.output)
    bank = copy.deepcopy(old)
    for split in bank:
        bank[split]["bridge"] = []
    for f in source.iterdir():
        if f.is_file() and f.name not in ["manifest.json", "bank.pt"]:
            shutil.copy2(f, a.output / f.name)
    e = Environment(assets, cfg)
    adc = adc_network(load(source / "adc.pt"))
    balance = BalanceActor()
    balance.load_state_dict(load(source / "balance.pt")["network"])
    records = []
    samples = []
    for split, seeds in [("train", [95120, 95121]), ("evaluation", [96120, 96121])]:
        for seed in seeds:
            obs, _ = e.reset(1.0, seed)
            crossed = set()
            while True:
                (obs, _), _, done, info = e.step(actions(adc, obs[None])[0])
                valid, detail = validate_state(e)
                if (
                    valid
                    and e._momentum_prefix_safe
                    and e.tracker.confirmed
                    and e.row["rear_clearance"] > 0.015
                ):
                    for target in [25, 35, 45]:
                        if target not in crossed and e.row["pitch"] >= np.deg2rad(
                            target
                        ):
                            state = copy.deepcopy(e.state())
                            bank[split]["bridge"].append(state)
                            samples.append((split, seed, target, state))
                            crossed.add(target)
                            s = e.momentum_sample
                            records.append(
                                dict(
                                    kind="actual_ADC",
                                    split=split,
                                    seed=seed,
                                    threshold_crossing=target,
                                    time=float(e.d.time),
                                    pitch=float(s["pitch"]),
                                    omega=float(s["pitch_velocity"]),
                                    energy=float(s["mechanical_energy"]),
                                    L_com=float(s["L_com"]),
                                    front_moment=float(s["front_moment_com"]),
                                    capture_distance=capture_distance(
                                        obs, data["centers"]
                                    ),
                                    front_angles=e.d.qpos[e.qids[:6]].tolist(),
                                    state_sha=digest_bytes(state["integration"]),
                                )
                            )
                if done:
                    break
    # Balance takeover from actual states, never from an invented intermediate pose.
    selected = [
        max((x for x in samples if x[0] == split and x[1] == seed), key=lambda x: x[2])
        for split, seed in [("train", 95120), ("evaluation", 96120)]
    ]
    for split, seed, threshold, state in selected:
        obs, _ = e.restore_start(copy.deepcopy(state), "partial")
        e.cfg.seconds = 6.2
        while True:
            with torch.no_grad():
                action = balance(torch.tensor(obs)).numpy()
            (obs, _), _, done, info = e.step(action)
            if done:
                break
        records.append(
            dict(
                kind="balance_takeover",
                split=split,
                seed=seed,
                pitch_sample=threshold,
                hold=e.multi_best_hold,
                success=bool(e.ever_hold_success),
                termination=info["termination"],
            )
        )
    # A tiny shooting study changes only bounded position commands in native dynamics.
    split, seed, threshold, start = selected[0]
    for candidate, (front, rear) in enumerate(
        [(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0), (0.08, 0.08)]
    ):
        obs, _ = e.restore_start(copy.deepcopy(start), "partial")
        states = []
        safe = True
        initial = copy.deepcopy(e.momentum_sample)
        for k in range(15):
            action = actions(adc, obs[None])[0]
            if k < 6:
                for i, name in enumerate(e.names):
                    if name.startswith(("FL_", "FR_")):
                        if "thigh" in name:
                            action[i] += front / e.cfg.action_scale
                        if "calf" in name:
                            action[i] -= front / e.cfg.action_scale
                    elif "thigh" in name:
                        action[i] += rear / e.cfg.action_scale
            (obs, _), _, done, info = e.step(np.clip(action, -1, 1))
            safe &= validate_state(e)[0] and e._momentum_prefix_safe
            states.append(copy.deepcopy(e.state()))
            if done:
                break
        if safe and len(states) == 15:
            bank["train"]["bridge"].append(states[-1])
        torch.save(states, a.output / f"shooting_{candidate}.pt")
        dt = e.m.opt.timestep
        records.append(
            dict(
                kind="native_shooting",
                candidate=candidate,
                front_offset_rad=front,
                rear_offset_rad=rear,
                valid_complete_prefix=bool(safe and len(states) == 15),
                seconds=len(states) * 0.02,
                initial_pitch=initial["pitch"],
                final_pitch=e.momentum_sample["pitch"],
                delta_L=e.momentum_sample["L_com"] - initial["L_com"],
                front_impulse=sum(r["front_moment_com"] * dt for r in e.physics_trace),
                rear_impulse=sum(r["rear_moment_com"] * dt for r in e.physics_trace),
                termination=info["termination"],
            )
        )
    if not bank["train"]["bridge"] or not bank["evaluation"]["bridge"]:
        raise RuntimeError("No valid actual bridge states")
    torch.save(bank, a.output / "bank.pt")
    # Endpoint reset identity and observation/control conventions are unchanged.
    for split in old:
        for before, after in zip(old[split]["balance"], bank[split]["balance"]):
            assert np.array_equal(before["integration"], after["integration"])
    provenance = dict(
        source_preparation=str(source),
        source_manifest_sha=digest(source / "manifest.json"),
        retention_certificate=str(a.certificate),
        retention_sha=digest(a.certificate),
        endpoint_configs_identical=digest(source / "config.json")
        == digest(a.output / "config.json"),
        competent_demonstrations_identical=digest(source / "demonstrations.npz")
        == digest(a.output / "demonstrations.npz"),
        endpoint_balance_states_unchanged=True,
        records=records,
    )
    save_json(a.output / "bridge_provenance.json", provenance)
    save_json(
        a.output / "manifest.json",
        {
            f.name: digest(f)
            for f in a.output.iterdir()
            if f.is_file() and f.name != "manifest.json"
        },
    )
    print(
        "Real bridge bank",
        [(split, len(b["bridge"])) for split, b in bank.items()],
        flush=True,
    )
    print(records, flush=True)


def digest_bytes(x):
    import hashlib

    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
