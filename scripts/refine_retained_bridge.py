"""Bounded native shooting preserving the exact ADC policy clock and full state."""

import argparse, copy, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    WANDB_MODE="disabled",
)
import torch
from scripts.unified_go2 import setup, actions
from scripts.build_retained_bridge import digest_bytes
from experiments.unified_go2.core import (
    Environment,
    adc_network,
    load,
    digest,
    save_json,
)
from experiments.front_catch.data import validate_state


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = Path("output/retained_go2_preparation_v1")
    cfg, data, bank, assets = setup(source, a.output)
    proof = json.loads((source / "bridge_provenance.json").read_text())
    for f in source.iterdir():
        if (
            f.is_file()
            and f.name not in ["bank.pt", "manifest.json"]
            and not f.name.startswith("shooting_")
        ):
            shutil.copy2(f, a.output / f.name)
    actual = {r["state_sha"] for r in proof["records"] if r["kind"] == "actual_ADC"}
    bank["train"]["bridge"] = [
        s for s in bank["train"]["bridge"] if digest_bytes(s["integration"]) in actual
    ]
    start = max(
        bank["train"]["bridge"], key=lambda s: s["values"]["momentum_sample"]["pitch"]
    )
    e = Environment(assets, cfg)
    adc = adc_network(load(source / "adc.pt"))
    records = []
    for candidate, (front, rear) in enumerate(
        [(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0), (0.08, 0.08)]
    ):
        e.restore(copy.deepcopy(start))
        e.cfg.seconds = 12.0
        obs, _ = e.observations()
        initial = copy.deepcopy(e.momentum_sample)
        safe = True
        states = []
        for k in range(15):
            command = actions(adc, obs[None])[0]
            if k < 6:
                for i, name in enumerate(e.names):
                    if name.startswith(("FL_", "FR_")) and "thigh" in name:
                        command[i] += front / 2
                    if name.startswith(("FL_", "FR_")) and "calf" in name:
                        command[i] -= front / 2
                    if name.startswith(("RL_", "RR_")) and "thigh" in name:
                        command[i] += rear / 2
            (obs, _), _, done, info = e.step(command.clip(-1, 1))
            safe &= validate_state(e)[0] and e._momentum_prefix_safe
            states.append(copy.deepcopy(e.state()))
            if done:
                break
        accepted = bool(safe and len(states) == 15)
        if accepted:
            bank["train"]["bridge"].extend(
                s
                for k, s in enumerate(states)
                if k % 3 == 2
                and s["values"]["momentum_sample"]["pitch"] >= initial["pitch"]
            )
        torch.save(states, a.output / f"shooting_{candidate}.pt")
        dt = e.m.opt.timestep
        records.append(
            dict(
                candidate=candidate,
                front_offset_rad=front,
                rear_offset_rad=rear,
                accepted=accepted,
                seconds=len(states) * 0.02,
                clock_preserved=True,
                initial_pitch=initial["pitch"],
                maximum_safe_pitch=max(
                    (
                        s["values"]["momentum_sample"]["pitch"]
                        for s in states
                        if s["values"].get("_momentum_prefix_safe", False)
                    ),
                    default=initial["pitch"],
                ),
                final_pitch=e.momentum_sample["pitch"],
                delta_L=e.momentum_sample["L_com"] - initial["L_com"],
                front_impulse=sum(r["front_moment_com"] * dt for r in e.physics_trace),
                rear_impulse=sum(r["rear_moment_com"] * dt for r in e.physics_trace),
            )
        )
    # Native state snapshots retain exact policy references and integration clocks.
    torch.save(bank, a.output / "bank.pt")
    save_json(a.output / "native_shooting.json", records)
    save_json(
        a.output / "manifest.json",
        {
            p.name: digest(p)
            for p in a.output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
    )
    print(
        "Exact-clock shooting",
        records,
        "bridge counts",
        len(bank["train"]["bridge"]),
        len(bank["evaluation"]["bridge"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
