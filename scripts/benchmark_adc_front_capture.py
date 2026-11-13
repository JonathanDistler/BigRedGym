"""Benchmark capture-enabled workload on the machine actually running training."""

import argparse
import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.data import network, config, Curriculum
from experiments.front_catch.vector import Vector
from experiments.front_catch.learning import collect, hp_defaults
from experiments.morphology.learning.environment import Assets
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    m = json.loads((a.preparation / "manifest.json").read_text())
    bank = torch.load(
        a.preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    assets = Assets(a.output / "assets")
    rows = []
    for n in (16, 32):
        net, parent = network(m["initial"])
        cfg = config(parent, "B")
        cfg.adc_enabled = True
        cfg.capture_enabled = True
        cfg.comparison_suite = "anticipatory_v1"
        with Vector(assets, {"D": cfg}, bank, n, seed=42, workers=a.workers) as v:
            obs, critic = v.reset("D", Curriculum().mixture)
            _, obs, critic, _, _ = collect(
                net, v, "D", obs, critic, 2, Curriculum().mixture, hp_defaults()
            )
            t = time.monotonic()
            collect(net, v, "D", obs, critic, 8, Curriculum().mixture, hp_defaults())
            elapsed = time.monotonic() - t
            rows.append(
                dict(
                    num_envs=n,
                    workers=a.workers,
                    transitions=n * 8,
                    seconds=elapsed,
                    steps_per_second=n * 8 / elapsed,
                    rss_mib=v.memory() / 1024**2,
                )
            )
    peak = max(r["steps_per_second"] for r in rows)
    selected = min(
        (r for r in rows if r["steps_per_second"] >= 0.95 * peak),
        key=lambda r: r["num_envs"],
    )
    dump(
        a.output / "selection.json",
        dict(
            **selected,
            cpu_count=os.cpu_count(),
            affinity=sorted(os.sched_getaffinity(0)),
            measurements=rows,
            hostname=os.uname().nodename,
        ),
    )
    print(selected)


if __name__ == "__main__":
    main()
