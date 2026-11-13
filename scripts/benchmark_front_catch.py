"""Measure CPU collection, memory and PPO time before choosing environment counts."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from experiments.front_catch.data import network, config, Curriculum
from experiments.front_catch.vector import Vector
from experiments.front_catch.learning import collect, hp_defaults
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import update
from scripts.morphology_feasibility import save_csv
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--suite", choices=("original", "anticipatory_v1"), default="original")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--counts", nargs="+", type=int, default=[8, 16, 32, 64])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
    )
    torch.set_num_threads(1)
    manifest = json.loads((a.preparation / "manifest.json").read_text())
    bank = torch.load(
        a.preparation / "reset_bank.pt", weights_only=False, map_location="cpu"
    )
    assets = Assets(a.output / "assets")
    rows = []
    mixture = Curriculum().mixture
    settings = [("serial", 8, 0)] + [
        ("process" if a.workers else "serial_vector", n, min(a.workers, n))
        for n in a.counts
    ]
    for backend, n, workers in settings:
        net, saved = network(manifest["initial"])
        hp = hp_defaults()
        hp.update(epochs=1, bc_weight=0.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
        cfg = config(saved, "C" if a.suite == "anticipatory_v1" else "A")
        cfg.comparison_suite = a.suite
        start = time.monotonic()
        with Vector(
            assets, {"A": cfg}, bank, n, seed=42, workers=workers
        ) as vec:
            obs, critic = vec.reset("A", mixture)
            startup = time.monotonic() - start
            _, obs, critic, _, _ = collect(net, vec, "A", obs, critic, 4, mixture, hp)
            start = time.monotonic()
            batch, obs, critic, episodes, samples = collect(
                net, vec, "A", obs, critic, a.horizon, mixture, hp
            )
            seconds = time.monotonic() - start
            memory = vec.memory()
            start = time.monotonic()
            stats = update(net, optimizer, batch, hp)
            update_seconds = time.monotonic() - start
            rows.append(
                dict(
                    comparison_suite=a.suite,
                    backend=backend,
                    num_envs=n,
                    workers=workers,
                    transitions=n * a.horizon,
                    collection_seconds=seconds,
                    steps_per_second=n * a.horizon / seconds,
                    aggregate_rss_mib=memory / 1024**2,
                    ppo_update_seconds=update_seconds,
                    startup_seconds=startup,
                    post_update_kl=stats["post_update_kl"],
                )
            )
            save_csv(a.output / "benchmark.csv", rows)
            print(rows[-1], flush=True)
    eligible = [
        r for r in rows if r["backend"] != "serial" and r["num_envs"] in (16, 32)
    ]
    # Prefer the smaller count if throughput differs by less than five percent.
    peak = max(r["steps_per_second"] for r in eligible)
    selected = min(
        (r for r in eligible if r["steps_per_second"] >= 0.95 * peak),
        key=lambda x: x["num_envs"],
    )
    dump(
        a.output / "selection.json",
        dict(
            num_envs=selected["num_envs"],
            workers=selected["workers"],
            rationale="Smallest 16/32 count within 5% of fastest measured rate",
            backend=(
                "CPU MuJoCo processes" if a.workers else "Serial CPU vector fallback"
            ),
            gpu_mjx_validated=False,
            memory_metric="Aggregate RSS; shared pages can be counted twice",
        ),
    )


if __name__ == "__main__":
    main()
